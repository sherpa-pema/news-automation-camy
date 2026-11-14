"""Timeline utility module for forced alignment, caption chunking, and subtitle rendering.

Provides:
- MMSAligner: Calls tools/timeline_worker.py in .venv-tts via subprocess.
- FakeAligner: Pure Python mock aligner for tests and dry-runs.
- Text tokenization and phonetic mapping (handling pronunciations and number expansion).
- Caption chunking (1-2 lines, <=42 chars, 0.9s-5.0s, weak-word avoidance).
- SRT generation and syntax verification.
- Preview video generation with burned-in subtitles via FFmpeg.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.utils.editor_budget import count_syllables_in_text
from src.utils.text_normalizer import (
    NUMBER_WORDS as NUMBER_WORDS,
    WEAK_WORDS as WEAK_WORDS,
    contains_devanagari as contains_devanagari,
    expand_digits as expand_digits,
    number_to_spoken_words as number_to_spoken_words,
    tokenize_for_alignment,
)


def prepare_scene_tokens(
    narration: str,
    pronunciations: dict[str, str] | None = None,
    romanizer: Callable[[str], str] | None = None,
    warnings_out: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Tokenizes raw narration text into word items with TTS spoken form and MMS tokens.

    Delegates to src.utils.text_normalizer.tokenize_for_alignment so VoiceAgent
    and TimelineAgent share identical normalization rules.
    """
    return tokenize_for_alignment(
        narration,
        pronunciations=pronunciations,
        romanizer=romanizer,
        warnings_out=warnings_out,
    )


def estimate_word_weight(word_item: dict[str, Any]) -> float:
    """Estimates relative spoken duration weight from syllable count, length, and punctuation."""
    spoken = str(word_item.get("tts") or "").strip()
    if not spoken:
        tokens = word_item.get("tokens") or []
        spoken = " ".join(tokens) if tokens else str(word_item.get("orig") or word_item.get("w") or "")
    orig = str(word_item.get("orig") or word_item.get("w") or spoken)

    syllables = count_syllables_in_text(spoken)
    alpha_chars = len(re.sub(r"[^a-zA-Z]", "", spoken))
    if alpha_chars == 0:
        # Fallback for non-Latin scripts (e.g. unromanized Devanagari)
        alpha_chars = len(re.sub(r"[\s\W_]+", "", orig))
        syllables = max(1, (alpha_chars + 1) // 2) if alpha_chars > 0 else 1

    weight = max(1.0, 0.7 * syllables + 0.15 * alpha_chars)

    if re.search(r"[.!?][\"']?$", orig):
        weight += 0.6
    elif re.search(r"[,;:\-][\"']?$", orig):
        weight += 0.3

    return weight


def allocate_weighted_intervals(
    words: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
    active_ratio: float = 1.0,
) -> list[tuple[float, float]]:
    """Allocates proportional [start, end] timestamps across [start_sec, end_sec] by word weight."""
    n = len(words)
    if n == 0:
        return []

    s_bound = round(start_sec, 3)
    e_bound = round(max(start_sec, end_sec), 3)
    total_dur = max(0.0, e_bound - s_bound)

    weights = [estimate_word_weight(w) for w in words]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * n
        total_weight = float(n)

    intervals: list[tuple[float, float]] = []
    cum_weight = 0.0
    prev_slot_end = s_bound

    for idx, wt in enumerate(weights):
        cum_weight += wt
        slot_start = prev_slot_end
        if idx == n - 1:
            slot_end = e_bound
        else:
            slot_end = round(s_bound + total_dur * (cum_weight / total_weight), 3)
            slot_end = max(slot_start, min(e_bound, slot_end))

        if active_ratio >= 1.0:
            w_end = slot_end
        else:
            w_end = round(slot_start + (slot_end - slot_start) * active_ratio, 3)
            w_end = max(slot_start, min(slot_end, w_end))

        intervals.append((slot_start, w_end))
        prev_slot_end = slot_end

    return intervals


def interpolate_unplaced_words(
    words: list[dict[str, Any]],
    scene_duration: float,
    pad_before: float = 0.4,
    pad_after: float = 0.6,
) -> tuple[list[dict[str, Any]], int]:
    """Interpolates timestamps for unplaced words proportionally by word weight from neighbors.

    Returns the updated words list and the number of unplaced words that were interpolated.
    """
    n = len(words)
    if n == 0:
        return words, 0

    unplaced_count = sum(1 for w in words if w.get("unplaced") or w.get("start") is None)
    if unplaced_count == 0:
        return words, 0

    # If all words are unplaced, distribute proportionally across speech interval
    if unplaced_count == n:
        speech_start = min(pad_before, max(0.0, scene_duration - 0.1)) if scene_duration > 0 else 0.0
        speech_end = max(speech_start + 0.1, scene_duration - pad_after)
        if scene_duration > 0:
            speech_end = min(scene_duration, speech_end)
        intervals = allocate_weighted_intervals(words, speech_start, speech_end, active_ratio=1.0)
        for w, (w_start, w_end) in zip(words, intervals):
            w["start"] = w_start
            w["end"] = w_end
            w["score"] = 0.0
            w["unplaced"] = True
        return words, unplaced_count

    # Identify consecutive runs of unplaced words
    i = 0
    while i < n:
        if words[i].get("start") is not None and not words[i].get("unplaced"):
            i += 1
            continue

        # Start of an unplaced run
        run_start = i
        while i < n and (words[i].get("start") is None or words[i].get("unplaced")):
            i += 1
        run_end = i  # exclusive

        run_len = run_end - run_start

        # Find bounds from left and right neighbors
        if run_start > 0:
            left_bound = float(words[run_start - 1]["end"])
        else:
            left_bound = min(pad_before, max(0.0, scene_duration - 0.1)) if scene_duration > 0 else 0.0

        if run_end < n:
            right_bound = float(words[run_end]["start"])
        else:
            right_bound = max(left_bound + 0.1, scene_duration - pad_after)
            if scene_duration > 0:
                right_bound = min(scene_duration, right_bound)

        available_time = max(0.02 * run_len, right_bound - left_bound)
        right_bound = left_bound + available_time

        run_slice = words[run_start:run_end]
        intervals = allocate_weighted_intervals(run_slice, left_bound, right_bound, active_ratio=1.0)

        for k, (w_start, w_end) in enumerate(intervals):
            idx = run_start + k
            words[idx]["start"] = w_start
            words[idx]["end"] = w_end
            words[idx]["score"] = 0.0
            words[idx]["unplaced"] = True

    return words, unplaced_count


class TimelineAligner(ABC):
    """Abstract interface for timeline forced aligners."""

    @abstractmethod
    def align_scenes(
        self,
        scene_jobs: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Aligns multiple scenes.

        Args:
            scene_jobs: list of dicts with [id, audio_path, words]
            settings: engine configuration dict

        Returns:
            dict containing status, model_load_time_sec, device, scenes dict.
        """
        pass


class MMSAligner(TimelineAligner):
    """MMS_FA forced aligner running via isolated subprocess in .venv-tts."""

    def __init__(
        self,
        python_path: str | Path = ".venv-tts/Scripts/python.exe",
        worker_script: str | Path = "tools/timeline_worker.py",
        timeout_sec: int = 300,
    ) -> None:
        self.python_path = Path(python_path)
        self.worker_script = Path(worker_script)
        self.timeout_sec = timeout_sec

    def align_scenes(
        self,
        scene_jobs: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        if not scene_jobs:
            return {"status": "success", "model_load_time_sec": 0.0, "scenes": {}}

        if not self.python_path.exists():
            raise FileNotFoundError(
                f"Subprocess Python interpreter not found at: {self.python_path}. "
                "Ensure .venv-tts is set up."
            )
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"Timeline worker script not found at: {self.worker_script}"
            )

        job_payload = {
            "model_path": settings.get("model_path", "models/MMS_FA/model.pt"),
            "device": settings.get("device", "cuda:0"),
            "dtype": settings.get("dtype", "float32"),
            "scenes": scene_jobs,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            job_file = Path(tmpdir) / "job.json"
            result_file = Path(tmpdir) / "result.json"

            with open(job_file, "w", encoding="utf-8") as f:
                json.dump(job_payload, f, indent=2)

            cmd = [
                str(self.python_path),
                str(self.worker_script),
                "--job-file",
                str(job_file),
                "--result-file",
                str(result_file),
            ]

            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            wall_time = time.perf_counter() - t0

            if proc.returncode != 0 or not result_file.exists():
                raise RuntimeError(
                    f"Timeline worker failed with code {proc.returncode}:\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

            with open(result_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)

            if result_data.get("status") != "success":
                raise RuntimeError(f"Timeline alignment error: {result_data.get('error')}")

            result_data["wall_time_sec"] = round(wall_time, 3)
            return result_data


class FakeAligner(TimelineAligner):
    """Pure Python mock aligner for unit testing and dry runs without dependencies."""

    def align_scenes(
        self,
        scene_jobs: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        scenes_res: dict[str, Any] = {}
        alignment_times: dict[str, float] = {}

        pad_before = settings.get("pad_before", 0.4)
        pad_after = settings.get("pad_after", 0.6)

        for job in scene_jobs:
            sc_id = job["id"]
            words_data = job.get("words", [])
            # Read or approximate duration
            duration_sec = job.get("duration_sec", 10.0)

            speech_start = min(pad_before, max(0.0, duration_sec - 0.1)) if duration_sec > 0 else 0.0
            speech_end = max(speech_start + 0.1, duration_sec - pad_after)
            if duration_sec > 0:
                speech_end = min(duration_sec, speech_end)

            aligned_words = []
            if words_data:
                intervals = allocate_weighted_intervals(
                    words_data,
                    start_sec=speech_start,
                    end_sec=speech_end,
                    active_ratio=0.9,
                )
                for w, (w_start, w_end) in zip(words_data, intervals):
                    aligned_words.append({
                        "w": w.get("orig", ""),
                        "tts": w.get("tts", ""),
                        "start": w_start,
                        "end": w_end,
                        "score": 0.95,
                        "unplaced": False,
                    })

            scenes_res[sc_id] = {
                "duration_sec": duration_sec,
                "words": aligned_words,
            }
            alignment_times[sc_id] = 0.01

        return {
            "status": "success",
            "model_load_time_sec": 0.01,
            "wall_time_sec": 0.02,
            "device": "cpu",
            "dtype": "float32",
            "alignment_times_sec": alignment_times,
            "scenes": scenes_res,
        }


def get_timeline_aligner(
    aligner_name: str,
    config: dict[str, Any] | None = None,
) -> TimelineAligner:
    """Factory creating the requested TimelineAligner."""
    cfg = config or {}
    name = aligner_name.lower().strip()
    if name in ("fake", "mock"):
        return FakeAligner()
    elif name in ("mms", "mms_fa", "mms-fa"):
        python_path = cfg.get("subprocess_python", ".venv-tts/Scripts/python.exe")
        worker_script = cfg.get("worker_script", "tools/timeline_worker.py")
        timeout_sec = cfg.get("worker_timeout_sec", 300)
        return MMSAligner(python_path=python_path, worker_script=worker_script, timeout_sec=timeout_sec)
    else:
        raise ValueError(f"Unsupported aligner: {aligner_name}")


def split_chunk_into_lines(chunk_words: list[dict[str, Any]], max_chars: int = 42) -> str:
    """Splits a list of words into 1 or 2 lines adhering to max_chars and weak word avoidance."""
    full_text = " ".join(w["w"] for w in chunk_words)
    if len(full_text) <= max_chars or len(chunk_words) == 1:
        return full_text

    # Candidate splits between word i and word i+1
    best_split = None
    best_penalty = float("inf")

    n = len(chunk_words)
    for i in range(1, n):
        line1 = " ".join(w["w"] for w in chunk_words[:i])
        line2 = " ".join(w["w"] for w in chunk_words[i:])

        # Hard constraint: line length
        len1, len2 = len(line1), len(line2)
        if len1 > max_chars * 1.2 or len2 > max_chars * 1.2:
            continue

        # Balance penalty
        penalty = abs(len1 - len2)

        # Length violation penalty
        if len1 > max_chars:
            penalty += (len1 - max_chars) * 15
        if len2 > max_chars:
            penalty += (len2 - max_chars) * 15

        # Penalty if line 1 ends with a weak word
        last_word_l1 = re.sub(r"[^\w]", "", chunk_words[i - 1]["w"].lower())
        if last_word_l1 in WEAK_WORDS:
            penalty += 25

        # Bonus if split is right after punctuation (e.g. comma or dash)
        if re.search(r"[,;:\-]$", chunk_words[i - 1]["w"]):
            penalty -= 20

        if penalty < best_penalty:
            best_penalty = penalty
            best_split = (line1, line2)

    if best_split:
        return f"{best_split[0]}\n{best_split[1]}"

    # Fallback: simple mid split
    mid = n // 2
    return f"{' '.join(w['w'] for w in chunk_words[:mid])}\n{' '.join(w['w'] for w in chunk_words[mid:])}"


def chunk_scene_words_into_captions(
    scene_words: list[dict[str, Any]],
    scene_id: str,
    scene_start: float,
    scene_end: float,
    start_caption_id: int = 1,
    max_chars_per_line: int = 42,
    min_duration_sec: float = 0.9,
    max_duration_sec: float = 5.0,
    lead_in_sec: float = 0.05,
    hold_after_sec: float = 0.15,
    min_gap_sec: float = 0.04,
) -> tuple[list[dict[str, Any]], int]:
    """Chunks scene words into formatted subtitle blocks with bounds and overlap prevention."""
    if not scene_words:
        return [], start_caption_id

    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []

    for word in scene_words:
        current_chunk.append(word)
        w_orig = word["w"]

        # Sentence end punctuation
        is_sentence_end = bool(re.search(r"[.!?]$", w_orig))
        # Phrase break punctuation
        is_phrase_break = bool(re.search(r"[,;:\-]$", w_orig))

        current_duration = current_chunk[-1]["end"] - current_chunk[0]["start"]
        current_text = " ".join(w["w"] for w in current_chunk)

        # Break condition 1: Sentence end
        if is_sentence_end:
            chunks.append(current_chunk)
            current_chunk = []
            continue

        # Break condition 2: Max duration exceeded
        if current_duration >= max_duration_sec:
            chunks.append(current_chunk)
            current_chunk = []
            continue

        # Break condition 3: Phrase break with sufficient duration or text length
        if is_phrase_break and (current_duration >= min_duration_sec or len(current_text) >= max_chars_per_line):
            chunks.append(current_chunk)
            current_chunk = []
            continue

        # Break condition 4: Text too long for 2 lines
        if len(current_text) >= max_chars_per_line * 2:
            chunks.append(current_chunk)
            current_chunk = []
            continue

    if current_chunk:
        chunks.append(current_chunk)

    # Build caption items with timing
    captions: list[dict[str, Any]] = []
    curr_id = start_caption_id
    last_end = scene_start

    for chunk in chunks:
        raw_start = chunk[0]["start"]
        raw_end = chunk[-1]["end"]

        c_start = max(scene_start, round(raw_start - lead_in_sec, 3))
        c_end = min(scene_end, round(raw_end + hold_after_sec, 3))

        # Enforce minimum duration
        if c_end - c_start < min_duration_sec:
            c_end = min(scene_end, round(c_start + min_duration_sec, 3))
            if c_end - c_start < min_duration_sec:
                c_start = max(scene_start, round(c_end - min_duration_sec, 3))

        # Prevent overlap with preceding caption
        if c_start < last_end + min_gap_sec:
            c_start = round(last_end + min_gap_sec, 3)
            if c_end <= c_start:
                c_end = min(scene_end, round(c_start + min_duration_sec, 3))

        c_text = split_chunk_into_lines(chunk, max_chars=max_chars_per_line)

        captions.append({
            "id": curr_id,
            "scene_id": scene_id,
            "start": round(c_start, 3),
            "end": round(c_end, 3),
            "duration": round(c_end - c_start, 3),
            "text": c_text,
            "words": [
                {
                    "w": w["w"],
                    "tts": w.get("tts", ""),
                    "start": w["start"],
                    "end": w["end"],
                    "score": w.get("score", 1.0),
                }
                for w in chunk
            ],
        })

        last_end = c_end
        curr_id += 1

    return captions, curr_id


def format_srt_time(seconds: float) -> str:
    """Formats seconds into SRT timestamp string HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    secs = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_content(captions: list[dict[str, Any]]) -> str:
    """Generates standard UTF-8 SubRip (SRT) format text."""
    blocks: list[str] = []
    for c in captions:
        idx = c["id"]
        start_str = format_srt_time(c["start"])
        end_str = format_srt_time(c["end"])
        text = c["text"]
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
    return "\n".join(blocks)


def validate_srt_syntax(srt_content: str) -> tuple[bool, str | None]:
    """Validates that SRT text satisfies strict SubRip syntax and monotonicity."""
    blocks = [b.strip() for b in srt_content.strip().split("\n\n") if b.strip()]
    if not blocks:
        return False, "SRT content is empty."

    prev_end_ms = -1
    for block_idx, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 3:
            return False, f"SRT block {block_idx} has fewer than 3 lines: {block}"

        # 1. Check index
        try:
            idx = int(lines[0].strip())
            if idx != block_idx:
                return False, f"SRT block index mismatch: expected {block_idx}, got {idx}"
        except ValueError:
            return False, f"SRT block {block_idx} index line is not an integer: {lines[0]}"

        # 2. Check time line
        time_match = re.match(
            r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})$",
            lines[1].strip(),
        )
        if not time_match:
            return False, f"SRT block {block_idx} timestamp line invalid: {lines[1]}"

        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
        start_ms = ((h1 * 60 + m1) * 60 + s1) * 1000 + ms1
        end_ms = ((h2 * 60 + m2) * 60 + s2) * 1000 + ms2

        if start_ms >= end_ms:
            return False, f"SRT block {block_idx} has start_ms >= end_ms ({start_ms} >= {end_ms})"

        if start_ms < prev_end_ms:
            return False, f"SRT block {block_idx} overlaps with previous block ({start_ms} < {prev_end_ms})"

        prev_end_ms = end_ms

        # 3. Check text lines
        text = "\n".join(lines[2:]).strip()
        if not text:
            return False, f"SRT block {block_idx} has empty subtitle text."

    return True, None


def generate_timeline_preview_video(
    narration_path: Path,
    srt_path: Path,
    output_mp4: Path,
    font_size: int = 22,
) -> dict[str, Any]:
    """Generates 1280x720 25fps black background preview video with burned-in captions."""
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    rel_narration = os.path.relpath(narration_path, output_mp4.parent)
    rel_srt = os.path.relpath(srt_path, output_mp4.parent)

    # Force style for crisp readable subtitles
    style = (
        f"Fontname=Arial,FontSize={font_size},"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,"
        "Outline=2,Alignment=2,MarginV=35"
    )
    filter_expr = f"subtitles={rel_srt}:force_style='{style}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:r=25",
        "-i", str(rel_narration),
        "-vf", filter_expr,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4.name),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(output_mp4.parent),
        capture_output=True,
        text=True,
    )
    wall_time = time.perf_counter() - t0

    if proc.returncode != 0 or not output_mp4.exists():
        raise RuntimeError(
            f"FFmpeg timeline preview generation failed (code {proc.returncode}):\n{proc.stderr}"
        )

    file_size_bytes = output_mp4.stat().st_size
    return {
        "file": str(output_mp4),
        "size_bytes": file_size_bytes,
        "render_time_sec": round(wall_time, 2),
    }
