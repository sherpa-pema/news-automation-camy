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
from typing import Any, TypedDict

from src.utils.editor_budget import count_syllables_in_text
from src.utils.text_normalizer import (
    NUMBER_WORDS as NUMBER_WORDS,
    WEAK_WORDS as WEAK_WORDS,
    contains_devanagari as contains_devanagari,
    expand_digits as expand_digits,
    number_to_spoken_words as number_to_spoken_words,
    tokenize_for_alignment,
)


class WordTiming(TypedDict, total=False):
    """Word timing model representing a single token in the timeline."""
    w: str
    tts: str
    start: float | None
    end: float | None
    score: float
    unplaced: bool
    low_confidence: bool
    tokens: list[str]


class SceneJob(TypedDict, total=False):
    """Specification of an alignment job for a single scene."""
    id: str
    audio_path: str
    words: list[dict[str, Any]]
    duration_sec: float
    start_sec: float
    end_sec: float
    pad_before: float
    pad_after: float
    headline: str
    narration: str


class CaptionCue(TypedDict, total=False):
    """A single subtitle caption cue."""
    id: int
    scene_id: str
    start: float
    end: float
    text: str
    words: list[str]


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

        # Resolve model path and all scene audio paths to absolute paths
        model_path_str = str(Path(settings.get("model_path", "models/MMS_FA/model.pt")).resolve())
        normalized_scenes: list[dict[str, Any]] = []
        total_audio_sec = 0.0
        for sc in scene_jobs:
            sc_copy = dict(sc)
            sc_copy["audio_path"] = str(Path(sc["audio_path"]).resolve())
            total_audio_sec += float(sc.get("duration_sec", 0.0))
            normalized_scenes.append(sc_copy)

        job_payload = {
            "model_path": model_path_str,
            "device": settings.get("device", "cuda:0"),
            "dtype": settings.get("dtype", "float32"),
            "scenes": normalized_scenes,
        }

        # Scale timeout by total audio length (at least timeout_sec, plus extra for long runs)
        effective_timeout = max(self.timeout_sec, int(total_audio_sec * 3.0) + 120)

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
                timeout=effective_timeout,
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


def can_fit_lines(chunk_words: list[dict[str, Any]], max_chars: int = 42, max_lines: int = 2) -> bool:
    """Checks whether chunk_words can be formatted within max_lines, each of max_chars length."""
    if not chunk_words:
        return True
    full_text = " ".join(w["w"] for w in chunk_words)
    if len(full_text) <= max_chars:
        return True
    if max_lines <= 1:
        return False
    n = len(chunk_words)
    for i in range(1, n):
        len1 = sum(len(w["w"]) for w in chunk_words[:i]) + (i - 1)
        len2 = sum(len(w["w"]) for w in chunk_words[i:]) + (n - i - 1)
        if len1 <= max_chars and len2 <= max_chars:
            return True
    return False


def split_chunk_into_lines(
    chunk_words: list[dict[str, Any]],
    max_chars: int = 42,
    max_lines: int = 2,
) -> str:
    """Splits a list of words into 1 or 2 lines adhering strictly to max_chars and weak word avoidance."""
    full_text = " ".join(w["w"] for w in chunk_words)
    if len(full_text) <= max_chars or len(chunk_words) == 1:
        return full_text

    best_split = None
    best_penalty = float("inf")

    n = len(chunk_words)
    for i in range(1, n):
        line1 = " ".join(w["w"] for w in chunk_words[:i])
        line2 = " ".join(w["w"] for w in chunk_words[i:])

        len1, len2 = len(line1), len(line2)
        if len1 > max_chars or len2 > max_chars:
            continue

        penalty = abs(len1 - len2) * 1.0

        last_word_l1 = re.sub(r"[^\w]", "", chunk_words[i - 1]["w"].lower())
        if last_word_l1 in WEAK_WORDS:
            penalty += 25.0

        if re.search(r"[,;:\-]$", chunk_words[i - 1]["w"]):
            penalty -= 20.0

        if penalty < best_penalty:
            best_penalty = penalty
            best_split = (line1, line2)

    if best_split:
        return f"{best_split[0]}\n{best_split[1]}"

    # Fallback: greedy packing respecting max_chars
    lines: list[str] = []
    curr_line: list[str] = []
    curr_len = 0
    for w in chunk_words:
        w_text = w["w"]
        w_len = len(w_text)
        if not curr_line:
            curr_line.append(w_text)
            curr_len = w_len
        elif curr_len + 1 + w_len <= max_chars:
            curr_line.append(w_text)
            curr_len += 1 + w_len
        else:
            lines.append(" ".join(curr_line))
            curr_line = [w_text]
            curr_len = w_len
    if curr_line:
        lines.append(" ".join(curr_line))

    return "\n".join(lines)


def split_sentence_into_chunks(
    words: list[dict[str, Any]],
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    min_duration_sec: float = 0.9,
    max_duration_sec: float = 5.0,
    target_cps: float = 17.0,
    max_cps: float = 20.0,
    lead_in_sec: float = 0.05,
    hold_after_sec: float = 0.15,
) -> list[list[dict[str, Any]]]:
    """Recursively splits a sentence that exceeds duration, line, or reading speed limits."""
    if not words:
        return []

    raw_dur = words[-1]["end"] - words[0]["start"]
    est_dur = raw_dur + lead_in_sec + hold_after_sec
    text = " ".join(w["w"] for w in words)
    est_cps = len(text) / max(min_duration_sec, est_dur)

    # If the whole sequence fits cleanly, don't split!
    if est_dur <= max_duration_sec and can_fit_lines(words, max_chars_per_line, max_lines) and est_cps <= max_cps:
        return [words]

    n = len(words)
    if n <= 1:
        return [words]

    best_k = None
    best_penalty = float("inf")

    conjunctions_prepositions = {
        "and", "but", "that", "with", "in", "to", "for", "on", "at", "from",
        "by", "as", "into", "through", "under", "over", "after", "before", "of"
    }

    for k in range(1, n):
        left = words[:k]
        right = words[k:]

        left_raw_dur = left[-1]["end"] - left[0]["start"]
        left_est_dur = left_raw_dur + lead_in_sec + hold_after_sec
        right_raw_dur = right[-1]["end"] - right[0]["start"]
        right_est_dur = right_raw_dur + lead_in_sec + hold_after_sec

        # Hard constraints for left chunk
        if left_est_dur > max_duration_sec or not can_fit_lines(left, max_chars_per_line, max_lines):
            continue

        left_text = " ".join(w["w"] for w in left)
        left_cps = len(left_text) / max(min_duration_sec, left_est_dur)
        right_text = " ".join(w["w"] for w in right)
        right_cps = len(right_text) / max(min_duration_sec, right_est_dur)

        penalty = 0.0

        # CPS penalty: target 17, max 20
        if left_cps > target_cps:
            penalty += (left_cps - target_cps) * 5.0
        if left_cps > max_cps:
            penalty += (left_cps - max_cps) * 200.0 + 300.0
        if right_cps > max_cps:
            penalty += (right_cps - max_cps) * 100.0

        # Avoid creating chunks with duration too short
        if left_est_dur < min_duration_sec:
            penalty += 150.0 * (min_duration_sec - left_est_dur)
        if right_est_dur < min_duration_sec:
            penalty += 150.0 * (min_duration_sec - right_est_dur)

        # Punctuation bonus on left end
        if re.search(r"[,;:\-]$", left[-1]["w"]):
            penalty -= 40.0

        # Weak word penalty on left end
        last_word_cleaned = re.sub(r"[^\w]", "", left[-1]["w"].lower())
        if last_word_cleaned in WEAK_WORDS:
            penalty += 50.0

        # Natural start bonus on right beginning
        first_right_cleaned = re.sub(r"[^\w]", "", right[0]["w"].lower())
        if first_right_cleaned in conjunctions_prepositions:
            penalty -= 20.0

        # Ideal length preference (~45-60 chars per cue)
        penalty += abs(len(left_text) - 52.0) * 0.6

        # Bonus if right also fits directly into a single cue!
        if right_est_dur <= max_duration_sec and can_fit_lines(right, max_chars_per_line, max_lines) and right_cps <= max_cps:
            penalty -= 60.0

        if penalty < best_penalty:
            best_penalty = penalty
            best_k = k

    if best_k is None:
        for k in range(n - 1, 0, -1):
            left = words[:k]
            left_est_dur = (left[-1]["end"] - left[0]["start"]) + lead_in_sec + hold_after_sec
            if left_est_dur <= max_duration_sec and can_fit_lines(left, max_chars_per_line, max_lines):
                best_k = k
                break
        if best_k is None:
            best_k = n // 2

    return split_sentence_into_chunks(
        words[:best_k], max_chars_per_line, max_lines, min_duration_sec,
        max_duration_sec, target_cps, max_cps, lead_in_sec, hold_after_sec
    ) + split_sentence_into_chunks(
        words[best_k:], max_chars_per_line, max_lines, min_duration_sec,
        max_duration_sec, target_cps, max_cps, lead_in_sec, hold_after_sec
    )


def chunk_scene_words_into_captions(
    scene_words: list[dict[str, Any]],
    scene_id: str,
    scene_start: float,
    scene_end: float,
    start_caption_id: int = 1,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    min_duration_sec: float = 0.9,
    max_duration_sec: float = 5.0,
    lead_in_sec: float = 0.05,
    hold_after_sec: float = 0.15,
    min_gap_sec: float = 0.04,
    target_cps: float = 17.0,
    max_cps: float = 20.0,
) -> tuple[list[dict[str, Any]], int]:
    """Chunks scene words into formatted subtitle blocks adhering strictly to bounds, duration, line, and CPS limits."""
    if not scene_words:
        return [], start_caption_id

    # 1. Group words into sentences
    sentences: list[list[dict[str, Any]]] = []
    curr_sent: list[dict[str, Any]] = []
    for w in scene_words:
        curr_sent.append(w)
        if re.search(r"[.!?।॥]$", w["w"]):
            sentences.append(curr_sent)
            curr_sent = []
    if curr_sent:
        sentences.append(curr_sent)

    # 2. Chunk each sentence
    raw_chunks: list[list[dict[str, Any]]] = []
    for sent in sentences:
        s_chunks = split_sentence_into_chunks(
            sent,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            target_cps=target_cps,
            max_cps=max_cps,
            lead_in_sec=lead_in_sec,
            hold_after_sec=hold_after_sec,
        )
        raw_chunks.extend(s_chunks)

    # 3. Post-process: merge adjacent short chunks if they fit within limits
    chunks: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(raw_chunks):
        c = raw_chunks[idx]
        c_dur = (c[-1]["end"] - c[0]["start"]) + lead_in_sec + hold_after_sec
        if idx + 1 < len(raw_chunks):
            next_c = raw_chunks[idx + 1]
            combined = c + next_c
            comb_dur = (combined[-1]["end"] - combined[0]["start"]) + lead_in_sec + hold_after_sec
            comb_text = " ".join(w["w"] for w in combined)
            comb_cps = len(comb_text) / max(min_duration_sec, comb_dur)
            if (c_dur < min_duration_sec and comb_dur <= max_duration_sec and
                can_fit_lines(combined, max_chars_per_line, max_lines) and comb_cps <= max_cps):
                chunks.append(combined)
                idx += 2
                continue
        # Also check if c is short and can merge backwards into chunks[-1]
        if chunks and c_dur < min_duration_sec:
            comb_prev = chunks[-1] + c
            comb_prev_dur = (comb_prev[-1]["end"] - comb_prev[0]["start"]) + lead_in_sec + hold_after_sec
            comb_prev_text = " ".join(w["w"] for w in comb_prev)
            comb_prev_cps = len(comb_prev_text) / max(min_duration_sec, comb_prev_dur)
            if (comb_prev_dur <= max_duration_sec and
                can_fit_lines(comb_prev, max_chars_per_line, max_lines) and comb_prev_cps <= max_cps):
                chunks[-1] = comb_prev
                idx += 1
                continue
        chunks.append(c)
        idx += 1

    # 4. Timing & Overlap Resolution
    captions: list[dict[str, Any]] = []
    curr_id = start_caption_id

    for i, chunk in enumerate(chunks):
        raw_start = chunk[0]["start"]
        raw_end = chunk[-1]["end"]
        text = split_chunk_into_lines(chunk, max_chars=max_chars_per_line, max_lines=max_lines)
        text_clean = text.replace("\n", " ")
        text_len = len(text_clean)

        c_start = max(scene_start, round(raw_start - lead_in_sec, 3))
        c_end = min(scene_end, round(raw_end + hold_after_sec, 3))

        # Check silence limit to next chunk or scene end
        if i + 1 < len(chunks):
            next_raw_start = chunks[i + 1][0]["start"]
            silence_limit = max(c_end, round(next_raw_start - lead_in_sec - min_gap_sec, 3))
        else:
            silence_limit = scene_end

        # Extend c_end into following silence to satisfy min_duration_sec
        if c_end - c_start < min_duration_sec:
            c_end = min(silence_limit, round(c_start + min_duration_sec, 3))

        # Extend c_end into following silence to satisfy target_cps or max_cps
        needed_dur = text_len / target_cps
        max_dur_cap = min(scene_end, round(c_start + max_duration_sec, 3))
        c_end = min(silence_limit, max_dur_cap, max(c_end, round(c_start + needed_dur, 3)))

        # Invariant: c_end >= raw_end
        c_end = max(c_end, raw_end)
        c_end = min(scene_end, c_end)

        # Resolve overlap with previous caption
        prev_limit = scene_start
        if captions:
            prev = captions[-1]
            prev_prev_limit = scene_start
            if len(captions) >= 2:
                prev_prev_limit = round(captions[-2]["end"] + min_gap_sec, 3)

            if prev["end"] > c_start - min_gap_sec:
                prev_raw_end = prev["words"][-1]["end"]
                capped_prev_end = round(c_start - min_gap_sec, 3)
                if capped_prev_end >= prev_raw_end:
                    prev["end"] = capped_prev_end
                else:
                    c_start = max(scene_start, round(prev_raw_end + min_gap_sec, 3))
                    if c_start > raw_start:
                        c_start = raw_start
                        prev["end"] = min(raw_start, prev["end"])
                    else:
                        prev["end"] = round(c_start - min_gap_sec, 3)

                prev["duration"] = round(prev["end"] - prev["start"], 3)
                # If capping reduced prev duration below max_cps or min_duration_sec,
                # extend prev["start"] into preceding silence if available
                prev_text_len = len(prev["text"].replace("\n", " "))
                prev_needed_dur = max(min_duration_sec, prev_text_len / max_cps)
                if prev["duration"] < prev_needed_dur and prev["start"] > prev_prev_limit:
                    prev_extra = round(prev_needed_dur - prev["duration"], 3)
                    prev_lead = min(round(prev["start"] - prev_prev_limit, 3), prev_extra)
                    prev["start"] = round(prev["start"] - prev_lead, 3)
                    prev["duration"] = round(prev["end"] - prev["start"], 3)

                # If prev duration is still too short, check if captions[-2] can cede end slack
                if prev["duration"] < prev_needed_dur and len(captions) >= 2:
                    p_prev = captions[-2]
                    p_raw_end = p_prev["words"][-1]["end"]
                    p_min_dur = max(min_duration_sec, len(p_prev["text"].replace("\n", " ")) / max_cps)
                    p_slack = max(0.0, round(p_prev["end"] - max(p_raw_end, p_prev["start"] + p_min_dur), 3))
                    needed = round(prev_needed_dur - prev["duration"], 3)
                    cede = min(p_slack, needed)
                    if cede > 0:
                        p_prev["end"] = round(p_prev["end"] - cede, 3)
                        p_prev["duration"] = round(p_prev["end"] - p_prev["start"], 3)
                        prev["start"] = round(prev["start"] - cede, 3)
                        prev["duration"] = round(prev["end"] - prev["start"], 3)

            prev_limit = round(prev["end"] + min_gap_sec, 3)

        # If current duration is too short for max_cps, extend c_start into preceding silence
        needed_dur = text_len / max_cps
        curr_dur = c_end - c_start
        if curr_dur < needed_dur and c_start > prev_limit:
            extra_needed = round(needed_dur - curr_dur, 3)
            lead_extra = min(round(c_start - prev_limit, 3), extra_needed)
            c_start = round(c_start - lead_extra, 3)
            curr_dur = c_end - c_start

        # If current duration is still too short for max_cps, check if prev["end"] can cede shared silence
        if curr_dur < needed_dur and captions:
            prev = captions[-1]
            prev_raw_end = prev["words"][-1]["end"]
            prev_text_len = len(prev["text"].replace("\n", " "))
            prev_min_dur = max(min_duration_sec, prev_text_len / max_cps)
            prev_slack = max(0.0, round(prev["end"] - max(prev_raw_end, prev["start"] + prev_min_dur), 3))
            extra_needed = round(needed_dur - curr_dur, 3)
            cede = min(prev_slack, extra_needed)
            if cede > 0:
                prev["end"] = round(prev["end"] - cede, 3)
                prev["duration"] = round(prev["end"] - prev["start"], 3)
                c_start = round(c_start - cede, 3)
                curr_dur = c_end - c_start

        c_dur = round(c_end - c_start, 3)

        # Invariant checks
        if c_start >= c_end:
            raise ValueError(
                f"Invalid cue timing in scene '{scene_id}': start ({c_start}s) >= end ({c_end}s) "
                f"for text: {repr(text)}"
            )

        captions.append({
            "id": curr_id,
            "scene_id": scene_id,
            "start": round(c_start, 3),
            "end": round(c_end, 3),
            "duration": c_dur,
            "text": text,
            "words": [
                {
                    "w": w["w"],
                    "tts": w.get("tts", ""),
                    "start": w["start"],
                    "end": w["end"],
                    "score": w.get("score", 1.0),
                    "unplaced": bool(w.get("unplaced", False)),
                    "low_confidence": bool(w.get("low_confidence", False)),
                }
                for w in chunk
            ],
        })
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


def audit_captions(
    captions: list[dict[str, Any]],
    scenes: list[dict[str, Any]] | None = None,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    min_duration_sec: float = 0.9,
    max_duration_sec: float = 5.0,
    target_cps: float = 17.0,
    max_cps: float = 20.0,
    late_threshold_ms: float = 50.0,
) -> dict[str, Any]:
    """Pure validation and audit function for generated captions.

    Checks:
        - Late starts (relative to first spoken word)
        - Overlaps between adjacent captions
        - Duration bounds (min_duration_sec, max_duration_sec)
        - Line constraints (max_chars_per_line, max_lines)
        - Reading speed (characters per second vs max_cps)
        - Word conservation across scenes (if scenes list provided)
    """
    violations: list[dict[str, Any]] = []
    per_scene: dict[str, dict[str, Any]] = {}
    cpss: list[float] = []

    late_count = 0
    late_50ms_count = 0
    max_late_ms = 0.0
    overlap_count = 0
    min_dur_count = 0
    max_dur_count = 0
    line_len_count = 0
    line_cnt_count = 0
    high_cps_count = 0
    word_cons_errors = 0

    prev_by_scene: dict[str, dict[str, Any]] = {}

    for c in captions:
        cid = c.get("id")
        sc_id = c.get("scene_id", "unknown")
        start = float(c.get("start", 0.0))
        end = float(c.get("end", 0.0))
        duration = float(c.get("duration", round(end - start, 3)))
        words = c.get("words", [])
        text = str(c.get("text", ""))

        if sc_id not in per_scene:
            per_scene[sc_id] = {
                "caption_count": 0,
                "word_count": 0,
                "max_late_ms": 0.0,
                "max_cps": 0.0,
                "overlaps": 0,
                "violations_count": 0,
            }
        sc_stat = per_scene[sc_id]
        sc_stat["caption_count"] += 1
        sc_stat["word_count"] += len(words)

        # 1. Start < End
        if start >= end:
            v = {
                "caption_id": cid,
                "scene_id": sc_id,
                "type": "inverted_timing",
                "detail": f"start ({start}s) >= end ({end}s)",
            }
            violations.append(v)
            sc_stat["violations_count"] += 1

        # 2. Late start
        if words:
            first_w_start = float(words[0].get("start", start))
            if start > first_w_start:
                late_ms = round((start - first_w_start) * 1000.0, 2)
                late_count += 1
                max_late_ms = max(max_late_ms, late_ms)
                sc_stat["max_late_ms"] = max(sc_stat["max_late_ms"], late_ms)
                if late_ms > late_threshold_ms:
                    late_50ms_count += 1
                    v = {
                        "caption_id": cid,
                        "scene_id": sc_id,
                        "type": "late_start",
                        "detail": f"Caption starts {late_ms}ms after first word ({start}s > {first_w_start}s)",
                        "late_ms": late_ms,
                    }
                    violations.append(v)
                    sc_stat["violations_count"] += 1

            last_w_end = float(words[-1].get("end", end))
            if end < last_w_end - 0.005:
                v = {
                    "caption_id": cid,
                    "scene_id": sc_id,
                    "type": "word_cut_off",
                    "detail": f"Caption ends before last word ends ({end}s < {last_w_end}s)",
                }
                violations.append(v)
                sc_stat["violations_count"] += 1

        # 3. Overlaps
        if sc_id in prev_by_scene:
            prev_c = prev_by_scene[sc_id]
            prev_end = float(prev_c.get("end", 0.0))
            if start < prev_end - 0.001:
                overlap_count += 1
                sc_stat["overlaps"] += 1
                v = {
                    "caption_id": cid,
                    "scene_id": sc_id,
                    "type": "overlap",
                    "detail": f"Caption start ({start}s) overlaps with previous caption {prev_c.get('id')} end ({prev_end}s)",
                }
                violations.append(v)
                sc_stat["violations_count"] += 1
        prev_by_scene[sc_id] = c

        # 4. Duration limits
        if duration < min_duration_sec - 0.05:
            min_dur_count += 1
            v = {
                "caption_id": cid,
                "scene_id": sc_id,
                "type": "duration_too_short",
                "detail": f"Duration {duration:.3f}s < minimum {min_duration_sec}s",
                "duration": duration,
            }
            violations.append(v)
            sc_stat["violations_count"] += 1

        if duration > max_duration_sec + 0.05:
            max_dur_count += 1
            v = {
                "caption_id": cid,
                "scene_id": sc_id,
                "type": "duration_too_long",
                "detail": f"Duration {duration:.3f}s > maximum {max_duration_sec}s",
                "duration": duration,
            }
            violations.append(v)
            sc_stat["violations_count"] += 1

        # 5. Line limits
        lines = text.split("\n")
        if len(lines) > max_lines:
            line_cnt_count += 1
            v = {
                "caption_id": cid,
                "scene_id": sc_id,
                "type": "line_count_exceeded",
                "detail": f"Caption has {len(lines)} lines > max {max_lines}",
                "line_count": len(lines),
            }
            violations.append(v)
            sc_stat["violations_count"] += 1

        for line_idx, line in enumerate(lines, start=1):
            if len(line) > max_chars_per_line:
                line_len_count += 1
                v = {
                    "caption_id": cid,
                    "scene_id": sc_id,
                    "type": "line_length_exceeded",
                    "detail": f"Line {line_idx} length {len(line)} > max {max_chars_per_line}: {repr(line)}",
                    "line_length": len(line),
                }
                violations.append(v)
                sc_stat["violations_count"] += 1

        # 6. Reading speed (CPS)
        clean_text = text.replace("\n", " ")
        cps = round(len(clean_text) / duration, 2) if duration > 0 else 0.0
        cpss.append(cps)
        sc_stat["max_cps"] = max(sc_stat["max_cps"], cps)

        if cps > max_cps + 0.05:
            high_cps_count += 1
            v = {
                "caption_id": cid,
                "scene_id": sc_id,
                "type": "high_cps",
                "detail": f"Reading speed {cps:.2f} CPS > max {max_cps} CPS for text: {repr(clean_text)}",
                "cps": cps,
            }
            violations.append(v)
            sc_stat["violations_count"] += 1

    # 7. Word conservation (if scenes provided)
    if scenes:
        for sc in scenes:
            sc_id = sc.get("id")
            sc_orig_words = [w["w"] for w in sc.get("words", [])]
            sc_cap_words = [w["w"] for cap in captions if cap.get("scene_id") == sc_id for w in cap.get("words", [])]
            if sc_orig_words != sc_cap_words:
                word_cons_errors += 1
                v = {
                    "scene_id": sc_id,
                    "type": "word_conservation_mismatch",
                    "detail": f"Scene words ({len(sc_orig_words)}) do not match caption words ({len(sc_cap_words)})",
                    "orig_count": len(sc_orig_words),
                    "cap_count": len(sc_cap_words),
                }
                violations.append(v)
                if sc_id in per_scene:
                    per_scene[sc_id]["violations_count"] += 1

    median_cps = round(sorted(cpss)[len(cpss) // 2], 2) if cpss else 0.0
    overall_max_cps = max(cpss) if cpss else 0.0

    summary = {
        "total_captions": len(captions),
        "total_words": sum(len(c.get("words", [])) for c in captions),
        "passed": len(violations) == 0,
        "violations_count": len(violations),
        "late_captions_count": late_count,
        "late_50ms_count": late_50ms_count,
        "max_late_ms": max_late_ms,
        "overlap_count": overlap_count,
        "min_duration_violations": min_dur_count,
        "max_duration_violations": max_dur_count,
        "line_length_violations": line_len_count,
        "line_count_violations": line_cnt_count,
        "high_cps_count": high_cps_count,
        "median_cps": median_cps,
        "max_cps": overall_max_cps,
        "word_conservation_errors": word_cons_errors,
        "verbatim_conserved": (word_cons_errors == 0),
    }

    return {
        "summary": summary,
        "per_scene": per_scene,
        "violations": violations,
    }


def generate_timeline_preview_video(
    narration_path: Path,
    srt_path: Path,
    output_mp4: Path,
    font_size: int = 22,
    fps: int = 10,
    preset: str = "veryfast",
) -> dict[str, Any]:
    """Generates 1280x720 preview video with burned-in captions using fast encoding settings."""
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
        "-i", f"color=c=black:s=1280x720:r={fps}",
        "-i", str(rel_narration),
        "-vf", filter_expr,
        "-c:v", "libx264",
        "-preset", preset,
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
