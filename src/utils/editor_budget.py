"""Word budget calculation calibrated to TTS speech rate."""
import json
from pathlib import Path
import re
from typing import Any

DEFAULT_WORDS_PER_SEC = 2.15
DEFAULT_SYLLABLES_PER_SEC = 3.3
DEFAULT_COMMA_PAUSE_SEC = 0.25
DEFAULT_PERIOD_PAUSE_SEC = 0.55


def count_syllables_in_word(word: str) -> int:
    """Deterministic English syllable counter heuristic."""
    cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1

    # Strip silent trailing e / es / ed carefully
    if cleaned.endswith("es") and not cleaned.endswith(("ses", "xes", "zes", "ches", "shes")):
        cleaned = cleaned[:-2]
    elif cleaned.endswith("ed") and not cleaned.endswith(("ted", "ded")):
        cleaned = cleaned[:-2]
    elif cleaned.endswith("e") and not (
        len(cleaned) >= 3 and cleaned.endswith("le") and cleaned[-3] not in "aeiouy"
    ):
        cleaned = cleaned[:-1]

    if not cleaned:
        return 1

    vowel_groups = re.findall(r"[aeiouy]+", cleaned)
    return max(1, len(vowel_groups))


def count_syllables_in_text(text: str) -> int:
    """Counts total estimated syllables in a text passage."""
    if not text:
        return 0
    return sum(count_syllables_in_word(tok) for tok in text.split())


def count_punctuation_pauses(text: str) -> tuple[int, int]:
    """Counts (comma_like_pauses, period_like_pauses) in text."""
    if not text:
        return 0, 0
    commas = len(re.findall(r"[,;:]", text))
    periods = len(re.findall(r"[.!?]+", text))
    return commas, periods


def estimate_scene_duration_sec(narration: str, budget: dict[str, Any]) -> float:
    """Estimates spoken duration blending word rate with syllable count + punctuation pauses."""
    text = (narration or "").strip()
    if not text:
        return 0.0
    words = len(text.split())
    if words == 0:
        return 0.0

    wps = float(budget.get("words_per_second", DEFAULT_WORDS_PER_SEC) or DEFAULT_WORDS_PER_SEC)
    sps = float(budget.get("syllables_per_second", DEFAULT_SYLLABLES_PER_SEC) or DEFAULT_SYLLABLES_PER_SEC)
    comma_pause = float(budget.get("comma_pause_sec", DEFAULT_COMMA_PAUSE_SEC))
    period_pause = float(budget.get("period_pause_sec", DEFAULT_PERIOD_PAUSE_SEC))

    syllables = count_syllables_in_text(text)
    commas, periods = count_punctuation_pauses(text)

    word_dur = words / wps
    syl_pause_dur = (syllables / sps) + (commas * comma_pause) + (periods * period_pause)
    return round(0.4 * word_dur + 0.6 * syl_pause_dur, 1)


def calculate_words_per_second_details(
    runs_dir: Path | str = "data/runs",
    default_wps: float = DEFAULT_WORDS_PER_SEC,
    default_sps: float = DEFAULT_SYLLABLES_PER_SEC,
    comma_pause_sec: float = DEFAULT_COMMA_PAUSE_SEC,
    period_pause_sec: float = DEFAULT_PERIOD_PAUSE_SEC,
) -> dict[str, Any]:
    """Finds the newest real voice_report.json and returns speech-rate details (wps & sps)."""
    runs_path = Path(runs_dir)
    voice_reports = list(runs_path.rglob("voice_report.json"))

    if not voice_reports:
        msg = f"Default estimate: {default_wps:.2f} wps / {default_sps:.2f} sps (no prior voice_report.json found)"
        return {
            "words_per_second": default_wps,
            "syllables_per_second": default_sps,
            "comma_pause_sec": comma_pause_sec,
            "period_pause_sec": period_pause_sec,
            "source": "default",
            "source_file": None,
            "calculation": f"default = {default_wps:.2f} words/sec, {default_sps:.2f} syl/sec (no prior voice_report.json found)",
            "message": msg,
        }

    # Pick the most recently modified report
    voice_reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    newest_report = voice_reports[0]

    try:
        with open(newest_report, "r", encoding="utf-8") as f:
            v_data = json.load(f)

        total_audio_sec = float(v_data.get("total_audio_duration_sec", 0.0))
        if total_audio_sec <= 0:
            msg = f"Default estimate: {default_wps:.2f} wps / {default_sps:.2f} sps (voice report had zero duration)"
            return {
                "words_per_second": default_wps,
                "syllables_per_second": default_sps,
                "comma_pause_sec": comma_pause_sec,
                "period_pause_sec": period_pause_sec,
                "source": "default",
                "source_file": newest_report.as_posix(),
                "calculation": f"default = {default_wps:.2f} words/sec, {default_sps:.2f} syl/sec (voice report had zero duration)",
                "message": msg,
            }

        # Find corresponding script.json to count actual words & syllables
        run_folder = newest_report.parent.parent
        script_candidates = [
            run_folder / "script" / "script.json",
            run_folder / "script.json",
            Path("tests/fixtures/sample_script.json"),
        ]
        total_words = 0
        total_syllables = 0
        total_commas = 0
        total_periods = 0
        for sc_p in script_candidates:
            if sc_p.exists():
                with open(sc_p, "r", encoding="utf-8") as sf:
                    s_data = json.load(sf)
                scenes = s_data.get("scenes", [])
                total_words = sum(len(sc.get("narration", "").split()) for sc in scenes)
                total_syllables = sum(count_syllables_in_text(sc.get("narration", "")) for sc in scenes)
                for sc in scenes:
                    c, p = count_punctuation_pauses(sc.get("narration", ""))
                    total_commas += c
                    total_periods += p
                break

        if total_words > 0:
            wps = round(total_words / total_audio_sec, 2)
            pause_sec = (total_commas * comma_pause_sec) + (total_periods * period_pause_sec)
            active_speech_sec = max(total_audio_sec - pause_sec, total_audio_sec * 0.5)
            sps = round(total_syllables / active_speech_sec, 2) if total_syllables > 0 else default_sps
            try:
                rel_report = newest_report.relative_to(runs_path.parent)
            except ValueError:
                rel_report = newest_report
            calculation = (
                f"{total_words} words / {total_audio_sec:.1f}s = {wps:.2f} words/sec; "
                f"{total_syllables} syllables / {active_speech_sec:.1f}s = {sps:.2f} syl/sec"
            )
            calc_msg = f"Calibrated from {rel_report}: {calculation}"
            return {
                "words_per_second": wps,
                "syllables_per_second": sps,
                "comma_pause_sec": comma_pause_sec,
                "period_pause_sec": period_pause_sec,
                "source": rel_report.as_posix(),
                "source_file": newest_report.as_posix(),
                "calculation": calculation,
                "message": calc_msg,
            }

    except Exception:
        pass

    msg = f"Default estimate: {default_wps:.2f} wps / {default_sps:.2f} sps (fallback due to read error)"
    return {
        "words_per_second": default_wps,
        "syllables_per_second": default_sps,
        "comma_pause_sec": comma_pause_sec,
        "period_pause_sec": period_pause_sec,
        "source": "default",
        "source_file": None,
        "calculation": f"default = {default_wps:.2f} words/sec, {default_sps:.2f} syl/sec (fallback due to read error)",
        "message": msg,
    }


def calculate_words_per_second(runs_dir: Path | str = "data/runs", default_wps: float = DEFAULT_WORDS_PER_SEC) -> tuple[float, str]:
    """Finds the newest real voice_report.json and calculates words-per-second."""
    details = calculate_words_per_second_details(runs_dir=runs_dir, default_wps=default_wps)
    return float(details["words_per_second"]), str(details["message"])


def compute_word_budget(
    wps: float,
    min_sec: float = 120.0,
    max_sec: float = 180.0,
    target_sec: float = 150.0,
    tolerance_pct: float = 0.10,
    sps: float = DEFAULT_SYLLABLES_PER_SEC,
    comma_pause_sec: float = DEFAULT_COMMA_PAUSE_SEC,
    period_pause_sec: float = DEFAULT_PERIOD_PAUSE_SEC,
) -> dict[str, Any]:
    """Computes total word budget and scene targets for target audio duration."""
    target_words = round(target_sec * wps)
    min_total_words = round(min_sec * wps * (1.0 - tolerance_pct))
    max_total_words = round(max_sec * wps * (1.0 + tolerance_pct))

    return {
        "words_per_second": wps,
        "syllables_per_second": sps,
        "comma_pause_sec": comma_pause_sec,
        "period_pause_sec": period_pause_sec,
        "target_duration_sec": target_sec,
        "duration_range_sec": [min_sec, max_sec],
        "target_total_words": target_words,
        "min_total_words": min_total_words,
        "max_total_words": max_total_words,
        "tolerance_pct": tolerance_pct,
        "scene_targets": {
            "intro": {"min_words": 15, "max_words": 40, "target_words": 25},
            "story": {"min_words": 20, "max_words": 60, "target_words": 50, "count": 5},
            "outro": {"min_words": 15, "max_words": 40, "target_words": 25},
        },
    }
