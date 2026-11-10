"""Word budget calculation calibrated to TTS speech rate."""
import json
from pathlib import Path
from typing import Any


def calculate_words_per_second(runs_dir: Path | str = "data/runs", default_wps: float = 2.15) -> tuple[float, str]:
    """Finds the newest real voice_report.json and calculates words-per-second."""
    runs_path = Path(runs_dir)
    voice_reports = list(runs_path.rglob("voice_report.json"))

    if not voice_reports:
        return default_wps, f"Default estimate: {default_wps:.2f} wps (no prior voice_report.json found)"

    # Pick the most recently modified report
    voice_reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    newest_report = voice_reports[0]

    try:
        with open(newest_report, "r", encoding="utf-8") as f:
            v_data = json.load(f)

        total_audio_sec = float(v_data.get("total_audio_duration_sec", 0.0))
        if total_audio_sec <= 0:
            return default_wps, f"Default estimate: {default_wps:.2f} wps (voice report had zero duration)"

        # Find corresponding script.json to count actual words
        run_folder = newest_report.parent.parent
        script_candidates = [
            run_folder / "script" / "script.json",
            run_folder / "script.json",
            Path("tests/fixtures/sample_script.json"),
        ]
        total_words = 0
        for sc_p in script_candidates:
            if sc_p.exists():
                with open(sc_p, "r", encoding="utf-8") as sf:
                    s_data = json.load(sf)
                total_words = sum(len(sc["narration"].split()) for sc in s_data.get("scenes", []))
                break

        if total_words > 0:
            wps = total_words / total_audio_sec
            calc_msg = (
                f"Calibrated from {newest_report.relative_to(runs_path.parent)}: "
                f"{total_words} words / {total_audio_sec:.1f}s = {wps:.2f} words/sec"
            )
            return round(wps, 2), calc_msg

    except Exception as e:
        pass

    return default_wps, f"Default estimate: {default_wps:.2f} wps (fallback due to read error)"


def compute_word_budget(
    wps: float,
    min_sec: float = 120.0,
    max_sec: float = 180.0,
    target_sec: float = 150.0,
    tolerance_pct: float = 0.10,
) -> dict[str, Any]:
    """Computes total word budget and scene targets for target audio duration."""
    target_words = round(target_sec * wps)
    min_total_words = round(min_sec * wps * (1.0 - tolerance_pct))
    max_total_words = round(max_sec * wps * (1.0 + tolerance_pct))

    return {
        "words_per_second": wps,
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
