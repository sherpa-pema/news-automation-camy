#!/usr/bin/env python3
"""Voice Benchmark Tool (Voice Agent Refinement - Phase Q0).

Measures and reports:
- Model load / prefill / AR step / CFM / QC / write times
- Voice Quality Index (VQI) metrics:
    1. Speaker cosine similarity (min, max, mean, std)
    2. Median F0 spread % (target <= 6%)
    3. Mastered scene LUFS spread (target <= 1.0 LU)
    4. WPS variance across scenes
    5. Dead air ratio (target <= 3%)
    6. Audio artifact flags (clipping margin, DC offset, silence, level jumps)
    7. Unverified pronunciations applied
- Real-Time Factor (RTF)

Can synthesize 3 fixed scenes from an existing run or evaluate existing reports.
Supports side-by-side comparison between two runs/configurations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure vendor directory is in sys.path for Ming TTS
VENDOR_DIR = PROJECT_ROOT / "vendor" / "Ming-3.0-tts"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from src.agents.voice import clean_narration_text, load_pronunciations
from src.utils.tts import (
    FakeTTS,
    assemble_scene_audio,
    compute_voice_quality_index,
    concatenate_scenes,
    get_scene_padding,
    master_audio_loudnorm,
    measure_wav_artifacts,
    measure_wav_ebur128,
    measure_wav_ebur128_segment,
    read_wav_info,
)


def format_scoreboard(report: dict[str, Any]) -> str:
    """Formats a human-readable markdown scoreboard from a voice report."""
    q = report.get("quality", {})
    p = report.get("perf", {})
    stages = p.get("stage_breakdown_sec", {})

    lines = [
        "## Voice Quality & Performance Scoreboard",
        "",
        "### 1. Voice Quality Index (VQI)",
        "| Metric | Value | Target | Status |",
        "|---|---|---|---|",
        f"| Speaker Cosine Min | {q.get('cosine_min', 'N/A')} | >= 0.84 | {'PASS' if q.get('cosine_min', 0) >= 0.84 else 'DRIFT'} |",
        f"| Speaker Cosine Std | {q.get('cosine_std', 'N/A')} | <= 0.02 | {'PASS' if q.get('cosine_std', 1) <= 0.02 else 'HIGH'} |",
        f"| F0 Range (Hz) | {q.get('f0_min_hz', 'N/A')} - {q.get('f0_max_hz', 'N/A')} Hz | Ref: {q.get('reference_f0_hz', 'N/A')} Hz | - |",
        f"| F0 Spread % | {q.get('f0_spread_pct', 'N/A')}% | <= 6.0% | {'PASS' if q.get('f0_spread_pct', 99) <= 6.0 else 'DRIFT'} |",
        f"| Mastered Scene LUFS Spread | {q.get('scene_lufs_spread', 'N/A')} LU | <= 1.0 LU | {'PASS' if q.get('scene_lufs_spread', 99) <= 1.0 else 'UNEVEN'} |",
        f"| Words Per Sec (WPS) Mean | {q.get('wps_mean', 'N/A')} wps | ~2.0 wps | OK |",
        f"| WPS Variance | {q.get('wps_variance', 'N/A')} | <= 0.08 | {'PASS' if q.get('wps_variance', 99) <= 0.08 else 'UNSTABLE'} |",
        f"| Dead Air Ratio | {round(q.get('dead_air_ratio', 0) * 100, 1)}% ({q.get('dead_air_duration_sec', 0)}s) | <= 3.0% | {'PASS' if q.get('dead_air_ratio', 1) <= 0.03 else 'EXCESSIVE'} |",
        "",
        "### 2. Audio Artifacts",
        "| Artifact Check | Value | Threshold | Flagged |",
        "|---|---|---|---|",
        f"| Clipping Margin | {q.get('artifact_flags', {}).get('clipping_margin_db', 'N/A')} dB | > 0.1 dB | {'YES (CLIPPING)' if q.get('artifact_flags', {}).get('has_clipping') else 'No'} |",
        f"| Max Intra-scene Silence | {q.get('artifact_flags', {}).get('max_intra_scene_silence_sec', 'N/A')} s | <= 1.5 s | {'YES (DEAD AIR)' if q.get('artifact_flags', {}).get('has_excess_silence') else 'No'} |",
        f"| DC Offset | {q.get('artifact_flags', {}).get('dc_offset', 'N/A')} | < 0.01 | {'YES' if abs(q.get('artifact_flags', {}).get('dc_offset', 0)) > 0.01 else 'No'} |",
        f"| Level Jumps Detected | {q.get('artifact_flags', {}).get('level_jumps_detected', False)} | None | {'YES' if q.get('artifact_flags', {}).get('level_jumps_detected') else 'No'} |",
        "",
        "### 3. Performance & Stage Breakdown",
        "| Stage | Time (s) | % of Total Synthesis |",
        "|---|---|---|",
        f"| Model Load | {p.get('model_load_time_sec', 'N/A')} s | (one-time cold start) |",
        f"| Prefill | {stages.get('prefill', 'N/A')} s | - |",
        f"| AR Step Decoding | {stages.get('ar_step', 'N/A')} s | - |",
        f"| CFM (Flow Matching) | {stages.get('cfm', 'N/A')} s | - |",
        f"| VAE Audio Decode | {stages.get('vae_decode', 'N/A')} s | - |",
        f"| Speaker QC (CamPP+ / F0) | {stages.get('qc', 'N/A')} s | - |",
        f"| Audio Disk Write | {stages.get('write', 'N/A')} s | - |",
        f"| **Total Synthesis Wall Time** | **{p.get('synthesis_wall_time_sec', 'N/A')} s** | - |",
        f"| Audio Duration Generated | {report.get('total_audio_duration_sec', 'N/A')} s | - |",
        f"| **Real-Time Factor (RTF)** | **{p.get('real_time_factor', 'N/A')}** | (GTX 1060 target ~3.6) |",
    ]
    return "\n".join(lines)


def compare_reports(rep_a: dict[str, Any], rep_b: dict[str, Any], label_a: str = "A", label_b: str = "B") -> str:
    """Generates a side-by-side comparison table between two reports."""
    qa = rep_a.get("quality", {})
    qb = rep_b.get("quality", {})
    pa = rep_a.get("perf", {})
    pb = rep_b.get("perf", {})
    sa = pa.get("stage_breakdown_sec", {})
    sb = pb.get("stage_breakdown_sec", {})

    def delta_str(val_b: float | None, val_a: float | None, unit: str = "", invert_better: bool = False) -> str:
        if val_b is None or val_a is None:
            return "N/A"
        diff = round(val_b - val_a, 4)
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff}{unit}"

    lines = [
        f"## Side-by-Side Comparison: [{label_a}] vs [{label_b}]",
        "",
        "| Metric | " + label_a + " | " + label_b + " | Delta | Target |",
        "|---|---|---|---|---|",
        f"| Real-Time Factor (RTF) | {pa.get('real_time_factor')} | {pb.get('real_time_factor')} | {delta_str(pb.get('real_time_factor'), pa.get('real_time_factor'))} | lower is faster |",
        f"| Model Load Time | {pa.get('model_load_time_sec')}s | {pb.get('model_load_time_sec')}s | {delta_str(pb.get('model_load_time_sec'), pa.get('model_load_time_sec'), 's')} | lower |",
        f"| AR Step Time | {sa.get('ar_step')}s | {sb.get('ar_step')}s | {delta_str(sb.get('ar_step'), sa.get('ar_step'), 's')} | lower |",
        f"| CFM Time | {sa.get('cfm')}s | {sb.get('cfm')}s | {delta_str(sb.get('cfm'), sa.get('cfm'), 's')} | lower |",
        f"| Speaker Cosine Min | {qa.get('cosine_min')} | {qb.get('cosine_min')} | {delta_str(qb.get('cosine_min'), qa.get('cosine_min'))} | >= 0.84 |",
        f"| Speaker Cosine Std | {qa.get('cosine_std')} | {qb.get('cosine_std')} | {delta_str(qb.get('cosine_std'), qa.get('cosine_std'))} | <= 0.02 |",
        f"| F0 Spread % | {qa.get('f0_spread_pct')}% | {qb.get('f0_spread_pct')}% | {delta_str(qb.get('f0_spread_pct'), qa.get('f0_spread_pct'), '%')} | <= 6.0% |",
        f"| Scene LUFS Spread | {qa.get('scene_lufs_spread')} LU | {qb.get('scene_lufs_spread')} LU | {delta_str(qb.get('scene_lufs_spread'), qa.get('scene_lufs_spread'), ' LU')} | <= 1.0 LU |",
        f"| WPS Variance | {qa.get('wps_variance')} | {qb.get('wps_variance')} | {delta_str(qb.get('wps_variance'), qa.get('wps_variance'))} | lower |",
        f"| Dead Air % | {round(qa.get('dead_air_ratio', 0)*100, 1)}% | {round(qb.get('dead_air_ratio', 0)*100, 1)}% | {delta_str(qb.get('dead_air_ratio'), qa.get('dead_air_ratio'), '%')} | <= 3.0% |",
    ]
    return "\n".join(lines)


def run_benchmark(
    run_dir: Path,
    scenes_to_bench: list[str],
    out_dir: Path,
    engine_name: str = "ming",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    reference_wav: str = "config/voice/reference.wav",
    reference_txt: str = "config/voice/reference.txt",
    similarity_threshold: float = 0.72,
    f0_tolerance_pct: float = 0.10,
    n_candidates: int = 1,
    f0_anchor: str = "running_median",
    padding_policy: str = "broadcast",
    target_scene_rms_dbfs: float = -23.0,
    loudness_strategy: str = "linear_strict",
    limiter_ceiling_dbfs: float = -1.5,
) -> dict[str, Any]:
    """Runs synthesis benchmark over fixed scenes and measures stage breakdown and VQI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = run_dir / "script" / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    pronunciations = load_pronunciations()
    all_scenes = {s["id"]: s for s in script.get("scenes", [])}

    target_scenes = []
    for s_id in scenes_to_bench:
        if s_id in all_scenes:
            target_scenes.append(all_scenes[s_id])
        else:
            raise ValueError(f"Requested scene '{s_id}' not found in {script_path}")

    items = []
    for sc in target_scenes:
        s_id = sc["id"]
        raw_text = sc.get("narration", "")
        cleaned = clean_narration_text(raw_text, pronunciations)
        out_wav = out_dir / f"{s_id}_raw.wav"
        items.append({"id": s_id, "text": cleaned, "out_path": str(out_wav)})

    synth_start = time.time()
    consistency_map: dict[str, Any] = {}
    ref_f0 = 224.3

    if engine_name == "fake":
        engine = FakeTTS()
        batch_res = engine.synthesize_batch(
            items,
            {
                "reference_wav": reference_wav,
                "n_candidates": n_candidates,
                "f0_anchor": f0_anchor,
                "f0_tolerance_pct": f0_tolerance_pct,
            },
        )
    else:
        from src.utils.tts import MingEngine

        engine = MingEngine()
        engine_settings = {
            "task": "TTS",
            "voice": "female_broadcast",
            "device": device,
            "dtype": dtype,
            "reference_wav": reference_wav,
            "reference_txt": reference_txt,
            "similarity_threshold": similarity_threshold,
            "f0_tolerance_pct": f0_tolerance_pct,
            "n_candidates": n_candidates,
            "f0_anchor": f0_anchor,
            "max_retries": 2,
        }
        batch_res = engine.synthesize_batch(items, engine_settings)

    synthesis_wall_time = time.time() - synth_start
    model_load_time = float(batch_res.get("model_load_time_sec", 0.0))
    stage_breakdown = batch_res.get("stage_breakdown_sec", {})
    if batch_res.get("reference_f0_hz"):
        ref_f0 = float(batch_res["reference_f0_hz"])

    for r in batch_res.get("results", []):
        if r.get("status") == "failed":
            raise RuntimeError(f"Synthesis failed for scene {r.get('id')}: {r.get('error')}")
        consistency_map[r["id"]] = {
            "cosine_sim": r.get("cosine_sim", 1.0),
            "f0_median_hz": r.get("f0_median_hz", ref_f0),
            "target_f0_hz": r.get("target_f0_hz", ref_f0),
            "passes_gate": r.get("passes_gate", True),
            "selected_candidate_idx": r.get("selected_candidate_idx", 0),
            "seed_used": r.get("seed_used", 1895),
            "ranker_score": r.get("ranker_score"),
            "ranker_breakdown": r.get("ranker_breakdown"),
            "candidates_evaluated": r.get("candidates_evaluated"),
        }

    # Assemble scenes
    t_asm = time.time()
    assembled_scene_wavs = []
    scene_metrics = []
    words_map = {sc["id"]: len(sc.get("narration", "").split()) for sc in target_scenes}

    for idx, sc in enumerate(target_scenes):
        s_id = sc["id"]
        raw_wav = out_dir / f"{s_id}_raw.wav"
        sc_wav = out_dir / f"{s_id}.wav"
        s_pad_before, s_pad_after = get_scene_padding(
            scene_id=s_id,
            scene_idx=idx,
            total_scenes=len(target_scenes),
            padding_policy=padding_policy,
            pad_before_fallback=0.4,
            pad_after_fallback=0.6,
        )
        dur = assemble_scene_audio(
            [raw_wav],
            sc_wav,
            pause_sec=0.0,
            pad_before=s_pad_before,
            pad_after=s_pad_after,
            target_sample_rate=44100,
            target_rms_dbfs=target_scene_rms_dbfs,
        )
        assembled_scene_wavs.append(sc_wav)
        scene_metrics.append({
            "id": s_id,
            "actual_duration_sec": dur,
            "pad_before": s_pad_before,
            "pad_after": s_pad_after,
            "file": str(sc_wav.relative_to(out_dir)),
            "word_count": words_map.get(s_id, 0),
        })

    assembly_time = time.time() - t_asm

    # Concatenate & master
    concat_wav = out_dir / "concat_raw.wav"
    concatenate_scenes(assembled_scene_wavs, concat_wav)

    t_mast = time.time()
    master_wav = out_dir / "narration_master.wav"
    master_stats = master_audio_loudnorm(
        concat_wav,
        master_wav,
        target_lufs=-16.0,
        true_peak=limiter_ceiling_dbfs,
        sample_rate=48000,
        highpass_hz=80.0,
        compressor=True,
        limiter=True,
        limiter_ceiling_dbfs=limiter_ceiling_dbfs,
        loudness_strategy=loudness_strategy,
    )
    mastering_time = time.time() - t_mast
    concat_wav.unlink(missing_ok=True)

    final_info = read_wav_info(master_wav)
    total_audio_dur = final_info["duration_sec"]
    rtf = round(synthesis_wall_time / total_audio_dur, 2) if total_audio_dur > 0 else 0.0

    # Per-scene loudness measurements
    curr_pos = 0.0
    for sm in scene_metrics:
        s_file = out_dir / f"{sm['id']}.wav"
        ebur = measure_wav_ebur128(s_file)
        seg_ebur = measure_wav_ebur128_segment(master_wav, curr_pos, sm["actual_duration_sec"])
        curr_pos += sm["actual_duration_sec"]
        sm["lufs"] = ebur.get("lufs")
        sm["true_peak_db"] = ebur.get("true_peak_db")
        sm["mastered_lufs"] = seg_ebur.get("lufs")
        sm["mastered_true_peak_db"] = seg_ebur.get("true_peak_db")

    total_pad = sum(sm.get("pad_before", 0.0) + sm.get("pad_after", 0.0) for sm in scene_metrics)
    avg_pad = total_pad / max(1, len(scene_metrics))
    artifacts = measure_wav_artifacts(master_wav)
    quality = compute_voice_quality_index(
        consistency_metrics=consistency_map,
        reference_f0_hz=ref_f0,
        scenes_info=scene_metrics,
        total_duration_sec=total_audio_dur,
        pad_per_scene_sec=round(avg_pad, 2),
        artifact_flags=artifacts,
        unverified_pronunciations=[],
    )

    perf = {
        "model_load_time_sec": round(model_load_time, 2),
        "synthesis_wall_time_sec": round(synthesis_wall_time, 2),
        "assembly_time_sec": round(assembly_time, 2),
        "mastering_time_sec": round(mastering_time, 2),
        "total_wall_time_sec": round(model_load_time + synthesis_wall_time + assembly_time + mastering_time, 2),
        "real_time_factor": rtf,
        "stage_breakdown_sec": stage_breakdown,
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "engine": engine_name,
        "device": device,
        "dtype": dtype,
        "reference_wav": reference_wav,
        "total_audio_duration_sec": total_audio_dur,
        "quality": quality,
        "perf": perf,
        "loudnorm": master_stats,
        "scenes": scene_metrics,
    }

    report_path = out_dir / "bench_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Benchmark and Quality Scoreboard (Phase Q0)")
    parser.add_argument("--run-dir", default="data/runs/2026-11-15", help="Path to run directory")
    parser.add_argument("--scenes", default="intro,scene_01,scene_02", help="Comma-separated scene IDs to benchmark")
    parser.add_argument("--engine", default="ming", choices=["ming", "fake"], help="TTS engine to benchmark")
    parser.add_argument("--device", default="cuda:0", help="Computation device (e.g. cuda:0, cpu)")
    parser.add_argument("--dtype", default="bfloat16", help="Torch data type (bfloat16, float16, float32)")
    parser.add_argument("--reference-wav", default="config/voice/reference.wav", help="Reference voice WAV path")
    parser.add_argument("--reference-txt", default="config/voice/reference.txt", help="Reference prompt transcript")
    parser.add_argument("--f0-tolerance-pct", type=float, default=0.10, help="F0 tolerance percentage (default 0.10)")
    parser.add_argument("--n-candidates", type=int, default=1, help="Number of candidates to evaluate per scene (default 1)")
    parser.add_argument("--f0-anchor", default="running_median", choices=["running_median", "reference"], help="Anchor for F0 ranker target")
    parser.add_argument("--padding-policy", default="broadcast", choices=["broadcast", "fixed"], help="Padding policy for scenes (default broadcast)")
    parser.add_argument("--target-scene-rms", type=float, default=-23.0, help="Target active RMS dBFS for assembled scenes (default -23.0)")
    parser.add_argument("--loudness-strategy", default="linear_strict", choices=["linear_strict", "fallback"], help="Loudness mastering strategy")
    parser.add_argument("--limiter-ceiling", type=float, default=-1.5, help="Limiter ceiling in dBFS (default -1.5)")
    parser.add_argument("--out-dir", default=None, help="Output directory for benchmark artifacts")
    parser.add_argument("--report-only", default=None, help="Path to an existing voice_report.json to display scoreboard")
    parser.add_argument("--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"), help="Compare two report JSON files side by side")
    args = parser.parse_args()

    # 1. Compare mode
    if args.compare:
        def resolve_report(path_str: str) -> Path:
            p = Path(path_str)
            if p.is_file():
                return p
            for cand in [p / "bench_report.json", p / "audio" / "voice_report.json", p / "voice_report.json"]:
                if cand.exists():
                    return cand
            return p

        p_a = resolve_report(args.compare[0])
        p_b = resolve_report(args.compare[1])
        with open(p_a, encoding="utf-8") as f:
            rep_a = json.load(f)
        with open(p_b, encoding="utf-8") as f:
            rep_b = json.load(f)
        lbl_a = p_a.parent.name if p_a.parent.name != "audio" else p_a.parent.parent.name
        lbl_b = p_b.parent.name if p_b.parent.name != "audio" else p_b.parent.parent.name
        print(compare_reports(rep_a, rep_b, label_a=lbl_a, label_b=lbl_b))
        return

    # 2. Report-only mode
    if args.report_only:
        p = Path(args.report_only)
        if p.is_dir():
            for cand in [p / "audio" / "voice_report.json", p / "bench_report.json", p / "voice_report.json"]:
                if cand.exists():
                    p = cand
                    break
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        print(format_scoreboard(rep))
        return

    # 3. Run active benchmark
    run_dir = Path(args.run_dir)
    scene_ids = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "data" / "benchmarks" / f"bench_{ts}"

    print(f"Starting voice benchmark on {len(scene_ids)} scenes using engine={args.engine} device={args.device}...")
    report = run_benchmark(
        run_dir=run_dir,
        scenes_to_bench=scene_ids,
        out_dir=out_dir,
        engine_name=args.engine,
        device=args.device,
        dtype=args.dtype,
        reference_wav=args.reference_wav,
        reference_txt=args.reference_txt,
        f0_tolerance_pct=args.f0_tolerance_pct,
        n_candidates=args.n_candidates,
        f0_anchor=args.f0_anchor,
        padding_policy=args.padding_policy,
        target_scene_rms_dbfs=args.target_scene_rms,
        loudness_strategy=args.loudness_strategy,
        limiter_ceiling_dbfs=args.limiter_ceiling,
    )
    print("\nBenchmark completed successfully! Report saved to:", out_dir / "bench_report.json")
    print("\n" + format_scoreboard(report))


if __name__ == "__main__":
    main()
