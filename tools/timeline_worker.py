"""Subprocess worker for MMS_FA forced alignment running inside .venv-tts.

Loads torchaudio MMS_FA pipeline and aligns pre-tokenized words against scene audio files.
Communication via JSON files: --job-file and --result-file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torchaudio.pipelines import MMS_FA
from torchaudio.pipelines._wav2vec2 import utils


def load_mms_model(model_path: str | Path | None, device: str = "cuda:0", dtype: str = "float32") -> tuple[torch.nn.Module, Any, Any]:
    """Loads MMS_FA model, tokenizer, and aligner."""
    bundle = MMS_FA
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    local_pt = Path(model_path) if model_path else None
    if local_pt and local_pt.exists() and local_pt.is_file():
        try:
            model = utils._get_model(bundle._model_type, bundle._params)
            state_dict = torch.load(str(local_pt), map_location="cpu")
            if bundle._remove_aux_axis:
                utils._remove_aux_axes(state_dict, bundle._remove_aux_axis)
            model.load_state_dict(state_dict)
            model = utils._extend_model(
                model,
                normalize_waveform=bundle._normalize_waveform,
                apply_log_softmax=True,
                append_star=True,
            )
        except Exception as e:
            print(f"Warning: Failed to load local model weights from '{local_pt}': {e}. Falling back to bundle.get_model().", file=sys.stderr)
            model = bundle.get_model()
    else:
        model = bundle.get_model()

    torch_dtype = getattr(torch, dtype, torch.float32)
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    return model, tokenizer, aligner


def align_scene(
    audio_path: str | Path,
    words_data: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    aligner: Any,
    device: str = "cuda:0",
    bundle_sr: int = 16000,
) -> dict[str, Any]:
    """Aligns word tokens to scene audio and returns word-level timestamps and scores."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = torchaudio.load(str(audio_path))
    if sr != bundle_sr:
        waveform = torchaudio.transforms.Resample(sr, bundle_sr)(waveform)
    # Ensure mono
    waveform = waveform.mean(dim=0, keepdim=True)

    duration_sec = waveform.shape[-1] / bundle_sr

    # Flatten tokens while tracking slice per word
    flat_tokens: list[str] = []
    word_slices: list[tuple[int, int]] = []

    for item in words_data:
        toks = item.get("tokens", [])
        start_idx = len(flat_tokens)
        flat_tokens.extend(toks)
        end_idx = len(flat_tokens)
        word_slices.append((start_idx, end_idx))

    if not flat_tokens:
        # No tokens to align
        return {
            "duration_sec": round(duration_sec, 3),
            "words": [
                {
                    "w": item.get("orig", ""),
                    "tts": item.get("tts", ""),
                    "start": None,
                    "end": None,
                    "score": 0.0,
                    "unplaced": True,
                }
                for item in words_data
            ],
        }

    # Run forward pass
    torch_dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device=device, dtype=torch_dtype))

    # Tokenize and align
    tokens = tokenizer(flat_tokens)
    spans = aligner(emissions[0], tokens)

    # Calculate frame duration
    num_frames = emissions.shape[1]
    frame_dur = duration_sec / num_frames if num_frames > 0 else 0.02

    aligned_words: list[dict[str, Any]] = []
    for item, (s_idx, e_idx) in zip(words_data, word_slices):
        sub_spans = [spans[k] for k in range(s_idx, e_idx) if k < len(spans) and spans[k]]
        if sub_spans:
            t_start = round(sub_spans[0][0].start * frame_dur, 3)
            t_end = round(sub_spans[-1][-1].end * frame_dur, 3)
            # Clip to duration
            t_start = max(0.0, min(duration_sec, t_start))
            t_end = max(t_start, min(duration_sec, t_end))

            all_scores = [s.score for sp in sub_spans for s in sp]
            avg_score = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

            aligned_words.append({
                "w": item.get("orig", ""),
                "tts": item.get("tts", ""),
                "start": t_start,
                "end": t_end,
                "score": avg_score,
                "unplaced": False,
            })
        else:
            aligned_words.append({
                "w": item.get("orig", ""),
                "tts": item.get("tts", ""),
                "start": None,
                "end": None,
                "score": 0.0,
                "unplaced": True,
            })

    return {
        "duration_sec": round(duration_sec, 3),
        "words": aligned_words,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Timeline Forced Alignment Worker")
    parser.add_argument("--job-file", required=True, help="Path to input job JSON file")
    parser.add_argument("--result-file", required=True, help="Path to output result JSON file")
    args = parser.parse_args()

    with open(args.job_file, "r", encoding="utf-8") as f:
        job = json.load(f)

    model_path = job.get("model_path")
    device = job.get("device", "cuda:0")
    dtype = job.get("dtype", "float32")
    scenes = job.get("scenes", [])

    device_fallback = False
    if not torch.cuda.is_available() and "cuda" in device:
        print(f"CUDA requested ('{device}') but not available; falling back to CPU.", file=sys.stderr)
        device = "cpu"
        device_fallback = True

    t0 = time.perf_counter()
    try:
        model, tokenizer, aligner = load_mms_model(model_path, device=device, dtype=dtype)
        load_time_sec = round(time.perf_counter() - t0, 3)
    except Exception as e:
        error_result = {
            "status": "error",
            "error": f"Failed to load MMS aligner model: {e}",
            "model_load_time_sec": round(time.perf_counter() - t0, 3),
        }
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)
        return 1

    scene_results: dict[str, Any] = {}
    alignment_times: dict[str, float] = {}

    for sc in scenes:
        sc_id = sc["id"]
        audio_path = sc["audio_path"]
        words_data = sc.get("words", [])

        t_scene_start = time.perf_counter()
        try:
            res = align_scene(
                audio_path=audio_path,
                words_data=words_data,
                model=model,
                tokenizer=tokenizer,
                aligner=aligner,
                device=device,
            )
            alignment_times[sc_id] = round(time.perf_counter() - t_scene_start, 3)
            scene_results[sc_id] = res
        except Exception as e:
            print(f"Warning: Failed to align scene '{sc_id}': {e}", file=sys.stderr)
            alignment_times[sc_id] = round(time.perf_counter() - t_scene_start, 3)
            scene_results[sc_id] = {
                "duration_sec": 0.0,
                "error": str(e),
                "words": [
                    {
                        "w": item.get("orig", ""),
                        "tts": item.get("tts", ""),
                        "start": None,
                        "end": None,
                        "score": 0.0,
                        "unplaced": True,
                    }
                    for item in words_data
                ],
            }

    out_payload = {
        "status": "success",
        "model_load_time_sec": load_time_sec,
        "device": device,
        "device_fallback": device_fallback,
        "dtype": dtype,
        "alignment_times_sec": alignment_times,
        "scenes": scene_results,
    }

    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
