"""Media normalization using ffmpeg, Pillow, and ffprobe."""
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from PIL import Image, ImageOps
import imagehash

logger = logging.getLogger("media.normalizer")


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_perceptual_hash(file_path: Path, is_video: bool = False) -> imagehash.ImageHash:
    """Computes perceptual hash for image or first frame of video."""
    if is_video:
        # Extract frame at 0.5s via ffmpeg to stdout pipe
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", "0.5",
            "-i", str(file_path),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        if proc.returncode == 0 and proc.stdout:
            import io
            img = Image.open(io.BytesIO(proc.stdout))
            return imagehash.phash(img)

    # For images or video fallback
    with Image.open(file_path) as img:
        return imagehash.phash(img)


def probe_media(file_path: Path) -> dict[str, Any]:
    """Probes media file using ffprobe to extract duration, resolution, fps, and check for audio streams."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(file_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    info = json.loads(proc.stdout)

    video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0

    # Parse fps
    fps = 0.0
    if video_stream and "r_frame_rate" in video_stream:
        r_fr = video_stream["r_frame_rate"]
        if "/" in r_fr:
            num, den = r_fr.split("/")
            fps = round(float(num) / float(den), 2) if float(den) > 0 else 0.0
        else:
            fps = float(r_fr)

    # Duration
    duration = 0.0
    if video_stream and "duration" in video_stream and video_stream["duration"]:
        duration = float(video_stream["duration"])
    elif "format" in info and "duration" in info["format"] and info["format"]["duration"]:
        duration = float(info["format"]["duration"])

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "has_audio": audio_stream is not None,
    }


def normalize_video(
    src_path: Path,
    dest_path: Path,
    target_duration: float,
    target_resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    source_duration: float | None = None,
) -> dict[str, Any]:
    """Normalizes video to target resolution, fps, duration, center-cropped, with no audio stream."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine start segment
    start_sec = 0.0
    if source_duration and source_duration > (target_duration + 2.0):
        start_sec = 1.0

    width, height = target_resolution
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", str(src_path),
        "-t", str(target_duration),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-an",
        "-movflags", "+faststart",
        str(dest_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg video normalization failed: {proc.stderr}")

    probe = probe_media(dest_path)
    probe["sha256"] = compute_sha256(dest_path)
    return probe


def normalize_image(
    src_path: Path,
    dest_path: Path,
    target_resolution: tuple[int, int] = (1920, 1080),
) -> dict[str, Any]:
    """Normalizes still image to target resolution JPEG with center crop."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img_rgb = img.convert("RGB")
        fitted = ImageOps.fit(img_rgb, target_resolution, method=Image.Resampling.LANCZOS)
        fitted.save(dest_path, "JPEG", quality=92)

    sha256_hash = compute_sha256(dest_path)
    return {
        "width": target_resolution[0],
        "height": target_resolution[1],
        "fps": None,
        "duration": None,
        "has_audio": False,
        "sha256": sha256_hash,
    }
