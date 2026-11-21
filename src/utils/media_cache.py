"""Caching, rate-limiting, and network utilities for the Media Agent."""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger("media.cache")


class MediaCacheManager:
    """Manages 24-hour API response caching and local downloaded file caching."""

    def __init__(
        self,
        cache_dir: str | Path = "data/media_cache",
        user_agent: str = "NepalNewsBot/1.0 (+https://github.com/nepal-news-bot; contact@nepalnewsbot.org)",
        ttl_seconds: int = 86400,  # 24 hours
        max_total_media_size_mb: float = 500.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.requests_dir = self.cache_dir / "requests"
        self.downloads_dir = self.cache_dir / "downloads"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.ttl_seconds = ttl_seconds
        self.max_total_media_size_mb = max_total_media_size_mb
        self._last_request_times: dict[str, float] = {}

    def _get_cache_path(self, key: str) -> Path:
        hashed = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.requests_dir / f"{hashed}.json"

    def _rate_limit(self, domain: str, min_interval: float) -> None:
        now = time.time()
        last_time = self._last_request_times.get(domain, 0.0)
        elapsed = now - last_time
        if elapsed < min_interval:
            sleep_sec = min_interval - elapsed
            time.sleep(sleep_sec)
        self._last_request_times[domain] = time.time()

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        min_interval: float = 0.5,
        domain_key: str | None = None,
        offline: bool = False,
    ) -> dict[str, Any] | None:
        """Fetches JSON with 24-hour file caching and retry logic."""
        full_headers = {"User-Agent": self.user_agent}
        if headers:
            full_headers.update(headers)

        # Normalize params for cache key
        norm_params = tuple(sorted((k, str(v)) for k, v in (params or {}).items()))
        cache_key = f"{url}?{norm_params}"
        cache_path = self._get_cache_path(cache_key)

        # Check cache
        if cache_path.exists():
            try:
                stat = cache_path.stat()
                age = time.time() - stat.st_mtime
                if offline or age < self.ttl_seconds:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache {cache_path}: {e}")

        if offline:
            logger.debug(f"Offline mode: cache miss for {url}")
            return None

        # Rate limit
        domain = domain_key or url.split("/")[2] if "//" in url else "default"
        self._rate_limit(domain, min_interval)

        # Request with retry for 5xx/connection errors; NEVER retry 4xx
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, params=params, headers=full_headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                    except Exception as e:
                        logger.warning(f"Failed to write cache {cache_path}: {e}")
                    return data
                elif 400 <= resp.status_code < 500:
                    logger.warning(f"Client error {resp.status_code} for {url} (will not retry)")
                    return None
                else:
                    logger.warning(f"Server error {resp.status_code} for {url} (attempt {attempt+1}/{max_retries+1})")
                    if attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(f"Network error for {url}: {e} (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                logger.error(f"Unexpected error requesting {url}: {e}")
                return None

        return None

    def download_file(
        self,
        url: str,
        target_filename: str | None = None,
        offline: bool = False,
        max_size_mb: float | None = None,
    ) -> Path | None:
        """Downloads a media file into the download cache and returns its local Path."""
        hashed = hashlib.sha256(url.encode("utf-8")).hexdigest()
        ext = Path(url.split("?")[0]).suffix or ".tmp"
        filename = target_filename or f"{hashed}{ext}"
        dest_path = self.downloads_dir / filename

        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path

        if offline:
            logger.debug(f"Offline mode: cached download not found for {url}")
            return None

        effective_max_mb = max_size_mb if max_size_mb is not None else self.max_total_media_size_mb
        max_bytes = int(effective_max_mb * 1024 * 1024) if effective_max_mb and effective_max_mb > 0 else None

        headers = {"User-Agent": self.user_agent}
        max_retries = 2
        for attempt in range(max_retries + 1):
            temp_path = dest_path.with_suffix(f".downloading.{os.getpid()}")
            try:
                resp = requests.get(url, headers=headers, stream=True, timeout=30)
                if resp.status_code == 200:
                    total_downloaded = 0
                    with open(temp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                total_downloaded += len(chunk)
                                if max_bytes and total_downloaded > max_bytes:
                                    logger.warning(f"Download exceeded max size ({max_bytes} bytes): {url}")
                                    return None
                                f.write(chunk)
                    temp_path.replace(dest_path)
                    return dest_path
                elif 400 <= resp.status_code < 500:
                    logger.warning(f"Download failed with client status {resp.status_code} for {url}")
                    return None
                else:
                    logger.warning(f"Download server status {resp.status_code} for {url} (attempt {attempt+1})")
                    if attempt < max_retries:
                        time.sleep(1.5)
            except Exception as e:
                logger.warning(f"Download error for {url}: {e} (attempt {attempt+1})")
                if attempt < max_retries:
                    time.sleep(1.5)
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

        return None

    def prune_cache(
        self,
        runs_dir: Path | str = "data/runs",
        max_gb: float = 5.0,
        keep_days: int = 7,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Prunes cached downloads not referenced by retained runs and older than keep_days.

        Never touches data/media_library/ or files referenced by any run in runs_dir.
        """
        runs_path = Path(runs_dir)
        referenced_hashes: set[str] = set()

        # Collect hashes and file paths referenced in retained runs
        if runs_path.exists():
            for manifest_file in runs_path.glob("**/media_manifest.json"):
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for shot in data.get("shots", []):
                        orig_url = shot.get("origin_url")
                        if orig_url:
                            orig_hash = hashlib.sha256(orig_url.encode("utf-8")).hexdigest()
                            referenced_hashes.add(orig_hash.lower())
                        cache_key = shot.get("cache_key")
                        if cache_key:
                            referenced_hashes.add(str(cache_key).lower())
                        raw_p = shot.get("raw_path")
                        if raw_p:
                            referenced_hashes.add(Path(raw_p).stem.lower())
                            referenced_hashes.add(Path(raw_p).name.lower())
                        sha = shot.get("sha256")
                        if sha:
                            referenced_hashes.add(sha.lower())
                except Exception as e:
                    logger.warning(f"Error reading manifest {manifest_file}: {e}")

        # Scan downloads dir
        if not self.downloads_dir.exists():
            return {"deleted_count": 0, "freed_mb": 0.0, "remaining_mb": 0.0, "files": [], "dry_run": dry_run}

        now = time.time()
        max_bytes = int(max_gb * 1024 * 1024 * 1024)
        keep_seconds = keep_days * 86400

        cached_files: list[tuple[Path, int, float]] = []
        total_size = 0

        for f in self.downloads_dir.iterdir():
            if f.is_file():
                try:
                    st = f.stat()
                    total_size += st.st_size
                    cached_files.append((f, st.st_size, st.st_mtime))
                except Exception:
                    pass

        # Sort oldest first
        cached_files.sort(key=lambda x: x[2])

        deleted_count = 0
        freed_bytes = 0
        target_files: list[str] = []

        for f_path, f_size, mtime in cached_files:
            age_sec = now - mtime
            is_stale = age_sec > keep_seconds
            is_over_cap = (total_size - freed_bytes) > max_bytes

            # Check if referenced
            f_stem = f_path.stem.lower()
            f_name = f_path.name.lower()
            if f_stem in referenced_hashes or f_name in referenced_hashes:
                continue

            if is_stale or is_over_cap:
                if dry_run:
                    deleted_count += 1
                    freed_bytes += f_size
                    target_files.append(str(f_path))
                else:
                    try:
                        f_path.unlink()
                        deleted_count += 1
                        freed_bytes += f_size
                        target_files.append(str(f_path))
                        logger.info(f"Pruned cached download: {f_path.name} ({f_size / (1024*1024):.2f} MB)")
                    except Exception as e:
                        logger.warning(f"Failed to delete {f_path}: {e}")

        remaining_bytes = max(0, total_size - freed_bytes)
        return {
            "deleted_count": deleted_count,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
            "remaining_mb": round(remaining_bytes / (1024 * 1024), 2),
            "files": target_files,
            "dry_run": dry_run,
        }
