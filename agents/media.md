# Media Agent Specification

## 1. Purpose
The Media Agent manages and normalizes high-quality visual assets (B-roll footage, photos, user-provided footage, web links, and 1080p graphic cards) for each scene in `script.json` using the scene's `visual_keywords` and the interactive Visual Picker. To ensure precision, eliminate irrelevant automated web results, and minimize latency, media is assigned 100% manually per run via the Visual Picker (direct file upload, media URL, graphic cards, or curated local footage). It validates file formats and licenses, downloads and normalizes assets locally, and compiles an attribution ledger in `media_manifest.json`.

## 2. Ground Rules
- **ZERO News Outlet Scraping**: NEVER download, scrape, or extract photos or video footage from news publishers (Onlinekhabar, Kathmandu Post, Republica, etc.).
- **Strict Licensing**: Visuals must come exclusively from free, open-licensed, or public domain sources (Pexels, Pixabay, Wikimedia Commons, CC-BY, CC0).
- **Attribution Ledger**: Every single asset must have its origin URL, license type, and creator attribution recorded in `media_manifest.json`.
- **Restricted yt-dlp Usage**: If using `yt-dlp`, it must ONLY download videos explicitly tagged under Creative Commons or Public Domain licenses; standard copyrighted videos are forbidden.
- **Concurrency**: Runs concurrently alongside the Voice Agent once `script.json` is ready.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/script.json`: Scene IDs and visual keywords.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/media_manifest.json`: Full manifest of downloaded assets, licenses, and attributions.
  - `data/runs/<YYYY-MM-DD>/media/<scene_id>.*`: Local image and video files.

## 4. JSON Schema Contract
- Must satisfy: [media_manifest.schema.json](../contracts/media_manifest.schema.json)
- Structure per item: `{ scene_id, file, type, origin_url, license, attribution }`

## 5. Tools & Modules
- Manual Assignment & Review: Interactive Visual Picker (`picker.html`), Web Office upload endpoints (`/api/runs/<date>/upload-media`, `/api/runs/<date>/validate-media-link`).
- Ingestion & Downloaders: `requests` for direct media URLs (with news outlet domain blocklist checks).
- Image / Video Normalizer: `ffmpeg` / `ffprobe` / `Pillow` to normalize all assets to target resolution (1920x1080), framerate (30fps), H.264 video (`-an`), or 1080p JPEG images.
- Graphic Card Renderer: Branded 1080p gradient card generator (`draw_card_image`).

## 6. Failure Modes
- **No Manual Media Supplied for Shot**: Defaults cleanly to the branded 1080p graphic card with the scene headline (`type: "card"`).
- **Corrupted or Incompatible Media**: Normalization filter detects unreadable media or bad codec and generates an alert fallback card.
- **Blocklisted Domain**: Download from news portals or copyright-restricted domains is strictly blocked.

## 7. Definition of Done Checklist
- [ ] At least one verified visual asset acquired for every scene.
- [ ] Zero assets acquired from news websites or unauthorized repositories.
- [ ] All downloaded media files exist locally in `media/` and are readable.
- [ ] `origin_url`, `license`, and `attribution` logged for every item.
- [ ] Output `media_manifest.json` validates against `media_manifest.schema.json`.
