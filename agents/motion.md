# Motion Agent Specification

## 1. Purpose
The Motion Agent orchestrates the graphic design, overlay planning, and camera motion blueprint for the broadcast video. Utilizing scene timings from `timeline.json` and collected assets from `media_manifest.json`, it generates broadcast overlay graphics (lower thirds, stat callouts, quote cards, headline splits, title/outro cards), formatted ASS captions, camera motion variety, and transition grammar, outputting `motion/render_plan.json`, `motion/captions.ass`, and `motion/overlays/*.png`.

## 2. Ground Rules
- **Mandatory On-Screen Attribution**: Every story scene must include an on-screen lower-third or featured card crediting the verified news outlet (e.g., "Source: The Kathmandu Post"), sanitized to prevent overflow.
- **Kinetic Continuity & Motion Variety**: Stills use alternating `zoom_in`/`zoom_out` camera motion rendered with supersampled lanczos downscaling, avoiding static freezes or sub-pixel stepping.
- **Broadcast Transition Grammar**: Adjacent story scenes cut cleanly (`cut`), while section boundaries (`intro -> story`, `story -> outro`, topic shifts) use calibrated `dip_to_black`.
- **Readability & Safe Zones**: Overlays and captions adhere to 5% action-safe television boundaries and maintain a strict >= 24px vertical clearance above the caption band.
- **Centralized Design Palette**: All font styles, layout rectangles, colors, and margins derive directly from `config/style.yaml`.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/timeline/timeline.json`: Scene boundaries and timing intervals.
  - `data/runs/<YYYY-MM-DD>/timeline/captions.json`: Timed transcript text segments.
  - `data/runs/<YYYY-MM-DD>/media/media_manifest.json`: Verified visuals and attributions per scene.
  - `data/runs/<YYYY-MM-DD>/script/script.json`: Scene headlines, categories, and source publisher names.
  - `data/runs/<YYYY-MM-DD>/audio/narration.wav`: Master narration audio track.
  - `config/style.yaml`: Design tokens, layout rectangles, and typography rules.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/motion/render_plan.json`: Complete declarative rendering blueprint.
  - `data/runs/<YYYY-MM-DD>/motion/captions.ass`: Complementary ASS subtitles.
  - `data/runs/<YYYY-MM-DD>/motion/overlays/*.png`: Broadcast graphics overlay assets.

## 4. JSON Schema Contract
- Must satisfy: [render_plan.schema.json](../contracts/render_plan.schema.json)
- Structure: `{ resolution, fps, duration_sec, scenes: [{ id, media, lower_third, caption_style, transition }], overlays }`

## 5. Tools & Modules
- **Semantic Scene Classifier** (`src/utils/scene_classifier.py`): Maps news story context into template archetypes (`title_card`, `stat_callout`, `quote_card`, `headline_split`, `cinematic_broll_full`, `outro_card`).
- **Overlay Planner & Layout Validator** (`src/utils/overlay_plan.py`): Single source of truth for overlay geometry, attribution sanitization, motion variety resolution, and pre-render safe-zone validation.
- **Motion Graphics Renderer** (`src/utils/motion_graphics.py`): High-resolution PIL/Pillow graphic generator for broadcast lower thirds, cards, tags, and subtitle styling.
- **Motion Timing & Transition Grammar** (`src/utils/motion_timing.py`): Resolves intra-story cuts vs section boundary dips.
- **Caption Styler** (`src/utils/captions.py`): Derives caption dimensions, line heights, and margins to guarantee safe lower-third clearance.

## 6. Failure Modes
- **Missing Scene Specification**: If any scene from `timeline.json` is omitted from `render_plan.json`, validation fails immediately.
- **Attribution Fallback**: If a story has a script source but lower-third credit falls back to channel name, QA inspection fails.
- **Safe Zone Collision**: Any card or text violating the 5% boundary, colliding with another card, or overlapping the caption band triggers validation failure.

## 7. Definition of Done Checklist
- [ ] Every scene from `timeline.json` has an entry in `render_plan.json`.
- [ ] On-screen source credit specified for every news story.
- [ ] Semantic scene recipe assigned (`title_card`, `stat_callout`, `quote_card`, `headline_split`, `cinematic_broll_full`, `outro_card`).
- [ ] Target resolution (e.g. `1920x1080`) and FPS (e.g. 30) defined.
- [ ] Layout validation passes with zero collisions and safe caption clearance.
- [ ] Output `render_plan.json` validates against `render_plan.schema.json`.
