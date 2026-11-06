# Motion Agent Specification

## 1. Purpose
The Motion Agent orchestrates the graphic design and motion composition plan for the broadcast video. Utilizing scene timings from `timeline.json` and collected assets from `media_manifest.json`, it defines lower thirds, headline cards, news ticker styling, transition curves, caption placements, and mandatory source credit overlays, outputting a complete `render_plan.json`.

## 2. Ground Rules
- **Mandatory On-Screen Attribution**: Every scene design must include a persistent lower-third or corner badge crediting the news outlet (e.g., "Source: The Kathmandu Post").
- **Kinetic Continuity**: Motion elements (transitions, lower-third arrivals) must feel cohesive, adhering to smooth easing curves rather than abrupt static jumps.
- **Timing Synchronization**: Transitions and graphic animations must strictly respect the scene boundaries provided in `timeline.json`.
- **Readability**: Ensure text overlays and captions maintain high contrast against dynamic background media.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/timeline.json`: Scene boundaries and timing intervals.
  - `data/runs/<YYYY-MM-DD>/media_manifest.json`: Verified visuals and attributions per scene.
  - `data/runs/<YYYY-MM-DD>/script.json`: Scene headlines and source publisher names.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/render_plan.json`: Complete declarative rendering blueprint.

## 4. JSON Schema Contract
- Must satisfy: [render_plan.schema.json](../contracts/render_plan.schema.json)
- Structure: `{ resolution, fps, scenes: [{ id, media, lower_third, caption_style, transition }] }`

## 5. Tools & Modules
- Design Rule Engine / LLM prompt: Generates motion parameters adhering to broadcast television standards.
- Color & Palette System: High-contrast news palette (navy/crimson/white/gold).
- Filter-Graph Generator: Synthesizes FFmpeg complex filter graphs or Remotion composition JSON.

## 6. Failure Modes
- **Missing Scene Specification**: If any scene from `timeline.json` is omitted from `render_plan.json`, validation fails immediately.
- **Transition Overlap Conflict**: Transition duration exceeding scene length triggers duration auto-clamping.
- **Unreadable Typography**: Missing font parameters or low-contrast presets trigger default fallback template.

## 7. Definition of Done Checklist
- [ ] Every scene from `timeline.json` has an entry in `render_plan.json`.
- [ ] On-screen source credit specified for every news story.
- [ ] Target resolution (e.g. `1920x1080`) and FPS (e.g. 30) defined.
- [ ] Transition types and durations specified between adjacent scenes.
- [ ] Output `render_plan.json` validates against `render_plan.schema.json`.
