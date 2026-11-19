"""Transition resolution and motion timing rules conforming to broadcast grammar."""
from __future__ import annotations

from typing import Any


def resolve_transitions(
    scene: dict[str, Any],
    prev_scene: dict[str, Any] | None = None,
    style_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolves scene-to-scene transition type and duration.

    Transition Grammar:
      - 'cut' for intra-story cutaways, same-topic adjacency, and adjacent story beats.
      - 'dip_to_black' for major section boundaries: intro -> story, story -> outro,
        or when explicit topic category changes.
      - 'fade_white' for outro if configured in style.yaml.
    """
    cfg = style_config or {}
    transitions_cfg = cfg.get("transitions", {})
    boundary_cfg = transitions_cfg.get("scene_boundary", {})
    dip_dur = float(boundary_cfg.get("duration_sec", 0.2))
    fade_out = float(boundary_cfg.get("fade_out_sec", dip_dur / 2.0))
    fade_in = float(boundary_cfg.get("fade_in_sec", dip_dur / 2.0))

    if prev_scene is None:
        # First scene of composition has no predecessor
        return {
            "type": "cut",
            "duration_sec": 0.0,
            "fade_out_sec": 0.0,
            "fade_in_sec": 0.0,
        }

    curr_type = scene.get("type", scene.get("shot_recipe", "story"))
    curr_id = scene.get("id", "")
    prev_type = prev_scene.get("type", prev_scene.get("shot_recipe", "story"))
    prev_id = prev_scene.get("id", "")

    # 1. Story -> Outro boundary
    if curr_type == "outro" or curr_id == "outro" or curr_type == "outro_card":
        outro_cfg = transitions_cfg.get("outro", {})
        if outro_cfg.get("type") == "fade_white":
            dur = float(outro_cfg.get("duration_sec", 0.3))
            return {
                "type": "fade_white",
                "duration_sec": dur,
                "fade_out_sec": dur / 2.0,
                "fade_in_sec": dur / 2.0,
            }
        return {
            "type": "dip_to_black",
            "duration_sec": dip_dur,
            "fade_out_sec": fade_out,
            "fade_in_sec": fade_in,
        }

    # 2. Intro -> Story boundary
    if prev_type == "intro" or prev_id == "intro" or prev_type == "title_card":
        return {
            "type": "dip_to_black",
            "duration_sec": dip_dur,
            "fade_out_sec": fade_out,
            "fade_in_sec": fade_in,
        }

    # 3. Topic or Section change
    curr_topic = scene.get("topic") or scene.get("category")
    prev_topic = prev_scene.get("topic") or prev_scene.get("category")
    if curr_topic and prev_topic and curr_topic != prev_topic:
        return {
            "type": "dip_to_black",
            "duration_sec": dip_dur,
            "fade_out_sec": fade_out,
            "fade_in_sec": fade_in,
        }

    # 4. Same topic, intra-story cutaway, or story adjacency: clean cut
    return {
        "type": "cut",
        "duration_sec": 0.0,
        "fade_out_sec": 0.0,
        "fade_in_sec": 0.0,
    }
