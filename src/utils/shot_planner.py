"""Shot planning algorithm: splits scenes into timed shots snapped to sentence boundaries."""
from typing import Any
from src.utils.media_models import PlannedScene, PlannedShot


def estimate_timeline_scenes_from_script(
    script_scenes: list[dict[str, Any]],
    words_per_second: float = 2.5,
    min_scene_sec: float = 4.0,
) -> list[dict[str, Any]]:
    """Builds provisional timeline-like scenes from script.json when audio timeline.json is not yet available.

    Estimates scene and sentence start/end timestamps based on narration word counts
    or estimated_duration_sec, enabling provisional shot planning for visual candidate discovery.
    """
    from src.agents.voice import split_into_sentences

    estimated_scenes: list[dict[str, Any]] = []
    current_time = 0.0

    for sc in script_scenes:
        sc_id = sc.get("id", "scene_00")
        narration = sc.get("narration", "").strip()

        # Determine estimated duration
        raw_dur = sc.get("estimated_duration_sec")
        if raw_dur is not None and float(raw_dur) > 0:
            scene_duration = round(float(raw_dur), 3)
        else:
            w_count = max(1, len(narration.split()))
            scene_duration = max(min_scene_sec, round(w_count / words_per_second, 3))

        scene_start = round(current_time, 3)
        scene_end = round(scene_start + scene_duration, 3)
        current_time = scene_end

        # Split narration into sentences and assign proportional timestamps
        sentences_text = split_into_sentences(narration) if narration else [sc.get("headline", sc_id)]
        if not sentences_text:
            sentences_text = [sc.get("headline", sc_id)]

        sentence_word_counts = [max(1, len(s.split())) for s in sentences_text]
        total_words = sum(sentence_word_counts)

        sentences: list[dict[str, Any]] = []
        sent_curr = scene_start
        for idx, s_text in enumerate(sentences_text):
            if idx == len(sentences_text) - 1:
                s_end = scene_end
            else:
                ratio = sentence_word_counts[idx] / total_words
                s_dur = round(scene_duration * ratio, 3)
                s_end = round(sent_curr + s_dur, 3)
                s_end = min(s_end, round(scene_end - 0.1, 3))

            sentences.append({
                "text": s_text,
                "start": round(sent_curr, 3),
                "end": round(s_end, 3),
            })
            sent_curr = s_end

        estimated_scenes.append({
            "id": sc_id,
            "start": scene_start,
            "end": scene_end,
            "duration": scene_duration,
            "sentences": sentences,
        })

    return estimated_scenes


def plan_shots_for_scenes(
    scenes: list[dict[str, Any]],
    target_duration: float = 7.0,
    min_duration: float = 4.0,
    max_duration: float = 10.0,
    snap_tolerance: float = 1.0,
) -> list[PlannedScene]:
    """Computes exact-tiled shot boundaries for each scene based on sentence timings."""
    planned_scenes: list[PlannedScene] = []

    for scene in scenes:
        scene_id = scene["id"]
        scene_start = round(float(scene["start"]), 3)
        scene_end = round(float(scene["end"]), 3)
        scene_duration = round(scene_end - scene_start, 3)

        if scene_duration <= 0:
            continue

        # Number of shots = round(scene duration / target_duration), at least 1
        num_shots = max(1, round(scene_duration / target_duration))

        # If 1 shot would be longer than max_duration, allow splitting if possible
        if num_shots == 1 and scene_duration >= min_duration * 2 and scene_duration > max_duration:
            num_shots = 2

        # Extract sentence end boundaries from timeline
        sentence_boundaries: list[float] = []
        for s in scene.get("sentences", []):
            s_end = round(float(s["end"]), 3)
            # Boundary must be strictly inside scene bounds with headroom for min_duration
            if (scene_start + min_duration) <= s_end <= (scene_end - min_duration):
                sentence_boundaries.append(s_end)

        # Compute ideal cut points
        avg_shot_len = scene_duration / num_shots
        ideal_cuts = [round(scene_start + (i * avg_shot_len), 3) for i in range(1, num_shots)]

        # Determine actual cut points, snapping to sentence boundaries if within tolerance
        actual_cuts: list[float] = []
        snapped_flags: list[bool] = []

        last_cut = scene_start
        for ideal in ideal_cuts:
            if (scene_end - last_cut) < 2 * min_duration:
                break
            best_snap = None
            min_dist = float("inf")

            for b in sentence_boundaries:
                dist = abs(b - ideal)
                if dist <= snap_tolerance and dist < min_dist:
                    # Verify boundary respects min_duration from last_cut and remaining duration
                    if (b - last_cut) >= min_duration and (scene_end - b) >= min_duration:
                        best_snap = b
                        min_dist = dist

            if best_snap is not None:
                actual_cuts.append(best_snap)
                snapped_flags.append(True)
                last_cut = best_snap
            else:
                # Keep ideal cut, ensuring min_duration spacing
                cut = max(last_cut + min_duration, min(scene_end - min_duration, ideal))
                if (cut - last_cut) < min_duration or (scene_end - cut) < min_duration:
                    break
                actual_cuts.append(round(cut, 3))
                snapped_flags.append(False)
                last_cut = cut

        # Construct shots tiling the scene exactly, deduplicating monotonic boundaries
        boundaries: list[float] = [scene_start]
        for b in actual_cuts:
            b_val = round(b, 3)
            if b_val > boundaries[-1] and (scene_end - b_val) >= min_duration:
                boundaries.append(b_val)
        if scene_end > boundaries[-1]:
            boundaries.append(round(scene_end, 3))
        elif len(boundaries) == 1:
            boundaries.append(round(scene_end, 3))
        shots: list[PlannedShot] = []

        for i in range(len(boundaries) - 1):
            s_start = boundaries[i]
            s_end = boundaries[i + 1]
            s_dur = round(s_end - s_start, 3)
            snapped = snapped_flags[i] if i < len(snapped_flags) else False
            ideal = ideal_cuts[i] if i < len(ideal_cuts) else None

            shots.append(
                PlannedShot(
                    scene_id=scene_id,
                    shot_index=i,
                    start=s_start,
                    end=s_end,
                    duration=s_dur,
                    ideal_cut=ideal,
                    snapped_to_sentence=snapped,
                )
            )

        planned_scenes.append(
            PlannedScene(
                scene_id=scene_id,
                start=scene_start,
                end=scene_end,
                duration=scene_duration,
                shots=shots,
            )
        )

    return planned_scenes
