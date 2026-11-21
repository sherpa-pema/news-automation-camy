"""Semantic Scene Classifier for Motion Graphics.

Replaces Shotcraft / Opus Prompts heuristics with a clean, robust classifier
that maps story content to HyperFrames composition templates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class SceneClassification:
    """Classification result for a scene's motion graphics template."""
    recipe: str
    template_name: str
    data: dict[str, Any] = field(default_factory=dict)
    has_featured_card: bool = False


def classify_scene(
    sc_script: dict[str, Any],
    has_video: bool = False,
) -> SceneClassification:
    """Classifies a scene into a HyperFrames motion graphics archetype.

    Supported archetypes:
      - title_card: Broadcast news intro title card.
      - outro_card: Outro card with attribution and synthetic voice disclosure.
      - stat_callout: Quantitative hero callouts with large metric, delta, and label.
      - quote_card: Authoritative spokesperson or official statement quote card.
      - headline_split: Multi-bullet structured briefing card.
      - lower_third / cinematic_broll_full: Default story overlay with animated lower-third strap.
    """
    sc_type = sc_script.get("type", "story")
    if sc_type in ("intro", "title_card"):
        return SceneClassification(
            recipe="title_card",
            template_name="title_card.html",
            data={},
            has_featured_card=True,
        )
    if sc_type in ("outro", "outro_card"):
        return SceneClassification(
            recipe="outro_card",
            template_name="outro_card.html",
            data={},
            has_featured_card=True,
        )

    headline = sc_script.get("headline", "")
    narration = sc_script.get("narration", "")
    full_text = f"{headline}. {narration}".strip()

    # 1. Quote Card heuristic (match true double quotes or directional single quotes; never plain ASCII apostrophe)
    quote_match = re.search(r'["“\u2018]([^"”\u2018\u2019]{15,100})["”\u2019]', full_text)
    if quote_match:
        quote_snippet = quote_match.group(1).strip()
        speaker = "Official Statement"
        # Check both "Name said/stated/told/भने/बताए" and "said/stated/according to/अनुसार Name"
        sp_pre = re.search(r'([A-Z\u0900-\u097F][a-z\u0900-\u097F]+(?:\s+[A-Z\u0900-\u097F][a-z\u0900-\u097F]+){1,3})\s+(?:said|stated|told|भने|बताए|दाबी गरे)', full_text)
        sp_post = re.search(r'(?:said|stated|told|according to|spokesperson|अनुसार|भनाइमा)\s+([A-Z\u0900-\u097F][a-z\u0900-\u097F]+(?:\s+[A-Z\u0900-\u097F][a-z\u0900-\u097F]+){1,3})', full_text)
        if sp_pre and sp_pre.group(1).strip() not in ("The", "This", "That", "Managing Director", "President", "Prime Minister", "मन्त्रालय", "विभाग"):
            speaker = sp_pre.group(1).strip()
        elif sp_post:
            speaker = sp_post.group(1).strip()
        elif sp_pre:
            speaker = sp_pre.group(1).strip()

        sources = sc_script.get("sources", [])
        src_str = sources[0] if isinstance(sources, list) and sources else "Verified Outlets"
        return SceneClassification(
            recipe="quote_card",
            template_name="quote_card.html",
            data={
                "quote": quote_snippet,
                "speaker": speaker,
                "title": "Spokesperson / Official",
                "source": src_str,
            },
            has_featured_card=True,
        )

    # 2. Stat Callout heuristic
    stat_val = None
    stat_delta = None

    curr_match = re.search(r'((?:Rs\.?|रू|USD|\$)\s*[\d,]+(?:\.\d+)?\s*(?:crore|billion|lakh|million|trillion|tola|करोड|अर्ब|लाख|हजार)?)', full_text, re.IGNORECASE)
    pct_match = re.search(r'([\+\-]?[\d]+(?:\.\d+)?\s*%)', full_text)
    mw_match = re.search(r'([\d,]+(?:\.\d+)?\s*(?:MW|megawatt|metric tons|km|tola|spots))', full_text, re.IGNORECASE)

    if curr_match:
        stat_val = curr_match.group(1).strip()
    elif pct_match:
        stat_val = pct_match.group(1).strip()
    elif mw_match:
        stat_val = mw_match.group(1).strip()

    if stat_val and len(stat_val) < 25:
        delta_match = re.search(r'(\b(?:increased|decreased|surged|fell|dropped|jumped|rose|up|down|record\s+high)\s+by\s+[\d\.]+%?)', full_text, re.IGNORECASE)
        if delta_match:
            stat_delta = delta_match.group(1).strip()
        elif "+" in stat_val or "-" in stat_val:
            stat_delta = "TREND"
        label = headline[:65]
        return SceneClassification(
            recipe="stat_callout",
            template_name="stat_callout.html",
            data={
                "value": stat_val,
                "label": label,
                "delta": stat_delta,
            },
            has_featured_card=True,
        )

    # 3. Headline Split heuristic (multi-bullet briefing)
    sentences = [s.strip() for s in re.split(r'[.!?।॥]+', narration) if s.strip()]
    if len(sentences) >= 3 and not has_video:
        kicker = sc_script.get("category") or "DEVELOPING"
        bullets = [sentences[0][:60], sentences[1][:60]]
        return SceneClassification(
            recipe="headline_split",
            template_name="headline_split.html",
            data={"kicker": kicker, "bullets": bullets},
            has_featured_card=True,
        )

    # 4. Default: Standard story with animated lower third
    return SceneClassification(
        recipe="cinematic_broll_full",
        template_name="lower_third.html",
        data={},
        has_featured_card=False,
    )
