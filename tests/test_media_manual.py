"""Tests for manual-first MediaAgent workflow: upload, link, and zero internet pulling."""
import json
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.agents.media import MediaAgent
from web_office.server import create_app


@pytest.fixture
def run_setup(tmp_path):
    run_date = "2026-11-13"
    run_dir = tmp_path / "data" / "runs" / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "script").mkdir(parents=True, exist_ok=True)
    (run_dir / "media" / "review").mkdir(parents=True, exist_ok=True)

    # script.json
    script_data = {
        "date": run_date,
        "origin": "fixture",
        "scenes": [
            {
                "id": "scene_01",
                "headline": "Kathmandu Tech Summit",
                "narration": "Kathmandu is hosting the annual tech innovation summit.",
                "visual_keywords": ["tech summit", "kathmandu conference"],
            }
        ],
    }
    (run_dir / "script" / "script.json").write_text(json.dumps(script_data), encoding="utf-8")

    # timeline.json
    timeline_data = {
        "date": run_date,
        "scenes": [
            {
                "id": "scene_01",
                "start": 0.0,
                "end": 8.0,
                "duration": 8.0,
                "sentences": [
                    {"text": "Kathmandu is hosting the annual tech innovation summit.", "start": 0.0, "end": 8.0}
                ],
            }
        ],
    }
    (run_dir / "timeline.json").write_text(json.dumps(timeline_data), encoding="utf-8")

    # manifest.json
    manifest_data = {
        "date": run_date,
        "stages": {
            "editor": {"status": "done"},
            "voice": {"status": "done", "metrics": {"disclose_synthetic_voice": True}},
            "timeline": {"status": "done"},
            "media": {"status": "pending"},
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    return tmp_path, run_dir, run_date


def test_candidates_mode_with_internet_search_disabled(run_setup):
    """Verify run_candidates produces instant card proposals and no web candidates when internet_search is false."""
    tmp_path, run_dir, run_date = run_setup
    agent = MediaAgent(run_dir, allow_fixture=True)
    # Ensure internet_search is False
    assert agent.selector.internet_search is False

    c_data = agent.run_candidates(candidates_k=6)
    assert c_data["total_shots"] > 0
    for sh in c_data["shots"]:
        assert sh["auto_pick_type"] == "card"
        assert len(sh["candidates"]) == 0

    picker_path = run_dir / "media" / "review" / "picker.html"
    assert picker_path.exists()
    picker_html = picker_path.read_text(encoding="utf-8")
    assert "Upload Video or Image" in picker_html
    assert "Direct Web Link" in picker_html
    assert "Branded 1080p Headline Card" in picker_html


def test_candidates_mode_without_timeline(run_setup):
    """Verify run_candidates works in parallel before timeline.json exists (provisional planning)."""
    tmp_path, run_dir, run_date = run_setup
    # Remove timeline.json to simulate running in parallel with voice
    (run_dir / "timeline.json").unlink()

    agent = MediaAgent(run_dir, allow_fixture=True)
    c_data = agent.run_candidates(candidates_k=6)
    assert c_data["total_shots"] > 0
    shot_plan_path = run_dir / "media" / "shot_plan.json"
    assert shot_plan_path.exists()
    plan = json.loads(shot_plan_path.read_text(encoding="utf-8"))
    assert plan.get("provisional") is True

    picker_path = run_dir / "media" / "review" / "picker.html"
    assert picker_path.exists()


def test_apply_picks_with_manual_upload(run_setup):
    """Verify apply_picks processes an uploaded image/video cleanly."""
    tmp_path, run_dir, run_date = run_setup

    # Create dummy user uploaded image
    uploads_dir = run_dir / "media" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    img_file = uploads_dir / "scene_01_00_custom.png"
    im = Image.new("RGB", (1280, 720), color=(100, 150, 200))
    im.save(img_file, "PNG")

    picks_data = {
        "date": run_date,
        "picks": {
            "scene_01_00": {
                "action": "upload",
                "scene_id": "scene_01",
                "shot_index": 0,
                "file_path": "media/uploads/scene_01_00_custom.png",
            }
        },
    }
    picks_file = run_dir / "media" / "review" / "picks.json"
    picks_file.write_text(json.dumps(picks_data), encoding="utf-8")

    agent = MediaAgent(run_dir, allow_fixture=True)
    outputs = agent.apply_picks(picks_file)
    assert "media/media_manifest.json" in outputs

    manifest = json.loads((run_dir / "media" / "media_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["shots"]) == 1
    shot = manifest["shots"][0]
    assert shot["type"] == "image"
    assert shot["source"] == "user-provided"
    assert shot["file"].startswith("media/shots/scene_01/")
    assert shot["sha256"] is not None
    assert (run_dir / shot["file"]).exists()


def test_web_office_upload_and_validate_endpoints(run_setup):
    """Verify web_office /upload-media and /validate-media-link endpoints."""
    tmp_path, run_dir, run_date = run_setup
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1. Test validate-media-link with blocklisted news domain
    res_bad = client.post(
        f"/api/runs/{run_date}/validate-media-link",
        json={"url": "https://kathmandupost.com/image.jpg"},
    )
    assert res_bad.status_code == 400
    assert "blocklist" in res_bad.json()["detail"].lower()

    # 2. Test validate-media-link with acceptable domain
    res_ok = client.post(
        f"/api/runs/{run_date}/validate-media-link",
        json={"url": "https://example.com/footage.mp4"},
    )
    assert res_ok.status_code == 200
    assert res_ok.json()["domain"] == "example.com"

    # 3. Test upload-media with dummy image
    test_img = tmp_path / "sample.jpg"
    Image.new("RGB", (640, 480), color=(50, 100, 150)).save(test_img, "JPEG")

    with open(test_img, "rb") as f:
        res_upload = client.post(
            f"/api/runs/{run_date}/upload-media",
            data={"shot_id": "scene_01_00"},
            files={"file": ("sample.jpg", f, "image/jpeg")},
        )
    assert res_upload.status_code == 200
    up_data = res_upload.json()
    assert up_data["ok"] is True
    assert up_data["shot_id"] == "scene_01_00"
    assert up_data["kind"] == "image"
    assert (run_dir / up_data["file_path"]).exists()
    assert (run_dir / "media" / "review" / "thumbs" / "scene_01_00_upload.jpg").exists()
