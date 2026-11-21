"""Unit tests for unattended notifier and orchestrator lifecycle alerts."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.orchestrator import Orchestrator
from src.utils.notifier import Notifier, get_lan_ip, get_web_office_url


def test_notifier_unconfigured_is_safe_noop() -> None:
    notifier = Notifier(telegram_token=None, telegram_chat_id=None, discord_webhook_url=None)
    assert notifier.is_configured is False
    # send_raw and notification methods should succeed without errors
    assert notifier.send_raw("Test message") is True
    assert notifier.notify_morning_ready("2026-11-21", 5, 120.0) is True
    assert notifier.notify_failure("2026-11-21", "render", "Crash") is True
    assert notifier.notify_publish_complete("2026-11-21", 125.0, "-16.0 LUFS, -1.0 dBTP", "final.mp4") is True


def test_notifier_telegram_dispatch() -> None:
    notifier = Notifier(
        telegram_token="123456:ABC-DEF",
        telegram_chat_id="987654321",
        discord_webhook_url=None,
    )
    assert notifier.is_configured is True

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        ok = notifier.notify_morning_ready(
            run_date="2026-11-21",
            story_count=4,
            estimated_duration_sec=145.2,
            office_url="http://192.168.1.100:8000",
        )
        assert ok is True
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert "api.telegram.org/bot123456:ABC-DEF/sendMessage" in req.full_url
        data = json.loads(req.data.decode("utf-8"))
        assert data["chat_id"] == "987654321"
        assert "Stories: 4" in data["text"]
        assert "192.168.1.100:8000" in data["text"]


def test_notifier_discord_dispatch() -> None:
    notifier = Notifier(
        telegram_token=None,
        telegram_chat_id=None,
        discord_webhook_url="https://discord.com/api/webhooks/123/xyz",
    )
    assert notifier.is_configured is True

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        ok = notifier.notify_failure(
            run_date="2026-11-21",
            failing_stage="scraper",
            error_snippet="Timeout error after 180s",
            redo_hint="python run_daily.py --reset-stage scraper",
        )
        assert ok is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://discord.com/api/webhooks/123/xyz"
        data = json.loads(req.data.decode("utf-8"))
        assert "Failing Stage: scraper" in data["content"]
        assert "Timeout error after 180s" in data["content"]


def test_notifier_publish_complete_dispatch() -> None:
    notifier = Notifier(
        discord_webhook_url="https://discord.com/api/webhooks/test",
    )
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        ok = notifier.notify_publish_complete(
            run_date="2026-11-21",
            final_duration_sec=152.0,
            lufs_summary="-16.2 LUFS, -1.2 dBTP",
            output_path="data/runs/2026-11-21/render/final.mp4",
        )
        assert ok is True
        req = mock_urlopen.call_args[0][0]
        data = json.loads(req.data.decode("utf-8"))
        assert "Final Duration: 2m 32s" in data["content"]
        assert "-16.2 LUFS, -1.2 dBTP" in data["content"]


def test_notifier_network_failure_is_non_fatal() -> None:
    notifier = Notifier(
        telegram_token="fake_tok",
        telegram_chat_id="fake_chat",
        discord_webhook_url="https://fake.url",
    )
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        # Should not raise exception
        ok = notifier.notify_failure("2026-11-21", "qa", "QA failed")
        assert ok is False


def test_orchestrator_notification_dispatch(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "pipeline.yaml").write_text(
        "execution:\n  runs_dir: 'runs'\nnotifications:\n  enabled: true\nstages: []\n",
        encoding="utf-8",
    )
    (config_dir / "sources.yaml").write_text("sources: {}\n", encoding="utf-8")

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)

    orch = Orchestrator(
        run_date="2026-11-21",
        config_dir=config_dir,
        runs_dir=runs_dir,
    )
    manifest = orch.initialize_run()
    mock_notifier = MagicMock()
    orch.notifier = mock_notifier

    orch._mark_run_failed(manifest, "FFmpeg encode failure", failing_stage="render")
    assert mock_notifier.notify_failure.called
    call_args = mock_notifier.notify_failure.call_args[1]
    assert call_args["failing_stage"] == "render"
    assert "FFmpeg encode failure" in call_args["error_snippet"]
