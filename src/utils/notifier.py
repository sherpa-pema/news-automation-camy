"""Lightweight webhook notifier supporting Telegram Bot API and Discord webhooks.

Configured exclusively via environment variables (never committed to YAML):
- TELEGRAM_BOT_TOKEN: Telegram Bot API token
- TELEGRAM_CHAT_ID: Telegram target chat/channel ID
- DISCORD_WEBHOOK_URL: Discord webhook URL
- WEB_OFFICE_URL: Optional override for Web Office LAN URL
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import sys
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def get_lan_ip() -> str:
    """Best-effort detection of local LAN IPv4 address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def get_web_office_url(configured_url: str | None = None) -> str:
    """Returns the effective Web Office URL (env var -> config -> LAN IP fallback)."""
    env_url = os.getenv("WEB_OFFICE_URL")
    if env_url:
        return env_url.rstrip("/")
    if configured_url:
        return configured_url.rstrip("/")
    lan_ip = get_lan_ip()
    return f"http://{lan_ip}:8000"


class Notifier:
    """Non-fatal notification dispatcher for Telegram and Discord."""

    def __init__(
        self,
        telegram_token: str | None = None,
        telegram_chat_id: str | None = None,
        discord_webhook_url: str | None = None,
        timeout_seconds: float = 5.0,
        enabled: bool = True,
        web_office_url: str | None = None,
    ) -> None:
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.discord_webhook_url = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.enabled = enabled
        self.web_office_url = get_web_office_url(web_office_url)

    @property
    def is_configured(self) -> bool:
        """True if at least one notification transport is configured."""
        has_tg = bool(self.telegram_token and self.telegram_chat_id)
        has_discord = bool(self.discord_webhook_url)
        return has_tg or has_discord

    def send_raw(self, message: str) -> bool:
        """Dispatches raw message to configured endpoints with non-fatal error handling."""
        if not self.enabled:
            logger.debug("Notifier disabled; skipping message.")
            return True

        if not self.is_configured:
            logger.debug("No Telegram or Discord credentials configured; skipping notification.")
            return True

        success = True

        # 1. Telegram Bot API
        if self.telegram_token and self.telegram_chat_id:
            try:
                tg_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": message,
                    "disable_web_page_preview": True,
                }
                req = urllib.request.Request(
                    tg_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    if resp.status >= 300:
                        logger.warning(f"Telegram webhook returned HTTP {resp.status}")
                        success = False
            except Exception as exc:
                logger.warning(f"Non-fatal notification error (Telegram): {exc}")
                success = False

        # 2. Discord Webhook
        if self.discord_webhook_url:
            try:
                payload = {
                    "content": message,
                }
                req = urllib.request.Request(
                    self.discord_webhook_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    if resp.status >= 300:
                        logger.warning(f"Discord webhook returned HTTP {resp.status}")
                        success = False
            except Exception as exc:
                logger.warning(f"Non-fatal notification error (Discord): {exc}")
                success = False

        return success

    def notify_morning_ready(
        self,
        run_date: str,
        story_count: int,
        estimated_duration_sec: float,
        office_url: str | None = None,
    ) -> bool:
        """Sends notification when morning editorial prep is awaiting approval."""
        url = office_url or self.web_office_url
        mins = int(estimated_duration_sec // 60)
        secs = int(estimated_duration_sec % 60)
        dur_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{estimated_duration_sec:.1f}s"

        msg = (
            f"🌅 Nepal News Bot: Morning Run Ready for Approval\n"
            f"• Date: {run_date}\n"
            f"• Stories: {story_count}\n"
            f"• Estimated Duration: {dur_str}\n"
            f"• Web Office: {url}\n\n"
            f"Action: Review script and approve in Web Office to proceed."
        )
        try:
            return self.send_raw(msg)
        except Exception as exc:
            logger.warning(f"Non-fatal morning notification failure: {exc}")
            return False

    def notify_failure(
        self,
        run_date: str,
        failing_stage: str,
        error_snippet: str,
        redo_hint: str | None = None,
    ) -> bool:
        """Sends alert on pipeline hard failure or interruption."""
        clean_snippet = (error_snippet or "Unknown error").strip()
        if len(clean_snippet) > 280:
            clean_snippet = clean_snippet[:277] + "..."

        hint = redo_hint or f"python run_daily.py --date {run_date} --reset-stage {failing_stage}"

        msg = (
            f"🚨 Nepal News Bot: Pipeline Failure\n"
            f"• Date: {run_date}\n"
            f"• Failing Stage: {failing_stage}\n"
            f"• Error: {clean_snippet}\n"
            f"• Redo Hint: {hint}"
        )
        try:
            return self.send_raw(msg)
        except Exception as exc:
            logger.warning(f"Non-fatal failure notification error: {exc}")
            return False

    def notify_publish_complete(
        self,
        run_date: str,
        final_duration_sec: float,
        lufs_summary: str,
        output_path: str,
    ) -> bool:
        """Sends notification when final QA has passed and broadcast is published."""
        mins = int(final_duration_sec // 60)
        secs = int(final_duration_sec % 60)
        dur_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{final_duration_sec:.1f}s"

        msg = (
            f"✅ Nepal News Bot: Broadcast Published!\n"
            f"• Date: {run_date}\n"
            f"• Final Duration: {dur_str}\n"
            f"• Audio Loudness: {lufs_summary}\n"
            f"• Deliverable: {output_path}"
        )
        try:
            return self.send_raw(msg)
        except Exception as exc:
            logger.warning(f"Non-fatal publish complete notification error: {exc}")
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Nepal News Bot webhook notifier.")
    parser.add_argument("--dry-run", action="store_true", help="Print messages without network requests.")
    parser.add_argument("--test", action="store_true", help="Dispatch test alerts to configured endpoints.")
    args = parser.parse_args()

    notifier = Notifier()
    office_url = notifier.web_office_url

    print("=== Notifier Configuration ===")
    print(f"Telegram Configured: {bool(notifier.telegram_token and notifier.telegram_chat_id)}")
    print(f"Discord Configured:  {bool(notifier.discord_webhook_url)}")
    print(f"Web Office URL:      {office_url}")

    if args.dry_run or not notifier.is_configured:
        print("\n--- Dry Run Alert Previews ---")
        class PrintNotifier(Notifier):
            def send_raw(self, message: str) -> bool:
                print(f"\n[ALERT PREVIEW]\n{message}")
                return True

        notifier_dry = PrintNotifier()
        notifier_dry.notify_morning_ready("2026-11-21", 5, 142.5, office_url)
        notifier_dry.notify_failure("2026-11-21", "render", "FFmpeg exited with returncode 1", "python run_daily.py --date 2026-11-21 --only render")
        notifier_dry.notify_publish_complete("2026-11-21", 145.2, "-16.2 LUFS, -1.2 dBTP", "data/runs/2026-11-21/render/final.mp4")
        print("\n[OK] Dry run formatting validated.")
        return 0

    if args.test:
        print("\nSending test notifications...")
        ok1 = notifier.notify_morning_ready("TEST-DATE", 5, 135.0, office_url)
        ok2 = notifier.notify_failure("TEST-DATE", "scraper", "Connection timed out", "python run_daily.py --date TEST-DATE --reset-stage scraper")
        ok3 = notifier.notify_publish_complete("TEST-DATE", 138.4, "-16.0 LUFS, -1.0 dBTP", "data/latest/final.mp4")
        print(f"Dispatch status: morning={ok1}, failure={ok2}, publish={ok3}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
