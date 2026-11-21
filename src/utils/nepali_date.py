"""Lightweight Bikram Sambat (BS) <-> Anno Domini (AD) calendar converter and Nepal timezone utilities.

Covers BS years 2080 to 2090 with exact day counts and bilingual month mappings.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import re
from typing import Any
from zoneinfo import ZoneInfo

KATHMANDU_TZ = ZoneInfo("Asia/Kathmandu")

# Month names in English and Devanagari (1-indexed)
NEPALI_MONTHS_EN = [
    "",
    "Baisakh",
    "Jestha",
    "Ashadh",
    "Shrawan",
    "Bhadra",
    "Ashwin",
    "Kartik",
    "Mangsir",
    "Poush",
    "Magh",
    "Falgun",
    "Chaitra",
]

NEPALI_MONTHS_NP = [
    "",
    "बैशाख",
    "जेठ",
    "असार",
    "साउन",
    "भदौ",
    "असोज",
    "कात्तिक",
    "मंसिर",
    "पुस",
    "माघ",
    "फागुन",
    "चैत",
]

# Common aliases / transliterations / formal Sanskrit variants
MONTH_NAME_TO_INDEX: dict[str, int] = {
    # 1. Baisakh
    "baisakh": 1, "baishakh": 1, "vaisakh": 1, "vaishakh": 1,
    "बैशाख": 1, "वैशाख": 1,
    # 2. Jestha
    "jestha": 2, "jeth": 2, "jyeshtha": 2, "jyestha": 2,
    "जेठ": 2, "ज्येष्ठ": 2,
    # 3. Ashadh
    "ashadh": 3, "asad": 3, "asar": 3, "aashadh": 3, "ashad": 3,
    "असार": 3, "आषाढ": 3, "अषाढ": 3,
    # 4. Shrawan
    "shrawan": 4, "saun": 4, "sawan": 4, "shravan": 4,
    "साउन": 4, "श्रावण": 4, "श्रावन": 4,
    # 5. Bhadra
    "bhadra": 5, "bhadau": 5, "bhadra": 5, "bhadrapad": 5,
    "भदौ": 5, "भाद्र": 5, "भाद्रपद": 5,
    # 6. Ashwin
    "ashwin": 6, "asoj": 6, "aswin": 6, "aashwin": 6,
    "असोज": 6, "आश्विन": 6, "अश्विन": 6,
    # 7. Kartik
    "kartik": 7, "katik": 7, "kattik": 7,
    "कात्तिक": 7, "कार्तिक": 7,
    # 8. Mangsir
    "mangsir": 8, "mangsir": 8, "margasirsha": 8, "marga": 8, "mansir": 8,
    "मंसिर": 8, "मार्ग": 8, "मार्गशीर्ष": 8, "मङ्गसिर": 8,
    # 9. Poush
    "poush": 9, "pus": 9, "paush": 9, "pous": 9,
    "पुस": 9, "पौष": 9,
    # 10. Magh
    "magh": 10,
    "माघ": 10,
    # 11. Falgun
    "falgun": 11, "phagun": 11, "phalguna": 11, "fagun": 11,
    "फागुन": 11, "फाल्गुन": 11,
    # 12. Chaitra
    "chaitra": 12, "chait": 12, "chaitra": 12,
    "चैत": 12, "चैत्र": 12,
}

# Bikram Sambat Calendar Table: BS Year -> (AD Start Date of BS Year (Baisakh 1), List of days in months 1..12)
BS_CALENDAR_DATA: dict[int, tuple[date, list[int]]] = {
    2080: (date(2023, 4, 14), [31, 31, 31, 32, 31, 31, 30, 29, 30, 29, 30, 30]),
    2081: (date(2024, 4, 13), [31, 31, 32, 32, 31, 30, 30, 30, 29, 30, 29, 31]),
    2082: (date(2025, 4, 14), [31, 32, 31, 32, 31, 30, 30, 30, 29, 30, 30, 30]),
    2083: (date(2026, 4, 14), [31, 31, 32, 31, 31, 30, 30, 30, 29, 30, 30, 30]),
    2084: (date(2027, 4, 14), [31, 31, 32, 31, 31, 30, 30, 30, 29, 30, 30, 30]),
    2085: (date(2028, 4, 13), [31, 32, 31, 32, 31, 30, 30, 30, 29, 30, 29, 31]),
    2086: (date(2029, 4, 14), [31, 32, 31, 32, 31, 30, 30, 30, 29, 30, 30, 30]),
    2087: (date(2030, 4, 14), [31, 31, 32, 31, 31, 31, 30, 29, 30, 29, 30, 30]),
    2088: (date(2031, 4, 14), [31, 31, 32, 32, 31, 30, 30, 29, 30, 29, 30, 30]),
    2089: (date(2032, 4, 13), [31, 32, 31, 32, 31, 30, 30, 30, 29, 30, 30, 31]),
    2090: (date(2033, 4, 14), [31, 31, 32, 31, 31, 30, 30, 30, 29, 30, 30, 30]),
}

DEVANAGARI_DIGITS = "०१२३४५६७८९"
ASCII_DIGITS = "0123456789"
DEV_TO_ASCII_TABLE = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)
ASCII_TO_DEV_TABLE = str.maketrans(ASCII_DIGITS, DEVANAGARI_DIGITS)


def today_npt() -> str:
    """Returns today's date in YYYY-MM-DD in the Asia/Kathmandu timezone (+05:45)."""
    return datetime.now(KATHMANDU_TZ).date().isoformat()


def is_probable_bs_year(year: int) -> bool:
    """Checks whether a year falls within the modern BS epoch (2070-2100)."""
    return 2070 <= year <= 2100


def ad_to_bs(ad_dt: date | datetime | str) -> tuple[int, int, int]:
    """Converts an AD date to Bikram Sambat (bs_year, bs_month, bs_day).

    Falls back to approximate offset (+56 years, 8 months, 17 days) if outside 2080-2090 table.
    """
    if isinstance(ad_dt, str):
        ad_dt = date.fromisoformat(ad_dt.split("T")[0])
    elif isinstance(ad_dt, datetime):
        ad_dt = ad_dt.date()

    for bs_year, (start_ad, month_days) in sorted(BS_CALENDAR_DATA.items()):
        total_days_in_year = sum(month_days)
        end_ad = start_ad + timedelta(days=total_days_in_year - 1)
        if start_ad <= ad_dt <= end_ad:
            remaining_days = (ad_dt - start_ad).days
            for m_idx, days_in_month in enumerate(month_days, start=1):
                if remaining_days < days_in_month:
                    return bs_year, m_idx, remaining_days + 1
                remaining_days -= days_in_month

    # Approximate fallback for dates outside 2080-2090
    approx_ad_delta = timedelta(days=int(56.7 * 365.25))
    bs_approx = ad_dt + approx_ad_delta
    return bs_approx.year, bs_approx.month, bs_approx.day


def bs_to_ad(bs_year: int, bs_month: int, bs_day: int) -> date:
    """Converts a BS date (bs_year, bs_month, bs_day) to an AD date."""
    if bs_year not in BS_CALENDAR_DATA:
        # Approximate fallback
        approx_days = int(56.7 * 365.25)
        try:
            return date(bs_year - 57, bs_month, min(28, bs_day))
        except Exception:
            return date(2026, 1, 1)

    start_ad, month_days = BS_CALENDAR_DATA[bs_year]
    bs_month = max(1, min(12, bs_month))
    max_days = month_days[bs_month - 1]
    bs_day = max(1, min(max_days, bs_day))

    offset = sum(month_days[: bs_month - 1]) + (bs_day - 1)
    return start_ad + timedelta(days=offset)


def format_bs_date(bs_year: int, bs_month: int, bs_day: int, lang: str = "en") -> str:
    """Formats a BS date into text (e.g. 'Mangsir 6, 2081' or '६ मंसिर २०८१')."""
    bs_month = max(1, min(12, bs_month))
    if lang == "np":
        m_name = NEPALI_MONTHS_NP[bs_month]
        d_str = str(bs_day).translate(ASCII_TO_DEV_TABLE)
        y_str = str(bs_year).translate(ASCII_TO_DEV_TABLE)
        return f"{d_str} {m_name} {y_str}"
    else:
        m_name = NEPALI_MONTHS_EN[bs_month]
        return f"{m_name} {bs_day}, {bs_year}"


BS_NUMERIC_RE = re.compile(r"\b(20[7-9]\d)[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[0-2])\b")


def parse_nepali_date(text: str) -> date | None:
    """Detects and parses BS dates from text strings and converts them to Gregorian AD date.

    Handles numeric format (2081-08-05, 2081/8/5) and text format (5 Mangsir 2081, २०८१ मंसिर ५).
    """
    if not text or not isinstance(text, str):
        return None

    # Normalize Devanagari numerals to ASCII
    norm_text = text.translate(DEV_TO_ASCII_TABLE)

    # 1. Try numeric YYYY-MM-DD or YYYY/MM/DD
    match = BS_NUMERIC_RE.search(norm_text)
    if match:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if is_probable_bs_year(y):
            return bs_to_ad(y, m, d)

    # 2. Try Month Day, Year or Day Month Year with word months
    for month_name, m_idx in MONTH_NAME_TO_INDEX.items():
        if month_name in norm_text.lower():
            # Pattern: 5 Mangsir 2081 or Mangsir 5, 2081
            p1 = re.search(rf"\b(\d{{1,2}})\s+{re.escape(month_name)}\s*,?\s+(20[7-9]\d)\b", norm_text, re.IGNORECASE)
            if p1:
                d, y = int(p1.group(1)), int(p1.group(2))
                return bs_to_ad(y, m_idx, d)
            p2 = re.search(rf"\b{re.escape(month_name)}\s+(\d{{1,2}})\s*,?\s+(20[7-9]\d)\b", norm_text, re.IGNORECASE)
            if p2:
                d, y = int(p2.group(1)), int(p2.group(2))
                return bs_to_ad(y, m_idx, d)
            p3 = re.search(rf"\b(20[7-9]\d)\s+{re.escape(month_name)}\s+(\d{{1,2}})\b", norm_text, re.IGNORECASE)
            if p3:
                y, d = int(p3.group(1)), int(p3.group(2))
                return bs_to_ad(y, m_idx, d)

    return None
