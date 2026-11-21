from datetime import date
from src.utils.nepali_date import (
    ad_to_bs,
    bs_to_ad,
    format_bs_date,
    is_probable_bs_year,
    parse_nepali_date,
    today_npt,
)


def test_nepali_date_conversion_known_milestones():
    # 2080-01-01 BS = 2023-04-14 AD
    assert bs_to_ad(2080, 1, 1) == date(2023, 4, 14)
    assert ad_to_bs(date(2023, 4, 14)) == (2080, 1, 1)

    # 2081-01-01 BS = 2024-04-13 AD
    assert bs_to_ad(2081, 1, 1) == date(2024, 4, 13)
    assert ad_to_bs(date(2024, 4, 13)) == (2081, 1, 1)

    # 2083-08-05 BS -> Check bidirectional conversion
    ad_dt = bs_to_ad(2083, 8, 5)
    assert ad_to_bs(ad_dt) == (2083, 8, 5)


def test_nepali_date_formatting():
    en_str = format_bs_date(2081, 8, 5, lang="en")
    assert en_str == "Mangsir 5, 2081"

    np_str = format_bs_date(2081, 8, 5, lang="np")
    assert "मंसिर" in np_str
    assert "२०८१" in np_str


def test_nepali_date_parsing():
    # Numeric
    parsed = parse_nepali_date("Published on 2081-08-05")
    assert parsed == bs_to_ad(2081, 8, 5)

    # Word month in English
    parsed_en = parse_nepali_date("Kathmandu, 5 Mangsir 2081 - Report")
    assert parsed_en == bs_to_ad(2081, 8, 5)

    # Word month in Devanagari numerals
    parsed_np = parse_nepali_date("काठमाडौं, २०८१ मंसिर ५ गते")
    assert parsed_np == bs_to_ad(2081, 8, 5)


def test_today_npt_and_helpers():
    t_str = today_npt()
    assert len(t_str) == 10
    assert t_str.count("-") == 2
    assert is_probable_bs_year(2081)
    assert not is_probable_bs_year(2024)
