"""Tests for public-release personal-data scanning."""

from scripts.public_release_check import PATTERNS, is_phone_number_like

_PHONE_LABEL = "phone-number-like personal data"


def _phone_match(line: str):
    match = PATTERNS[_PHONE_LABEL].search(line)
    assert match is not None
    return match


def test_numeric_benchmark_sequence_is_not_a_phone_number():
    line = "python scripts/benchmark.py --seq-lens 64 128 256 512 1024 --runs 5"

    assert is_phone_number_like(line, _phone_match(line)) is False


def test_formatted_phone_number_is_detected():
    line = "contact: " + "-".join(("819", "555", "0123"))

    assert is_phone_number_like(line, _phone_match(line)) is True


def test_contiguous_phone_number_is_detected():
    line = "".join(("819", "555", "0123"))

    assert is_phone_number_like(line, _phone_match(line)) is True


def test_standalone_space_separated_phone_number_is_detected():
    line = " ".join(("819", "555", "0123"))

    assert is_phone_number_like(line, _phone_match(line)) is True
