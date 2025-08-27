import re
import pytest

from src.links.router import _generate_short_code


def test_generate_short_code_format():
    code = _generate_short_code()
    assert isinstance(code, str)
    assert len(code) == 7
    assert re.match(r"^[A-Za-z0-9]{7}$", code)
    assert not code.isdigit()
    assert not code.isalpha()


def test_generate_short_code_randomness():
    codes = { _generate_short_code() for _ in range(500) }
    assert len(codes) > 490


