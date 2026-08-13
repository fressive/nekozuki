"""Tests for the coarse trick search + detail API (routes.py)."""

import re
import sys
from pathlib import Path

# Ensure project src is importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.routes import _trick_id, parse_trick_section, search_tricks_in

_SAMPLE = """## Authentication Bypass via Tautology

Description: Inject OR 1=1 into the login query to bypass authentication.

Conditions: Login uses raw string concat; No WAF;

Implementation:
- Submit admin' OR 1=1 --
- Observe login succeeds

Key code/payload:

```
admin' OR '1'='1
```

Example:

```
admin' or 1=1 -- login as admin
```

Detection signs:
- Response differs for valid/invalid

Example challenge: CTF 2024 - Easy Login"""


def _indexed(tricks: list[dict]) -> list[dict]:
    """Attach the internal token fields the search scorer expects."""
    for t in tricks:
        t["id"] = _trick_id(t["technique_name"], t["title"])
        t["content"] = t["title"] + " " + (t.get("description") or "")
        t["_title_tokens"] = frozenset(re.findall(r"\w+", t["title"].lower()))
        t["_content_tokens"] = frozenset(re.findall(r"\w+", t["content"].lower()))
    return tricks


def test_parse_trick_section():
    t = parse_trick_section("sql_injection", "Authentication Bypass via Tautology", _SAMPLE)
    assert t["technique_name"] == "sql_injection"
    assert "OR 1=1" in t["description"]
    assert t["conditions"] == ["Login uses raw string concat", "No WAF"]
    assert t["implementation_steps"] == ["Submit admin' OR 1=1 --", "Observe login succeeds"]
    assert "admin' OR '1'='1" in t["key_code"]
    assert "admin' or 1=1" in t["example"]
    assert t["detection_signs"] == ["Response differs for valid/invalid"]
    assert t["example_challenge"] == "CTF 2024 - Easy Login"


def test_search_finds_by_title_and_description():
    tricks = _indexed([
        parse_trick_section("sql_injection", "Authentication Bypass via Tautology", _SAMPLE),
        parse_trick_section("sql_injection", "Blind SQLi via time delay",
                            "## Blind SQLi via time delay\n\nDescription: Use SLEEP to infer data."),
        parse_trick_section("crypto", "XOR single byte brute force",
                            "## XOR single byte brute force\n\nDescription: Try 256 keys."),
    ])
    res = search_tricks_in(tricks, "tautology login bypass", limit=5)
    assert res, "expected at least one hit"
    assert res[0]["title"] == "Authentication Bypass via Tautology"
    assert "id" in res[0]
    assert "description" in res[0]
    # Full content is returned so no detail request is needed.
    assert "conditions" in res[0] and res[0]["conditions"]
    assert "implementation_steps" in res[0] and res[0]["implementation_steps"]
    assert "key_code" in res[0]
    assert "content" in res[0] and res[0]["content"]

    res2 = search_tricks_in(tricks, "sleep time", limit=5)
    assert res2 and res2[0]["title"] == "Blind SQLi via time delay"

    assert search_tricks_in(tricks, "", limit=5) == []
    assert search_tricks_in(tricks, "zzzznonexistent", limit=5) == []


def test_trick_id_is_stable():
    assert _trick_id("a", "Trick Title") == _trick_id("a", "Trick Title")
    assert _trick_id("a", "Trick Title") != _trick_id("b", "Trick Title")
    assert _trick_id("sql_injection", "Authentication Bypass via Tautology").startswith("sql_injection::")
