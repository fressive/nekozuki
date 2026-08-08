"""Tests for the configuration/connectivity diagnostics (nekozuki test)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.diagnostics import (
    FAIL,
    OK,
    WARN,
    _mask,
    check_data_file,
    check_directories,
    check_embedding_key,
    check_llm_key,
    render_report,
    run_checks,
)


def test_mask_hides_key():
    """API keys are masked for display."""
    assert _mask("sk-ant-1234abcd") == "sk-a...abcd"
    assert _mask("") == "(empty)"
    assert _mask("short") == "***"


def test_data_file_check(tmp_path, monkeypatch):
    """Data file check reports the writeup count."""
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps([{"url": "x"}, {"url": "y"}]))
    monkeypatch.setattr(settings, "data_path", data_path)

    result = check_data_file()
    assert result.status == OK
    assert "2" in result.detail


def test_data_file_missing(tmp_path, monkeypatch):
    """Missing data file is a failure."""
    monkeypatch.setattr(settings, "data_path", tmp_path / "nope.json")
    result = check_data_file()
    assert result.status == FAIL


def test_key_checks(monkeypatch):
    """Missing API keys are flagged as failures."""
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "embedding_api_key", "")
    assert check_llm_key().status == FAIL
    assert check_embedding_key().status == FAIL


def test_directory_check(tmp_path, monkeypatch):
    """Directories are reported writable."""
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "vectors_dir", tmp_path / "vectors")
    assert check_directories().status == OK


def test_run_checks_offline_exit_code(tmp_path, monkeypatch):
    """run_checks with connectivity disabled returns exit 0 when configured."""
    # Valid configuration
    monkeypatch.setattr(settings, "llm_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "embedding_api_key", "sk-test")
    monkeypatch.setattr(settings, "data_path", tmp_path / "data.json")
    (tmp_path / "data.json").write_text(json.dumps([]))
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "vectors_dir", tmp_path / "vectors")

    results, exit_code = run_checks(include_connectivity=False)
    assert exit_code == 0
    assert not any(r.status == FAIL for r in results)


def test_run_checks_offline_fails_without_keys(tmp_path, monkeypatch):
    """Missing keys produce a non-zero exit even offline."""
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "embedding_api_key", "")
    monkeypatch.setattr(settings, "data_path", tmp_path / "data.json")
    (tmp_path / "data.json").write_text(json.dumps([]))
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "vectors_dir", tmp_path / "vectors")

    results, exit_code = run_checks(include_connectivity=False)
    assert exit_code == 1
    assert sum(r.status == FAIL for r in results) == 2  # both keys


def test_render_report_has_summary():
    """The report ends with a status summary line."""
    from src.diagnostics import CheckResult
    report = render_report([
        CheckResult("a", OK, "x"),
        CheckResult("b", WARN, "y"),
    ])
    assert "[ OK ] a: x" in report
    assert "[WARN] b: y" in report
    assert "1 ok" in report and "1 warn" in report