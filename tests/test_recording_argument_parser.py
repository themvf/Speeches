import pytest

import source_health
from source_health import RecordingArgumentParser, _extract_cli_flag_value, categorize_error


def test_extract_cli_flag_value_space_separated(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script.py", "--connector", "krebs_on_security_article"])
    assert _extract_cli_flag_value("--connector") == "krebs_on_security_article"


def test_extract_cli_flag_value_equals_separated(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script.py", "--source-kind=sec_speech"])
    assert _extract_cli_flag_value("--source-kind") == "sec_speech"


def test_extract_cli_flag_value_missing_returns_empty(monkeypatch):
    monkeypatch.setattr("sys.argv", ["script.py", "--dry-run"])
    assert _extract_cli_flag_value("--connector") == ""


def test_recording_parser_logs_before_raising_systemexit(monkeypatch):
    logged = []
    monkeypatch.setattr(source_health, "record_source_health", lambda payload: logged.append(payload))
    monkeypatch.setattr("sys.argv", ["script.py", "--connector", "totally_fake_connector"])

    parser = RecordingArgumentParser(description="test")
    parser.add_argument("--connector", required=True, choices=["a", "b"])

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--connector", "totally_fake_connector"])

    # CLI behavior (exit code) must be unchanged.
    assert exc_info.value.code == 2
    # But now a failure entry was logged where previously nothing was.
    assert len(logged) == 1
    payload = logged[0]
    assert payload["ok"] is False
    assert payload["command"] == "argparse_error"
    assert payload["connector"] == "totally_fake_connector"
    assert "invalid choice" in payload["error"]


def test_recording_parser_extracts_source_kind_for_subparser_style_flags(monkeypatch):
    logged = []
    monkeypatch.setattr(source_health, "record_source_health", lambda payload: logged.append(payload))
    monkeypatch.setattr("sys.argv", ["script.py", "enrich", "--source-kind", "sec_speech", "--provider", "bogus"])

    parser = RecordingArgumentParser(description="test")
    parser.add_argument("--source-kind", default="")
    parser.add_argument("--provider", choices=["openai", "deepseek"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--source-kind", "sec_speech", "--provider", "bogus"])

    assert logged[0]["source_kind"] == "sec_speech"


def test_recording_parser_logging_failure_does_not_mask_original_error(monkeypatch):
    def _raise(_payload):
        raise RuntimeError("GCS unreachable")

    monkeypatch.setattr(source_health, "record_source_health", _raise)
    monkeypatch.setattr("sys.argv", ["script.py", "--connector", "bogus"])

    parser = RecordingArgumentParser(description="test")
    parser.add_argument("--connector", required=True, choices=["a", "b"])

    # The original argparse SystemExit must still propagate even though the
    # health-logging attempt itself raised.
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--connector", "bogus"])
    assert exc_info.value.code == 2


def test_recording_parser_end_to_end_produces_failing_run_entry(monkeypatch):
    """Full pipeline: parser.error() -> record_source_health() -> build_run_entry()
    -> categorize_error(), proving the resulting entry actually lands in the
    same shape the failing/stale/quiet dashboard consumes."""
    from source_health import build_run_entry

    captured_entries = []

    def fake_record(payload):
        captured_entries.append(build_run_entry(payload))

    monkeypatch.setattr(source_health, "record_source_health", fake_record)
    monkeypatch.setattr("sys.argv", ["script.py", "--connector", "totally_fake_connector"])

    parser = RecordingArgumentParser(description="test")
    parser.add_argument("--connector", required=True, choices=["a", "b"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--connector", "totally_fake_connector"])

    entry = captured_entries[0]
    assert entry["source_key"] == "totally_fake_connector"
    assert entry["status"] == "failed"
    assert entry["error_category"] == "invalid_choice"


def test_categorize_error_invalid_choice_vs_generic_cli_usage_error():
    invalid_choice_summary = {"command": "argparse_error", "error": "argument --provider: invalid choice: 'x'"}
    assert categorize_error(invalid_choice_summary["error"], invalid_choice_summary) == "invalid_choice"

    missing_arg_summary = {"command": "argparse_error", "error": "the following arguments are required: --connector"}
    assert categorize_error(missing_arg_summary["error"], missing_arg_summary) == "cli_usage_error"
