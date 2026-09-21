"""The guard in conftest.py exists so a dead database cannot produce a green suite.

It is tested by running pytest in a subprocess against a throwaway suite, because the guard acts
on session startup and on report outcomes — neither is observable from inside the run it governs.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")

SUITE = textwrap.dedent('''
    import pytest
    def test_skips_for_a_missing_database(): pytest.skip("Disposable Postgres not configured")
    def test_skips_for_something_else(): pytest.skip("no AVX512 on this platform")
    def test_passes(): assert True
''')


def _run(tmp_path, env_overrides):
    suite = tmp_path / "tests"
    suite.mkdir(exist_ok=True)
    shutil.copy(CONFTEST, suite / "conftest.py")   # the real guard, not a copy that can drift
    (suite / "test_sample.py").write_text(SUITE)
    env = {k: v for k, v in os.environ.items() if k not in ("REQUIRE_DB_TESTS", "CRYPTO_SOCIAL_TEST_DATABASE_URL")}
    env.update(env_overrides)
    return subprocess.run([sys.executable, "-m", "pytest", "tests/test_sample.py", "-q"],
                          cwd=tmp_path, env=env, capture_output=True, text=True)


def test_without_the_flag_a_database_skip_stays_a_skip(tmp_path):
    out = _run(tmp_path, {})
    assert out.returncode == 0, out.stdout
    assert "2 skipped" in out.stdout and "1 passed" in out.stdout


def test_the_flag_without_a_url_refuses_to_run_at_all(tmp_path):
    out = _run(tmp_path, {"REQUIRE_DB_TESTS": "1"})
    assert out.returncode != 0
    # It must refuse before running anything, so "0 tests ran" cannot read as success.
    assert "is not" in (out.stdout + out.stderr)
    assert "passed" not in out.stdout


def test_the_flag_with_an_unreachable_url_refuses_to_run(tmp_path):
    out = _run(tmp_path, {"REQUIRE_DB_TESTS": "1",
                          "CRYPTO_SOCIAL_TEST_DATABASE_URL": "postgresql://nobody@127.0.0.1:59998/absent"})
    assert out.returncode != 0
    assert "unreachable" in (out.stdout + out.stderr)


@pytest.mark.skipif(not os.environ.get("CRYPTO_SOCIAL_TEST_DATABASE_URL"), reason="Disposable Postgres not configured")
def test_with_a_live_database_only_the_database_skip_becomes_a_failure(tmp_path):
    out = _run(tmp_path, {"REQUIRE_DB_TESTS": "1",
                          "CRYPTO_SOCIAL_TEST_DATABASE_URL": os.environ["CRYPTO_SOCIAL_TEST_DATABASE_URL"]})
    assert out.returncode != 0
    assert "1 failed" in out.stdout and "1 passed" in out.stdout and "1 skipped" in out.stdout
    assert "skipped for a missing database" in out.stdout      # names why it failed
    assert "test_skips_for_something_else" not in out.stdout   # an unrelated skip is left alone


def test_the_flag_accepts_the_usual_falsey_spellings(tmp_path):
    for value in ("0", "false", "no", ""):
        out = _run(tmp_path, {"REQUIRE_DB_TESTS": value})
        assert out.returncode == 0, f"REQUIRE_DB_TESTS={value!r} should not enforce: {out.stdout}"
