"""Stop a database-backed test from skipping unnoticed.

Several suites here gate on CRYPTO_SOCIAL_TEST_DATABASE_URL and call pytest.skip when it is
absent. That is right locally. In CI it is dangerous: if the Postgres service fails to start, or
the variable is dropped from a workflow, every one of those tests skips and the job reports
GREEN having exercised no database at all. Green-because-nothing-ran is worse than red, because
nobody investigates it.

Set REQUIRE_DB_TESTS=1 (CI does) and this turns that silence into a failure: the session refuses
to start unless the database is reachable, and any test that skips for a missing database is
reported as a failure rather than a skip.
"""
import os

import pytest

DB_ENV = "CRYPTO_SOCIAL_TEST_DATABASE_URL"
REQUIRE_ENV = "REQUIRE_DB_TESTS"
# Matched against a skip reason, lowercased. These are the wordings the suites use today; a new
# one that does not match still cannot hide, because the session check below runs first.
DB_SKIP_MARKERS = ("postgres not configured", "disposable postgres not configured")


def _required():
    return os.environ.get(REQUIRE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def pytest_configure(config):
    """Fail before any test runs, so a dead service cannot be mistaken for a passing suite."""
    if not _required():
        return
    url = os.environ.get(DB_ENV)
    if not url:
        raise pytest.UsageError(
            f"{REQUIRE_ENV} is set but {DB_ENV} is not. The database-backed tests would all skip "
            f"and the run would report success having tested no database.")
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - a packaging problem, not a test outcome
        raise pytest.UsageError(f"{REQUIRE_ENV} is set but psycopg2 is missing: {exc}")
    try:
        psycopg2.connect(url, connect_timeout=10).close()
    except Exception as exc:
        raise pytest.UsageError(
            f"{REQUIRE_ENV} is set but {DB_ENV} is unreachable, so every database-backed test "
            f"would skip: {type(exc).__name__} {exc}".strip())


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Report a database skip as a failure while the requirement is in force."""
    outcome = yield
    report = outcome.get_result()
    if not _required() or not report.skipped:
        return
    reason = str(report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr).lower()
    if any(marker in reason for marker in DB_SKIP_MARKERS):
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} skipped for a missing database while {REQUIRE_ENV} is set. "
            f"The suite must exercise the database in this environment, not skip past it. "
            f"Original reason: {reason}")
