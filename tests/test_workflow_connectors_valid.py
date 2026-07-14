"""Guard against scheduled workflows referencing connectors/source-kinds the
pipeline no longer supports.

A July 2026 merge silently deleted 16 connector definitions from
run_connector_extraction_pipeline.py while leaving the GitHub Actions
workflows that scheduled them in place. Every scheduled run then failed fast
on an argparse "invalid choice" error, before any extraction or source-health
logging — so the regression was invisible for days. This test turns that
failure mode into a red CI check.
"""

import glob
import os
import re

import run_connector_extraction_pipeline as pipeline

WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".github", "workflows")

# source_kind values that are legitimately produced outside the connector
# extraction pipeline (e.g. the NewsAPI ingest path in
# run_financial_news_pipeline.py), so they are valid --source-kind arguments
# even though they are not members of SUPPORTED_CONNECTORS.
KNOWN_NON_CONNECTOR_SOURCE_KINDS = {
    "newsapi_article",
}

# Matrix list-item value, e.g. "  - connector: krebs_on_security_article".
_MATRIX_RE = re.compile(r"^\s*-\s+(connector|source_kind):\s+\"?([A-Za-z0-9_]+)\"?\s*$")
# Inline CLI flag, e.g. '--connector "krebs_on_security_article"'.
_CLI_RE = re.compile(r"--(connector|source-kind)\s+\"?([A-Za-z0-9_$][A-Za-z0-9_{}. ]*?)\"?(?:\s|\\|$)")


def _is_template(value: str) -> bool:
    # Values like "${{ matrix.connector }}" are resolved from the matrix at run
    # time; the concrete values live in the matrix list-item lines instead.
    return "$" in value or "{" in value


def _collect_references():
    connectors = {}
    source_kinds = {}
    for path in sorted(glob.glob(os.path.join(WORKFLOWS_DIR, "*.yml"))):
        name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                matrix = _MATRIX_RE.match(line)
                if matrix:
                    kind, value = matrix.group(1), matrix.group(2)
                    bucket = connectors if kind == "connector" else source_kinds
                    bucket.setdefault(value, set()).add(name)
                    continue
                for cli in _CLI_RE.finditer(line):
                    flag, value = cli.group(1), cli.group(2).strip()
                    if _is_template(value):
                        continue
                    bucket = connectors if flag == "connector" else source_kinds
                    bucket.setdefault(value, set()).add(name)
    return connectors, source_kinds


def test_all_workflow_connectors_are_supported():
    connectors, _ = _collect_references()
    unsupported = {
        connector: sorted(files)
        for connector, files in connectors.items()
        if connector not in pipeline.SUPPORTED_CONNECTORS
    }
    assert not unsupported, (
        "Workflows reference connectors not in SUPPORTED_CONNECTORS "
        f"(pipeline can't run them): {unsupported}"
    )


def test_all_workflow_source_kinds_are_known():
    _, source_kinds = _collect_references()
    valid = pipeline.SUPPORTED_CONNECTORS | KNOWN_NON_CONNECTOR_SOURCE_KINDS
    unknown = {
        source_kind: sorted(files)
        for source_kind, files in source_kinds.items()
        if source_kind not in valid
    }
    assert not unknown, (
        "Workflows reference --source-kind values that are neither a supported "
        f"connector nor a known non-connector source kind: {unknown}"
    )


def test_guard_actually_found_references():
    # Sanity check that the parser is extracting something, so a future change
    # to workflow formatting doesn't silently turn this guard into a no-op.
    connectors, source_kinds = _collect_references()
    assert len(connectors) >= 20, f"Expected many connector references, found {len(connectors)}"
    assert len(source_kinds) >= 20, f"Expected many source_kind references, found {len(source_kinds)}"


def _workflow_text(filename: str) -> str:
    with open(os.path.join(WORKFLOWS_DIR, filename), "r", encoding="utf-8") as handle:
        return handle.read()


def test_high_frequency_enrichment_steps_are_gated_on_extraction_changes():
    workflows = {
        "bloomberg-public-hourly.yml": "Run Bloomberg Enrichment",
        "substack-public-2hour.yml": "Run Substack Enrichment",
    }
    for filename, step_name in workflows.items():
        text = _workflow_text(filename)
        pattern = (
            rf"- name: {re.escape(step_name)}\r?\n"
            r"\s+if: steps\.extraction_summary\.outputs\.changed_count != '0'"
        )
        assert re.search(pattern, text), f"{filename} must gate {step_name} on changed_count"


def test_substack_public_workflow_runs_hourly():
    text = _workflow_text("substack-public-2hour.yml")

    assert re.search(r'^\s*- cron: "17 \* \* \* \*"\s*$', text, flags=re.MULTILINE)
    assert "name: Substack Public Sources Hourly" in text
    assert "group: substack-public-hourly" in text
