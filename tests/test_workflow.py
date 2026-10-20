"""The CI workflow is code too: parse it and check the parts the quality gate depends on."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "eval.yml"


def load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def steps() -> list[dict]:
    return load()["jobs"]["eval"]["steps"]


def run_text() -> str:
    return "\n".join(s.get("run", "") for s in steps())


def test_triggers_on_relevant_pull_requests_and_manually():
    on = load()[True]  # YAML 1.1 reads the bare key `on` as boolean True
    assert set(on["pull_request"]["paths"]) >= {"src/**", "evals/**", "pyproject.toml", "uv.lock"}
    assert "workflow_dispatch" in on
    assert "pull_request_target" not in on  # would expose secrets to code from forks


def test_least_privilege_permissions():
    assert load()["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_postgres_service_matches_local_image():
    services = load()["jobs"]["eval"]["services"]
    assert services["db"]["image"] == "paradedb/paradedb:0.26.2-pg18"


def test_pipeline_order():
    text = run_text()
    order = ["gh release download eval-snapshot", "copilot init-db", "copilot evals snapshot import",
             "copilot evals run --ci-only"]
    positions = [text.index(s) for s in order]
    assert positions == sorted(positions)
    assert "--yes" in text


def test_actions_are_pinned_and_caches_set():
    uses = [s["uses"] for s in steps() if "uses" in s]
    assert all("@" in u and not u.endswith("@main") and not u.endswith("@master") for u in uses)
    assert any(u.startswith("astral-sh/setup-uv@") for u in uses)
    assert any(u.startswith("actions/cache@") for u in uses)  # fastembed reranker model


def test_budget_is_capped_per_run():
    assert "copilot spend --raw" in run_text() and "COPILOT_BUDGET_USD" in run_text()


def test_report_is_posted_and_gate_failure_fails_the_job():
    text = run_text()
    assert "gh pr comment" in text and "GITHUB_STEP_SUMMARY" in text
    last = steps()[-1]
    assert "exit 1" in last["run"] and "if" in last
