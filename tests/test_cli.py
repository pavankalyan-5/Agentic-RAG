import pytest
from typer.testing import CliRunner

from copilot import cli
from copilot.config import get_settings
from fakes import FakeEmbedder

runner = CliRunner()


@pytest.fixture
def cli_env(seeded_db, settings, monkeypatch):
    monkeypatch.setenv("COPILOT_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("COPILOT_BUDGET_USD", "8")
    monkeypatch.setattr(cli, "build_embedder", lambda conn, s: FakeEmbedder())
    get_settings.cache_clear()
    yield seeded_db
    get_settings.cache_clear()


def test_search_command_prints_results(cli_env):
    r = runner.invoke(cli.app, ["search", "ERR_TIMEOUT", "--mode", "sparse", "-k", "2"])
    assert r.exit_code == 0, r.output
    assert "ERR_TIMEOUT" in r.output and "[issue #11]" in r.output
    assert "https://github.com/o/r/" in r.output


def test_search_type_and_state_filters(cli_env):
    r = runner.invoke(cli.app, ["search", "background tasks", "--type", "pr", "--state", "merged"])
    assert r.exit_code == 0, r.output
    assert "[pr #12]" in r.output and "[issue" not in r.output


def test_search_no_results_message(cli_env):
    r = runner.invoke(cli.app, ["search", "zzzqqq", "--mode", "sparse"])
    assert r.exit_code == 0 and "No results" in r.output


def test_spend_command(cli_env):
    r = runner.invoke(cli.app, ["spend"])
    assert r.exit_code == 0 and "of $8.00" in r.output


def test_sync_full_flag_is_passed_through(cli_env, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "sync_repo", lambda *a, full=False, **kw: seen.setdefault("full", full) and {})
    monkeypatch.setenv("COPILOT_DEMO_REPO", "o/r")
    get_settings.cache_clear()
    runner.invoke(cli.app, ["sync", "--full"])
    assert seen["full"] is True


def test_init_db_is_safe_to_rerun(cli_env):
    r = runner.invoke(cli.app, ["init-db"])
    assert r.exit_code == 0 and "Schema ready" in r.output
