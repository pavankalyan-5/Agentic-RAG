from copilot.config import Settings


def test_settings_reads_prefixed_env(monkeypatch):
    monkeypatch.setenv("COPILOT_DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("COPILOT_BUDGET_USD", "3.5")
    s = Settings(_env_file=None)
    assert s.database_url == "postgresql://x/y"
    assert s.budget_usd == 3.5
    assert s.embedding_model == "text-embedding-3-small"
    assert s.embedding_dim == 1536
