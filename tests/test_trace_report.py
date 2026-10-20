from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from copilot.observability.report import compute_report, percentile, render_markdown, rows_from_phoenix

T0 = datetime(2026, 10, 20, tzinfo=timezone.utc)


def span(name, trace, start_s, dur_s, kind="CHAIN", **attrs):
    return {"name": name, "span_kind": kind, "trace_id": trace,
            "start_time": T0 + timedelta(seconds=start_s), "end_time": T0 + timedelta(seconds=start_s + dur_s),
            "attributes": attrs}


def turn(trace, plan_s, answer_s, cost=0.001, cached=False):
    return [
        span("agent_turn", trace, 0, plan_s + answer_s + 0.1),
        span("plan", trace, 0, plan_s),
        span("llm.plan", trace, 0, plan_s, "LLM", **{"llm.model_name": "gpt-4.1-nano", "llm.token_count.prompt": 900,
             "llm.token_count.completion": 70, "copilot.cost_usd": 0.0 if cached else cost / 10,
             "copilot.cache_hit": cached}),
        span("synthesize", trace, plan_s, answer_s),
        span("llm.answer", trace, plan_s, answer_s, "LLM", **{"llm.model_name": "gpt-4.1-mini",
             "llm.token_count.prompt": 1700, "llm.token_count.completion": 280,
             "copilot.cost_usd": 0.0 if cached else cost, "copilot.cache_hit": cached}),
    ]


def test_percentile_nearest_rank():
    values = list(range(1, 11))
    assert (percentile(values, 50), percentile(values, 90), percentile(values, 99)) == (5, 9, 10)
    assert percentile([7.0], 90) == 7.0


def test_percentiles_expose_the_slow_tail_the_mean_hides():
    times = [2, 2, 3, 3, 3, 3, 4, 4, 5, 31]
    assert sum(times) / len(times) == 6.0
    assert (percentile(times, 50), percentile(times, 90), percentile(times, 99)) == (3, 5, 31)


def test_report_latency_tokens_cost_and_cache():
    spans = turn("t1", 1.0, 2.0) + turn("t2", 3.0, 4.0) + turn("t3", 0.1, 0.1, cached=True)
    r = compute_report(spans)
    assert r["turns"] == 3
    assert r["latency_s"]["llm.answer"]["p50"] == pytest.approx(2.0)
    assert r["latency_s"]["llm.answer"]["count"] == 3
    assert r["turn_latency_s"]["p90"] == pytest.approx(7.1)
    assert r["tokens_by_model"]["gpt-4.1-mini"] == {"prompt": 5100, "completion": 840}
    assert r["cost_usd"]["total"] == pytest.approx(2 * 0.0011)
    assert r["cost_usd"]["per_turn_mean"] == pytest.approx(2 * 0.0011 / 3)
    assert r["cache_hit_rate"] == pytest.approx(2 / 6)  # 2 of 6 LLM calls were cache hits


def test_share_of_turn_time_shows_the_bottleneck():
    r = compute_report(turn("t1", 1.0, 3.0))
    assert r["share_of_turn_time"]["llm.answer"] == pytest.approx(3.0 / 4.1)


def test_empty_input_does_not_crash():
    r = compute_report([])
    assert r["turns"] == 0 and r["cost_usd"]["total"] == 0 and r["cache_hit_rate"] is None
    assert "No traces" in render_markdown(r)


def test_turn_without_llm_calls_is_handled():
    r = compute_report([span("agent_turn", "t1", 0, 0.2), span("plan", "t1", 0, 0.1)])
    assert r["turns"] == 1 and r["cost_usd"]["per_turn_mean"] == 0 and r["tokens_by_model"] == {}


def test_rows_from_phoenix_handles_flat_and_nested_attributes():
    # Phoenix flattens attributes it knows (llm.*) but nests unknown namespaces (copilot.*) in one dict.
    df = pd.DataFrame([{
        "name": "llm.answer", "span_kind": "LLM", "context.trace_id": "t1",
        "start_time": T0, "end_time": T0 + timedelta(seconds=2),
        "attributes.llm.model_name": "gpt-4.1-mini", "attributes.llm.token_count.prompt": 100.0,
        "attributes.llm.token_count.completion": float("nan"),
        "attributes.copilot": {"cost_usd": 0.0004, "cache_hit": False, "operation": "answer"},
    }])
    row = rows_from_phoenix(df)[0]
    assert row["trace_id"] == "t1"
    assert row["attributes"]["copilot.cost_usd"] == 0.0004 and row["attributes"]["copilot.cache_hit"] is False
    assert row["attributes"]["llm.token_count.prompt"] == 100
    assert "llm.token_count.completion" not in row["attributes"]  # NaN means "not set"


def test_markdown_lists_slowest_steps_first():
    md = render_markdown(compute_report(turn("t1", 1.0, 3.0) + turn("t2", 1.2, 2.8)))
    assert md.index("llm.answer") < md.index("llm.plan")
    assert "p50" in md and "/turn" in md
