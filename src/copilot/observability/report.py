"""Turn traces into numbers: latency percentiles per step, tokens per model, cost per turn, cache rate.

Input rows (one per span): {"name", "span_kind", "trace_id", "start_time", "end_time", "attributes"} with
flat dotted attribute keys. rows_from_phoenix() converts Phoenix's dataframe into that shape.
"""

import math
from collections import defaultdict
from datetime import datetime


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: sort, take the value at position ceil(p% * n). Exact for small samples."""
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _stats(values: list[float]) -> dict:
    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": percentile(values, 50), "p90": percentile(values, 90), "p99": percentile(values, 99)}


def _seconds(row: dict) -> float:
    return (row["end_time"] - row["start_time"]).total_seconds()


def compute_report(spans: list[dict]) -> dict:
    turns = [s for s in spans if s["name"] == "agent_turn"]
    turn_traces = {s["trace_id"] for s in turns}

    durations: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        durations[s["name"]].append(_seconds(s))

    tokens: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0})
    cost_by_trace: dict[str, float] = defaultdict(float)
    cache_flags: list[bool] = []
    for s in spans:
        a = s["attributes"]
        if s["span_kind"] == "LLM" and a.get("llm.model_name"):
            model = tokens[a["llm.model_name"]]
            model["prompt"] += int(a.get("llm.token_count.prompt") or 0)
            model["completion"] += int(a.get("llm.token_count.completion") or 0)
        if a.get("copilot.cost_usd") is not None:
            cost_by_trace[s["trace_id"]] += float(a["copilot.cost_usd"])
        if a.get("copilot.cache_hit") is not None:
            cache_flags.append(bool(a["copilot.cache_hit"]))

    turn_costs = [cost_by_trace.get(t, 0.0) for t in turn_traces]
    total_turn_time = sum(_seconds(s) for s in turns)
    share = {name: sum(v) / total_turn_time for name, v in durations.items()
             if name != "agent_turn" and total_turn_time > 0}

    return {
        "turns": len(turns),
        "turn_latency_s": _stats([_seconds(s) for s in turns]) if turns else None,
        "latency_s": {name: _stats(v) for name, v in durations.items() if name != "agent_turn"},
        "share_of_turn_time": share,
        "tokens_by_model": {m: dict(v) for m, v in tokens.items()},
        "cost_usd": {
            "total": sum(turn_costs),
            "per_turn_mean": sum(turn_costs) / len(turn_costs) if turn_costs else 0,
            "per_turn_p90": percentile(turn_costs, 90) if turn_costs else 0,
        },
        "cache_hit_rate": sum(cache_flags) / len(cache_flags) if cache_flags else None,
    }


def rows_from_phoenix(df) -> list[dict]:
    """Phoenix flattens attributes it knows ("attributes.llm.model_name") but nests unknown namespaces
    in one dict ("attributes.copilot" = {"cost_usd": ...}). Normalize both to flat dotted keys."""
    rows = []
    for record in df.to_dict("records"):
        attributes: dict = {}
        for column, value in record.items():
            if column.startswith("attributes."):
                _flatten(column[len("attributes."):], value, attributes)
        rows.append({
            "name": record["name"], "span_kind": record.get("span_kind"),
            "trace_id": record.get("context.trace_id"),
            "start_time": _as_datetime(record["start_time"]), "end_time": _as_datetime(record["end_time"]),
            "attributes": attributes,
        })
    return rows


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}", inner, out)
    elif value is not None and not (isinstance(value, float) and math.isnan(value)):
        out[prefix] = value


def _as_datetime(value) -> datetime:
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else value


def render_markdown(report: dict, title: str = "Trace report") -> str:
    if not report["turns"]:
        return f"# {title}\n\nNo traces in this window."
    t = report["turn_latency_s"]
    lines = [
        f"# {title}", "",
        f"**{report['turns']} turns** · turn latency p50 **{t['p50']:.2f}s**, p90 **{t['p90']:.2f}s**, "
        f"p99 **{t['p99']:.2f}s** · cost **${report['cost_usd']['per_turn_mean']:.5f}/turn** "
        f"(p90 ${report['cost_usd']['per_turn_p90']:.5f}, total ${report['cost_usd']['total']:.5f})"
        + (f" · cache hit rate **{report['cache_hit_rate']:.0%}**" if report["cache_hit_rate"] is not None else ""),
        "", "## Where the time goes (slowest first)", "",
        "| step | count | p50 (s) | p90 (s) | p99 (s) | share of turn time |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in sorted(report["latency_s"].items(), key=lambda kv: kv[1]["p50"], reverse=True):
        share = report["share_of_turn_time"].get(name, 0)
        lines.append(f"| `{name}` | {s['count']} | {s['p50']:.3f} | {s['p90']:.3f} | {s['p99']:.3f} | {share:.0%} |")
    if report["tokens_by_model"]:
        lines += ["", "## Tokens by model", "", "| model | prompt | completion |", "|---|---:|---:|"]
        for model, tok in sorted(report["tokens_by_model"].items()):
            lines.append(f"| `{model}` | {tok['prompt']:,} | {tok['completion']:,} |")
    lines += ["", "_Percentiles use the nearest-rank method. Cost and cache data come from gateway spans; "
              "steps nest, so shares overlap (e.g. `synthesize` contains `llm.answer`)._"]
    return "\n".join(lines)
