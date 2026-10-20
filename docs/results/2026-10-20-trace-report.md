# Trace report (last 24 h)

**6 turns** · turn latency p50 **1.92s**, p90 **6.55s**, p99 **6.55s** · cost **$0.00035/turn** (p90 $0.00074, total $0.00209) · cache hit rate **56%**

## Where the time goes (slowest first)

| step | count | p50 (s) | p90 (s) | p99 (s) | share of turn time |
|---|---:|---:|---:|---:|---:|
| `llm.answer` | 5 | 1.585 | 2.674 | 2.674 | 32% |
| `plan` | 6 | 1.243 | 3.049 | 3.049 | 43% |
| `llm.plan` | 6 | 1.243 | 3.049 | 3.049 | 43% |
| `rerank` | 5 | 0.769 | 0.847 | 0.847 | 16% |
| `tool.pull_request_read` | 2 | 0.704 | 0.734 | 0.734 | 7% |
| `search` | 5 | 0.042 | 0.070 | 0.070 | 1% |
| `llm.embed` | 5 | 0.006 | 0.007 | 0.007 | 0% |
| `synthesize` | 6 | 0.005 | 2.675 | 2.675 | 32% |
| `verify` | 6 | 0.000 | 0.000 | 0.000 | 0% |
| `retrieve` | 6 | 0.000 | 1.388 | 1.388 | 17% |
| `respond` | 6 | 0.000 | 0.000 | 0.000 | 0% |
| `live_lookup` | 6 | 0.000 | 0.734 | 0.734 | 7% |

## Tokens by model

| model | prompt | completion |
|---|---:|---:|
| `gpt-4.1-mini` | 2,271 | 480 |
| `gpt-4.1-nano` | 3,500 | 158 |

_Percentiles use the nearest-rank method. Cost and cache data come from gateway spans; steps nest, so shares overlap (e.g. `synthesize` contains `llm.answer`)._
