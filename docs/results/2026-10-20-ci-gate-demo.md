# CI quality gate on real pull requests (2026-10-20)

Workflow: `.github/workflows/eval.yml` — restore the `eval-snapshot` release (index + LLM cache) into a fresh
ParadeDB service, run the 16 CI cases, post the report as a PR comment, fail on `evals/thresholds.json`.

| | PR #1 `demo/gate-pass` | PR #2 `demo/gate-fail` |
|---|---|---|
| Change | comment-only edit in `src/` (+ `.gitignore`) | `rerank_min_score` −1.0 → 3.0 ("stricter cutoff to reduce hallucinations") |
| Gate | **PASSED** | **FAILED: answered_rate 0.82 < 0.90** |
| Wall time | 2 min 23 s | 3 min 1 s |
| OpenAI cost | $0.0000 (every call a snapshot-cache hit) | $0.0478 (changed contexts → new answers + grading) |
| answered_rate | 0.91 | 0.82 |
| correctness | 0.77 | 0.72 (docs 0.76 → 0.56) |
| faithfulness | 0.96 | 0.95 |
| hit_rate@5 | 1.00 | 1.00 |

Every workflow step succeeded in both runs; PR #2 failed only at "Fail on quality gate".

## What the failing PR shows
- The change sounds safe and helps one metric's story ("fewer irrelevant chunks"), but it drops relevant
  chunks too: docs-12 (lifespan state) and docs-20 started answering "I don't know".
- **Correctness alone would not have blocked it** (0.72 is above the 0.60 threshold). It was caught by
  `answered_rate`, the threshold added after the first smoke run passed a clearly bad agent
  (`2026-10-20-evals-ci-smoke.md`).
- Retrieval hit rate stayed 1.00: the retrieval eval measures ranking without the cutoff, so only the
  end-to-end answer eval sees this regression.
