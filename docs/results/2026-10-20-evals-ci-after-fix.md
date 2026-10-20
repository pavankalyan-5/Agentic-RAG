# Eval-driven fix: before vs after (same 15 CI cases, same judge)

Fixes (from root causes measured in the first run, `2026-10-20-evals-ci-smoke.md`):
1. A planner "no search" is trusted only for messages of ≤ 6 words or when live lookups answer it (it had skipped 6/38 real questions).
2. Planner source-type filters dropped; state filters kept (it had limited 11/15 history questions to docs).
3. Answer rule: when several PRs tackle the same problem, only a merged one is the fix.

| metric | before | after |
|---|---:|---:|
| answered_rate | 0.64 | **0.91** |
| correctness | 0.32 | **0.75** |
| contextual_precision | 0.66 | 0.77 |
| faithfulness | 1.00 | 0.98 |
| answer_relevancy | 0.98 | 0.94 |
| unanswerable / adversarial | 1.00 / 1.00 | 1.00 / 1.00 |
| gate (strict thresholds) | fail | **pass** |

Still wrong: hist-06 and hist-14 credit a closed PR (#3590, #3355) even with fix 3. In hist-06 the merged PR #3671 was in context at rank 8; in hist-14 the merged #3491 wasn't retrieved (same title as closed #3258). docs-20 still declines.

## Round 2: one chunk per document (retrieval fix)

Probe: in hist-14 three chunks of closed PR #3355 filled 3 of the 4 context slots; merged #3491 was 1.7–3 points behind. In hist-06 merged #3671 is at hybrid rank 25 (outside the 10-candidate pool).

Sweep over 38 answerable questions (hybrid k=10, keep 4; local models, $0):

| change | hit@4 | merged fix in context (6 cases) | distinct docs in context |
|---|---:|---:|---:|
| none | 0.974 | 3/6 | 2.55 |
| merged bonus +0.5 / +1 | 0.974 | 3/6 | 2.4 |
| merged bonus +2 | 0.974 | 5/6 | 2.45 |
| max 2 chunks per doc | 0.974 | 4/6 | 2.74 |
| **max 1 chunk per doc** | 0.974 | **5/6** | **3.26** |

Chosen: max 1 chunk per document (neutral to PR state; no tuned bonus). Answer eval, same 15 cases:

| metric | round 1 | round 2 |
|---|---:|---:|
| correctness | 0.75 | 0.75 |
| history correctness | 0.65 | **0.78** |
| contextual_precision | 0.77 | **0.81** |
| contextual_recall | 0.93 | **0.97** |
| answer_relevancy | 0.94 | 0.98 |
| faithfulness | 0.98 | 0.96 |

hist-14 now names merged #3491. hist-06 unchanged (not retrievable by the plain question). New miss: multi-02 calls a closed BaseHTTPMiddleware PR "merged" despite its state attribute. With 11 judged cases, ±0.02 is noise.

---
# Answer evaluation (15 cases, CI subset)

**Quality gate: PASSED**

| metric | value |
|---|---:|
| cases | 15 |
| faithfulness | 0.9571 |
| answer_relevancy | 0.9833 |
| contextual_precision | 0.8085 |
| contextual_recall | 0.9667 |
| correctness | 0.7455 |
| hallucination_rate | 0.0429 |
| unanswerable_correct | 1.0000 |
| adversarial_safe | 1.0000 |
| answered_rate | 0.9091 |
| cost_usd | 0.0653 |
| mean_seconds | 3.0641 |
| hit_rate@5 | 1.0000 |
| mrr@5 | 1.0000 |

## By category

| category | cases | correctness | faithfulness | rule checks passed |
|---|---:|---:|---:|---:|
| adversarial | 2 | – | – | 2/2 |
| docs | 5 | 0.72 | 1.00 | – |
| history | 4 | 0.78 | 0.89 | – |
| multi_part | 2 | 0.75 | 1.00 | – |
| unanswerable | 2 | – | – | 2/2 |

## Cases to look at (6)

**docs-12** (docs): How can a lifespan handler share resources, like a database pool, with request handlers?
- contextual_recall 0.67: The score is 0.67 because sentences 1 and 2 are well supported by node 1, which discusses lifespan handlers and sharing state dictionaries accessible in request handlers. However, sentence 3 about using TestClient as a context manager to trigger lifespan during tests is not supported by any node in the retrieval context, lowering the overall recall score.

**docs-20** (docs): How can I access and handle uploaded files in a Starlette HTTP request, and what attributes and methods does the UploadFile object provide?
- said it didn't know, but the golden set says this is answerable

**hist-06** (history): Which pull request actually fixed the bug where a URL path starting with // was treated as the host when a request has no Host header?
- contextual_precision 0.12: The score is 0.12 because the first seven nodes in the retrieval contexts are irrelevant, as they either describe fixes that were not merged or do not correspond to the merged fix PR #3671, which is only mentioned in the eighth node. This relevant node is ranked too low (eighth position), causing the low score. However, the presence of this relevant node at all prevents the score from being zero, reflecting some correct retrieval despite poor ranking.
- correctness 0.20: The actual output identifies PR #3590 as the fix, while the expected output states that the merged fix was PR #3671. This is a direct contradiction regarding which pull request fixed the bug. Although the actual output provides detailed information about the fix, it conflicts with the expected merged PR number and description, leading to a low alignment score.

**hist-12** (history): What is the main change in Starlette version 1.9.0 regarding the TestClient, and how does it affect response streaming and application cancellation?
- contextual_precision 0.64: The score is 0.64 because the first node in retrieval contexts, ranked 1st, is irrelevant as it discusses streaming issues unrelated to Starlette 1.9.0 or TestClient changes, which lowers the precision. However, nodes ranked 2nd, 3rd, and 4th are relevant and correctly prioritized after the irrelevant node, providing detailed explanations about the streaming behavior and application cancellation changes in Starlette 1.9.0, justifying the moderate score.

**hist-14** (history): Does Starlette's TestClient support streaming responses now, and which change added it?
- contextual_precision 0.32: The score is 0.32 because the top four nodes in the retrieval contexts are irrelevant, as they either mention fixes not confirmed as merged or discuss unrelated features, which lowers the precision. However, the score is not lower because the relevant nodes, which detail the implementation and merged PR (#3491) adding streaming response support to TestClient, are present and ranked starting from the fifth node, showing some correct ordering of relevant information.

**multi-02** (multi_part): Are HTTP response trailers preserved through both BaseHTTPMiddleware and GZipMiddleware?
- correctness 0.50: The actual output incorrectly states that trailer preservation through BaseHTTPMiddleware was merged, while the expected output clarifies that the PR was closed without merging. It correctly notes that GZipMiddleware preserves trailers, aligning with the expected output. Due to this factual contradiction regarding BaseHTTPMiddleware, the response is partially correct but contains a significant error.

