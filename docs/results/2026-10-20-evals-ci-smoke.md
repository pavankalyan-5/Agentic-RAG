# Answer evaluation (15 cases, CI subset)

**Quality gate: PASSED**

| metric | value |
|---|---:|
| cases | 15 |
| faithfulness | 1.0000 |
| answer_relevancy | 0.9778 |
| contextual_precision | 0.6585 |
| contextual_recall | 0.9333 |
| correctness | 0.3182 |
| hallucination_rate | 0.0000 |
| unanswerable_correct | 1.0000 |
| adversarial_safe | 1.0000 |
| answered_rate | 0.6364 |
| cost_usd | 0.0532 |
| mean_seconds | 4.1532 |
| hit_rate@5 | 1.0000 |
| mrr@5 | 1.0000 |

## By category

| category | cases | correctness | faithfulness | rule checks passed |
|---|---:|---:|---:|---:|
| adversarial | 2 | – | – | 2/2 |
| docs | 5 | 0.58 | 1.00 | – |
| history | 4 | 0.15 | 1.00 | – |
| multi_part | 2 | 0.00 | – | – |
| unanswerable | 2 | – | – | 2/2 |

## Cases to look at (9)

**docs-05** (docs): How can I use Starlette to send HTTP 103 Early Hints to preload resources before the final response is ready?
- said it didn't know, but the golden set says this is answerable

**docs-12** (docs): How can a lifespan handler share resources, like a database pool, with request handlers?
- contextual_recall 0.67: The score is 0.67 because sentences 1 and 2 are well supported by node 1, which discusses lifespan handlers and sharing state dictionaries accessible in request handlers. However, sentence 3 about using TestClient as a context manager to trigger lifespan during tests is not supported by any node in the retrieval context, lowering the overall recall score.

**docs-20** (docs): How can I access and handle uploaded files in a Starlette HTTP request, and what attributes and methods does the UploadFile object provide?
- said it didn't know, but the golden set says this is answerable

**hist-06** (history): Which pull request actually fixed the bug where a URL path starting with // was treated as the host when a request has no Host header?
- contextual_precision 0.25: The score is 0.25 because only the fourth node in the retrieval contexts is relevant and it is ranked last, while the first three nodes are irrelevant and ranked higher. The first node is ranked first despite being irrelevant, as it fixes the bug but was not the final merged fix. The fourth node, although relevant and providing detailed explanation and regression tests, is ranked last, lowering the score. The score is not zero because there is a relevant node present, but its low ranking reduces the overall contextual precision.
- correctness 0.30: The actual output incorrectly identifies PR #3590 as the fix, while the expected output states that PR #3671 was the merged fix. It also mentions other PRs (#3595 and #3611) being closed without merging, which aligns with the expected output's note about multiple competing PRs. However, the key factual discrepancy about which PR actually fixed the bug results in a significant contradiction, lowering the score.

**hist-12** (history): What is the main change in Starlette version 1.9.0 regarding the TestClient, and how does it affect response streaming and application cancellation?
- said it didn't know, but the golden set says this is answerable

**hist-14** (history): Does Starlette's TestClient support streaming responses now, and which change added it?
- contextual_precision 0.29: The score is 0.29 because the top four nodes in retrieval contexts are irrelevant, discussing unrelated features or general introductions without mentioning streaming response support in TestClient. However, the relevant nodes appear later (ranks 5, 7, and 8) and provide direct information about the addition of streaming response support and the specific pull request (#3355) that implemented it, justifying the partial precision score.
- correctness 0.30: The actual output correctly states that TestClient now supports streaming responses, aligning with the input query. However, it incorrectly attributes the feature addition to PR #3355, which was actually an earlier attempt that was closed without merging, as per the expected output. The expected output specifies PR #3491 as the merged change that added streaming support, along with additional context about the release version and earlier attempts. Since the actual output contradicts the expected output regarding which PR added the feature, it cannot receive a high score despite correctly acknowledging the functionality's existence.

**hist-20** (history): Is there a fix for FormData equality when a field has multiple file uploads, and has it been merged?
- said it didn't know, but the golden set says this is answerable

**multi-01** (multi_part): What are the correct constructor signatures for the WebSocket and Request classes in Starlette, and how do they differ from the previously documented versions?
- said it didn't know, but the golden set says this is answerable

**multi-02** (multi_part): Are HTTP response trailers preserved through both BaseHTTPMiddleware and GZipMiddleware?
- said it didn't know, but the golden set says this is answerable

