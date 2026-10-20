# Answer evaluation (50 cases)

**Quality gate: PASSED**

| metric | value |
|---|---:|
| cases | 50 |
| faithfulness | 0.9599 |
| answer_relevancy | 0.9754 |
| contextual_precision | 0.8438 |
| contextual_recall | 0.9905 |
| correctness | 0.8553 |
| hallucination_rate | 0.0401 |
| unanswerable_correct | 1.0000 |
| adversarial_safe | 1.0000 |
| answered_rate | 0.9211 |
| cost_usd | 0.0010 |
| mean_seconds | 0.9308 |
| hit_rate@5 | 1.0000 |
| mrr@5 | 0.9539 |

## By category

| category | cases | correctness | faithfulness | rule checks passed |
|---|---:|---:|---:|---:|
| adversarial | 6 | – | – | 6/6 |
| docs | 24 | 0.85 | 0.98 | – |
| history | 11 | 0.87 | 0.91 | – |
| multi_part | 3 | 0.83 | 1.00 | – |
| unanswerable | 6 | – | – | 6/6 |

## RAG vs no-RAG (correctness on answerable cases)

| system | correctness |
|---|---:|
| **with retrieval (this agent)** | 0.86 |
| model alone (no retrieval) | 0.65 |

## Cases to look at (13)

**docs-08** (docs): How can I run one or multiple background tasks after sending a response in Starlette, and what happens if one of the tasks fails?
- said it didn't know, but the golden set says this is answerable

**docs-10** (docs): How do I declare typed path parameters in Starlette routes and build URLs back from route names?
- said it didn't know, but the golden set says this is answerable

**docs-11** (docs): How do I set up routes and startup/shutdown logic when creating a Starlette application?
- contextual_precision 0.46: The score is 0.46 because while relevant nodes such as the 2nd, 4th, and 8th nodes in retrieval contexts provide useful information on setting up routes and startup/shutdown logic, several irrelevant nodes like the 1st and 3rd nodes, which discuss unrelated topics such as API schemas and middleware, are ranked higher or intermixed among relevant ones. This mixing of irrelevant nodes early in the ranking lowers the score, but the presence of key relevant nodes within the top ranks keeps it from being lower.

**docs-17** (docs): How can I use Starlette to push a static CSS file to the client using HTTP/2 or HTTP/3 server push?
- contextual_precision 0.67: The score is 0.67 because the first and sixth nodes in retrieval contexts are correctly ranked higher due to their direct relevance, with the sixth node providing a specific example of pushing a CSS file using 'Request.send_push_promise'. However, the presence of irrelevant nodes like the second and third nodes, which offer only general introductions without mentioning server push, ranked above some relevant nodes, prevents a higher score.

**docs-20** (docs): How can I access and handle uploaded files in a Starlette HTTP request, and what attributes and methods does the UploadFile object provide?
- said it didn't know, but the golden set says this is answerable

**docs-23** (docs): How does Starlette handle configuration loading and what are the benefits of using its Config class for environment variables and secrets?
- contextual_recall 0.67: The score is 0.67 because sentences 1, 2, 3, and 6 in the expected output are well supported by node 1 in the retrieval context, which details configuration loading from environment variables and .env files, casting support, Secret wrapping, and environment variable prefixes. However, sentences 4 and 5 about raising errors when no configuration is found and the environ mapping for safe environment variable access are not supported by any node, lowering the overall recall.

**docs-27** (docs): How can I customize error responses for HTTP and WebSocket exceptions in a Starlette application, and what is the difference between handled exceptions and errors?
- contextual_precision 0.67: The score is 0.67 because the first node in retrieval contexts, ranked highest, is relevant and provides detailed guidance on customizing error responses, while the second node is irrelevant as it focuses narrowly on authentication errors. The sixth node, which is relevant and explains the difference between handled exceptions and errors, is ranked lower (sixth), below several irrelevant nodes such as those about CORS middleware and repository issues. This misranking of relevant nodes among irrelevant ones prevents a higher score, but the presence of two relevant nodes within the top ranks maintains a moderate precision.

**hist-02** (history): What are the key user-facing changes introduced in Starlette version 1.8.0, especially regarding Python version requirements and GZipMiddleware behavior?
- correctness 0.50: The actual output correctly identifies the updated Python version requirement to 3.11 but incorrectly states it as the minimum required version without mentioning the dropping of Python 3.10 support. It fails to mention the exclusion of native gRPC media types from GZipMiddleware compression and the immediate sending of the http.response.start event for certain responses, which are key user-facing changes. Additionally, it introduces details about HTTP 103 Early Hints and various GZipMiddleware parameters not referenced in the expected output, which do not contradict but are extraneous. Overall, the actual output partially aligns with the expected output but misses critical details about GZipMiddleware behavior changes, leading to a moderate score.

**hist-05** (history): Where can I find the current documentation for Authlib's integration with Starlette, given that the old link returns a 404 error?
- faithfulness 0.50: The score is 0.50 because the actual output implies the previous link returned a 404 error, whereas the retrieval context only states it was a broken link without specifying a 404 error, indicating a partial mismatch.

**hist-06** (history): Which pull request actually fixed the bug where a URL path starting with // was treated as the host when a request has no Host header?
- contextual_precision 0.00: The score is 0.00 because all nodes in the retrieval contexts are ranked higher despite being irrelevant, as each node's reason explains that although they address the bug, none correspond to the merged fix PR #3671. This misranking of irrelevant nodes at the top results in the lowest possible score.
- correctness 0.30: The actual output incorrectly identifies PR #3590 as the merged fix, while the expected output states that PR #3671 was the merged fix. Although it correctly notes multiple PRs were opened and closed without merging, the key factual detail about which PR was merged contradicts the expected output. Therefore, the response does not align well with the evaluation steps.

**hist-12** (history): What is the main change in Starlette version 1.9.0 regarding the TestClient, and how does it affect response streaming and application cancellation?
- contextual_precision 0.64: The score is 0.64 because the first node in retrieval contexts, ranked 1st, is irrelevant as it discusses streaming issues unrelated to Starlette 1.9.0 or TestClient changes, which lowers the precision. However, nodes ranked 2nd, 3rd, and 4th are relevant and correctly prioritized after the irrelevant node, providing detailed explanations about the streaming behavior and application cancellation changes in Starlette 1.9.0, justifying the moderate score.

**hist-14** (history): Does Starlette's TestClient support streaming responses now, and which change added it?
- contextual_precision 0.32: The score is 0.32 because the top four nodes in the retrieval contexts are irrelevant, as they either mention fixes not confirmed as merged or discuss unrelated features, which lowers the precision. However, the score is not lower because the relevant nodes, which detail the implementation and merged PR (#3491) adding streaming response support to TestClient, are present and ranked starting from the fifth node, showing some correct ordering of relevant information.

**multi-02** (multi_part): Are HTTP response trailers preserved through both BaseHTTPMiddleware and GZipMiddleware?
- correctness 0.50: The actual output incorrectly states that trailer preservation through BaseHTTPMiddleware was merged, while the expected output clarifies that the PR was closed without merging. It correctly notes that GZipMiddleware preserves trailers, aligning with the expected output. Due to this factual contradiction regarding BaseHTTPMiddleware, the response is partially correct but contains a significant error.

