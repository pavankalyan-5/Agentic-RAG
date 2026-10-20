# Retrieval evaluation (top 5 documents)

| strategy | hit rate@5 | MRR@5 | recall@5 | cases |
|---|---:|---:|---:|---:|
| dense | 0.95 | 0.91 | 0.93 | 38 |
| sparse | 0.95 | 0.79 | 0.93 | 38 |
| hybrid | 1.00 | 0.90 | 0.99 | 38 |
| hybrid+rerank | 1.00 | 0.94 | 0.99 | 38 |

## Hit rate by category

| strategy | docs | history | multi_part |
|---|---:|---:|---:|
| dense | 1.00 (23) | 0.83 (12) | 1.00 (3) |
| sparse | 0.91 (23) | 1.00 (12) | 1.00 (3) |
| hybrid | 1.00 (23) | 1.00 (12) | 1.00 (3) |
| hybrid+rerank | 1.00 (23) | 1.00 (12) | 1.00 (3) |

## Misses (expected document not in the top 5)

**dense** (2): `hist-02` (What are the key user-facing changes introduced in Starlette…), `hist-12` (What is the main change in Starlette version 1.9.0 regarding…)

**sparse** (2): `docs-07` (How do I accept a WebSocket connection and send and receive …), `docs-19` (How can I create class-based HTTP and WebSocket endpoints in…)

**hybrid** (0): none

**hybrid+rerank** (0): none

_Cost of this run: $0.00002 (query embeddings)._
