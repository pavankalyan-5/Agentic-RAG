"""Drive a running `copilot serve` over HTTP, like any client would (used for the plan-3 demos).

  uv run python scripts/demo_service.py ask "question" [--thread ID]   # follows progress, prints answer
  uv run python scripts/demo_service.py approve THREAD [--reject]
  uv run python scripts/demo_service.py status THREAD

Uses the first maintainer key from COPILOT_API_KEYS and http://127.0.0.1:8000 (override with DEMO_URL).
"""

import json
import os
import sys
import time

import httpx

from copilot.config import get_settings

URL = os.environ.get("DEMO_URL", "http://127.0.0.1:8000")
KEY = next(k for k, role in get_settings().api_keys.items() if role == "maintainer")
H = {"X-API-Key": KEY}


def follow(client: httpx.Client, thread: str) -> None:
    t0 = time.perf_counter()
    with client.stream("GET", f"{URL}/api/v1/threads/{thread}/events", headers=H, timeout=None) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                e = json.loads(line[6:])
                label = e.get("label") or f"status → {e['status']}"
                print(f"  +{time.perf_counter() - t0:5.2f}s  {label}", flush=True)


def show(client: httpx.Client, thread: str) -> dict:
    snap = client.get(f"{URL}/api/v1/threads/{thread}", headers=H).json()
    print(f"\nstatus: {snap['status']}")
    if snap.get("answer"):
        print("\n" + snap["answer"]["text"])
        for s in snap["answer"]["sources"]:
            print(f"  [{s['chunk_id']}] {s['doc_type']} {s.get('number') or ''} {s['title']} — {s['url']}")
    if snap.get("pending_action"):
        p = snap["pending_action"]
        print(f"\n⏸ pending: {p['kind']} on #{p['issue_number']} in {p['repo']}\n---\n{p['body']}\n---")
    if snap.get("action_result"):
        print(f"\naction_result: {snap['action_result']}")
    return snap


def main() -> None:
    cmd, args = sys.argv[1], sys.argv[2:]
    with httpx.Client(timeout=30) as client:
        if cmd == "ask":
            question = args[0]
            thread = args[args.index("--thread") + 1] if "--thread" in args else \
                client.post(f"{URL}/api/v1/threads", headers=H).json()["thread_id"]
            print(f"thread {thread}: {question}")
            r = client.post(f"{URL}/api/v1/threads/{thread}/messages", headers=H, json={"question": question})
            print(f"POST /messages → {r.status_code}")
            follow(client, thread)
            show(client, thread)
        elif cmd == "approve":
            thread = args[0]
            r = client.post(f"{URL}/api/v1/threads/{thread}/approve", headers=H,
                            json={"approved": "--reject" not in args})
            print(f"POST /approve → {r.status_code} {r.json()}")
            if r.status_code == 202:
                follow(client, thread)
                show(client, thread)
        elif cmd == "status":
            show(client, args[0])


if __name__ == "__main__":
    main()
