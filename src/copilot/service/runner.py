"""Runs agent turns in the background, tracks their status, and resumes runs that died mid-way.

Durability model:
- LangGraph checkpoints the conversation after every node (Postgres).
- thread_metadata holds each thread's status plus a *lease* (lease_until) owned by one worker
  (lease_owner, a random fencing token). While a run is in progress a heartbeat renews the lease;
  if the process dies the lease expires.
- recover() (at startup and periodically) atomically claims every 'running' thread whose lease
  expired, gives it a NEW owner token, and continues it from its last checkpoint.
- Fencing: every lease renewal and status write is conditional on the caller's token. A worker
  whose lease was taken over (it stalled, or its node outlived the lease) finds 0 rows updated,
  stops at the next node boundary, and never overwrites the new owner's status.
- The question is stored with the claim (pending_input), so recovery can replay a question that
  LangGraph never checkpointed, and can tell a finished turn from an unfinished one.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime, timezone

from langgraph.types import Command
from psycopg.types.json import Jsonb

from copilot.agent.graph import is_awaiting_approval, new_turn_input, turn_result
from copilot.db.connection import connect

log = logging.getLogger("copilot.runner")

NODE_LABELS = {
    "plan": "Planning searches",
    "retrieve": "Searching docs, issues and PRs",
    "live_lookup": "Checking GitHub",
    "synthesize": "Writing the answer",
    "verify": "Verifying citations",
    "propose_action": "Drafting the GitHub action",
    "execute_action": "Posting to GitHub",
    "respond": "Done",
}
# A turn paused or stopped at one of these must be finished (or decided) before a new question:
# starting a new turn would silently discard an action the user asked for or already approved.
_ACTION_STEPS = {"approve", "execute_action"}


class ThreadBusy(Exception):
    """The thread is already running a turn, waiting for an approval, or has an unfinished action."""


class NotAwaiting(Exception):
    """The thread isn't in the state this request needs (e.g. a second Approve click)."""


class LeaseLost(Exception):
    """Another worker owns this run now; stop without writing anything."""


class RunDeadline(Exception):
    """The run exceeded its overall time budget (a hung worker must not hold a lease forever)."""


class InlineExecutor(Executor):
    """Runs submitted work immediately in the caller's thread (tests, CLI)."""

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # recorded by _execute already; keep the Future contract
            future.set_exception(exc)
        return future


class _StopSignal:
    """Set by the heartbeat when the run must stop (lease lost or deadline passed)."""

    def __init__(self):
        self.reason: str | None = None

    def check(self) -> None:
        if self.reason == "lease lost":
            raise LeaseLost()
        if self.reason == "deadline":
            raise RunDeadline("run deadline exceeded")


class ThreadRunner:
    def __init__(
        self,
        graph_factory: Callable[[], AbstractContextManager],
        database_url: str,
        executor: Executor | None = None,
        lease_s: float = 60,
        run_deadline_s: float = 900,
    ):
        # graph_factory() yields a graph with its OWN db connection: runs happen on worker threads,
        # and a psycopg connection must not be shared between threads.
        self.graph_factory = graph_factory
        self.database_url = database_url
        self.executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="copilot-run")
        self.lease_s = lease_s
        self.heartbeat_s = lease_s / 3
        self.run_deadline_s = run_deadline_s
        self._events: dict[str, list[dict]] = {}
        self._inflight: set[str] = set()  # threads submitted in THIS process (from submit time)
        self._lock = threading.Lock()

    # ---------- public API ----------

    def create_thread(self, thread_id: str, owner: str | None = None) -> None:
        with connect(self.database_url) as conn:
            conn.execute(
                "insert into thread_metadata (thread_id, status, owner) values (%s, 'idle', %s)"
                " on conflict do nothing",
                (thread_id, owner),
            )
            conn.commit()

    def owner_of(self, thread_id: str) -> tuple[bool, str | None]:
        """(exists, owner fingerprint)."""
        with connect(self.database_url) as conn:
            row = conn.execute("select owner from thread_metadata where thread_id = %s", (thread_id,)).fetchone()
        return (row is not None, row["owner"] if row else None)

    def start_turn(self, thread_id: str, question: str) -> None:
        with self.graph_factory() as graph:
            if _ACTION_STEPS & set(graph.get_state(_config(thread_id)).next):
                raise ThreadBusy(f"{thread_id}: decide or retry the pending GitHub action first")
        with connect(self.database_url) as conn:
            conn.execute(
                "insert into thread_metadata (thread_id, status) values (%s, 'idle') on conflict do nothing",
                (thread_id,),
            )
            token = self._claim(conn, thread_id, "status not in ('running', 'awaiting_approval')",
                                pending_input={"question": question})
        if not token:
            raise ThreadBusy(thread_id)
        self._submit(thread_id, token, {"input": new_turn_input(question)})

    def decide(self, thread_id: str, decision: dict) -> None:
        with connect(self.database_url) as conn:
            token = self._claim(conn, thread_id, "status = 'awaiting_approval'")
        if not token:
            raise NotAwaiting(thread_id)
        self._submit(thread_id, token, {"input": Command(resume=decision)})

    def retry(self, thread_id: str) -> None:
        """Continue a failed turn from its last checkpoint (or replay its question)."""
        with connect(self.database_url) as conn:
            token = self._claim(conn, thread_id, "status = 'failed'")
            row = conn.execute("select pending_input from thread_metadata where thread_id = %s", (thread_id,)).fetchone()
        if not token:
            raise NotAwaiting(thread_id)
        self._submit(thread_id, token, {"recover": row["pending_input"]})

    def recover(self) -> list[str]:
        """Resume runs whose worker died. Safe to call from several processes at once."""
        with self._lock:
            mine = list(self._inflight)
        token = uuid.uuid4().hex
        with connect(self.database_url) as conn:
            rows = conn.execute(
                "update thread_metadata set lease_until = now() + make_interval(secs => %s),"
                " lease_owner = %s, updated_at = now()"
                " where status = 'running' and lease_until < now() and not (thread_id = any(%s))"
                " returning thread_id, pending_input",
                (self.lease_s, token, mine),
            ).fetchall()
            conn.commit()
        for row in rows:
            log.warning("recovering thread %s (its previous worker stopped renewing the lease)", row["thread_id"])
            self._submit(row["thread_id"], token, {"recover": row["pending_input"]})
        return [r["thread_id"] for r in rows]

    def snapshot(self, thread_id: str) -> dict:
        with connect(self.database_url) as conn:
            meta = conn.execute(
                "select status, last_error from thread_metadata where thread_id = %s", (thread_id,)
            ).fetchone()
        status = meta["status"] if meta else "idle"
        with self.graph_factory() as graph:
            values = graph.get_state(_config(thread_id)).values
            result = turn_result(graph, thread_id) if values.get("answer") else None
        error = None
        if meta and meta["last_error"]:
            # Details stay in the database/server log for operators; clients get the error type.
            error = f"The run failed ({meta['last_error'].split(':', 1)[0]}). Details are in the server log."
        return {
            "thread_id": thread_id,
            "status": status,
            "last_error": error,
            "messages": values.get("messages", []),
            # While a turn is running, the previous turn's answer would be misleading.
            "answer": result if status != "running" else None,
            "pending_action": values.get("pending_action") if status == "awaiting_approval" else None,
            "action_result": values.get("action_result") if status != "running" else None,
            "events": self.events_since(thread_id, 0),
        }

    def status(self, thread_id: str, conn=None) -> str:
        """Cheap status check (DB only), for polling. Pass a connection to reuse it."""
        if conn is None:
            with connect(self.database_url) as own:
                return self.status(thread_id, own)
        row = conn.execute("select status from thread_metadata where thread_id = %s", (thread_id,)).fetchone()
        conn.commit()  # end the read transaction so the next poll sees fresh data
        return row["status"] if row else "idle"

    def events_since(self, thread_id: str, index: int) -> list[dict]:
        with self._lock:
            return list(self._events.get(thread_id, [])[index:])

    def close(self) -> None:
        """Let in-flight runs finish (call before closing the MCP clients they use)."""
        self.executor.shutdown(wait=True)

    # ---------- internals ----------

    def _claim(self, conn, thread_id: str, condition: str, pending_input: dict | None = None) -> str | None:
        token = uuid.uuid4().hex
        row = conn.execute(
            "update thread_metadata set status = 'running', last_error = null, updated_at = now(),"
            " lease_until = now() + make_interval(secs => %s), lease_owner = %s,"
            " pending_input = coalesce(%s, pending_input)"
            f" where thread_id = %s and {condition} returning thread_id",
            (self.lease_s, token, Jsonb(pending_input) if pending_input else None, thread_id),
        ).fetchone()
        conn.commit()
        return token if row else None

    def _submit(self, thread_id: str, token: str, work: dict) -> None:
        with self._lock:
            self._inflight.add(thread_id)  # from submit time, so a queued run isn't "recovered"
        self._reset_events(thread_id)
        signal, finished = _StopSignal(), threading.Event()
        # The heartbeat starts now (not when a worker picks the run up): queued runs stay alive too.
        threading.Thread(target=self._heartbeat, args=(thread_id, token, signal, finished),
                         daemon=True, name=f"lease-{thread_id}").start()
        self.executor.submit(self._execute, thread_id, token, work, signal, finished)

    def _heartbeat(self, thread_id: str, token: str, signal: _StopSignal, finished: threading.Event) -> None:
        deadline = time.monotonic() + self.run_deadline_s
        try:
            with connect(self.database_url) as conn:  # its own connection: never shared across threads
                while not finished.wait(self.heartbeat_s):
                    if time.monotonic() > deadline:
                        signal.reason = "deadline"
                        return
                    if not self._renew(conn, thread_id, token):
                        signal.reason = "lease lost"
                        return
        except Exception:
            log.exception("heartbeat for %s failed; the lease will expire and the run can be recovered", thread_id)

    def _execute(self, thread_id: str, token: str, work: dict, signal: _StopSignal,
                 finished: threading.Event) -> None:
        try:
            with self.graph_factory() as graph, connect(self.database_url) as conn:
                graph_input = work["input"] if "input" in work else _recovery_input(graph, thread_id, work["recover"])
                if graph_input is not _ALREADY_FINISHED:
                    for update in graph.stream(graph_input, _config(thread_id), stream_mode="updates"):
                        for node in update:
                            if node in NODE_LABELS:
                                self._event(thread_id, node)
                        signal.check()
                        if not self._renew(conn, thread_id, token):
                            raise LeaseLost()
                final = "awaiting_approval" if is_awaiting_approval(graph, thread_id) else "done"
                self._finish(conn, thread_id, token, final)
        except LeaseLost:
            log.warning("thread %s: lease taken over by another worker; stopped without writing status", thread_id)
        except Exception as exc:
            log.exception("thread %s failed", thread_id)
            with connect(self.database_url) as conn:
                self._finish(conn, thread_id, token, "failed", error=f"{type(exc).__name__}: {exc}"[:500])
        finally:
            finished.set()
            with self._lock:
                self._inflight.discard(thread_id)

    def _event(self, thread_id: str, node: str) -> None:
        with self._lock:
            self._events.setdefault(thread_id, []).append(
                {"node": node, "label": NODE_LABELS[node], "at": datetime.now(timezone.utc).isoformat()}
            )

    def _reset_events(self, thread_id: str) -> None:
        with self._lock:
            self._events[thread_id] = []

    def _renew(self, conn, thread_id: str, token: str) -> bool:
        row = conn.execute(
            "update thread_metadata set lease_until = now() + make_interval(secs => %s), updated_at = now()"
            " where thread_id = %s and lease_owner = %s and status = 'running' returning 1",
            (self.lease_s, thread_id, token),
        ).fetchone()
        conn.commit()
        return row is not None

    @staticmethod
    def _finish(conn, thread_id: str, token: str, status: str, error: str | None = None) -> None:
        # Only the current lease owner may write a final status (fencing). A finished turn no longer
        # needs its question for replay; a failed one keeps it so retry/recovery can use it.
        conn.execute(
            "update thread_metadata set status = %s, last_error = %s, lease_until = null, lease_owner = null,"
            " pending_input = case when %s in ('done', 'awaiting_approval') then null else pending_input end,"
            " updated_at = now() where thread_id = %s and lease_owner = %s",
            (status, error, status, thread_id, token),
        )
        conn.commit()


_ALREADY_FINISHED = object()


def _recovery_input(graph, thread_id: str, pending: dict | None):
    """What to feed the graph when resuming a run we didn't start ourselves.

    - The checkpoint has unfinished steps for the SAME question  -> continue (None).
    - The question never reached LangGraph (died right after it was accepted), or the checkpoint
      belongs to an older turn                                     -> replay the question.
    - The turn's answer is already in the checkpoint               -> nothing to do, just mark done.
    """
    snap = graph.get_state(_config(thread_id))
    question = (pending or {}).get("question")
    if snap.next and (question is None or snap.values.get("question") == question):
        return None
    if question is None:
        return _ALREADY_FINISHED
    msgs = snap.values.get("messages") or []
    finished = len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[-2]["content"] == question
    return _ALREADY_FINISHED if finished else new_turn_input(question)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
