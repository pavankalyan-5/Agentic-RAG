"""HTTP API: threads, questions, approvals, and a Server-Sent Events progress stream."""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from copilot.config import Settings
from copilot.db.connection import connect
from copilot.service.runner import NotAwaiting, ThreadBusy, ThreadRunner

log = logging.getLogger("copilot.api")
ThreadId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
# The chat page is self-contained: no external scripts, can't be framed (clickjacking the Approve button).
PAGE_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
                               " img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self';"
                               " frame-ancestors 'none'",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
STATIC = FilePath(__file__).parent / "static"
POLL_S = 0.5


class Caller(BaseModel):
    role: Literal["reader", "maintainer"]
    fingerprint: str  # identifies the key in audit logs without storing the key itself


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Decision(BaseModel):
    approved: bool
    body: str | None = Field(default=None, min_length=1, max_length=10_000)  # optional edited text


def create_app(runner: ThreadRunner, settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Resume runs that died with a previous server process. Not only at startup: after a quick
        # restart the dead run's lease is still valid, so keep checking until it expires.
        async def recovery_loop():
            while True:
                await asyncio.sleep(settings.recover_interval_s)
                try:
                    await asyncio.to_thread(runner.recover)
                except Exception:
                    log.exception("recovery pass failed; retrying next interval")  # e.g. DB briefly down

        await asyncio.to_thread(runner.recover)  # first pass before serving requests
        task = asyncio.create_task(recovery_loop())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Dev Docs Copilot", lifespan=lifespan)

    def caller(x_api_key: Annotated[str | None, Header()] = None) -> Caller:
        given = (x_api_key or "").encode()
        role = None
        for key, key_role in settings.api_keys.items():  # constant-time: no timing hints about valid keys
            if hmac.compare_digest(given, key.encode()):
                role = key_role
        if role is None or not given:
            raise HTTPException(401, "missing or invalid X-API-Key")
        return Caller(role=role, fingerprint="key:" + hashlib.sha256(given).hexdigest()[:10])

    Authed = Annotated[Caller, Depends(caller)]

    def visible(thread_id: str, who: Caller) -> None:
        """Readers see only threads they created; maintainers see all. Others' threads are 404, not 403,
        so a reader can't learn which ids exist. Threads with no owner (CLI/recovery) are maintainer-only."""
        exists, owner = runner.owner_of(thread_id)
        if not exists or (who.role != "maintainer" and owner != who.fingerprint):
            raise HTTPException(404, "thread not found")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers=PAGE_HEADERS)

    @app.get("/api/v1/info")
    def info(who: Authed) -> dict:
        """What the UI needs to describe itself; also lets it validate the key."""
        return {"repo": settings.demo_repo, "sandbox_repo": settings.sandbox_repo or None, "role": who.role}

    @app.post("/api/v1/threads", status_code=201)
    def create_thread(who: Authed) -> dict:
        thread_id = uuid.uuid4().hex  # 128 bits: not guessable
        runner.create_thread(thread_id, owner=who.fingerprint)
        return {"thread_id": thread_id}

    @app.get("/api/v1/threads/{thread_id}")
    def get_thread(thread_id: ThreadId, who: Authed) -> dict:
        visible(thread_id, who)
        return runner.snapshot(thread_id)

    @app.post("/api/v1/threads/{thread_id}/messages", status_code=202)
    def ask(thread_id: ThreadId, body: Question, who: Authed) -> dict:
        visible(thread_id, who)
        try:
            runner.start_turn(thread_id, body.question)
        except ThreadBusy:
            raise HTTPException(409, "this thread is busy, waiting for an approval, or has an unfinished action")
        return {"status": "accepted"}

    @app.post("/api/v1/threads/{thread_id}/retry", status_code=202)
    def retry(thread_id: ThreadId, who: Authed) -> dict:
        visible(thread_id, who)
        try:
            runner.retry(thread_id)
        except NotAwaiting:
            raise HTTPException(409, "only a failed run can be retried")
        return {"status": "accepted"}

    @app.post("/api/v1/threads/{thread_id}/approve", status_code=202)
    def approve(thread_id: ThreadId, body: Decision, who: Authed) -> dict:
        visible(thread_id, who)
        if who.role != "maintainer":
            raise HTTPException(403, "only maintainers can approve or reject actions")
        decision = {"approved": body.approved, "by": who.fingerprint, "role": who.role, "body": body.body}
        try:
            runner.decide(thread_id, decision)
        except NotAwaiting:
            raise HTTPException(409, "nothing is waiting for approval (already decided?)")
        return {"status": "accepted"}

    @app.get("/api/v1/threads/{thread_id}/events")
    async def events(thread_id: ThreadId, who: Authed, after: Annotated[int, Query(ge=0)] = 0):
        await asyncio.to_thread(visible, thread_id, who)

        async def stream():
            index = after
            conn = await asyncio.to_thread(connect, runner.database_url)  # one per stream, same DB as the runner
            try:
                while True:
                    for event in runner.events_since(thread_id, index):
                        index += 1
                        yield f"data: {json.dumps(event)}\n\n"
                    status = await asyncio.to_thread(runner.status, thread_id, conn)
                    if status != "running":
                        # Drain events written between the last read and the status change, then finish.
                        for event in runner.events_since(thread_id, index):
                            index += 1
                            yield f"data: {json.dumps(event)}\n\n"
                        yield f"data: {json.dumps({'type': 'status', 'status': status})}\n\n"
                        return
                    await asyncio.sleep(POLL_S)
            finally:
                await asyncio.to_thread(conn.close)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
