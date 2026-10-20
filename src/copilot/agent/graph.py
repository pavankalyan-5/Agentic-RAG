"""Wires the nodes into a LangGraph state machine with Postgres checkpoints and retries.

    START → plan → retrieve → live_lookup → synthesize → verify ─(done)─┬──────────────────────────→ respond → END
                                              ▲            │            │                                ▲
                                              └──(retry)───┘            └→ propose_action → approve ⏸ → execute_action
                                       (one self-correction for bad citations)   (only when the user asked for an action;
                                                                                  approve pauses until a human decides)
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from copilot.agent.nodes import (
    AgentDeps, action_outcome_text, approve_node, execute_action_node, live_lookup_node, plan_node,
    propose_action_node, retrieve_node, synthesize_node, verify_node,
)
from copilot.agent.state import AgentState
from copilot.llm.gateway import LLMOutputError
from copilot.observability.tracing import set_attributes, span, traced_node


class ThreadAwaitingApproval(Exception):
    """A new question was sent to a thread whose previous turn is paused for approval."""


def respond_node(state: dict) -> dict:
    """Append the final answer to the conversation, inside the same checkpointed run.

    content = answer + action outcome (what the planner reads as history); text/sources/action are
    kept separately so a UI can render citations and outcomes for every past answer."""
    answer, action = state["answer"], state.get("action_result")
    return {"messages": [{
        "role": "assistant",
        "content": answer["text"] + action_outcome_text(action),
        "text": answer["text"],
        "sources": answer["sources"],
        "action": action,
    }]}


def build_graph(deps: AgentDeps, checkpointer, retry_initial_interval: float = 0.5):
    # Only malformed/refused model output is retried here. HTTP errors are already retried in the
    # gateway, and BudgetExceeded must never be retried.
    llm_retry = RetryPolicy(max_attempts=3, initial_interval=retry_initial_interval, retry_on=LLMOutputError)

    g = StateGraph(AgentState)
    def node(name, fn, wants_config=False):
        return traced_node(name, fn, wants_config=wants_config)  # one span per node, nesting its work

    g.add_node("plan", node("plan", partial(plan_node, deps=deps)), retry_policy=llm_retry)
    g.add_node("retrieve", node("retrieve", partial(retrieve_node, deps=deps)))
    g.add_node("live_lookup", node("live_lookup", partial(live_lookup_node, deps=deps)))
    g.add_node("synthesize", node("synthesize", partial(synthesize_node, deps=deps)), retry_policy=llm_retry)
    g.add_node("verify", node("verify", partial(verify_node, deps=deps)))
    g.add_node("propose_action", node("propose_action", partial(propose_action_node, deps=deps), wants_config=True),
               retry_policy=llm_retry)
    g.add_node("approve", approve_node)  # not wrapped: interrupt() unwinds the node by raising
    g.add_node("execute_action", node("execute_action", partial(execute_action_node, deps=deps)))
    g.add_node("respond", node("respond", respond_node))

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "live_lookup")
    g.add_edge("live_lookup", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", _after_verify, ["synthesize", "propose_action", "respond"])
    g.add_conditional_edges("propose_action", lambda s: "approve" if s.get("pending_action") else "respond",
                            ["approve", "respond"])
    g.add_edge("approve", "execute_action")
    g.add_edge("execute_action", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


def _after_verify(state: dict) -> str:
    if state["next_step"] == "retry":
        return "synthesize"
    return "propose_action" if state.get("requested_action") else "respond"


@contextmanager
def open_checkpointer(database_url: str) -> Iterator[PostgresSaver]:
    with PostgresSaver.from_conn_string(database_url) as saver:
        saver.setup()  # idempotent: creates LangGraph's checkpoint tables on first use
        yield saver


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def is_awaiting_approval(graph, thread_id: str) -> bool:
    return graph.get_state(_config(thread_id)).next == ("approve",)


def run_turn(graph, thread_id: str, question: str) -> dict:
    """Run one question. Returns the answer dict plus "status" ("done" | "awaiting_approval"),
    "pending_action" and "action_result"."""
    if is_awaiting_approval(graph, thread_id):
        # Starting a new turn would silently discard the paused action.
        raise ThreadAwaitingApproval(thread_id)
    with turn_span(thread_id, question) as s:
        graph.invoke(new_turn_input(question), config=_config(thread_id))
        return finish_turn_span(s, turn_result(graph, thread_id))


def new_turn_input(question: str) -> dict:
    return {
        "question": question,
        "messages": [{"role": "user", "content": question}],  # appended by the reducer
        # Per-turn fields are reset so nothing leaks from the previous turn in this thread.
        "plan": [], "retrieved_docs": [], "plan_live": [], "live_docs": [], "live_error": None,
        "draft": None, "answer": None, "error_context": None, "correction_attempts": 0, "next_step": "",
        "requested_action": None, "pending_action": None, "approval": None, "action_result": None,
    }


def resume_turn(graph, thread_id: str, decision: dict) -> dict:
    """Continue a turn paused at `approve` with the human's decision."""
    with turn_span(thread_id, None, resumed=True) as s:
        graph.invoke(Command(resume=decision), config=_config(thread_id))
        return finish_turn_span(s, turn_result(graph, thread_id))


def turn_span(thread_id: str, question: str | None, resumed: bool = False):
    """Root span of one turn. session.id groups a conversation's turns in Phoenix."""
    return span("agent_turn", "CHAIN", session__id=thread_id, input__value=question, copilot__resumed=resumed)


def finish_turn_span(s, result: dict) -> dict:
    set_attributes(s, output__value=result.get("text"), copilot__status=result.get("status"),
                   copilot__sources=len(result.get("sources") or []))
    return result


def turn_result(graph, thread_id: str) -> dict:
    values = graph.get_state(_config(thread_id)).values
    answer = dict(values.get("answer") or {})
    result = values.get("action_result")
    awaiting = is_awaiting_approval(graph, thread_id)
    if result and not awaiting:
        answer["text"] = answer.get("text", "") + action_outcome_text(result)
    return {**answer, "status": "awaiting_approval" if awaiting else "done",
            "pending_action": values.get("pending_action"), "action_result": result}
