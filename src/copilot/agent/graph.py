"""Wires the nodes into a LangGraph state machine with Postgres checkpoints and retries.

    START → plan → retrieve → synthesize → verify ─(done)→ respond → END
                                  ▲            │
                                  └──(retry)───┘   (one self-correction for bad citations)
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from copilot.agent.nodes import AgentDeps, plan_node, retrieve_node, synthesize_node, verify_node
from copilot.agent.state import AgentState
from copilot.llm.gateway import LLMOutputError


def respond_node(state: dict) -> dict:
    """Append the final answer to the conversation, inside the same checkpointed run."""
    return {"messages": [{"role": "assistant", "content": state["answer"]["text"]}]}


def build_graph(deps: AgentDeps, checkpointer, retry_initial_interval: float = 0.5):
    # Only malformed/refused model output is retried here. HTTP errors are already retried in the
    # gateway, and BudgetExceeded must never be retried.
    llm_retry = RetryPolicy(max_attempts=3, initial_interval=retry_initial_interval, retry_on=LLMOutputError)

    g = StateGraph(AgentState)
    g.add_node("plan", partial(plan_node, deps=deps), retry_policy=llm_retry)
    g.add_node("retrieve", partial(retrieve_node, deps=deps))
    g.add_node("synthesize", partial(synthesize_node, deps=deps), retry_policy=llm_retry)
    g.add_node("verify", partial(verify_node, deps=deps))
    g.add_node("respond", respond_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", lambda s: s["next_step"], {"retry": "synthesize", "done": "respond"})
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


@contextmanager
def open_checkpointer(database_url: str) -> Iterator[PostgresSaver]:
    with PostgresSaver.from_conn_string(database_url) as saver:
        saver.setup()  # idempotent: creates LangGraph's checkpoint tables on first use
        yield saver


def run_turn(graph, thread_id: str, question: str) -> dict:
    """Run one question in a conversation thread and return the final answer dict."""
    turn_input = {
        "question": question,
        "messages": [{"role": "user", "content": question}],  # appended by the reducer
        # Per-turn fields are reset so nothing leaks from the previous turn in this thread.
        "plan": [], "retrieved_docs": [], "draft": None, "answer": None,
        "error_context": None, "correction_attempts": 0, "next_step": "",
    }
    final = graph.invoke(turn_input, config={"configurable": {"thread_id": thread_id}})
    return final["answer"]
