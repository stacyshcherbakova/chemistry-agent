"""Streamlit front end for the chemistry agent, for trying it with real people.

Run from the repo root:

    poetry install --with app
    poetry run streamlit run app/streamlit_app.py

Every solve is appended to app/usage_log.jsonl as one JSON line, and any
feedback is appended later as a separate line that points back to the solve
by its id. The file is append-only and gitignored.
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import streamlit as st

from chemistry_agent.agents.orchestrator import Orchestrator
from chemistry_agent.config import AgentConfig
from chemistry_agent.models import SolveEvent, SolveOutcome

LOG_PATH = Path(__file__).resolve().parent / "usage_log.jsonl"
LEVELS = ["auto", "gcse", "alevel", "university"]
LEVEL_LABELS = {
    "auto": "Auto (let the planner decide)",
    "gcse": "GCSE",
    "alevel": "A-level",
    "university": "University",
}
MAX_QUESTION_CHARS = 2000

# Sonnet as solver matched Opus on the benchmark at about two-thirds of the
# cost (see the README's Results); planner and verifier stay on the default.
SOLVER_MODEL = os.environ.get("CHEM_APP_SOLVER_MODEL", "claude-sonnet-5")
# Every question spends the owner's API credit, so cap questions per browser
# session. Override with CHEM_APP_MAX_SOLVES.
MAX_SOLVES_PER_SESSION = int(os.environ.get("CHEM_APP_MAX_SOLVES", "10"))


# --------------------------------------------------------------------------- #
# API key (same rules as the CLI: .env from the working directory upwards,
# real environment variables win)
# --------------------------------------------------------------------------- #


def load_dotenv_like_cli() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def has_api_key() -> bool:
    return "ANTHROPIC_API_KEY" in os.environ or "ANTHROPIC_AUTH_TOKEN" in os.environ


# --------------------------------------------------------------------------- #
# Logging: one JSON object per line, append only. Never write credentials.
# --------------------------------------------------------------------------- #


@st.cache_resource
def log_lock() -> threading.Lock:
    # Shared across all browser sessions served by this process.
    return threading.Lock()


def append_log(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with log_lock():
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def solve_record(solve_id: str, question: str, level: str, outcome: SolveOutcome) -> dict:
    answer = outcome.answer
    return {
        "type": "solve",
        "id": solve_id,
        "timestamp": now_iso(),
        "question": question,
        "level": level,
        "formatted_answer": answer.formatted_answer if answer else None,
        "value": answer.value if answer else None,
        "units": answer.units if answer else None,
        "insufficient_data": bool(getattr(answer, "insufficient_data", False)),
        "verifier_approved": outcome.verdict.approved if outcome.verdict else None,
        "attempts": outcome.attempts,
        "tool_calls": outcome.tool_calls,
        "cost_usd": outcome.cost_usd,
        "error": outcome.error,
    }


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def md(text: str) -> str:
    """Escape characters Streamlit markdown would otherwise interpret ($ is LaTeX)."""
    return re.sub(r"([\\`*_\[\]#|~$])", r"\\\1", text or "")


def make_observer(status):
    """Write orchestrator progress events into an st.status container.

    The orchestrator calls this synchronously from the solve loop, which runs
    in the script thread, so writing to Streamlit elements here is safe.
    """

    def observer(event: SolveEvent) -> None:
        p = event.payload
        if event.kind == "plan_ready":
            steps = "\n".join(
                f"{i}. {md(s)}" for i, s in enumerate(p.get("steps", []), start=1)
            )
            target = md(p.get("target_quantity") or "?")
            units = p.get("expected_units")
            status.markdown(
                f"**Plan** ({p.get('difficulty', '?')}, effort {p.get('effort', '?')})\n\n"
                f"{steps}\n\nTarget: {target}" + (f" ({md(units)})" if units else "")
            )
            status.update(label="Plan ready. Solving...")
        elif event.kind == "attempt_start":
            n, total = p.get("attempt", "?"), p.get("max_attempts", "?")
            status.markdown(f"**Attempt {n} of {total}**")
            status.update(label=f"Solving (attempt {n} of {total})...")
        elif event.kind == "tool_call":
            status.markdown(f"- tool: `{p.get('name', '?')}`")
        elif event.kind == "answer_submitted":
            status.markdown(f"Answer submitted: {md(p.get('formatted_answer', ''))}")
            status.update(label="Verifying the answer...")
        elif event.kind == "verdict":
            if p.get("approved"):
                status.markdown(":green[Verifier approved]")
            else:
                issues = "; ".join(p.get("issues", [])) or "no details"
                status.markdown(f":orange[Verifier rejected:] {md(issues)}")
        elif event.kind == "error":
            status.markdown(
                f":red[Error in {p.get('stage', '?')}:] {md(p.get('message', ''))}"
            )

    return observer


def run_solve(question: str, level: str) -> SolveOutcome:
    with st.status("Planning...", expanded=True) as status:
        try:
            config = AgentConfig(solver_model=SOLVER_MODEL)
            orchestrator = Orchestrator(config=config, observer=make_observer(status))
            outcome = orchestrator.solve(
                question, level_hint=None if level == "auto" else level
            )
        except Exception as exc:  # the orchestrator catches agent errors; this is a backstop
            outcome = SolveOutcome(prompt=question, error=f"app: {exc}")
        state = "complete" if outcome.success else "error"
        label = "Done" if outcome.success else "Finished with problems"
        status.update(label=label, state=state, expanded=False)
    return outcome


def render_result(outcome: SolveOutcome) -> None:
    answer = outcome.answer

    if answer is None:
        st.error("No answer was produced.")
    elif getattr(answer, "insufficient_data", False):
        st.warning("The agent says this question can't be answered as written.")
        st.write(answer.formatted_answer)  # explains what is missing or wrong
    else:
        st.markdown(f"### {md(answer.formatted_answer)}")

    if answer is not None:
        if outcome.verdict is None:
            st.info("Not verified: the verifier could not check this answer.")
        elif outcome.verdict.approved:
            st.success("The verifier approved this answer.")
        else:
            issues = "; ".join(outcome.verdict.issues) or "no details given"
            st.warning(f"The verifier did not approve this answer: {issues}")

    if outcome.error:
        st.error(f"Error: {outcome.error}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Attempts", outcome.attempts)
    c2.metric("Tool calls", len(outcome.tool_calls))
    c3.metric("Cost", f"${outcome.cost_usd:.4f}")

    with st.expander("How it got there"):
        if outcome.plan is not None:
            plan = outcome.plan
            st.markdown(
                f"**Plan** (difficulty: {plan.difficulty}, topic: {plan.topic})"
            )
            st.markdown(
                "\n".join(
                    f"{i}. {md(s.description)}" for i, s in enumerate(plan.steps, start=1)
                )
            )
            target = md(plan.target_quantity)
            if plan.expected_units:
                target += f" ({md(plan.expected_units)})"
            st.markdown(f"Target: {target}")
        if outcome.tool_calls:
            st.markdown("**Tool calls, in order**")
            st.markdown(
                "\n".join(
                    f"{i}. `{name}`" for i, name in enumerate(outcome.tool_calls, start=1)
                )
            )
        if answer is not None:
            st.markdown("**Reasoning**")
            st.write(answer.reasoning_summary)


def render_feedback(solve_id: str) -> None:
    sent: set = st.session_state.setdefault("feedback_sent", set())
    if solve_id in sent:
        st.caption("Thanks, your feedback was recorded.")
        return

    with st.form(key=f"feedback_form_{solve_id}"):
        st.markdown("**Was this answer right?**")
        rating = st.feedback("thumbs", key=f"thumbs_{solve_id}")
        comment = st.text_area(
            "Comment (optional)",
            placeholder="e.g. the expected answer was 0.25 mol, or the question was misread",
            max_chars=1000,
        )
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        if rating is None and not comment.strip():
            st.warning("Pick a thumbs up or down, or write a comment.")
            return
        try:
            append_log(
                {
                    "type": "feedback",
                    "solve_id": solve_id,
                    "timestamp": now_iso(),
                    "rating": None if rating is None else ("up" if rating == 1 else "down"),
                    "comment": comment.strip() or None,
                }
            )
        except OSError as exc:
            st.error(f"Could not save your feedback: {exc}")
            return
        sent.add(solve_id)
        st.success("Thanks, your feedback was recorded.")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Chemistry Agent", page_icon="⚗️")

with st.sidebar:
    st.header("What this is")
    st.markdown(
        "A multi-agent chemistry solver built on Claude. A **planner** breaks the "
        "question into steps, a **solver** does every calculation with real tools "
        "(no mental arithmetic), and a **verifier** checks the answer before it is "
        "shown. If the verifier rejects it, the solver tries again.\n\n"
        "It covers GCSE, A-level and intro-university chemistry. It can still be "
        "wrong, so please say so with the feedback buttons."
    )
    st.header("Cost")
    st.markdown(
        "Each question calls the Anthropic API and typically costs about "
        "$0.08 to $0.15; harder questions and retries cost more. The exact cost "
        f"is shown with each answer. You can ask up to {MAX_SOLVES_PER_SESSION} "
        "questions per session."
    )
    st.header("Privacy")
    st.markdown(
        "Questions, answers and feedback are saved so the owner can see where it "
        "goes wrong. Don't type anything personal."
    )

st.title("Chemistry Agent")
st.write("Ask a chemistry calculation question and watch the agents work it out.")

load_dotenv_like_cli()
key_ok = has_api_key()
if not key_ok:
    st.error(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY in the environment "
        "or in a .env file in the repo, then restart the app."
    )

question = st.text_area(
    "Question",
    placeholder="e.g. What mass of CaO is produced when 25.0 g of CaCO3 is heated?",
    height=120,
    max_chars=MAX_QUESTION_CHARS,
)
level = st.selectbox("Level", LEVELS, format_func=LEVEL_LABELS.get)
solves_used = st.session_state.get("solves_used", 0)
limit_reached = solves_used >= MAX_SOLVES_PER_SESSION
if limit_reached:
    st.info(
        f"You've used all {MAX_SOLVES_PER_SESSION} questions for this session. "
        "Thanks for trying it."
    )
solve_clicked = st.button("Solve", type="primary", disabled=not key_ok or limit_reached)

if solve_clicked:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        st.session_state["solves_used"] = solves_used + 1
        solve_id = uuid.uuid4().hex
        outcome = run_solve(question.strip(), level)
        try:
            append_log(solve_record(solve_id, question.strip(), level, outcome))
        except OSError as exc:
            st.caption(f"(Could not write the usage log: {exc})")
        st.session_state["result"] = {"id": solve_id, "outcome": outcome}

result: Optional[dict] = st.session_state.get("result")
if result is not None:
    st.divider()
    render_result(result["outcome"])
    st.divider()
    render_feedback(result["id"])
