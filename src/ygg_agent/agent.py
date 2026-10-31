# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""The loop: the model writes one SQL statement, the database decides, the verdict goes back.

No MCP, no tool schemas in the model's context, no JSON-RPC between the model and its
data: the model writes SQL, which is what it knows best, and pg_agent_gate verifies it
against PostgreSQL itself. Measured against the same gate reached through MCP, this loop
used 8x fewer prompt tokens with the same zero breaches (agent-gate-demo/compare).

    agent = Agent(dsn="postgresql://billing_agent:...@db/app",
                  model=OpenAICompatible("http://127.0.0.1:8080/v1"))
    result = agent.run("Mark order 4 as cancelled.")
    result.outcome    # done | pending | refused | gave_up
    result.steps      # every statement, the gate's verdict, the rows it touched

Every proposal, verdict and commit is also an Event handed to the on_event callbacks:
that is where an audit sink, an approval queue or a fleet control plane plugs in.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .gate import Gate, describe_catalog, first_failure
from .model import Model
from .sql import extract_sql, is_done

SYSTEM = (
    "You operate a PostgreSQL database. Each turn, reply with exactly ONE SQL statement in the "
    "PostgreSQL dialect and nothing else -- or the single word DONE once the task is complete. "
    "The database verifies every statement before it runs and tells you what it did, or why "
    "it refused. Use only the relations listed. If a statement is refused, do not try to get "
    "around the reason: change the approach or reply DONE."
)
FEEDBACK = ("The database answered: {verdict}\n"
            "If the task is complete, reply DONE. Otherwise reply with ONE SQL statement.")


@dataclass
class Event:
    """One thing that happened, for on_event callbacks. kind: proposed | refused | kept |
    read | aborted | rolled_back | pending | no_sql | stopped | done | gave_up."""
    kind: str
    agent: str
    task: str
    sql: str = ""
    proposal: int | None = None
    reason: str = ""
    rows_affected: int | None = None
    at: float = field(default_factory=time.time)


@dataclass
class Step:
    sql: str
    proposal: int | None
    verdict: str            # refused | kept | read | aborted | rolled_back | pending | no_sql
    reason: str = ""
    rows_affected: int | None = None
    rows: list = field(default_factory=list)


@dataclass
class Result:
    task: str
    outcome: str            # done | pending | refused (max_refusals in a row) | gave_up
    steps: list[Step] = field(default_factory=list)
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    pending: int | None = None   # dry_run mode: the proposal waiting for a person to commit it

    @property
    def kept(self) -> list[Step]:
        return [s for s in self.steps if s.verdict == "kept"]


class Agent:
    """mode="commit": every statement the gate accepts is committed (writes are kept, reads run).
    mode="dry_run": nothing is kept; the first write the gate accepts is shown and returned as
    `pending`, for a person (or agent_gate.approve() and a commit) to decide."""

    def __init__(self, model: Model, dsn: str | None = None, gate: Gate | None = None, *,
                 mode: str = "commit", max_turns: int = 8, max_refusals: int = 3,
                 max_rows_shown: int = 20,
                 system: str = SYSTEM, on_event: list[Callable[[Event], None]] | None = None):
        if mode not in ("commit", "dry_run"):
            raise ValueError("mode is 'commit' or 'dry_run'")
        if (dsn is None) == (gate is None):
            raise ValueError("give exactly one of dsn or gate")
        self.gate = gate or Gate.connect(dsn)
        self.model = model
        self.mode = mode
        self.max_turns = max_turns
        # An agent the database keeps saying no to stops, instead of probing the gate with
        # variations until it runs out of turns: measured with a plain Qwen3-8B, "delete every
        # item" became eight ever-wider DELETEs, every one aborted. 0 disables the stop.
        self.max_refusals = max_refusals
        self.max_rows_shown = max_rows_shown
        self.system = system
        self.on_event = list(on_event or [])
        self._catalog: str | None = None

    def catalog(self) -> str:
        if self._catalog is None:
            self._catalog = describe_catalog(self.gate.discover())
        return self._catalog

    def run(self, task: str, intent: str | None = None) -> Result:
        intent = (intent or task)[:500]
        result = Result(task=task, outcome="gave_up")
        refused_in_a_row = 0
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Relations you may use:\n{self.catalog()}\n\nTask: {task}"}]
        for _ in range(self.max_turns):
            reply = self.model.complete(messages)
            result.model_calls += 1
            result.prompt_tokens += reply.prompt_tokens
            result.completion_tokens += reply.completion_tokens
            messages.append({"role": "assistant", "content": reply.text})
            if is_done(reply.text):
                result.outcome = "done"
                self._emit("done", task)
                return result
            sql = extract_sql(reply.text)
            if not sql:
                result.steps.append(Step("", None, "no_sql"))
                self._emit("no_sql", task)
                verdict = "no SQL statement found in your reply."
            else:
                step = self._act(task, sql, intent)
                result.steps.append(step)
                if step.verdict == "pending":
                    result.outcome = "pending"
                    result.pending = step.proposal
                    return result
                refused_in_a_row = refused_in_a_row + 1 if step.verdict in ("refused", "aborted") else 0
                if self.max_refusals and refused_in_a_row >= self.max_refusals:
                    result.outcome = "refused"
                    self._emit("stopped", task, reason=f"{refused_in_a_row} refusals in a row")
                    return result
                verdict = self._say(step)
            messages.append({"role": "user", "content": FEEDBACK.format(verdict=verdict)})
        self._emit("gave_up", task)
        return result

    def close(self) -> None:
        self.gate.close()

    # ------------------------------------------------------------------ #
    def _act(self, task: str, sql: str, intent: str) -> Step:
        p = self.gate.propose(sql, intent)
        pid = p.get("proposal")
        if not p.get("ok") or pid is None:
            reason = first_failure(p)
            self._emit("refused", task, sql, pid, reason)
            return Step(sql, pid, "refused", reason)
        self._emit("proposed", task, sql, pid)
        if self.mode == "dry_run":
            out = self.gate.dry_run(pid)
            if p.get("kind") == "write" and out.get("outcome") == "rolled_back":
                step = self._step(sql, pid, out, "pending")
                self._emit("pending", task, sql, pid, rows_affected=step.rows_affected)
                return step
        else:
            out = self.gate.commit(pid)
        step = self._step(sql, pid, out, out.get("outcome") or "aborted")
        self._emit(step.verdict, task, sql, pid, step.reason, step.rows_affected)
        return step

    def _step(self, sql: str, pid: int, out: dict, verdict: str) -> Step:
        return Step(sql, pid, verdict, out.get("reason") or "", out.get("rows_affected"),
                    list(out.get("rows") or [])[: self.max_rows_shown])

    def _say(self, step: Step) -> str:
        if step.verdict == "refused":
            return f"refused before running -- {step.reason}"
        if step.verdict == "aborted":
            return f"it ran and was undone -- {step.reason}"
        if step.verdict == "read":
            return f"{len(step.rows)} row(s): {json.dumps(step.rows, default=str)}"
        if step.verdict == "kept":
            return f"kept; {step.rows_affected} row(s) changed."
        return f"{step.verdict}; {step.reason}".strip("; ")

    def _emit(self, kind: str, task: str, sql: str = "", proposal: int | None = None,
              reason: str = "", rows_affected: int | None = None) -> None:
        ev = Event(kind, self.gate.agent, task, sql, proposal, reason, rows_affected)
        for cb in self.on_event:
            cb(ev)
