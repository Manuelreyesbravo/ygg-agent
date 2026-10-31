# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""The database side: pg_agent_gate's verbs, over PostgreSQL's own protocol.

This module runs nothing but the gate's verbs. It never sends the model's SQL as a
statement of its own: the SQL travels as an argument to agent_gate.propose(), and the
database decides whether it runs. And it refuses to work at all unless the session is
behind the gate -- a connection that is not the agent's own (a superuser, an application
role, or a role that SET ROLE'd into the agent's) would run whatever it was given.
"""
from __future__ import annotations

import psycopg


class NotBehindGate(RuntimeError):
    """The connection is not an agent session that pg_agent_gate enforces."""


class Gate:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn
        self.whoami = self._call("select agent_gate.whoami()")
        if not (self.whoami.get("is_agent") and self.whoami.get("enforced")):
            raise NotBehindGate(
                f"this session is not behind the gate (role {self.whoami.get('role')!r}, "
                f"is_agent={self.whoami.get('is_agent')}, enforced={self.whoami.get('enforced')}). "
                "Connect AS the registered agent role -- not as another role, and not with "
                "SET ROLE, which does not put a session behind the gate.")
        self.agent = self.whoami.get("agent")

    @classmethod
    def connect(cls, dsn: str, **kwargs) -> "Gate":
        try:
            conn = psycopg.connect(dsn, autocommit=True, **kwargs)
        except psycopg.Error as e:
            raise NotBehindGate(f"could not connect: {e}") from e
        try:
            return cls(conn)
        except (psycopg.Error, NotBehindGate) as e:
            conn.close()
            if isinstance(e, NotBehindGate):
                raise
            raise NotBehindGate(f"pg_agent_gate is not reachable here: {e}") from e

    def discover(self, filter: str | None = None, limit: int = 50) -> dict:
        return self._call("select agent_gate.discover(%s, %s)", (filter, limit))

    def propose(self, sql: str, intent: str) -> dict:
        return self._call("select agent_gate.propose(%s, %s)", (sql, intent))

    def dry_run(self, proposal: int) -> dict:
        return self._call("select agent_gate.dry_run(%s)", (proposal,))

    def commit(self, proposal: int) -> dict:
        return self._call("select agent_gate.commit(%s)", (proposal,))

    def close(self) -> None:
        self.conn.close()

    def _call(self, query: str, args: tuple = ()) -> dict:
        row = self.conn.execute(query, args).fetchone()
        return row[0] if row and isinstance(row[0], dict) else {}


def first_failure(result: dict) -> str:
    for c in result.get("checks") or []:
        if not c.get("passed", True):
            return f"{c.get('check')}: {c.get('detail')}"
    return result.get("error") or "refused"


def describe_catalog(catalog: dict) -> str:
    """discover()'s answer as compact lines a model reads well: one relation per line."""
    lines = []
    for r in catalog.get("relations") or []:
        cols = []
        pk = set(r.get("primary_key") or [])
        for c in r.get("columns") or []:
            bits = [c["name"], c["type"]]
            if c["name"] in pk:
                bits.append("pk")
            if c.get("not_null") and c["name"] not in pk:
                bits.append("not null")
            if c.get("default"):
                bits.append(f"default {c['default']}")
            cols.append(" ".join(bits))
        line = f"{r['relation']} ({', '.join(cols)})"
        if r.get("privileges"):
            line += f" -- you may: {', '.join(r['privileges'])}"
        for fk in r.get("foreign_keys") or []:
            line += f"; {fk.get('definition')}"
        lines.append(line)
    return "\n".join(lines)
