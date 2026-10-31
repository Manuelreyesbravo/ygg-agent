# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""A throwaway database with pg_agent_gate, an agent role, and a superuser to check from.

YGG_AGENT_TEST_DSN: a SUPERUSER connection URL to a PostgreSQL with pg_agent_gate preloaded,
e.g. postgresql://postgres:pw@127.0.0.1:5432/postgres (the ygg-pg image works). The tests
create and drop their own database and role.
"""
import os
import secrets
import urllib.parse

import psycopg
import pytest

DB = "ygg_agent_test"
ROLE = "ygg_agent_test_role"
AGENT = "sdk_test"


def _with(dsn: str, **parts) -> str:
    u = urllib.parse.urlsplit(dsn)
    user = parts.get("user", u.username or "")
    pw = parts.get("password", u.password)
    host = u.hostname + (f":{u.port}" if u.port else "")
    auth = user + (f":{urllib.parse.quote(pw)}" if pw else "")
    path = "/" + parts.get("db", u.path.lstrip("/"))
    return urllib.parse.urlunsplit((u.scheme, f"{auth}@{host}", path, u.query, ""))


@pytest.fixture(scope="session")
def admin_dsn():
    dsn = os.environ.get("YGG_AGENT_TEST_DSN")
    if not dsn:
        pytest.skip("set YGG_AGENT_TEST_DSN to a superuser URL of a PostgreSQL with pg_agent_gate")
    return dsn


@pytest.fixture(scope="session")
def db(admin_dsn):
    pw = secrets.token_hex(12)
    with psycopg.connect(admin_dsn, autocommit=True) as c:
        c.execute(f"drop database if exists {DB}")
        c.execute(f"drop role if exists {ROLE}")
        c.execute(f"create database {DB}")
        c.execute(f"create role {ROLE} login password '{pw}'")
    su_dsn = _with(admin_dsn, db=DB)
    with psycopg.connect(su_dsn, autocommit=True) as c:
        c.execute("create extension pg_agent_gate")
        c.execute(f"grant usage on schema public to {ROLE}")
        c.execute(f"select agent_gate.register_agent('{AGENT}', '{ROLE}', 'the ygg-agent test agent', "
                  "p_max_rows => 3)")
    yield {"su": su_dsn, "agent": _with(admin_dsn, db=DB, user=ROLE, password=pw)}
    with psycopg.connect(admin_dsn, autocommit=True) as c:
        c.execute(f"drop database if exists {DB} with (force)")
        c.execute(f"drop role if exists {ROLE}")


@pytest.fixture
def world(db):
    """Ten rows in public.items, and the agent's whole write budget back."""
    with psycopg.connect(db["su"], autocommit=True) as c:
        c.execute("drop table if exists items")
        c.execute("create table items (id int primary key, name text, qty int not null default 0)")
        c.execute("insert into items select g, 'item ' || g, g from generate_series(1, 10) g")
        c.execute(f"grant select, insert, update, delete on items to {ROLE}")
        if c.execute("select 1 from pg_proc where proname = 'reset_budget'").fetchone():
            c.execute("select agent_gate.reset_budget(%s)", (AGENT,))
    return db


def su_value(db, query):
    with psycopg.connect(db["su"], autocommit=True) as c:
        return c.execute(query).fetchone()[0]
