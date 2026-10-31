# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""The loop against a real pg_agent_gate. A scripted model plays the LLM, so every case is
deterministic; every claim about the database is checked from a superuser connection."""
import psycopg
import pytest

from ygg_agent import Agent, Gate, NotBehindGate, ScriptedModel

from .conftest import ROLE, su_value


def agent(world, replies, **kw):
    return Agent(ScriptedModel(replies), dsn=world["agent"], **kw)


def test_refuses_a_session_that_is_not_behind_the_gate(world):
    with pytest.raises(NotBehindGate, match="not behind the gate"):
        Agent(ScriptedModel([]), dsn=world["su"])


def test_refuses_set_role_into_the_agent(world):
    # The hole in a broker that SET ROLEs into an agent: that session is not behind the gate,
    # and the SDK must not run on it.
    with psycopg.connect(world["su"], autocommit=True) as c:
        c.execute(f"set role {ROLE}")
        with pytest.raises(NotBehindGate):
            Gate(c)


def test_a_write_is_kept_and_the_model_is_told(world):
    a = agent(world, ["UPDATE items SET qty = 99 WHERE id = 1", "DONE"])
    r = a.run("Set the quantity of item 1 to 99.")
    assert r.outcome == "done"
    assert [s.verdict for s in r.steps] == ["kept"]
    assert r.steps[0].rows_affected == 1
    assert su_value(world, "select qty from items where id = 1") == 99
    assert "kept; 1 row(s) changed" in a.model.seen[-1][-1]["content"]


def test_the_catalog_reaches_the_model(world):
    a = agent(world, ["DONE"])
    a.run("Nothing.")
    first_user = a.model.seen[0][1]["content"]
    assert "public.items (id integer pk" in first_user
    assert "you may: select, insert, update, delete" in first_user


def test_a_refusal_goes_back_and_the_model_can_change_course(world):
    a = agent(world, ["DROP TABLE items", "UPDATE items SET qty = 0 WHERE id = 2", "DONE"])
    r = a.run("Clear item 2.")
    assert [s.verdict for s in r.steps] == ["refused", "kept"]
    assert "kind_allowed" in r.steps[0].reason
    assert "refused before running" in a.model.seen[1][-1]["content"]
    assert su_value(world, "select count(*) from items") == 10      # the table is still there


def test_over_the_row_limit_is_aborted_and_nothing_changes(world):
    r = agent(world, ["DELETE FROM items", "DONE"]).run("Delete everything.")
    assert [s.verdict for s in r.steps] == ["aborted"]
    assert su_value(world, "select count(*) from items") == 10


def test_the_budget_holds_across_turns(world):
    # max_rows 3 and, unset, a budget of 3 per hour: two slices of 2 cannot both be kept.
    r = agent(world, ["DELETE FROM items WHERE id IN (1, 2)",
                      "DELETE FROM items WHERE id IN (3, 4)", "DONE"]).run("Delete items 1 to 4.")
    assert [s.verdict for s in r.steps] == ["kept", "aborted"]
    assert "write budget" in r.steps[1].reason
    assert su_value(world, "select count(*) from items") == 8


def test_a_read_feeds_rows_back(world):
    a = agent(world, ["SELECT id, qty FROM items WHERE qty > 8 ORDER BY id",
                      "UPDATE items SET qty = 0 WHERE id = 9", "DONE"])
    r = a.run("Zero the first item with more than 8.")
    assert [s.verdict for s in r.steps] == ["read", "kept"]
    assert r.steps[0].rows == [{"id": 9, "qty": 9}, {"id": 10, "qty": 10}]
    assert '"id": 9' in a.model.seen[1][-1]["content"]


def test_dry_run_keeps_nothing_and_returns_the_pending_proposal(world):
    r = agent(world, ["UPDATE items SET qty = 5 WHERE id = 3"], mode="dry_run").run("Set item 3 to 5.")
    assert r.outcome == "pending" and r.pending is not None
    assert r.steps[0].rows_affected == 1
    assert su_value(world, "select qty from items where id = 3") == 3
    with psycopg.connect(world["agent"], autocommit=True) as c:   # a person decides: commit it
        out = c.execute("select agent_gate.commit(%s)", (r.pending,)).fetchone()[0]
    assert out["outcome"] == "kept"
    assert su_value(world, "select qty from items where id = 3") == 5


def test_a_reply_without_sql_is_said_so(world):
    a = agent(world, ["I am not sure.", "DONE"])
    r = a.run("Do something.")
    assert [s.verdict for s in r.steps] == ["no_sql"]
    assert "no SQL statement found" in a.model.seen[1][-1]["content"]


def test_it_gives_up_after_max_turns(world):
    r = agent(world, ["select 1 where false"] * 5, max_turns=3).run("Look around.")
    assert r.outcome == "gave_up" and len(r.steps) == 3


def test_it_stops_after_refusals_in_a_row(world):
    seen = []
    r = agent(world, ["DROP TABLE items", "DELETE FROM items", "TRUNCATE items", "DROP TABLE items"],
              on_event=[seen.append]).run("Get rid of the items.")
    assert r.outcome == "refused"
    assert [s.verdict for s in r.steps] == ["refused", "aborted", "refused"]
    assert seen[-1].kind == "stopped"
    assert su_value(world, "select count(*) from items") == 10


def test_a_kept_step_resets_the_refusal_count(world):
    r = agent(world, ["DROP TABLE items", "DROP TABLE items",
                      "UPDATE items SET qty = 1 WHERE id = 1",
                      "DROP TABLE items", "DROP TABLE items", "DONE"]).run("Mixed.")
    assert r.outcome == "done"
    assert [s.verdict for s in r.steps] == ["refused", "refused", "kept", "refused", "refused"]


def test_events_are_emitted_in_order(world):
    seen = []
    agent(world, ["DROP TABLE items", "UPDATE items SET qty = 1 WHERE id = 1", "DONE"],
          on_event=[seen.append]).run("Set item 1 to 1.")
    assert [e.kind for e in seen] == ["refused", "proposed", "kept", "done"]
    assert all(e.agent == "sdk_test" for e in seen)
    assert seen[2].rows_affected == 1 and seen[2].proposal is not None


def test_every_act_is_in_the_gate_record(world):
    r = agent(world, ["UPDATE items SET qty = 7 WHERE id = 7", "DONE"]).run("Set item 7 to 7.")
    pid = r.steps[0].proposal
    assert su_value(world, f"select outcome from agent_gate_internal.executions "
                           f"where proposal = {pid} and mode = 'commit'") == "kept"
