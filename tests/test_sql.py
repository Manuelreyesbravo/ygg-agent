# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
from ygg_agent import extract_sql, is_done


def test_plain_statement():
    assert extract_sql("UPDATE t SET a = 1 WHERE id = 2;") == "UPDATE t SET a = 1 WHERE id = 2"


def test_fence_and_prose():
    reply = "Sure, here it is:\n```sql\nselect * from t;\n```\nThis reads t."
    assert extract_sql(reply) == "select * from t"


def test_prose_then_sql_without_fence():
    assert extract_sql("To do that:\nDELETE FROM t WHERE id = 1;\nDone.") == "DELETE FROM t WHERE id = 1"


def test_think_block_is_dropped():
    assert extract_sql("<think>DROP TABLE t;</think>\nselect 1;") == "select 1"


def test_only_the_first_statement():
    assert extract_sql("select 1; drop table t;") == "select 1"


def test_semicolon_inside_a_string_or_comment_does_not_cut():
    assert extract_sql("update t set s = 'a;b' -- x;y\n where id = 1; select 2") == \
        "update t set s = 'a;b' -- x;y\n where id = 1"
    assert extract_sql("select $$a;b$$, 'it''s;' from t;") == "select $$a;b$$, 'it''s;' from t"


def test_no_sql():
    assert extract_sql("I cannot help with that.") == ""
    assert extract_sql("") == ""


def test_done():
    assert is_done("DONE")
    assert is_done(" done. ")
    assert is_done("<think>finished</think>\n**DONE**")
    assert not is_done("DONE with the update: UPDATE t SET a = 1")
    assert not is_done("select 1")


def test_a_think_block_that_never_closes():
    # Seen with a Qwen3 GGUF run in process under /no_think: the block opens, and the answer
    # is inside it. A server's reasoning parser hides this; LlamaCpp does not.
    from ygg_agent.sql import extract_sql, is_done
    assert is_done("<think>\n\nDONE")
    assert is_done("<think>\n\n**DONE.**")
    assert extract_sql("<think>\n\nUPDATE t SET a = 1 WHERE id = 4;") == "UPDATE t SET a = 1 WHERE id = 4"
    assert extract_sql("<think>\n```sql\nSELECT 1;\n```") == "SELECT 1"
    # The start of a chain of thought is not an answer.
    assert not is_done("<think>\nLet me check whether the task is DONE")
    assert extract_sql("<think>\nThe user wants every row gone, so I would") == ""
