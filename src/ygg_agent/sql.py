# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""Getting ONE statement out of what a model wrote. The gate refuses more than one anyway;
this only strips what models wrap around it: a reasoning block, a code fence, prose."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:sql|postgresql|pgsql)?\s*\n?(.*?)```", re.S | re.I)
_STARTS = re.compile(r"^\s*(select|with|insert|update|delete|merge|values|table|create|alter|drop|"
                     r"truncate|grant|revoke|comment|copy|call|do|lock|set|reset|explain|vacuum|"
                     r"analyze|refresh|reindex|cluster)\b", re.I)


def is_done(text: str) -> bool:
    """The model says the task is finished: a reply that is only DONE (any case, punctuation)."""
    body = _strip_think(text).strip().strip("`*.!").strip()
    return body.upper() == "DONE"


def extract_sql(text: str) -> str:
    """The first SQL statement in a model's reply, without its terminating semicolon; '' if none."""
    body = _strip_think(text).strip()
    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    else:
        # Prose before the statement: start at the first line that opens like SQL.
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if _STARTS.match(line):
                body = "\n".join(lines[i:])
                break
        else:
            return ""
    stmt = _first_statement(body).strip()
    return stmt if _STARTS.match(stmt) else ""


def _strip_think(text: str) -> str:
    # Reasoning models (Qwen3, DeepSeek-R1) may put a <think> block before the answer.
    return text.rsplit("</think>", 1)[1] if "</think>" in text else text


def _first_statement(sql: str) -> str:
    """Up to the first semicolon that is not inside a string, identifier or comment."""
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c in ("'", '"'):
            j = i + 1
            while j < n:
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:   # doubled quote is an escaped quote
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c == "$":
            m = re.match(r"\$[A-Za-z_]*\$", sql[i:])
            if m:
                tag = m.group(0)
                j = sql.find(tag, i + len(tag))
                i = n if j < 0 else j + len(tag)
            else:
                i += 1
        elif c == ";":
            return sql[:i]
        else:
            i += 1
    return sql
