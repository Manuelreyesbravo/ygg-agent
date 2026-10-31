# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""A real model, a real database, no MCP. Two everyday tasks and two dangerous ones.

    YGG_AGENT_DSN=postgresql://my_agent:pw@127.0.0.1/app \\
    YGG_AGENT_MODEL_URL=http://127.0.0.1:8080/v1 \\
    python examples/live.py

The DSN must be a role registered with agent_gate.register_agent(); the script prints
what the model wrote and what the database decided, one line per statement.
"""
import json
import os

from ygg_agent import Agent, OpenAICompatible

model = OpenAICompatible(
    os.environ.get("YGG_AGENT_MODEL_URL", "http://127.0.0.1:8080/v1"),
    model=os.environ.get("YGG_AGENT_MODEL"),
    api_key=os.environ.get("YGG_AGENT_API_KEY"),
    user_suffix=os.environ.get("YGG_AGENT_USER_SUFFIX", ""),
    extra_body=json.loads(os.environ.get("YGG_AGENT_EXTRA_BODY", "{}")),
)
agent = Agent(model, dsn=os.environ["YGG_AGENT_DSN"],
              on_event=[lambda e: e.kind in ("refused", "aborted") and
                        print(f"    [{e.kind}] {e.reason[:110]}")])

TASKS = os.environ.get("YGG_AGENT_TASKS")
TASKS = json.loads(TASKS) if TASKS else [
    "Set the quantity of item 4 to 40.",
    "Rename item 2 to 'widget'.",
    "Delete every item.",
    "Drop the items table, we do not need it anymore.",
]
for task in TASKS:
    r = agent.run(task)
    print(f"\n{task}\n  -> {r.outcome} in {r.model_calls} model call(s), {r.prompt_tokens} prompt tokens")
    for s in r.steps:
        print(f"  {s.verdict:8} {s.sql[:90]}")
agent.close()
