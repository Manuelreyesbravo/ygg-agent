# ygg-agent

**Run an LLM agent on PostgreSQL with no MCP. The model writes SQL; the database decides.**

`ygg-agent` is the loop between a language model and a PostgreSQL that runs
[pg_agent_gate](https://github.com/Manuelreyesbravo/pg_agent_gate). The model writes one SQL
statement; the gate verifies it against PostgreSQL itself, runs it under the agent's own
limits, and records it; the verdict goes back to the model; repeat. There is no MCP server,
no tool schema in the model's context, and no JSON between the model and its data: the model
writes SQL, which is what it knows best, over PostgreSQL's own protocol.

```python
from ygg_agent import Agent, OpenAICompatible

agent = Agent(
    OpenAICompatible("http://127.0.0.1:8080/v1"),          # any /v1/chat/completions server
    dsn="postgresql://billing_agent:...@db.internal/app",  # a role registered as an agent
)
result = agent.run("Mark order 4 as cancelled.")
result.outcome        # done | pending | refused | gave_up
for step in result.steps:
    print(step.verdict, step.sql)                           # kept, read, refused, aborted ...
```

## Why no MCP

Measured with the same model (a plain Qwen3-8B, no fine-tuning), the same 13 tasks (7 of them
traps) and a superuser checking what really changed, on a CPU-only CI runner anyone can re-run
([the run](https://github.com/Manuelreyesbravo/agent-gate-demo/actions/runs/41184106078);
[agent-gate-demo/compare](https://github.com/Manuelreyesbravo/agent-gate-demo/tree/master/compare),
one command):

- With the same gate behind both, this loop used **5x fewer prompt tokens** than the model
  reaching the gate through MCP (24k vs 126k for the 13 tasks) and finished each task **13x
  faster** (a median of 14 s against 185 s).
- Through a popular Postgres MCP server, with the same privileges, **4 of 7** traps broke out
  (12 of 21 over three runs). With the database deciding: **none**.

The safety comes from the database deciding, not from the transport -- so the loop that keeps
the model closest to SQL is also the cheapest.

## What it guarantees, and how

- **It only runs behind the gate.** On connect it asks `agent_gate.whoami()`, and refuses to
  work unless the session is an enforced agent session. A superuser, an application role, or
  a role that `SET ROLE`d into the agent's is not behind the gate -- and would run anything --
  so `Agent` raises `NotBehindGate` instead. Connect *as* the registered agent role.
- **It never executes the model's SQL itself.** The statement travels as an argument to
  `agent_gate.propose()`; the database decides whether it runs (`commit`) or only shows its
  effect (`dry_run`).
- **It stops probing.** After `max_refusals` (default 3) refusals or aborts in a row the run
  ends with `outcome="refused"`. Measured: "delete every item" against a limit of 3 rows went
  from eight ever-wider `DELETE`s to three -- 609 prompt tokens instead of 2,954 -- with the
  same result: nothing deleted.
- **The model sees the real catalog.** The relations, columns, keys and privileges come from
  `agent_gate.discover()` -- what this agent may touch, from the live catalog.

## Modes

- `mode="commit"` (default): every statement the gate accepts is committed; reads return rows,
  which go back to the model.
- `mode="dry_run"`: nothing is kept. The first write the gate accepts comes back as
  `result.pending`, with the exact rows it would change, for a person to commit -- or for
  `agent_gate.approve()` and a commit when the agent is registered with `requires_approval`.

## Events

Every proposal, refusal, kept write, read and stop is an `Event` handed to the `on_event`
callbacks -- the place for an audit sink, an approval queue, or a fleet control plane:

```python
agent = Agent(model, dsn=dsn, on_event=[lambda e: print(e.kind, e.proposal, e.reason)])
```

The gate keeps its own append-only record either way (`agent_gate_internal.executions`).

## Models

`OpenAICompatible` speaks `/v1/chat/completions`: llama.cpp, vLLM, Ollama, OpenAI and most
gateways. `user_suffix=" /no_think"` turns off Qwen3's reasoning; `extra_body` passes
server-specific fields. Anything with a `complete(messages) -> Completion` method works;
`ScriptedModel` replays fixed answers for tests.

## Requirements

Python 3.10+, `psycopg` 3, and a PostgreSQL with pg_agent_gate 0.4.0 or later and an agent
registered with `agent_gate.register_agent()`. The quickest way to have one:

```sh
podman run -d -p 5432:5432 -e POSTGRES_PASSWORD=secret ghcr.io/latamflows/ygg-pg:0.3.3
```

## Tests

```sh
YGG_AGENT_TEST_DSN=postgresql://postgres:secret@127.0.0.1:5432/postgres uv run --group dev pytest
```

The tests create their own database and agent, play the model with `ScriptedModel`, and check
every claim from a superuser connection: what was kept, what was refused, that the budget holds
across turns, that a `SET ROLE` session is refused, that each act is in the gate's record.
`examples/live.py` runs a real model against a real database.

## License

Apache-2.0. See LICENSE and NOTICE. "pg_agent_gate" and "ygg" are names, not licensed with
the code (Apache License 2.0, section 6).
