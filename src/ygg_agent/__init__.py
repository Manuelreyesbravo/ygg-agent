# Copyright 2026 Manuel Reyes Bravo
# SPDX-License-Identifier: Apache-2.0
"""ygg-agent: run an LLM agent on PostgreSQL through pg_agent_gate, with no MCP.

The model writes SQL; the database decides. See Agent."""
from .agent import SYSTEM, Agent, Event, Result, Step
from .gate import Gate, NotBehindGate, describe_catalog
from .model import Completion, Model, ModelError, OpenAICompatible, ScriptedModel
from .sql import extract_sql, is_done

__version__ = "0.1.0"
__all__ = ["Agent", "Event", "Result", "Step", "SYSTEM", "Gate", "NotBehindGate",
           "describe_catalog", "Completion", "Model", "ModelError", "OpenAICompatible",
           "ScriptedModel", "extract_sql", "is_done", "__version__"]
