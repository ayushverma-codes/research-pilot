"""
Shared agent state.

This is the single structured object that flows through every node of the
graph (Planner -> Researcher -> Reporter, and later Critic). Every node reads
from it and returns a partial update to it. Keeping it as one Pydantic model
makes the whole pipeline explainable and easy to inspect/log/test.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single piece of evidence gathered by a tool."""
    url: str
    title: str = ""
    snippet: str = ""


class ToolCallRecord(BaseModel):
    """Record of one tool invocation, for logging / demo / debugging."""
    tool: str
    input: str
    success: bool
    summary: str = ""


class AgentState(BaseModel):
    # --- input ---
    user_goal: str

    # --- planning ---
    plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    completed_steps: List[str] = Field(default_factory=list)

    # --- research results ---
    findings: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)

    # --- bookkeeping ---
    tool_history: List[ToolCallRecord] = Field(default_factory=list)
    critique: Optional[Dict[str, Any]] = None
    iteration: int = 0

    # --- output ---
    final_report: str = ""
