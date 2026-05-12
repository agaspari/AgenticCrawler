"""
State schemas for AI Browser agents.

Defines the typed state structures used by the orchestrator, navigator,
and parser agents. Uses Pydantic-style TypedDict with LangGraph reducers
for safe concurrent state updates.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# ── Global Orchestrator State ─────────────────────────────────────────────────

class GlobalState(TypedDict):
    """
    Top-level state shared across the entire graph.

    Attributes:
        messages: Conversation history (managed by LangGraph's add_messages reducer).
        target_url: The root URL provided by the user to start crawling.
        task_description: What the user wants the agent to accomplish.
        discovered_urls: URLs found during navigation that are relevant to the task.
        parsed_results: Structured data extracted by parser subagents.
                        Uses operator.add reducer for safe parallel merging.
        output_path: Where the final results were saved.
        status: Current status of the overall workflow.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    target_url: str
    task_description: str
    discovered_urls: Annotated[list[str], operator.add]
    parsed_results: Annotated[list[dict[str, Any]], operator.add]
    output_path: str
    status: str


# ── Navigator Subagent State ──────────────────────────────────────────────────

class NavigatorState(TypedDict):
    """
    State for the Navigator subagent, which explores a site to find
    relevant pages.

    Attributes:
        messages: Conversation history for this subagent.
        target_url: The starting URL to explore.
        task_description: What pages the navigator should look for.
        discovered_urls: URLs found that match the task criteria.
        current_url: The page the browser is currently on.
        page_summary: LLM-generated summary of the current page.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    target_url: str
    task_description: str
    discovered_urls: Annotated[list[str], operator.add]
    current_url: str
    page_summary: str


# ── Parser Subagent State ─────────────────────────────────────────────────────

class ParserState(TypedDict):
    """
    State for a Parser subagent, which extracts structured data from
    a specific page.

    Attributes:
        messages: Conversation history for this subagent.
        page_url: The specific URL this parser should extract data from.
        task_description: What data to extract from the page.
        extracted_data: The structured data extracted from the page.
        raw_content: The cleaned text/HTML content of the page.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    page_url: str
    task_description: str
    extracted_data: list[dict[str, Any]]
    raw_content: str


# ── Send Payload ──────────────────────────────────────────────────────────────

@dataclass
class ParserTaskPayload:
    """
    Payload sent via LangGraph's Send API to fan out parser subagents.

    This is the input each parser subagent receives when the orchestrator
    dispatches parallel parsing tasks.
    """
    page_url: str
    task_description: str
    extraction_schema: str | None = None  # Optional: Pydantic schema name
