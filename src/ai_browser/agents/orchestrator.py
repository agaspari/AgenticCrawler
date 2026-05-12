"""
Orchestrator — multi-agent supervisor graph for AI Browser.

This is the top-level graph that coordinates the Navigator and Parser
subagents using LangGraph's Send API for parallel execution.

Flow:
    START -> navigate_site -> route_parsers -> [parse_page x N] -> aggregate_results -> save_results -> END

The Send API enables dynamic fan-out: the orchestrator discovers N relevant
URLs via the Navigator, then dispatches N Parser instances in parallel
(bounded by MAX_CONCURRENCY).

Usage:
    from ai_browser.agents.orchestrator import run_multi_agent

    result = await run_multi_agent(
        url="https://company.com",
        task="Find all job listings and extract them",
    )
"""

from __future__ import annotations

import json
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ai_browser.agents.navigator import run_navigator
from ai_browser.agents.parser import parse_page
from ai_browser.config import OUTPUT_DIR, get_llm
from ai_browser.utils.runtime import runtime


# ── Orchestrator State ────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    """
    State for the multi-agent orchestrator graph.

    Uses operator.add reducers on list fields so that parallel parser
    branches can safely merge their results.
    """
    target_url: str
    task_description: str
    discovered_urls: list[str]
    # Each parser appends to this via the reducer
    parsed_results: Annotated[list[dict[str, Any]], operator.add]
    output_path: str
    status: str


# ── State for individual parse_page fan-out ───────────────────────────────────

class ParsePageInput(TypedDict):
    """Input state for a single parse_page invocation via Send."""
    target_url: str
    task_description: str
    discovered_urls: list[str]
    parsed_results: Annotated[list[dict[str, Any]], operator.add]
    output_path: str
    status: str
    # The specific URL this parser instance should process
    page_url: str


# ── Graph Nodes ───────────────────────────────────────────────────────────────

async def navigate_site(state: OrchestratorState) -> dict:
    """
    Node: Run the Navigator subagent to explore the site and discover URLs.
    """
    print("\n" + "=" * 50)
    print("PHASE: Navigation")
    print("=" * 50)

    urls = await run_navigator(
        url=state["target_url"],
        task=state["task_description"],
    )

    if not urls:
        print("[Orchestrator] Navigator found no relevant URLs.")
        # Fall back to using the target URL itself
        urls = [state["target_url"]]

    return {
        "discovered_urls": urls,
        "status": f"discovered_{len(urls)}_urls",
    }


def route_parsers(state: OrchestratorState) -> list[Send]:
    """
    Conditional edge: Fan out Parser instances via Send API.

    Creates one Send per discovered URL. Each Send dispatches to the
    'parse_page_node' with the URL to process.
    """
    urls = state.get("discovered_urls", [])

    if not urls:
        # No URLs to parse — go directly to save
        return [Send("aggregate_results", state)]

    print(f"\n{'=' * 50}")
    print(f"PHASE: Parsing ({len(urls)} page(s) in parallel)")
    print("=" * 50)

    return [
        Send("parse_page_node", {**state, "page_url": url})
        for url in urls
    ]


async def parse_page_node(state: ParsePageInput) -> dict:
    """
    Node: Parse a single page and extract structured data.

    Each instance runs in its own browser context (managed by the
    Parser's acquire_context). Multiple instances run in parallel,
    bounded by MAX_CONCURRENCY.
    """
    url = state["page_url"]
    task = state["task_description"]

    results = await parse_page(url=url, task=task)

    # Tag each result with its source URL
    for r in results:
        if isinstance(r, dict):
            r["_source_url"] = url

    return {"parsed_results": results}


async def aggregate_results(state: OrchestratorState) -> dict:
    """
    Node: Aggregate all parsed results and report summary.
    """
    results = state.get("parsed_results", [])

    print(f"\n{'=' * 50}")
    print(f"PHASE: Aggregation")
    print(f"{'=' * 50}")
    print(f"[Orchestrator] Total records extracted: {len(results)}")

    # Report errors
    errors = [r for r in results if isinstance(r, dict) and "error" in r]
    if errors:
        print(f"[Orchestrator] {len(errors)} page(s) had errors:")
        for e in errors:
            print(f"  - {e.get('url', 'unknown')}: {e.get('error', 'unknown error')}")

    clean_count = len([r for r in results if isinstance(r, dict) and "error" not in r])
    return {
        "status": f"aggregated_{clean_count}_records",
    }


async def save_results(state: OrchestratorState) -> dict:
    """
    Node: Save the aggregated results to JSON and CSV files.
    """
    results = state.get("parsed_results", [])

    print(f"\n{'=' * 50}")
    print(f"PHASE: Saving Results")
    print("=" * 50)

    if not results:
        print("[Orchestrator] No results to save.")
        return {"status": "complete_no_results", "output_path": ""}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = OUTPUT_DIR / f"results_{timestamp}.json"
    output = {
        "metadata": {
            "target_url": state["target_url"],
            "task": state["task_description"],
            "urls_parsed": state.get("discovered_urls", []),
            "total_records": len(results),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "data": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"[Orchestrator] Saved {len(results)} record(s) to: {json_path}")

    # Also save CSV if results are flat dicts
    csv_path = ""
    try:
        import csv

        csv_path_obj = OUTPUT_DIR / f"results_{timestamp}.csv"
        # Collect all unique keys
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in results:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)

        with open(csv_path_obj, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                if isinstance(r, dict):
                    # Flatten any nested values to strings
                    flat = {k: str(v) if not isinstance(v, str) else v for k, v in r.items()}
                    writer.writerow(flat)

        csv_path = str(csv_path_obj)
        print(f"[Orchestrator] Saved CSV to: {csv_path_obj}")
    except Exception as e:
        print(f"[Orchestrator] CSV save skipped: {e}")

    return {
        "status": "complete",
        "output_path": str(json_path),
    }


# ── Graph Assembly ────────────────────────────────────────────────────────────

def create_orchestrator():
    """
    Build and compile the multi-agent orchestrator graph.

    Graph structure:
        START -> navigate_site -> route_parsers -> [parse_page_node x N] -> aggregate_results -> save_results -> END
    """
    builder = StateGraph(OrchestratorState)

    # Add nodes
    builder.add_node("navigate_site", navigate_site)
    builder.add_node("parse_page_node", parse_page_node)
    builder.add_node("aggregate_results", aggregate_results)
    builder.add_node("save_results", save_results)

    # Define edges
    builder.add_edge(START, "navigate_site")
    builder.add_conditional_edges("navigate_site", route_parsers)
    builder.add_edge("parse_page_node", "aggregate_results")
    builder.add_edge("aggregate_results", "save_results")
    builder.add_edge("save_results", END)

    return builder.compile()


# ── Top-Level Runner ──────────────────────────────────────────────────────────

async def run_multi_agent(
    url: str,
    task: str,
    model: str | None = None,
    headless: bool | None = None,
) -> dict:
    """
    Run the full multi-agent pipeline end-to-end.

    1. Initialize browser
    2. Navigator explores the site
    3. Parsers extract data from discovered pages (in parallel)
    4. Results are aggregated and saved
    5. Browser is cleaned up

    Args:
        url: The starting URL.
        task: What to extract (natural language).
        model: Optional LLM model override.
        headless: Optional headless mode override.

    Returns:
        The final orchestrator state.
    """
    await runtime.initialize(headless=headless)

    try:
        graph = create_orchestrator()

        initial_state: OrchestratorState = {
            "target_url": url,
            "task_description": task,
            "discovered_urls": [],
            "parsed_results": [],
            "output_path": "",
            "status": "started",
        }

        print("\n" + "=" * 60)
        print("AI Browser — Multi-Agent Pipeline")
        print("=" * 60)
        print(f"URL:  {url}")
        print(f"Task: {task}")
        print("=" * 60)

        result = await graph.ainvoke(initial_state)

        print(f"\n{'=' * 60}")
        print("Pipeline Complete!")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Output: {result.get('output_path', 'none')}")
        print(f"Records: {len(result.get('parsed_results', []))}")
        print("=" * 60)

        return result

    finally:
        await runtime.shutdown()
