"""
Navigator subagent for AI Browser.

The Navigator's job is to explore a website and discover pages relevant
to the user's task. It starts at a given URL, examines navigation menus
and links, follows promising paths, and returns a list of discovered URLs.

This is a ReAct agent with browser navigation and DOM reading tools —
it can click, scroll, and follow links to explore the site.

Usage:
    from ai_browser.agents.navigator import run_navigator

    urls = await run_navigator(
        url="https://company.com",
        task="Find pages containing job listings",
    )
    # urls = ["https://company.com/careers", "https://company.com/careers/engineering"]
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ai_browser.config import get_llm
from ai_browser.tools.browser import BROWSER_TOOLS
from ai_browser.tools.dom import (
    extract_page_text,
    extract_links,
    extract_elements_by_selector,
    get_page_title,
)
from ai_browser.tools.utility import log_observation, plan_next_steps

# Navigator gets browser + a subset of DOM tools (no structured extraction — that's the Parser's job)
NAVIGATOR_TOOLS = BROWSER_TOOLS + [
    extract_page_text,
    extract_links,
    extract_elements_by_selector,
    get_page_title,
    log_observation,
    plan_next_steps,
]


NAVIGATOR_PROMPT = """You are a Navigator agent. Your sole purpose is to explore \
a website and discover pages relevant to the user's task.

## Your Mission
Starting from the given URL, find all pages that are relevant to the task.
You are NOT responsible for extracting data — a separate Parser agent will do that.
Your job is purely to DISCOVER and RETURN relevant page URLs.

## Strategy
1. Start at the target URL.
2. Extract all links from the page, paying close attention to navigation menus.
3. Identify links that look relevant to the task (e.g., "Careers", "Jobs", "Open Positions").
4. Navigate to promising pages to verify they contain relevant content.
5. Check for sub-pages, pagination, or category filters.
6. Log observations as you explore.

## Output Format
When you've finished exploring, your FINAL message must contain a JSON block
with the discovered URLs. Format it exactly like this:

```json
{"discovered_urls": ["https://example.com/page1", "https://example.com/page2"]}
```

Include ONLY URLs that are directly relevant to the task. Do not include
the starting URL unless it itself contains relevant content.

## Rules
- Be thorough but efficient. Don't visit every page — use link text and URL
  patterns to identify relevant pages.
- If the site has pagination, note the pattern but don't click through every page.
  Instead, include the first page URL and note the pagination pattern.
- Maximum exploration depth: 3 clicks from the starting page.
- Always return at least 1 URL if possible — even if it's just the starting URL
  with relevant content.
"""


def create_navigator(model: str | None = None):
    """Create a Navigator ReAct agent."""
    llm = get_llm(model=model)
    return create_react_agent(
        model=llm,
        tools=NAVIGATOR_TOOLS,
        prompt=NAVIGATOR_PROMPT,
    )


async def run_navigator(
    url: str,
    task: str,
    model: str | None = None,
) -> list[str]:
    """
    Run the Navigator to explore a site and discover relevant URLs.

    Args:
        url: The starting URL to explore.
        task: Description of what pages to look for.
        model: Optional LLM model override.

    Returns:
        List of discovered URLs relevant to the task.
    """
    navigator = create_navigator(model=model)

    user_message = (
        f"Starting URL: {url}\n\n"
        f"Task: {task}\n\n"
        f"Explore the site starting from this URL and find all pages "
        f"relevant to the task. Return the discovered URLs in your final message."
    )

    print("\n[Navigator] Starting site exploration...")
    print(f"[Navigator] Target: {url}")
    print(f"[Navigator] Looking for: {task}")

    result = await navigator.ainvoke(
        {"messages": [HumanMessage(content=user_message)]},
    )

    # Extract URLs from the navigator's final message
    discovered_urls = _parse_urls_from_response(result)

    print(f"[Navigator] Discovered {len(discovered_urls)} relevant URL(s):")
    for u in discovered_urls:
        print(f"  - {u}")

    return discovered_urls


def _parse_urls_from_response(result: dict) -> list[str]:
    """
    Extract discovered URLs from the navigator's response.

    Tries to parse a JSON block first, then falls back to extracting
    URLs from the message text.
    """
    if not result.get("messages"):
        return []

    # Get the last AI message
    last_message = result["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Try parsing JSON block
    json_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict) and "discovered_urls" in data:
                return data["discovered_urls"]
        except json.JSONDecodeError:
            pass

    # Try parsing inline JSON
    json_match = re.search(r'\{[^{}]*"discovered_urls"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data.get("discovered_urls", [])
        except json.JSONDecodeError:
            pass

    # Fallback: extract any URLs from the text
    url_pattern = r'https?://[^\s\'"<>\])]+'
    urls = re.findall(url_pattern, content)
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        # Clean trailing punctuation
        u = u.rstrip(".,;:!?)")
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    return unique_urls
