"""
Parser subagent for AI Browser.

The Parser's job is to visit a single page and extract structured data.
Each Parser gets its own isolated browser context, so multiple parsers
can run in parallel without interference.

Unlike the Navigator (which uses the ReAct tool loop), the Parser operates
directly with Playwright and the LLM for efficiency — it doesn't need to
"explore" or make navigation decisions.

Usage:
    from ai_browser.agents.parser import parse_page

    results = await parse_page(
        url="https://company.com/careers",
        task="Extract all job postings with title, location, and URL",
    )
    # results = [{"title": "Engineer", "location": "Remote", "url": "..."}, ...]
"""

from __future__ import annotations

import json
from typing import Any

from ai_browser.config import get_llm
from ai_browser.utils.html_cleaner import HTMLCleaner
from ai_browser.utils.runtime import runtime

_cleaner = HTMLCleaner(remove_boilerplate=True)


async def parse_page(
    url: str,
    task: str,
    extraction_schema: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Parse a single page and extract structured data.

    Creates an isolated browser context, navigates to the URL, cleans
    the HTML, and uses the LLM to extract structured data.

    Args:
        url: The page URL to parse.
        task: Description of what data to extract (e.g., "Extract all
              job postings with title, location, department, and URL").
        extraction_schema: Optional JSON schema string describing the
                          expected output format.
        model: Optional LLM model override.

    Returns:
        List of extracted data records (dicts).
    """
    if not runtime.browser_manager:
        raise RuntimeError("Runtime not initialized.")

    print(f"  [Parser] Parsing: {url}")

    # Create an isolated browser context for this parser
    async with runtime.browser_manager.acquire_context() as ctx:
        page = await ctx.new_page()

        try:
            # Navigate to the page
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            status = response.status if response else "unknown"
            print(f"  [Parser] Loaded ({status}): {url}")

            # Wait a moment for any dynamic content
            await page.wait_for_timeout(2000)

            # Get and clean the HTML
            html = await page.content()
            text = _cleaner.clean(html)

            # Also extract links for context (job pages often have links to detail pages)
            links = _cleaner.extract_links(html, base_url=url)
            link_context = ""
            if links:
                link_lines = [f"- [{l['text'][:60]}]({l['href']})" for l in links[:30]]
                link_context = f"\n\nLinks found on the page:\n" + "\n".join(link_lines)

            # Truncate text if needed
            max_chars = 12000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[... content truncated]"

            # Build the extraction prompt
            schema_instruction = ""
            if extraction_schema:
                schema_instruction = (
                    f"\n\nExtract data matching this schema:\n{extraction_schema}"
                )

            prompt = (
                f"You are a data extraction specialist. Extract structured data "
                f"from the following web page content.\n\n"
                f"**Extraction task:** {task}{schema_instruction}\n\n"
                f"**Page URL:** {url}\n\n"
                f"**Page content:**\n{text}"
                f"{link_context}\n\n"
                f"**Instructions:**\n"
                f"- Return a JSON array of objects, where each object represents "
                f"one extracted record.\n"
                f"- Include all relevant fields you can find.\n"
                f"- If detail URLs exist for each record, include them.\n"
                f"- If no matching data is found, return an empty array: []\n"
                f"- Return ONLY the JSON array, no other text.\n"
            )

            llm = get_llm(model=model)
            response = await llm.ainvoke(prompt)
            content = response.content

            # Parse the JSON response
            results = _parse_json_response(content)
            print(f"  [Parser] Extracted {len(results)} record(s) from {url}")
            return results

        except Exception as e:
            print(f"  [Parser] Error parsing {url}: {e}")
            return [{"error": str(e), "url": url}]


def _parse_json_response(content: str) -> list[dict[str, Any]]:
    """
    Parse a JSON array from the LLM's response.

    Handles various response formats:
    - Raw JSON array
    - JSON wrapped in ```json blocks
    - JSON object with a data key containing an array
    """
    import re

    # Try to extract from code block first
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)

    content = content.strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Look for an array value in the dict
            for key in ["data", "results", "jobs", "items", "records", "listings"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If it's a single record, wrap it
            return [data]
        return []
    except json.JSONDecodeError:
        # Try to find any JSON array in the response
        array_match = re.search(r'\[.*\]', content, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        print(f"  [Parser] Warning: Could not parse JSON from LLM response")
        return [{"raw_response": content}]
