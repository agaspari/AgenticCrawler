"""
DOM reading and parsing tools for AI Browser agents.

These tools give the agent the ability to read and understand page content.
They combine Playwright's raw HTML access with BeautifulSoup cleaning and
LLM-powered structured extraction.

Tools in this module focus on *reading* the page, not *interacting* with it.
For clicking, typing, and navigation, see browser.py.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel

from ai_browser.utils.html_cleaner import HTMLCleaner
from ai_browser.utils.runtime import runtime

# Shared cleaner instance — strips noise by default
_cleaner = HTMLCleaner(remove_boilerplate=True)
# Cleaner that preserves navigation elements (useful for finding links in navbars)
_cleaner_with_nav = HTMLCleaner(remove_boilerplate=False)


@tool
async def extract_page_text(selector: str | None = None) -> str:
    """
    Extract all visible text from the current page, cleaned of scripts,
    styles, and boilerplate (nav, header, footer).

    Use this when you need to understand the main content of a page.
    The text is cleaned and compressed to save tokens.

    Args:
        selector: Optional CSS selector to narrow extraction to a specific
                  section (e.g., "main", "#content", ".job-listings").
                  If not provided, extracts text from the entire page body.

    Returns:
        The cleaned, readable text content of the page.
    """
    page = runtime.get_page()
    html = await page.content()
    text = _cleaner.clean(html, selector=selector)

    # Truncate if extremely long to avoid blowing up context
    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated, {len(text) - max_chars} chars omitted]"

    return text


@tool
async def extract_links(
    selector: str | None = None,
    include_nav: bool = True,
) -> str:
    """
    Extract all hyperlinks from the current page.

    Use this when you need to discover what pages are linked from the
    current page — for example, finding a 'Careers' link in the navigation,
    or finding individual job posting links on a listings page.

    Args:
        selector: Optional CSS selector to limit link extraction to a
                  specific section (e.g., "nav", "main", ".sidebar").
        include_nav: If True (default), includes links from nav/header/footer.
                     Set to False to get only main content links.

    Returns:
        A formatted list of links with their text and URLs.
    """
    page = runtime.get_page()
    html = await page.content()
    base_url = page.url

    cleaner = _cleaner_with_nav if include_nav else _cleaner
    links = cleaner.extract_links(html, base_url=base_url, selector=selector)

    if not links:
        return "No links found on the page" + (f" matching selector '{selector}'" if selector else "") + "."

    # Format as a readable list
    lines = [f"Found {len(links)} links:\n"]
    for i, link in enumerate(links, 1):
        text = link["text"][:80] if link["text"] else "(no text)"
        href = link["href"]
        lines.append(f"  {i}. [{text}] -> {href}")

    return "\n".join(lines)


@tool
async def extract_elements_by_selector(
    selector: str,
    attributes: list[str] | None = None,
) -> str:
    """
    Extract elements matching a CSS selector, with optional attributes.

    Use this when you need to find specific elements on the page, like
    job cards, list items, or data entries. You can request specific
    HTML attributes (href, class, data-*, etc.) for each match.

    Args:
        selector: CSS selector to match (e.g., ".job-card", "li.posting").
                  NOTE: This uses BeautifulSoup, so Playwright-specific
                  pseudo-classes like ":has-text()" are NOT supported. Use
                  standard CSS selectors only.
        attributes: Optional list of HTML attributes to extract from each
                    element (e.g., ["href", "class", "data-job-id"]).
                    Text content is always included.

    Returns:
        A formatted list of matching elements with their text and attributes.
    """
    page = runtime.get_page()
    html = await page.content()
    try:
        elements = _cleaner.extract_elements(html, selector, attributes=attributes)
    except Exception as e:
        return f"Error extracting elements: invalid CSS selector '{selector}'. Exception: {e}"

    if not elements:
        return f"No elements found matching selector '{selector}'."

    lines = [f"Found {len(elements)} elements matching '{selector}':\n"]
    for i, el in enumerate(elements, 1):
        parts = [f"  {i}. text: \"{el.get('text', '')[:100]}\""]
        for key, val in el.items():
            if key != "text" and val:
                parts.append(f"     {key}: \"{val}\"")
        lines.append("\n".join(parts))

    return "\n".join(lines)


@tool
async def get_page_html(selector: str | None = None) -> str:
    """
    Get the raw HTML of the current page or a specific section.

    Use this sparingly — it returns raw HTML which uses many tokens.
    Prefer extract_page_text or extract_elements_by_selector for most tasks.
    This tool is useful when you need to see the exact HTML structure
    to construct better CSS selectors.

    Args:
        selector: Optional CSS selector to get HTML of a specific section.
                  If not provided, returns the full page HTML (truncated).

    Returns:
        The raw HTML string (truncated to 10,000 characters if needed).
    """
    page = runtime.get_page()

    if selector:
        element = await page.query_selector(selector)
        if element:
            html = await element.inner_html()
        else:
            return f"No element found matching selector '{selector}'."
    else:
        html = await page.content()

    max_chars = 10000
    if len(html) > max_chars:
        html = html[:max_chars] + f"\n\n<!-- ... truncated, {len(html) - max_chars} chars omitted -->"

    return html


@tool
async def extract_structured_data(
    data_description: str,
    selector: str | None = None,
) -> str:
    """
    Use an LLM to extract structured data from the current page content.

    This is the most powerful extraction tool. It reads the page content,
    sends it to the LLM with your description of what to extract, and
    returns structured JSON data.

    Use this when you need to pull specific information from a page
    (e.g., job titles, prices, names) and the page structure is complex
    or inconsistent.

    Args:
        data_description: A clear description of what data to extract.
            Be specific about the fields you want. Examples:
            - "Extract all job postings with title, location, and URL"
            - "Extract the company name and main contact email"
            - "Extract product names and prices from the listing"
        selector: Optional CSS selector to narrow the page section to extract from.

    Returns:
        A JSON-formatted string with the extracted structured data.
    """
    from ai_browser.config import get_llm

    page = runtime.get_page()
    html = await page.content()
    text = _cleaner.clean(html, selector=selector)

    # Truncate if needed
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars]

    llm = get_llm()
    prompt = (
        f"Extract the following data from this web page content:\n\n"
        f"**What to extract:** {data_description}\n\n"
        f"**Page URL:** {page.url}\n\n"
        f"**Page content:**\n{text}\n\n"
        f"Return the extracted data as a JSON object or array. "
        f"If no matching data is found, return an empty array []."
    )

    response = await llm.ainvoke(prompt)
    return response.content


@tool
async def get_page_title() -> str:
    """
    Get the title of the current page.

    Quick way to understand what page you're on without extracting
    all the text.

    Returns:
        The page title.
    """
    page = runtime.get_page()
    html = await page.content()
    title = _cleaner.get_page_title(html)
    return f"Page title: {title}" if title else "Page has no title."


# ── Tool Collection ───────────────────────────────────────────────────────────
# Convenience list for registering all DOM tools with an agent.

DOM_TOOLS = [
    extract_page_text,
    extract_links,
    extract_elements_by_selector,
    get_page_html,
    extract_structured_data,
    get_page_title,
]
