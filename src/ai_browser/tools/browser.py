"""
Browser navigation tools for AI Browser agents.

These tools give the agent the ability to control a headless (or headed)
browser — navigating to URLs, clicking elements, typing, scrolling, and
taking screenshots. Each tool operates on the current active page from
the shared RuntimeContext.

All tools are async and decorated with @tool for LangChain compatibility.
"""

from __future__ import annotations

import base64
from langchain_core.tools import tool

from ai_browser.utils.runtime import runtime


@tool
async def navigate_to_url(url: str) -> str:
    """
    Navigate the browser to a specific URL.

    Use this tool when you need to visit a new web page. The URL should be
    a complete URL including the protocol (https://).

    Args:
        url: The full URL to navigate to (e.g., "https://example.com/careers").

    Returns:
        A confirmation message with the page title and final URL.
    """
    page = runtime.get_page()
    response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)

    title = await page.title()
    status = response.status if response else "unknown"

    return (
        f"Navigated to: {page.url}\n"
        f"Page title: {title}\n"
        f"HTTP status: {status}"
    )


@tool
async def go_back() -> str:
    """
    Navigate back to the previous page in browser history.

    Use this when you need to return to a page you were on before.

    Returns:
        A confirmation message with the new page title and URL.
    """
    page = runtime.get_page()
    await page.go_back(wait_until="domcontentloaded", timeout=15000)

    title = await page.title()
    return f"Went back to: {page.url}\nPage title: {title}"


@tool
async def get_current_url() -> str:
    """
    Get the URL of the page the browser is currently on.

    Use this to check where you are before deciding what to do next.

    Returns:
        The current page URL.
    """
    page = runtime.get_page()
    return f"Current URL: {page.url}"


@tool
async def click_element(selector: str) -> str:
    """
    Click on an element identified by a CSS selector.

    Use this to interact with buttons, links, or other clickable elements.
    The selector should be a valid CSS selector that uniquely identifies
    the target element.

    Examples of good selectors:
        - "a[href='/careers']" — a link pointing to /careers
        - "button.submit" — a button with class 'submit'
        - "#load-more" — an element with id 'load-more'
        - "a:has-text('Careers')" — a link containing text 'Careers' (Playwright-specific)

    Args:
        selector: CSS selector or Playwright selector for the element to click.

    Returns:
        Confirmation that the element was clicked, with the new page URL.
    """
    page = runtime.get_page()
    try:
        await page.click(selector, timeout=10000)
        # Wait a moment for any navigation or dynamic content to load
        await page.wait_for_load_state("domcontentloaded", timeout=10000)

        title = await page.title()
        return (
            f"Clicked element matching '{selector}'\n"
            f"Current URL: {page.url}\n"
            f"Page title: {title}"
        )
    except Exception as e:
        return (
            f"Failed to click element '{selector}': {e}\n"
            f"The element may not exist, may not be visible, or the selector may be wrong.\n"
            f"Try a different selector or use extract_links to find the correct target."
        )


@tool
async def type_text(selector: str, text: str) -> str:
    """
    Type text into an input field identified by a CSS selector.

    Use this to fill in search boxes, forms, or other text inputs.

    Args:
        selector: CSS selector for the input element.
        text: The text to type into the input.

    Returns:
        Confirmation that the text was entered.
    """
    page = runtime.get_page()
    await page.fill(selector, text, timeout=10000)

    return f"Typed '{text}' into element matching '{selector}'"


@tool
async def screenshot_page(full_page: bool = False) -> str:
    """
    Take a screenshot of the current page for visual inspection.

    Use this when you need to visually understand the page layout or
    verify that you're on the correct page. The screenshot is saved to
    the output directory.

    Args:
        full_page: If True, capture the entire scrollable page.
                   If False (default), capture only the visible viewport.

    Returns:
        A message confirming the screenshot was taken, with the file path.
    """
    from ai_browser.config import OUTPUT_DIR

    page = runtime.get_page()
    filename = f"screenshot_{page.url.split('/')[-1][:30] or 'page'}.png"
    filepath = OUTPUT_DIR / filename

    await page.screenshot(path=str(filepath), full_page=full_page)
    return f"Screenshot saved to: {filepath}"


@tool
async def scroll_page(direction: str = "down", amount: int = 500) -> str:
    """
    Scroll the page up or down by a specified number of pixels.

    Use this when content is below the fold or you need to trigger
    lazy-loaded elements.

    Args:
        direction: Either "up" or "down" (default: "down").
        amount: Number of pixels to scroll (default: 500).

    Returns:
        Confirmation of the scroll action.
    """
    page = runtime.get_page()
    delta = amount if direction == "down" else -amount
    await page.mouse.wheel(0, delta)
    # Give lazy-loaded content a moment to render
    await page.wait_for_timeout(1000)

    return f"Scrolled {direction} by {amount}px"


@tool
async def wait_for_selector(selector: str, timeout: int = 10000) -> str:
    """
    Wait for a specific element to appear on the page.

    Use this when you expect an element to load dynamically (e.g., after
    clicking a button or scrolling). The tool will wait up to the specified
    timeout for the element to become visible.

    Args:
        selector: CSS selector for the element to wait for.
        timeout: Maximum time to wait in milliseconds (default: 10000).

    Returns:
        Confirmation that the element appeared, or a timeout message.
    """
    page = runtime.get_page()
    try:
        await page.wait_for_selector(selector, timeout=timeout, state="visible")
        return f"Element '{selector}' is now visible on the page."
    except Exception:
        return f"Timeout: Element '{selector}' did not appear within {timeout}ms."


# ── Tool Collection ───────────────────────────────────────────────────────────
# Convenience list for registering all browser tools with an agent.

BROWSER_TOOLS = [
    navigate_to_url,
    go_back,
    get_current_url,
    click_element,
    type_text,
    screenshot_page,
    scroll_page,
    wait_for_selector,
]
