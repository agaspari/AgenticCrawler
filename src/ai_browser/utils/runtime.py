"""
Runtime context for AI Browser tools.

Provides a shared runtime context that tools use to access the browser
manager and the current active page. This module acts as the glue between
the LangGraph agent loop and the Playwright browser instance.

The context is initialized once before the agent starts, and all tools
reference it through the module-level `runtime` instance.

Usage:
    from ai_browser.utils.runtime import runtime

    # Initialize before agent runs
    await runtime.initialize()

    # Tools use it internally
    page = runtime.get_page()

    # Cleanup after agent finishes
    await runtime.shutdown()
"""

from __future__ import annotations

from playwright.async_api import BrowserContext, Page

from ai_browser.utils.browser_manager import BrowserManager


class RuntimeContext:
    """
    Shared runtime state accessible to all tools.

    Manages the browser manager lifecycle and tracks the current active
    page that navigation tools operate on. Supports multiple named pages
    for subagent isolation.

    Attributes:
        browser_manager: The shared BrowserManager instance.
    """

    def __init__(self) -> None:
        self.browser_manager: BrowserManager | None = None
        self._default_context: BrowserContext | None = None
        self._default_page: Page | None = None
        self._pages: dict[str, Page] = {}  # Named pages for subagent isolation

    async def initialize(
        self,
        headless: bool | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        """
        Start the browser and create a default page.

        Args:
            headless: Override headless setting from config.
            max_concurrency: Override concurrency limit from config.
        """
        self.browser_manager = BrowserManager(
            headless=headless,
            max_concurrency=max_concurrency,
        )
        await self.browser_manager.start()

        # Create a default context and page for single-agent use
        self._default_context = await self.browser_manager.new_context()
        self._default_page = await self._default_context.new_page()

    async def shutdown(self) -> None:
        """Stop the browser and clean up all resources."""
        self._pages.clear()
        self._default_page = None
        self._default_context = None
        if self.browser_manager:
            await self.browser_manager.stop()
            self.browser_manager = None

    def get_page(self, name: str | None = None) -> Page:
        """
        Get the active page for the current agent.

        Args:
            name: Optional name for a specific subagent's page.
                  If None, returns the default page.

        Returns:
            The active Playwright Page.

        Raises:
            RuntimeError: If the runtime hasn't been initialized.
        """
        if name and name in self._pages:
            return self._pages[name]

        if self._default_page is None:
            raise RuntimeError(
                "Runtime not initialized. Call 'await runtime.initialize()' first."
            )
        return self._default_page

    async def create_page(self, name: str) -> Page:
        """
        Create a named page in a new isolated context.

        Used by subagents that need their own browsing session
        (e.g., parallel parser subagents).

        Args:
            name: Unique identifier for this page.

        Returns:
            A new Page in its own BrowserContext.
        """
        if not self.browser_manager:
            raise RuntimeError(
                "Runtime not initialized. Call 'await runtime.initialize()' first."
            )
        ctx = await self.browser_manager.new_context()
        page = await ctx.new_page()
        self._pages[name] = page
        return page

    async def close_page(self, name: str) -> None:
        """
        Close a named page and its context.

        Args:
            name: The page identifier to close.
        """
        if name in self._pages:
            page = self._pages.pop(name)
            context = page.context
            await page.close()
            await context.close()

    @property
    def is_initialized(self) -> bool:
        """Check if the runtime has been initialized."""
        return (
            self.browser_manager is not None
            and self.browser_manager.is_running
        )

    async def __aenter__(self) -> RuntimeContext:
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.shutdown()


# ── Module-level singleton ────────────────────────────────────────────────────
# All tools import this instance to access the browser.
runtime = RuntimeContext()
