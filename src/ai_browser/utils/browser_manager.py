"""
Browser lifecycle manager for AI Browser.

Manages Playwright browser instances and contexts. Designed as an async
context manager for clean setup/teardown. Supports both headless and headed
modes, and provides isolated browser contexts for parallel subagent execution.

Usage:
    async with BrowserManager() as manager:
        page = await manager.new_page()
        await page.goto("https://example.com")
        content = await page.content()
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from ai_browser.config import BROWSER_HEADLESS, MAX_CONCURRENCY


class BrowserManager:
    """
    Manages the Playwright browser lifecycle.

    Creates a single browser instance and hands out isolated BrowserContexts
    for each subagent. This allows parallel page processing without
    cookie/session leakage between agents.

    Attributes:
        headless: Whether to run the browser in headless mode.
        max_concurrency: Maximum number of concurrent browser contexts.
    """

    def __init__(
        self,
        headless: bool | None = None,
        max_concurrency: int | None = None,
    ):
        self.headless = headless if headless is not None else BROWSER_HEADLESS
        self.max_concurrency = max_concurrency or MAX_CONCURRENCY
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._contexts: list[BrowserContext] = []
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def start(self) -> None:
        """Launch Playwright and the browser instance."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
        )

    async def stop(self) -> None:
        """Close all contexts, the browser, and Playwright."""
        for ctx in self._contexts:
            try:
                await ctx.close()
            except Exception:
                pass  # Context may already be closed
        self._contexts.clear()

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self) -> BrowserManager:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def new_context(self, **kwargs) -> BrowserContext:
        """
        Create a new isolated browser context.

        Each context has its own cookies, localStorage, and cache —
        perfect for giving each subagent an isolated browsing session.

        Args:
            **kwargs: Additional arguments passed to browser.new_context()
                      (e.g. viewport, user_agent, locale).

        Returns:
            A new BrowserContext instance.
        """
        if not self._browser:
            raise RuntimeError("BrowserManager not started. Use 'async with' or call start().")

        ctx = await self._browser.new_context(**kwargs)
        self._contexts.append(ctx)
        return ctx

    async def new_page(self, context: BrowserContext | None = None, **kwargs) -> Page:
        """
        Create a new page, optionally within a specific context.

        If no context is provided, a new default context is created.

        Args:
            context: An existing BrowserContext to open the page in.
            **kwargs: Arguments passed to new_context() if creating one.

        Returns:
            A new Page instance.
        """
        if context is None:
            context = await self.new_context(**kwargs)
        return await context.new_page()

    @asynccontextmanager
    async def acquire_context(self, **kwargs) -> AsyncGenerator[BrowserContext, None]:
        """
        Acquire a browser context with concurrency limiting.

        Uses a semaphore to ensure no more than max_concurrency contexts
        are active simultaneously. Automatically cleans up on exit.

        Usage:
            async with manager.acquire_context() as ctx:
                page = await ctx.new_page()
                await page.goto(url)
        """
        await self._semaphore.acquire()
        ctx = await self.new_context(**kwargs)
        try:
            yield ctx
        finally:
            try:
                await ctx.close()
                self._contexts.remove(ctx)
            except (ValueError, Exception):
                pass
            self._semaphore.release()

    @property
    def is_running(self) -> bool:
        """Check if the browser is currently running."""
        return self._browser is not None and self._browser.is_connected()
