"""
AI Browser tools — browser, DOM, storage, and utility tools.

All tools are importable from this package:
    from ai_browser.tools import BROWSER_TOOLS, DOM_TOOLS, STORAGE_TOOLS, ALL_TOOLS
"""

from ai_browser.tools.browser import BROWSER_TOOLS
from ai_browser.tools.dom import DOM_TOOLS
from ai_browser.tools.storage import STORAGE_TOOLS

# Combined list of all available tools
ALL_TOOLS = BROWSER_TOOLS + DOM_TOOLS + STORAGE_TOOLS

__all__ = [
    "BROWSER_TOOLS",
    "DOM_TOOLS",
    "STORAGE_TOOLS",
    "ALL_TOOLS",
]
