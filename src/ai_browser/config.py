"""
Configuration module for AI Browser.

Loads environment variables and exposes typed settings used across the project.
All configurable values live in .env and are accessed through this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# ── Load .env from project root ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── LLM Settings ─────────────────────────────────────────────────────────────
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "openai:gpt-4o-mini")


def get_llm(model: str | None = None, **kwargs) -> BaseChatModel:
    """
    Return a configured LLM instance via LangChain's init_chat_model.

    The model string follows the format "provider:model_name", e.g.:
        - "openai:gpt-4o-mini"
        - "anthropic:claude-sonnet-4-20250514"
        - "google-genai:gemini-2.0-flash"

    Args:
        model: Model string override. Defaults to DEFAULT_MODEL from .env.
        **kwargs: Additional arguments forwarded to init_chat_model
                  (e.g. temperature, max_tokens).

    Returns:
        A configured BaseChatModel instance.
    """
    model_str = model or DEFAULT_MODEL
    return init_chat_model(model_str, **kwargs)


# ── Browser Settings ─────────────────────────────────────────────────────────
BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "5"))


# ── Output Settings ──────────────────────────────────────────────────────────
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(_PROJECT_ROOT / "output")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── LangSmith ────────────────────────────────────────────────────────────────
# LangSmith tracing is auto-configured by environment variables:
#   LANGSMITH_API_KEY, LANGSMITH_TRACING, LANGSMITH_PROJECT
# No explicit code needed — LangChain picks these up automatically.
LANGSMITH_ENABLED: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
