"""
CLI entry point for the multi-agent AI Browser pipeline.

Usage:
    uv run python scripts/run_multi_agent.py --url "https://example.com" --task "Find and extract all job listings"
    uv run python scripts/run_multi_agent.py --url "https://example.com" --task "Extract job postings" --headed
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_browser.agents.orchestrator import run_multi_agent


def main():
    parser = argparse.ArgumentParser(
        description="AI Browser — Multi-Agent Pipeline (Navigator + Parallel Parsers)",
    )

    parser.add_argument("--url", required=True, help="Starting URL to explore.")
    parser.add_argument("--task", required=True, help="What to extract (natural language).")
    parser.add_argument("--model", default=None, help='LLM model string (e.g., "openai:gpt-4o").')
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode.")

    args = parser.parse_args()

    result = asyncio.run(
        run_multi_agent(
            url=args.url,
            task=args.task,
            model=args.model,
            headless=not args.headed,
        )
    )


if __name__ == "__main__":
    main()
