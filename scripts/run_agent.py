"""
CLI entry point for running the AI Browser agent.

Usage:
    uv run python scripts/run_agent.py --url "https://example.com" --task "Find and extract all job listings"
    uv run python scripts/run_agent.py --url "https://example.com/careers" --task "Extract all jobs" --model "anthropic:claude-sonnet-4-20250514"
    uv run python scripts/run_agent.py --url "https://example.com" --task "Summarize the page" --headed
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path so we can import ai_browser
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_browser.agents.single import run_agent


def main():
    parser = argparse.ArgumentParser(
        description="AI Browser Agent — Navigate, extract, and save web data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url "https://example.com" --task "Summarize this page"
  %(prog)s --url "https://company.com/careers" --task "Extract all job listings and save to JSON"
  %(prog)s --url "https://example.com" --task "Find the careers page" --headed --model "anthropic:claude-sonnet-4-20250514"
        """,
    )

    parser.add_argument(
        "--url",
        required=True,
        help="The starting URL for the agent to navigate to.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Natural language description of what to accomplish.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help='LLM model string (e.g., "openai:gpt-4o", "anthropic:claude-sonnet-4-20250514"). Defaults to .env DEFAULT_MODEL.',
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        default=False,
        help="Run the browser in headed (visible) mode for debugging.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AI Browser Agent")
    print("=" * 60)
    print(f"URL:   {args.url}")
    print(f"Task:  {args.task}")
    print(f"Model: {args.model or '(default from .env)'}")
    print(f"Mode:  {'headed (visible)' if args.headed else 'headless'}")
    print("=" * 60)
    print()

    headless = not args.headed

    result = asyncio.run(
        run_agent(
            url=args.url,
            task=args.task,
            model=args.model,
            headless=headless,
        )
    )

    # Print final message
    print("\n" + "=" * 60)
    print("Agent finished.")
    if result and "messages" in result:
        final_msg = result["messages"][-1]
        if hasattr(final_msg, "content"):
            print(f"\nFinal summary:\n{final_msg.content}")
    print("=" * 60)


if __name__ == "__main__":
    main()
