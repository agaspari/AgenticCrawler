"""
Single ReAct agent for AI Browser.

This is the Phase 4 agent — a single ReAct loop with all 21 tools.
It can navigate sites, extract content, and save results autonomously.

In Phase 5 this will be replaced by the multi-agent orchestrator/navigator/parser
architecture, but this single agent is useful for:
  - Testing individual tools end-to-end
  - Simple scraping tasks that don't need parallelism
  - Learning how the tools work together

Usage:
    from ai_browser.agents.single import create_agent, run_agent

    result = await run_agent(
        url="https://example.com",
        task="Find the careers page and extract all job listings",
    )
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from ai_browser.config import get_llm
from ai_browser.tools import ALL_TOOLS
from ai_browser.utils.runtime import runtime


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert web browsing agent. You navigate websites, \
extract information, and save structured data.

## Your Capabilities
You have tools for:
1. **Browser Navigation** — navigate to URLs, click elements, type text, scroll, take screenshots
2. **DOM Reading** — extract text, links, elements by CSS selector, and structured data
3. **Storage** — save data to JSON or CSV files, read saved files, list saved files
4. **Reasoning** — summarize pages, plan next steps, log observations

## How to Work
1. **Plan first**: Before taking action, think about what you need to do.
   Use `plan_next_steps` to organize your approach.
2. **Navigate strategically**: Start at the given URL, extract links, and
   identify which pages are relevant to the task.
3. **Extract carefully**: Use `extract_page_text` or `extract_links` to understand
   page content. Use `extract_structured_data` for complex extraction tasks.
4. **Save results**: Always save extracted data using `save_to_json_file` or
   `save_to_csv_file` before finishing.
5. **Be thorough**: Check for pagination, dynamically loaded content, and
   alternative navigation paths.

## Important Rules
- Always start by navigating to the target URL.
- Use CSS selectors or Playwright selectors to interact with specific elements.
- When looking for links, try `extract_links` with `include_nav=True` first
  to find navigation menus.
- If a page has lots of content, use a CSS selector to narrow your extraction.
- Log important observations as you go so you don't lose context.
- When you're done, provide a clear summary of what you found and where you saved it.
"""


# ── Agent Factory ─────────────────────────────────────────────────────────────

def create_agent(
    model: str | None = None,
    tools: list | None = None,
    checkpointer: bool = True,
):
    """
    Create a ReAct agent with browser tools.

    Args:
        model: LLM model string override (e.g., "openai:gpt-4o").
               Defaults to DEFAULT_MODEL from .env.
        tools: Optional custom tool list. Defaults to ALL_TOOLS.
        checkpointer: If True, uses InMemorySaver for conversation persistence.

    Returns:
        A compiled LangGraph agent ready to invoke.
    """
    llm = get_llm(model=model)
    agent_tools = tools or ALL_TOOLS

    memory = InMemorySaver() if checkpointer else None

    agent = create_react_agent(
        model=llm,
        tools=agent_tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    return agent


# ── Convenience Runner ────────────────────────────────────────────────────────

async def run_agent(
    url: str,
    task: str,
    model: str | None = None,
    headless: bool | None = None,
    thread_id: str = "default",
) -> dict:
    """
    Run the single agent end-to-end: initialize browser, execute task, cleanup.

    This is the high-level entry point for simple tasks.

    Args:
        url: The starting URL for the agent to navigate to.
        task: A natural language description of what to accomplish.
        model: Optional LLM model string override.
        headless: Optional override for headless browser mode.
        thread_id: Thread ID for conversation persistence.

    Returns:
        The final agent state containing messages and results.
    """
    # Initialize the browser runtime
    await runtime.initialize(headless=headless)

    try:
        agent = create_agent(model=model)

        # Compose the user message
        user_message = (
            f"Target URL: {url}\n\n"
            f"Task: {task}\n\n"
            f"Please navigate to the URL and complete the task. "
            f"Save any extracted data to files."
        )

        config = {"configurable": {"thread_id": thread_id}}

        # Stream the agent execution so we can see progress
        final_state = None
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            version="v2",
        ):
            kind = event["event"]

            # Print tool calls for visibility
            if kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", "")
                print(f"  -> Tool: {tool_name}")
                if tool_input:
                    # Truncate long inputs
                    input_str = str(tool_input)
                    if len(input_str) > 120:
                        input_str = input_str[:120] + "..."
                    print(f"     Input: {input_str}")

            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", "")
                if output:
                    output_str = str(output)
                    if len(output_str) > 200:
                        output_str = output_str[:200] + "..."
                    print(f"     Output: {output_str}")

            elif kind == "on_chat_model_stream":
                # Print the LLM's thinking/response as it streams
                chunk = event.get("data", {}).get("chunk", None)
                if chunk and hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        print(chunk.content, end="", flush=True)

        # Get final state
        final_state = await agent.ainvoke(
            {"messages": [HumanMessage(content="Provide a final summary of what you accomplished.")]},
            config=config,
        )

        return final_state

    finally:
        await runtime.shutdown()
