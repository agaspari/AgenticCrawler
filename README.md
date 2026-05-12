# AI Browser 🌐🤖

A general-purpose AI browser/crawler agent built with **LangChain**, **LangGraph**, and **Playwright**.

Give it a URL and a task — it navigates, explores, extracts structured data, and saves the results.

## Architecture

```
User Input (URL + Task)
        │
        ▼
┌─────────────────┐
│   Orchestrator  │ ← Supervisor agent: plans, routes, aggregates
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌────────┐
│Navigator│ │Parser×N│ ← Fan-out via LangGraph Send API
└─────────  └────────┘
                │
                ▼
         ┌──────────┐
         │  Storage  │ ← JSON, CSV (extensible to Sheets, DB, etc.)
         └──────────┘
```

## Features

- 🌐 **Browser Automation** — Playwright-powered navigation, clicking, typing, screenshots
- 📄 **DOM Extraction** — BeautifulSoup cleaning + LLM-powered structured extraction
- 🤖 **Multi-Agent** — Orchestrator/Navigator/Parser subgraphs with parallel execution
- 💾 **Flexible Storage** — JSON, CSV (extensible to Google Sheets, databases)
- 🔧 **Multi-LLM** — OpenAI, Anthropic, Google Gemini via `init_chat_model()`
- 📊 **Observable** — LangSmith tracing built in

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run the Job Scraper (Test Case)

```bash
uv run python scripts/run_job_scraper.py --url "https://example.com/careers"
```

## Configuration

All settings are configurable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `openai:gpt-4o-mini` | LLM model string (`provider:model`) |
| `BROWSER_HEADLESS` | `true` | Run browser in headless mode |
| `MAX_CONCURRENCY` | `5` | Max parallel browser contexts |
| `OUTPUT_DIR` | `output` | Directory for saved results |
| `LANGSMITH_TRACING` | `true` | Enable LangSmith tracing |

## Project Structure

```
src/ai_browser/
├── config.py              # Settings & env loading
├── state.py               # LangGraph state schemas
├── tools/                 # Agent tools (browser, DOM, storage, utility)
├── agents/                # LangGraph subgraphs (orchestrator, navigator, parser)
├── schemas/               # Pydantic extraction schemas
└── utils/                 # Browser manager, HTML cleaner
```

## License

MIT
