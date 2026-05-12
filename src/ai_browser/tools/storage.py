"""
Storage tools for AI Browser agents.

These tools give the agent the ability to persist extracted data to the
local filesystem. Currently supports JSON and CSV formats, with an
extensible design for adding Google Sheets, databases, etc. later.

All output files are saved to the configured OUTPUT_DIR (from .env).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ai_browser.config import OUTPUT_DIR


def _resolve_path(filename: str) -> Path:
    """
    Resolve a filename to an absolute path inside OUTPUT_DIR.

    If the filename already contains path separators, treat it as
    relative to OUTPUT_DIR. Ensures the parent directory exists.
    """
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@tool
def save_to_json_file(
    data: list[dict[str, Any]] | dict[str, Any],
    filename: str | None = None,
    label: str = "",
) -> str:
    """
    Save structured data to a JSON file in the output directory.

    Use this when you have extracted data (e.g., a list of job postings)
    and want to persist it. The data can be a single dict or a list of dicts.

    Args:
        data: The structured data to save. Can be a dict or list of dicts.
        filename: Optional filename (e.g., "jobs.json"). If not provided,
                  a timestamped filename is generated automatically.
        label: Optional label to include in the auto-generated filename
               (e.g., "careers" produces "careers_20260511_213500.json").

    Returns:
        Confirmation message with the file path and number of records saved.
    """
    if not filename:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = f"{label}_" if label else ""
        filename = f"{prefix}{timestamp}.json"

    if not filename.endswith(".json"):
        filename += ".json"

    path = _resolve_path(filename)

    # Wrap single dict in metadata envelope
    if isinstance(data, dict):
        output = data
        record_count = 1
    else:
        output = {
            "metadata": {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(data),
                "label": label,
            },
            "data": data,
        }
        record_count = len(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return f"Saved {record_count} record(s) to: {path}"


@tool
def save_to_csv_file(
    data: list[dict[str, Any]],
    filename: str | None = None,
    label: str = "",
) -> str:
    """
    Save a list of records to a CSV file in the output directory.

    Use this when you want tabular output (e.g., for spreadsheets).
    Each dict in the list becomes a row; keys become column headers.

    Args:
        data: List of dicts to save. All dicts should have the same keys.
        filename: Optional filename (e.g., "jobs.csv"). If not provided,
                  a timestamped filename is generated automatically.
        label: Optional label for the auto-generated filename.

    Returns:
        Confirmation message with the file path and number of rows saved.
    """
    if not data:
        return "No data to save — the list is empty."

    if not filename:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = f"{label}_" if label else ""
        filename = f"{prefix}{timestamp}.csv"

    if not filename.endswith(".csv"):
        filename += ".csv"

    path = _resolve_path(filename)

    # Collect all unique keys across all records for headers
    headers: list[str] = []
    seen: set[str] = set()
    for row in data:
        for key in row.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)

    return f"Saved {len(data)} row(s) to: {path}"


@tool
def read_from_file(filename: str) -> str:
    """
    Read the contents of a previously saved file from the output directory.

    Use this to review data that was saved earlier, or to check what
    has already been extracted before deciding on next steps.

    Args:
        filename: The filename to read (e.g., "jobs.json", "results.csv").
                  Relative to the output directory.

    Returns:
        The file contents as a string. JSON files are pretty-printed.
    """
    path = _resolve_path(filename)

    if not path.exists():
        return f"File not found: {path}"

    content = path.read_text(encoding="utf-8")

    # Pretty-print JSON if applicable
    if path.suffix == ".json":
        try:
            parsed = json.loads(content)
            content = json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass  # Return raw content if not valid JSON

    # Truncate very large files
    max_chars = 10000
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[... truncated, {len(content) - max_chars} chars omitted]"

    return content


@tool
def list_saved_files() -> str:
    """
    List all files in the output directory.

    Use this to see what data has already been saved, before deciding
    whether to overwrite or create new files.

    Returns:
        A formatted list of saved files with sizes.
    """
    files = sorted(OUTPUT_DIR.glob("*"))
    files = [f for f in files if f.is_file() and f.name != ".gitkeep"]

    if not files:
        return "No saved files found in the output directory."

    lines = [f"Saved files in {OUTPUT_DIR}:\n"]
    for f in files:
        size_kb = f.stat().st_size / 1024
        lines.append(f"  - {f.name} ({size_kb:.1f} KB)")

    return "\n".join(lines)


# ── Tool Collection ───────────────────────────────────────────────────────────

STORAGE_TOOLS = [
    save_to_json_file,
    save_to_csv_file,
    read_from_file,
    list_saved_files,
]
