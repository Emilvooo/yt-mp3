# AGENTS.md — yt-mp3

## Project Overview

YouTube audio downloader and converter. Single-file Python/FastAPI backend (`main.py`) with a
vanilla HTML/CSS/JS frontend (`templates/index.html`). Uses yt-dlp + ffmpeg for downloading and
converting audio. Optional LLM-powered metadata parsing via OpenRouter API.

**Stack:** Python 3.12, FastAPI, uvicorn, yt-dlp, ffmpeg, httpx, Jinja2

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (development)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run the server (production)
uvicorn main:app --host 0.0.0.0 --port 8000

# Docker build and run
docker build -t yt-mp3 .
docker run -p 8000:8000 yt-mp3
```

### System Dependencies

- `ffmpeg` must be installed (audio conversion, probing, thumbnail embedding)
- `yt-dlp` is installed via pip but depends on system ffmpeg

### Environment Variables

| Variable             | Required | Purpose                                            |
|----------------------|----------|----------------------------------------------------|
| `OPENROUTER_API_KEY` | No       | Enables LLM metadata parsing (artist/title cleanup)|

## Tests

No test suite exists. No test framework is configured. If adding tests:

```bash
# Recommended: use pytest with httpx for async FastAPI testing
pip install pytest pytest-asyncio httpx
pytest                     # run all tests
pytest tests/test_foo.py   # run a single test file
pytest -k "test_name"      # run a single test by name
```

## Linting & Formatting

No linter or formatter is configured. If adding one, use ruff:

```bash
pip install ruff
ruff check .               # lint
ruff format .              # format
```

## Code Style Guidelines

### Python (`main.py`)

**Imports** — stdlib first (one per line), blank line, then third-party. Use `from` imports
for specific symbols. No wildcard imports.

```python
import asyncio
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
```

**Naming:**
- Functions: `snake_case`; prefix private/internal helpers with underscore (`_fetch_video_info`)
- Route handlers: plain `snake_case` (`index`, `download`, `progress`)
- Constants: `UPPER_SNAKE_CASE` (`TASK_TTL`, `VALID_FORMATS`)
- Variables: `snake_case`

**Types:**
- Use Python 3.10+ syntax: `str | None`, `Path | None` (not `Optional[str]`)
- Use built-in generics: `dict[str, dict]`, `list[str]`, `tuple[Path | None, str | None]`
- Type hints on all function signatures (parameters + return types)

**Formatting:**
- 4-space indentation
- Double quotes for all strings
- No semicolons
- ~120 char line length (soft limit)
- Two blank lines between top-level functions/classes
- No trailing commas required

**No comments or docstrings** — code should be self-documenting. Only add comments for
genuinely complex logic that can't be simplified.

**No docblocks** — never add documentation blocks (PHPDoc, JSDoc, Python docstrings) to
functions or classes.

**Error handling:**
- Use `try/except Exception` with fallback return values (empty dict, `None`, etc.)
- Store errors in task dict: `task["status"] = "error"`, `task["error"] = str(e)`
- Return HTTP errors as `JSONResponse({"error": "message"}, status_code=XXX)`
- No custom exception classes
- No logging (yet)

**Architecture patterns:**
- In-memory task store (`tasks` dict), no database
- Async throughout: `async def` for all I/O functions
- `asyncio.create_subprocess_exec()` for external process calls (yt-dlp, ffmpeg)
- Background tasks via `asyncio.create_task()`
- SSE (Server-Sent Events) for progress streaming to frontend
- Fallback strategy: stream pipeline fails -> download to file -> convert from file

**Function style:**
- Use `def` / `async def` declarations (no lambdas for named functions)
- Inner functions for SSE event streams and cleanup callbacks
- Group related statements: no blank line between variable assignment and immediate use

### JavaScript (inline in `templates/index.html`)

- Vanilla JS, no frameworks
- `const` for DOM refs and constants; `let` for mutable state
- Arrow functions for callbacks; regular `function` for named utilities
- `camelCase` naming
- Semicolons used consistently
- Single quotes for strings
- `async/await` for fetch; `EventSource` for SSE

### CSS (inline in `templates/index.html`)

- CSS custom properties in `:root`
- Flat class naming (`.dl-btn`, `.preview-thumb`, `.progress-fill`)
- Dark theme with accent color `#ff3434`

## Project Structure

```
yt-mp3/
  main.py              # All backend logic (FastAPI app, routes, download pipeline)
  templates/
    index.html         # Single-page frontend (HTML + inline CSS + inline JS)
  requirements.txt     # Python dependencies
  Dockerfile           # Container definition (python:3.12-slim + ffmpeg)
  .gitignore
  AGENTS.md
```

## Key Endpoints

| Method | Path               | Purpose                              |
|--------|--------------------|--------------------------------------|
| GET    | `/`                | Serve the frontend                   |
| GET    | `/info?url=`       | Fetch video metadata (title, etc.)   |
| POST   | `/download`        | Start a download task (returns task_id) |
| GET    | `/progress/{id}`   | SSE stream of download/convert progress |
| GET    | `/file/{id}`       | Download the converted audio file    |
| GET    | `/robots.txt`      | Robots exclusion                     |

## Conventions

- **Minimal changes**: only modify what's needed
- **Verify first**: understand the code before implementing changes
- **Concise over verbose**: prefer one-liners when readability allows
- **Simple over clever**: avoid unnecessary abstractions
- **Performance-aware**: efficient solutions, no premature optimization
- **Self-documenting code**: no comments unless complex logic requires it
