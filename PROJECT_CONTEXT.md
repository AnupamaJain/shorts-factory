# 🧠 Shorts-Factory Project Context

## Project Goal
An automated, multi-agent AI pipeline for generating highly engaging, production-grade YouTube Shorts and Instagram Reels about finance and algorithmic trading.

## Architecture & Workflows
The project relies on a strictly defined `LangGraph` state machine orchestration.
The pipeline follows this exact node sequence:
1. **Researcher**: Queries the local BM25 RAG database (`agents/rag.py`) for historical data.
2. **Scriptwriter**: Drafts a highly engaging 60-second script (`llama-3.3-70b-versatile`).
3. **Quality Evaluator**: Uses `DeepEval` (G-Eval programmatic metric) to grade the hook, retention, and authority on a 1-10 scale. Routes back to the Scriptwriter if the score is below threshold.
4. **Guardrail Reviewer**: Ensures strict financial compliance (no guaranteed returns).
5. **Director**: Configures visual pacing and calls the video rendering tools.
6. **Social Media Manager**: Drafts click-worthy Instagram captions and hashtags.
7. **Publisher**: Autonomously uploads the final video to Instagram Reels using `instagrapi` if enabled in the config.

## Global Configuration
All AI behavioral parameters, metrics, thresholds, and server names are centralized in `/shorts_config.yaml`.
**DO NOT hardcode thresholds or criteria** in the python files. Always use `SHORTS_CONFIG.get()`.

## Core Engines
- **Standard Video Engine** (`core/viral_storyteller.py`): Requires a base background `.mp4` video. Relies heavily on exact word synchronization via `faster_whisper`.
- **Text-to-Video Engine** (`core/text_to_video.py`): Completely AI-generated videos. Downloads images dynamically from Pollinations API and uses Ken Burns pan/zoom effects.
- **MCP Server** (`servers/mcp_server.py`): Exposes the video generation and social media tools to external Model Context Protocol clients (like Claude Desktop).

## Critical Developer Guidelines
1. **MacOS Compat**: Always ensure `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is set in core scripts to prevent OpenMP crashes.
2. **Pathing**: Run all scripts from the root directory with `export PYTHONPATH=.`.
3. **API Rate Limits**: The project uses free-tier Groq and Pollinations API. Assume rate limits are tight. Do not build loops that infinitely poll APIs.
4. **Environment**: Ensure `.env` is loaded using `python-dotenv`. Required variables: `GROQ_API_KEY`.
