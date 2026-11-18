# AGENTS.md

A super minimal Streamlit app plus an MCP server for local image generation with diffusers (Z-Image Turbo, FLUX.2 [klein] 4B, SDXL). Terms are defined in [GLOSSARY.md](GLOSSARY.md).

## Commands

- `uv sync`: install dependencies
- `uv run streamlit run main.py`: run the app
- `uv run python mcp_server.py`: run the MCP server (streamable HTTP on 127.0.0.1:8765; `--port`, `--preload <model-slug>`)
- `uv run pytest`: run the tests (no GPU needed)

## Layout

- `main.py`: Streamlit UI
- `sd2/generate.py`: models table, generation, and saving to `outputs/` with a settings .txt; no Streamlit dependency
- `mcp_server.py`: the MCP server
- `tests/`: pytest tests
- `outputs/`: generated images and settings files; gitignored

## Conventions

- Python 3.12. Add dependencies with `uv add`, then regenerate `requirements.txt` with the `uv export` command in the README.
- Keep the README in sync: it documents the models, licences, memory notes, the app and the server.
- Adding a model means a row in `PIPELINES`, plus defaults and hints, a slug in `MODEL_SLUGS`, the README models table and its licence, and the slug list in the README's MCP section.
- Model downloads are big; don't trigger them in tests.
