FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8501

CMD ["sh", "-c", "uv run python -m utils.load_qdrant_collection && uv run streamlit run src/app/app_launcher.py --server.address=0.0.0.0"]