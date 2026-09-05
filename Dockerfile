FROM python:3.10-slim

RUN apt update && apt install -y curl procps git cmake build-essential

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy everything
COPY . /app

# Remove any existing .venv and create environment
RUN rm -rf .venv && uv sync --python 3.12

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG

CMD ["uv", "run", "python", "web_app.py"]
