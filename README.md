# Cross-Environment Cooperation Enables Zero-shot Multi-agent Coordination

This is the human-AI experiment code from the paper [Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](https://arxiv.org/abs/2504.12714), which explores how environment diversity can build agents capable of robust cooperation with humans. To learn more, check out the [project website](https://kjha02.github.io/publication/cross-env-coop).

## Installation

To get started, install dependencies using uv:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --python 3.12
```

## Counter Circuit Experiment

To run the counter circuit experiment, run the following command:

```bash
uv run python web_app.py 'counter_circuit'
```

## Coordination Ring Experiment

To run the coordination ring experiment, run the following command:

```bash
uv run python web_app.py 'coord_ring'
```

## Local GPU development with Docker

The development container uses `wilf93/jax_ued:v4`, GPU 0, and mounts this
repository at `/app`. Build and start the Counter Circuit experiment with:

```bash
docker compose up --build
```

After the first build, `docker compose up` is enough. Open
<http://localhost:8080> in a browser.

To run the Coordination Ring experiment instead:

```bash
EXPERIMENT=coord_ring docker compose up
```

Stop and remove the container with `docker compose down`.

## Deploying online with fly.io

**Prerequisites**: Install the [fly CLI](https://fly.io/docs/hands-on/install-flyctl/)

```bash
# Login to fly.io
flyctl auth login

# setup configuration
flyctl launch --dockerfile Dockerfile --name overcooked-cec --config fly.toml --vm-size 'performance-8x' --yes

# deploy to servers/update deployment
flyctl deploy --config fly.toml

# scale to multiple regions (optional, for decreasing latency)
flyctl scale count 10 --config fly.toml --region "iad,sea,lax,den" --yes

# to see logs of run
flyctl logs --config fly.toml
```

**Note:** [fly.io pricing](https://fly.io/docs/about/pricing/)
