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

## Run the Experiment

Edit `LAYOUTS_TO_TEST` and `ALGORITHMS_TO_TEST` near the top of `web_app.py`,
then run:

```bash
uv run python web_app.py
```

All selected algorithm-layout pairs are tested in one participant session.
Layout selection is controlled only by `LAYOUTS_TO_TEST`; command-line layout
arguments are not used.

## Extended Analysis

Generate per-user, per-map, per-algorithm, and total CSV/JSON/PNG reports:

```bash
uv run python analysis_extend.py
```

Reports are written under `analysis/`. The paper-oriented summaries include
mean recipes made, human-AI collisions, seven qualitative ratings, standard
errors, Pearson survey correlations, Cronbach's alpha, and pairwise Welch
t-tests. Runs collected before the paper-metric logger was added remain
readable, but their collision values are marked as unavailable.

New experiment runs explicitly record the selected model checkpoint, human and
AI actions, delivery events, agent positions, and movement collisions. A
collision means that the agents tried to enter the same cell or swap places.

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
