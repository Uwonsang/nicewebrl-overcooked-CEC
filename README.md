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

## Model x Human-proxy Cross-play

Run the six requested trained-policy families against every map-specific BC
human-proxy seed, in both agent seats:

```bash
uv run python model_human_proxy_xp.py
```

IPPO seed 4 is excluded; IPPO therefore uses seeds 0, 1, 2, 3, 5, and 6.

The default trained-model root is
`/mnt/nas/wonsang/crossenv_ued/models/ICRL`. By default, the output path is
inferred beside `models/`, so that root writes to
`/mnt/nas/wonsang/crossenv_ued/proxy_data/`. A root mounted at
`/app/nas/wonsang/crossenv_ued/models/ICRL` instead writes to
`/app/nas/wonsang/crossenv_ued/proxy_data/`. Each rollout uses the same
length-prefixed MessagePack `EnvStage` records as the browser experiment. The
BC action occupies the human-action fields; reaction times and demographics
are null. A compact `summary.csv` and a run `manifest.json` are also written.

For a small deterministic check before a full sweep:

```bash
uv run python model_human_proxy_xp.py \
  --models ippo --layouts coord_ring \
  --model-seeds 0 --human-proxy-seeds 0 \
  --episodes 1 --max-timesteps 10
```

Existing rollout files are skipped so interrupted sweeps can be resumed. Pass
`--overwrite` to regenerate them. Use `--episodes N` for repeated stochastic
rollouts per model/proxy/seat pair.

The generated records can be passed directly to the existing analysis:

```bash
uv run python analysis_extend.py \
  --data-dir /mnt/nas/wonsang/crossenv_ued/proxy_data \
  --output-dir analysis/xp_human_proxy
```

To run one algorithm-layout pair at a time in separate processes, use the
provided sweep script:

```bash
./run_model_human_proxy_xp.sh
```

Alternatively, pass one algorithm name and the common ICRL model root. The
script finds the appropriate algorithm and layout directories automatically:

```bash
./run_model_human_proxy_xp.sh \
  ippo /mnt/nas/wonsang/crossenv_ued/models/ICRL
```

The script uses `uv run --no-sync`, so it does not reinstall packages between
algorithm-layout runs. Run `uv sync --python 3.12` once beforehand when setting
up a new environment.

The same ICRL root works for shared CEC checkpoints:

```bash
./run_model_human_proxy_xp.sh \
  cec_64 /mnt/nas/wonsang/crossenv_ued/models/ICRL
```

The second argument can alternatively be an algorithm directory, one layout
directory, a shared CEC seed-directory root, or one `.pkl` file. When using a
layout directory or file with a map-specific model, select the matching layout
via the `LAYOUTS` environment variable.

The sweep can be restricted or redirected with environment variables. Extra
arguments are forwarded to `model_human_proxy_xp.py`:

```bash
ALGORITHMS="ippo cec_64" \
LAYOUTS="coord_ring cramped_room" \
EPISODES=1 \
OUTPUT_DIR=data/xp_subset \
./run_model_human_proxy_xp.sh --model-seeds 0 1 --human-proxy-seeds 0 1
```

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
