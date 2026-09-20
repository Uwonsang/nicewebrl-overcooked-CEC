"""
W&B Sweep for ippo_general_population.py hyperparameter search.
Sweeps over: LR, VF_COEF, NUM_MINIBATCHES, ENT_COEF, layout, FCP, e3t_beta

Usage (run from repo root):
  # 1. Create sweep (prints sweep ID)
  python -m baselines.CEC.sweep.sweep_ippo_population --create

  # 2. Join an existing sweep
  python -m baselines.CEC.sweep.sweep_ippo_population --run --sweep_id <ID>

  # 3. Create + immediately start one agent
  python -m baselines.CEC.sweep.sweep_ippo_population --create --run
"""
import copy
import os
import argparse
import time
import jax
import wandb

# ── Sweep configuration ────────────────────────────────────────────────────────
# W&B sweep params must be flat — nested keys (layout) are applied
# manually in run_sweep_agent() below.
SWEEP_CONFIG = {
    "method": "grid",   # "grid" | "random" | "bayes"
    "metric": {
        "name": "returns",
        "goal": "maximize",
    },
    "parameters": {
        "LR": {
            "values": [1e-4, 3e-4, 1e-3],
        },
        "VF_COEF": {
            "values": [1],
        },
        "NUM_MINIBATCHES": {
            "values": [2, 4],
        },
        # nested under ENV_KWARGS — applied manually
        "layout": {
            "values": [
                "cramped_room_9",
                "asymm_advantages_9",
                "coord_ring_9",
                "counter_circuit_9",
                "forced_coord_9",
            ],
        },
    },
}

# ── Base config (mirrors repro_config/ippo_final_baseline.yaml) ───────────────
BASE_CONFIG = {
    "ENV_NAME": "overcooked",
    "LR": 1e-3,
    "NUM_ENVS": 256,
    "NUM_SEEDS": 1,
    "NUM_STEPS": 256,
    "FC_DIM_SIZE": 256,
    "GRU_HIDDEN_DIM": 256,
    "TOTAL_TIMESTEPS": 5e7,
    "MAX_TRAIN_STEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 2,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "SCALE_CLIP_EPS": False,
    "ENT_COEF": 0.005,
    "VF_COEF": 1.0,
    "MOA_COEF": 1.0,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": True,
    "SEED": 0,
    "ENV_KWARGS": {
        "random_reset": False,
        "max_steps": 400,
        "check_held_out": False,
        "debug": False,
        "partial_obs": False,
        "incentivize_strat": 2,
        "shuffle_inv_and_pot": False,
        "layout": "cramped_room_9",
        "random_reset_fn": "reset_all",
    },
    "TRAINING": True,
    "TRAIN_KWARGS": {
        "ckpt_id": 0,
        "overwrite_ckpt": True,
        "finetune": False,
        "e3t_beta": 0.55,
    },
    "TEST_KWARGS": {
        "beta": 1.0,
        "argmax": False,
        "num_trajs": 2,
        "plot": False,
        "self_play": False,
        "ik": False,
        "debug": False,
        "use_ckpt": True,
    },
    "CONV_NET": True,
    "LSTM": True,
    "FCP": False,
    "FCP_KWARGS": {"train_oracle": True},
    "ENTITY": "overcooked_ai",
    "PROJECT": "crossenv_ued_short",
    "WANDB_MODE": "online",
    "model_name": "IPPO",
}


def run_sweep_agent():
    """Single sweep trial — called by wandb.agent."""
    from baselines.CEC.ippo_general_population import make_train

    run = wandb.init()
    sp = dict(wandb.config)
    sweep_params = SWEEP_CONFIG["parameters"]

    # Deep-copy so each trial has an independent config
    config = copy.deepcopy(BASE_CONFIG)

    # Only apply parameters that are explicitly present in SWEEP_CONFIG.
    flat_param_keys = ["LR", "VF_COEF", "NUM_MINIBATCHES", "ENT_COEF", "FCP"]
    for key in flat_param_keys:
        if key in sweep_params:
            config[key] = sp[key]

    if "layout" in sweep_params:
        config["ENV_KWARGS"]["layout"] = sp["layout"]

    if "e3t_beta" in sweep_params:
        config["TRAIN_KWARGS"]["e3t_beta"] = sp["e3t_beta"]

    layout_name = config["ENV_KWARGS"]["layout"]
    fcp_tag = "fcp" if config["FCP"] else "nofcp"
    xpid = "sweep-%s" % time.strftime("%Y%m%d-%H%M%S")
    filepath = (
        f"ckpts/ippo/{config['ENV_NAME']}/{layout_name}"
        f"/ik{config['ENV_KWARGS']['random_reset']}"
        f"/{config['ENV_KWARGS']['random_reset_fn']}/{fcp_tag}/{xpid}"
    )
    os.makedirs(filepath, exist_ok=True)

    run.name = (
        f"{layout_name}_{fcp_tag}"
        f"_lr{config['LR']:.0e}_vf{config['VF_COEF']}"
        f"_mb{config['NUM_MINIBATCHES']}_ent{config['ENT_COEF']:.3f}"
    )

    if "e3t_beta" in sweep_params:
        run.name += f"_e3t{config['TRAIN_KWARGS']['e3t_beta']}"

    print(
        f"[sweep] layout={layout_name}  FCP={config['FCP']}\n"
        f"        LR={config['LR']}  VF_COEF={config['VF_COEF']}  "
        f"NUM_MINIBATCHES={config['NUM_MINIBATCHES']}  ENT_COEF={config['ENT_COEF']}\n"
        f"        ckpt dir: {filepath}"
    )

    if "e3t_beta" in sweep_params:
        print(f"        e3t_beta={config['TRAIN_KWARGS']['e3t_beta']}")

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config, 0, filepath), device=jax.devices()[0])
    train_jit(rng, None, 0)


def create_sweep(entity: str, project: str) -> str:
    sweep_id = wandb.sweep(SWEEP_CONFIG, entity=entity, project=project)
    print(f"[sweep] Created sweep: {sweep_id}")
    print(f"[sweep] View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    return sweep_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create",   action="store_true", help="Create a new W&B sweep")
    parser.add_argument("--run",      action="store_true", help="Start a sweep agent")
    parser.add_argument("--sweep_id", type=str,  default=None, help="Existing sweep ID to join")
    parser.add_argument("--count",    type=int,  default=None, help="Max trials per agent (default: unlimited)")
    parser.add_argument("--entity",   type=str,  default=BASE_CONFIG["ENTITY"])
    parser.add_argument("--project",  type=str,  default=BASE_CONFIG["PROJECT"])
    args = parser.parse_args()

    if not args.create and not args.run:
        parser.print_help()
        return

    sweep_id = args.sweep_id
    if args.create:
        sweep_id = create_sweep(args.entity, args.project)

    if args.run:
        if sweep_id is None:
            raise ValueError("Provide --sweep_id or use --create --run together.")
        full_sweep_id = f"{args.entity}/{args.project}/{sweep_id}"
        print(f"[sweep] Starting agent for {full_sweep_id}")
        wandb.agent(full_sweep_id, function=run_sweep_agent, count=args.count)


if __name__ == "__main__":
    main()
