"""
W&B Sweep for fcp_general.py hyperparameter search.
Sweeps over: LR, VF_COEF, NUM_MINIBATCHES, ENT_COEF, layout

FCP requires frozen partner params loaded from disk before training.
Provide --ckpt_root pointing to your saved IPPO/SP checkpoints, e.g.:
  ckpt_root = "our_ckpts"   (expects our_ckpts/<layout>/ikFalse/reset_all/seed<s>/)

Usage (run from repo root):
  python -m baselines.CEC.sweep.sweep_fcp --create --ckpt_root our_ckpts
  python -m baselines.CEC.sweep.sweep_fcp --run --sweep_id <ID> --ckpt_root our_ckpts
  python -m baselines.CEC.sweep.sweep_fcp --create --run --ckpt_root our_ckpts
"""
import copy
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import wandb

SWEEP_CONFIG = {
    "method": "bayes",   # "grid" | "random" | "bayes"
    "metric": {
        "name": "returns",
        "goal": "maximize",
    },
    "parameters": {
        "LR": {
            "values": [1e-4, 3e-4, 1e-3],
        },
        "VF_COEF": {
            "values": [0.25, 0.5],
        },
        "NUM_MINIBATCHES": {
            "values": [4, 8, 16],
        },
        "ENT_COEF": {
            "values": [0.005, 0.01],
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


# mirrors repro_config/fcp_final_baseline.yaml
BASE_CONFIG = {
    "ENV_NAME": "overcooked",
    "LR": 3e-4,
    "NUM_ENVS": 256,
    "NUM_SEEDS": 1,
    "NUM_STEPS": 400,
    "FC_DIM_SIZE": 256,
    "GRU_HIDDEN_DIM": 256,
    "TOTAL_TIMESTEPS": 3e9,
    "MAX_TRAIN_STEPS": 3e9,
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
        "layout": "coord_ring_9",
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
    "FCP_KWARGS": {"train_oracle": False},
    "ENTITY": "overcooked_ai",
    "PROJECT": "crossenv_ued_test",
    "WANDB_MODE": "online",
    "model_name": "FCP",
}

# Checkpoint IDs and matching update steps used when loading frozen SP partners.
# Mirrors the logic in fcp_general.py main().
CKPT_ID_LIST      = [0, 10, 19]
UPDATE_STEP_LIST  = [1204, 13244, 22887]
SEED_LIST         = range(6)


def load_frozen_params(layout: str, ckpt_root: str) -> tuple:
    """Load all available SP partner checkpoints for the given layout.

    Returns (frozen_param_stack, num_stacked_params).
    """
    custom_path = os.path.join(ckpt_root, layout, "ikFalse", "reset_all")
    frozen_params = []
    for ckpt_id, update_step in zip(CKPT_ID_LIST, UPDATE_STEP_LIST):
        for seed in SEED_LIST:
            candidate = os.path.join(
                custom_path, f"seed{seed}", f"seed{seed}_ckpt{ckpt_id}_update{update_step}.pkl"
            )
            if os.path.exists(candidate):
                with open(candidate, "rb") as f:
                    ckpt = pickle.load(f)
                    frozen_params.append(ckpt["params"])

    if len(frozen_params) == 0:
        raise FileNotFoundError(
            f"No frozen partner checkpoints found under {custom_path}.\n"
            f"Expected files like: seed<s>/seed<s>_ckpt<c>_update<u>.pkl"
        )

    frozen_param_stack = jax.tree.map(lambda *x: jnp.stack(x), *frozen_params)
    return frozen_param_stack, len(frozen_params)


# Module-level storage so run_sweep_agent can access ckpt_root without argparse
_ckpt_root: str = "our_ckpts"


def run_sweep_agent():
    from baselines.CEC.fcp_general import make_train

    run = wandb.init()
    sp = dict(wandb.config)

    config = copy.deepcopy(BASE_CONFIG)

    # flat params
    config["LR"]              = sp["LR"]
    config["VF_COEF"]         = sp["VF_COEF"]
    config["NUM_MINIBATCHES"] = sp["NUM_MINIBATCHES"]
    config["ENT_COEF"]        = sp["ENT_COEF"]

    # nested params
    config["ENV_KWARGS"]["layout"] = sp["layout"]

    layout_name = sp["layout"]
    xpid = "sweep-%s" % time.strftime("%Y%m%d-%H%M%S")
    filepath = (
        f"ckpts/fcp/{config['ENV_NAME']}/{layout_name}"
        f"/ik{config['ENV_KWARGS']['random_reset']}"
        f"/{config['ENV_KWARGS']['random_reset_fn']}/{xpid}"
    )
    os.makedirs(filepath, exist_ok=True)

    run.name = (
        f"{layout_name}"
        f"_lr{sp['LR']:.0e}_vf{sp['VF_COEF']}"
        f"_mb{sp['NUM_MINIBATCHES']}_ent{sp['ENT_COEF']:.3f}"
    )

    print(
        f"[sweep/fcp] layout={layout_name}\n"
        f"            LR={sp['LR']}  VF_COEF={sp['VF_COEF']}\n"
        f"            NUM_MINIBATCHES={sp['NUM_MINIBATCHES']}  ENT_COEF={sp['ENT_COEF']}\n"
        f"            ckpt dir: {filepath}"
    )

    frozen_param_stack, num_stacked_params = load_frozen_params(layout_name, _ckpt_root)
    print(f"[sweep/fcp] loaded {num_stacked_params} frozen partner params")

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config, 0), device=jax.devices()[0])
    out = train_jit(rng, frozen_param_stack, None, 0, num_stacked_params)

    # fcp_general.make_train has no filepath arg — save checkpoint here
    runner_state = out["runner_state"]
    train_state = runner_state[0]
    rng_out = runner_state[-1]
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    with open(f"{filepath}/seed{config['SEED']}_ckpt0_fcp_updates{num_updates}.pkl", "wb") as f:
        pickle.dump({"key": rng_out, "params": train_state.params, "update_steps": num_updates}, f)


def create_sweep(entity: str, project: str) -> str:
    sweep_id = wandb.sweep(SWEEP_CONFIG, entity=entity, project=project)
    print(f"[sweep/fcp] Created sweep: {sweep_id}")
    print(f"[sweep/fcp] View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    return sweep_id


def main():
    global _ckpt_root

    parser = argparse.ArgumentParser()
    parser.add_argument("--create",    action="store_true")
    parser.add_argument("--run",       action="store_true")
    parser.add_argument("--sweep_id",  type=str, default=None)
    parser.add_argument("--count",     type=int, default=None)
    parser.add_argument("--entity",    type=str, default=BASE_CONFIG["ENTITY"])
    parser.add_argument("--project",   type=str, default=BASE_CONFIG["PROJECT"])
    parser.add_argument(
        "--ckpt_root", type=str, default="our_ckpts",
        help="Root directory of frozen SP partner checkpoints "
             "(expects <root>/<layout>/ikFalse/reset_all/seed<s>/)",
    )
    args = parser.parse_args()

    _ckpt_root = args.ckpt_root

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
        print(f"[sweep/fcp] Starting agent for {full_sweep_id}")
        wandb.agent(full_sweep_id, function=run_sweep_agent, count=args.count)


if __name__ == "__main__":
    main()
