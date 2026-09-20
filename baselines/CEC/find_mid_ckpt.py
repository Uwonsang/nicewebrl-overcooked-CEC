"""
Post-processing script that converts ippo_general_population_v2.py checkpoints
into the init/mid/final format expected by fcp_general.py.

  ckpt_0        -> init
  ckpt closest to final/2 -> mid
  best (final)  -> final

Saves to a new output directory: <out_dir>/seed{seed}/seed{seed}_{init,mid,final}.pkl

Usage:
    python find_mid_ckpt.py --dir <ckpt_dir> --seed <seed> [--out <out_dir>]

Example:
    python find_mid_ckpt.py \\
        --dir ckpts/ippo/overcooked/cramped_room_9/ikTrue/random/lr-20240101-120000 \\
        --seed 0 \\
        --out ckpts/ippo/overcooked/cramped_room_9/fcp_pool
"""

import os
import glob
import pickle
import argparse


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def find_mid_ckpt(ckpt_dir: str, seed: int):
    # --- init: ckpt_0 or init.pkl ---
    init_path = os.path.join(ckpt_dir, f"seed{seed}_ckpt_0.pkl")
    if not os.path.exists(init_path):
        init_path = os.path.join(ckpt_dir, f"seed{seed}_init.pkl")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Init checkpoint not found (tried _ckpt_0.pkl and _init.pkl)")
    init_ckpt = load_pkl(init_path)
    print(f"Init  return: {init_ckpt.get('returns', 0):.2f}")

    # --- final: best ---
    final_path = os.path.join(ckpt_dir, f"seed{seed}_best.pkl")
    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Final checkpoint not found: {final_path}")
    final_ckpt = load_pkl(final_path)
    best_return = final_ckpt["returns"]
    print(f"Final return: {best_return:.2f}")

    # --- mid: ckpt closest to best/2 ---
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, f"seed{seed}_ckpt_*.pkl")))
    if not ckpt_files:
        raise FileNotFoundError(f"No ckpt files found in {ckpt_dir} for seed {seed}")

    ckpts = [(f, load_pkl(f)) for f in ckpt_files if "returns" in load_pkl(f)]
    mid_target = best_return / 2
    mid_file, mid_ckpt = min(ckpts, key=lambda fc: abs(fc[1]["returns"] - mid_target))
    print(f"Mid   target: {mid_target:.2f}  ->  {mid_ckpt['returns']:.2f}  ({os.path.basename(mid_file)})")

    # --- save to <ckpt_dir>/fcp_pool/seed{seed}/ ---
    out_dir = os.path.join(ckpt_dir, "fcp_pool")
    for name, ckpt in [("init", init_ckpt), ("mid", mid_ckpt), ("final", final_ckpt)]:
        out_path = os.path.join(out_dir, f"seed{seed}_{name}.pkl")
        save_pkl(out_path, ckpt)
        print(f"Saved {name} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing v2 checkpoints")
    parser.add_argument("--seed", type=int, required=True, help="Seed number")
    args = parser.parse_args()

    find_mid_ckpt(args.dir, args.seed)


if __name__ == "__main__":
    main()
