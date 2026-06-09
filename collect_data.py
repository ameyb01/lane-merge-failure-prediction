"""
collect_data.py
===============
Collect latent states and hand-crafted features from DreamerV3 episodes
of the carla_workzone_merge_complex task.

Features:
- Saves every 10 episodes — crash safe
- Resumes from last saved episode if run again
- Final save on completion or crash via atexit

Usage:
    python collect_data.py \
        --checkpoint ./logdir/carla_workzone_merge_complex_120k/checkpoint.ckpt \
        --task carla_workzone_merge_complex \
        --num_episodes 500 \
        --output_dir ./data/workzone_complex
"""

import argparse
import atexit
import os
import warnings
import numpy as np
import ruamel.yaml as yaml

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

import embodied
import car_dreamer
import dreamerv3


# =========================================================
# Feature keys — 16 infrastructure-observable features
# =========================================================
FEATURE_KEYS = [
    "ego_x",               # lateral position
    "ego_y",               # longitudinal position
    "vy",                  # forward velocity
    "speed_norm",          # total speed
    "lateral_change_rate", # d(ego_x)/dt — merge commitment signal
    "dist_to_closure",     # distance to Cybertruck
    "gap_size",            # space between bg1 and closure
    "gap_closing_rate",    # how fast gap_size is changing
    "bg_y",                # bg1 position (lead vehicle, ahead)
    "bg_speed",            # bg1 speed
    "gap12_size",          # actual merge gap between bg2 and bg1
    "gap12_closing_rate",  # how fast merge gap is closing
    "bg2_y",               # bg2 position (following vehicle, behind)
    "bg2_speed",           # bg2 speed
    "follower_dist",       # distance from follower to ego
    "follower_speed",      # follower speed — pressure magnitude
]


# =========================================================
# Save / Load helpers
# =========================================================
def save(output_dir, all_latents, all_features, all_labels, all_lengths):
    """Save current progress to disk. Overwrites previous save."""
    if len(all_labels) == 0:
        return
    N       = len(all_labels)
    max_len = max(len(s) for s in all_latents)
    lat_dim = len(all_latents[0][0])  # auto-detect from actual latent size
    feat_dim = len(FEATURE_KEYS)

    latents_arr  = np.zeros((N, max_len, lat_dim),  dtype=np.float32)
    features_arr = np.zeros((N, max_len, feat_dim), dtype=np.float32)

    for i, (lats, feats) in enumerate(zip(all_latents, all_features)):
        T = min(len(lats), max_len)
        if T > 0:
            latents_arr[i, :T] = np.stack(lats[:T])
        T2 = min(len(feats), max_len)
        if T2 > 0:
            features_arr[i, :T2] = feats[:T2]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "latents.npy"),  latents_arr)
    np.save(os.path.join(output_dir, "features.npy"), features_arr)
    np.save(os.path.join(output_dir, "labels.npy"),   np.array(all_labels,  dtype=np.int32))
    np.save(os.path.join(output_dir, "lengths.npy"),  np.array(all_lengths, dtype=np.int32))

    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        for name in FEATURE_KEYS:
            f.write(name + "\n")

    print(f"[SAVED] {N} episodes saved to {output_dir}")


def load_existing(output_dir):
    """
    Load previously saved episodes if they exist.
    Returns (all_latents, all_features, all_labels, all_lengths)
    or empty lists if nothing saved yet.
    """
    labels_path = os.path.join(output_dir, "labels.npy")
    if not os.path.exists(labels_path):
        print("No existing data found — starting fresh.")
        return [], [], [], []

    labels  = np.load(os.path.join(output_dir, "labels.npy")).tolist()
    lengths = np.load(os.path.join(output_dir, "lengths.npy")).tolist()
    latents_arr  = np.load(os.path.join(output_dir, "latents.npy"))
    features_arr = np.load(os.path.join(output_dir, "features.npy"))

    all_latents  = []
    all_features = []
    for i, L in enumerate(lengths):
        L = int(L)
        all_latents.append(list(latents_arr[i, :L]))
        all_features.append(features_arr[i, :L])

    print(f"[RESUME] Loaded {len(labels)} existing episodes from {output_dir}")
    return all_latents, all_features, labels, [int(l) for l in lengths]


# =========================================================
# Helpers
# =========================================================
def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def extract_latent(state):
    (latent, _), _, _ = state
    deter = np.array(latent["deter"]).reshape(-1)
    stoch = np.array(latent["stoch"]).reshape(-1)
    return np.concatenate([deter, stoch])


def get_label(ep_info):
    if "success" in ep_info:
        vals = np.array(ep_info["success"]).flatten()
        if vals.any():
            return 1
    return 0


def extract_feature_matrix(ep_info, length):
    T = length
    F = len(FEATURE_KEYS)
    matrix = np.zeros((T, F), dtype=np.float32)
    for j, key in enumerate(FEATURE_KEYS):
        if key in ep_info:
            vals = np.array(ep_info[key]).flatten().astype(np.float32)
            t = min(T, len(vals))
            matrix[:t, j] = vals[:t]
    return matrix


def print_summary(output_dir, all_labels):
    N = len(all_labels)
    print(f"\n{'='*55}")
    print(f"  Data Collection Complete")
    print(f"{'='*55}")
    print(f"  Episodes : {N}")
    print(f"  Success  : {sum(all_labels)} ({sum(all_labels)/N*100:.1f}%)")
    print(f"  Failure  : {N-sum(all_labels)} ({(N-sum(all_labels))/N*100:.1f}%)")
    print(f"  Saved to : {output_dir}")
    print(f"{'='*55}")


# =========================================================
# Main
# =========================================================
def collect(args):
    # --- Load existing progress ---
    all_latents, all_features, all_labels, all_lengths = load_existing(args.output_dir)
    episodes_done = len(all_labels)
    episodes_remaining = args.num_episodes - episodes_done

    if episodes_remaining <= 0:
        print(f"Already have {episodes_done} episodes — target reached!")
        print_summary(args.output_dir, all_labels)
        return

    print(f"Need {episodes_remaining} more episodes (have {episodes_done}/{args.num_episodes})\n")

    # --- Config ---
    model_configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "dreamerv3/dreamerv3.yaml").read()
    )
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    argv = [f"--task={args.task}"]
    parsed, _ = embodied.Flags(task=[args.task]).parse_known(argv)
    for name in parsed.task:
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    config = embodied.Flags(config).parse([])

    dreamerv3_config = config.dreamerv3

    from embodied.envs import from_gym
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.checkpoint, keys=["agent"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Features ({len(FEATURE_KEYS)}): {FEATURE_KEYS}\n")

    # Register atexit — saves on crash or normal exit
    atexit.register(save, args.output_dir, all_latents, all_features, all_labels, all_lengths)

    episode_count = [episodes_done]
    current_latents = []

    def on_step(tran, inf, worker):
        step.increment()

    def on_episode(ep, ep_info, worker):
        label  = get_label(ep_info)
        length = len(ep["reward"]) - 1

        lats = current_latents[:]
        current_latents.clear()

        feat_matrix = extract_feature_matrix(ep_info, length)

        all_latents.append(lats)
        all_features.append(feat_matrix)
        all_labels.append(label)
        all_lengths.append(length)

        episode_count[0] += 1
        n = episode_count[0]
        outcome = "SUCCESS" if label == 1 else "FAILURE"
        print(f"Ep {n:3d}/{args.num_episodes} | len={length:4d} | {outcome} | "
              f"success={sum(all_labels)}/{n} ({sum(all_labels)/n*100:.1f}%)")

        # Save every 10 episodes
        if n % 10 == 0:
            save(args.output_dir, all_latents, all_features, all_labels, all_lengths)

    def policy(obs, state, mode="eval"):
        outs, new_state = agent.policy(obs, state, mode=mode)
        current_latents.append(extract_latent(new_state))
        return outs, new_state

    driver = embodied.Driver(env)
    driver.on_step(on_step)
    driver.on_episode(on_episode)

    print(f"Collecting {episodes_remaining} more episodes...\n")
    while episode_count[0] < args.num_episodes:
        driver(policy, steps=100)

    # Final save
    save(args.output_dir, all_latents, all_features, all_labels, all_lengths)
    print_summary(args.output_dir, all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--task",         type=str, default="carla_workzone_merge_complex")
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--output_dir",   type=str, default="./data/workzone_complex")
    args = parser.parse_args()
    collect(args)
