"""
merge_datasets.py
=================
Merge run1 (failures only) + run2 (all episodes) into one balanced dataset.

Run1: 413 success + 87 failure  → keep failures only (87)
Run2: 401 success + 99 failure  → keep all (500)
─────────────────────────────────────────────────────
Final: 401 success + 186 failure → 68/32 split
       587 total episodes

Usage:
    python merge_datasets.py \
        --run1 ./data/workzone_complex \
        --run2 ./data/workzone_complex_run2 \
        --output ./data/workzone_complex_merged
"""

import argparse
import os
import numpy as np


def load_dataset(data_dir):
    labels   = np.load(os.path.join(data_dir, "labels.npy"))
    lengths  = np.load(os.path.join(data_dir, "lengths.npy"))
    latents  = np.load(os.path.join(data_dir, "latents.npy"))
    features = np.load(os.path.join(data_dir, "features.npy"))
    print(f"[LOAD] {data_dir}: {len(labels)} episodes "
          f"(success={labels.sum()}, failure={(labels==0).sum()})")
    return latents, features, labels, lengths


def merge(run1_dir, run2_dir, output_dir):
    # Load both runs
    lat1, feat1, lab1, len1 = load_dataset(run1_dir)
    lat2, feat2, lab2, len2 = load_dataset(run2_dir)

    # From run1 keep only failures (label == 0)
    fail_idx = np.where(lab1 == 0)[0]
    lat1_f   = lat1[fail_idx]
    feat1_f  = feat1[fail_idx]
    lab1_f   = lab1[fail_idx]
    len1_f   = len1[fail_idx]
    print(f"[FILTER] Run1: keeping {len(fail_idx)} failures, "
          f"discarding {(lab1==1).sum()} successes")

    # From run2 keep everything
    print(f"[KEEP] Run2: keeping all {len(lab2)} episodes")

    # Pad to same max_len across both
    max_len  = max(lat1_f.shape[1], lat2.shape[1])

    def pad(arr, max_len):
        N, T, D = arr.shape
        if T == max_len:
            return arr
        padded = np.zeros((N, max_len, D), dtype=arr.dtype)
        padded[:, :T, :] = arr
        return padded

    lat1_f  = pad(lat1_f,  max_len)
    feat1_f = pad(feat1_f, max_len)
    lat2    = pad(lat2,    max_len)
    feat2   = pad(feat2,   max_len)

    # Concatenate run1 failures + all of run2
    latents_final  = np.concatenate([lat1_f,  lat2],  axis=0)
    features_final = np.concatenate([feat1_f, feat2], axis=0)
    labels_final   = np.concatenate([lab1_f,  lab2],  axis=0)
    lengths_final  = np.concatenate([len1_f,  len2],  axis=0)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "latents.npy"),  latents_final)
    np.save(os.path.join(output_dir, "features.npy"), features_final)
    np.save(os.path.join(output_dir, "labels.npy"),   labels_final)
    np.save(os.path.join(output_dir, "lengths.npy"),  lengths_final)

    # Copy feature names from run1
    src = os.path.join(run1_dir, "feature_names.txt")
    dst = os.path.join(output_dir, "feature_names.txt")
    if os.path.exists(src):
        with open(src) as f:
            names = f.read()
        with open(dst, "w") as f:
            f.write(names)

    N = len(labels_final)
    S = int(labels_final.sum())
    F = int((labels_final == 0).sum())

    print(f"\n{'='*55}")
    print(f"  Merged Dataset")
    print(f"{'='*55}")
    print(f"  Total episodes : {N}")
    print(f"  Success        : {S} ({S/N*100:.1f}%)")
    print(f"  Failure        : {F} ({F/N*100:.1f}%)")
    print(f"  Shape latents  : {latents_final.shape}")
    print(f"  Shape features : {features_final.shape}")
    print(f"  Saved to       : {output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1",   type=str, default="./data/workzone_complex")
    parser.add_argument("--run2",   type=str, default="./data/workzone_complex_run2")
    parser.add_argument("--output", type=str, default="./data/workzone_complex_merged")
    args = parser.parse_args()
    merge(args.run1, args.run2, args.output)
