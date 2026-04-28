#!/home/joel/miniforge3/envs/snakemake/bin/python
"""
Precomputes a simplex embedding for simplex_3d.py and saves it to disk.

Usage (from project root):
    python interactibles/precompute_simplex.py --name diverse_k5
    python interactibles/precompute_simplex.py --name nsclc
    python interactibles/precompute_simplex.py --name my_set --config my_set.json

Output:  results/simplex/{name}.npz   — compressed arrays  (~1 MB)
         results/simplex/{name}.json  — human-readable metadata

--config accepts a JSON file of the form:
    {"Label 1": "UMLS_ID_1", "Label 2": "UMLS_ID_2", ...}
"""

import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from diff_prof.diffusion_profiles import DiffusionProfiles
from msi.msi import MSI

# ── Named disease sets ────────────────────────────────────────────────────────

DISEASE_SETS = {

    "diverse_k5": {
        "Breast Carcinoma":     "C0678222",
        "NSCLC":                "C0007131",
        "Alzheimer's":          "C0002395",
        "Type 2 Diabetes":      "C0011860",
        "Rheumatoid Arthritis": "C0003873",
    },

    "nsclc": {
        "NSCLC":                   "C0007131",   # parent
        "Small Cell Lung Cancer":  "C0149925",   # SCLC — contrast
        "Lung Neoplasms":          "C0024121",   # broader lung cancer
        "Adenocarcinoma":          "C0001418",   # NSCLC histotype 1 (~40%)
        "Squamous Cell Carcinoma": "C0007137",   # NSCLC histotype 2 (~25%)
    },

}

# ─────────────────────────────────────────────────────────────────────────────

def precompute(profiles_map, name, out_dir="results/simplex"):
    t0 = time.time()

    print("Loading MSI...")
    msi = MSI()
    msi.load()

    print("Loading diffusion profiles...")
    dp = DiffusionProfiles(
        alpha=None, max_iter=None, tol=None, weights=None,
        num_cores=None, save_load_file_path="results/",
    )
    msi.load_saved_node_idx_mapping_and_nodelist(dp.save_load_file_path)
    dp.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

    p        = dp.drug_or_indication2diffusion_profile
    labels   = list(profiles_map.keys())
    K        = len(labels)
    node_ids = list(msi.nodelist)
    N        = len(node_ids)

    # Validate IDs
    missing = [v for v in profiles_map.values() if v not in p]
    if missing:
        raise ValueError(f"These IDs have no saved diffusion profile: {missing}")

    print(f"Computing barycentric embedding  (N={N}, K={K})...")
    P_mat = np.stack([p[v] for v in profiles_map.values()], axis=1)  # (N, K)
    W     = P_mat / np.maximum(P_mat.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    W_c   = (W - 1.0 / K).astype(np.float32)

    eps       = np.finfo(float).tiny
    M         = P_mat.mean(axis=1)
    relevance = (M * np.log(N * np.maximum(M, eps))).astype(np.float32)
    dominant  = np.argmax(W, axis=1).astype(np.int8)

    os.makedirs(out_dir, exist_ok=True)
    stem      = os.path.join(out_dir, name)
    npz_path  = stem + ".npz"
    json_path = stem + ".json"

    np.savez_compressed(npz_path, W_c=W_c, relevance=relevance, dominant=dominant)

    meta = {
        "name":       name,
        "labels":     labels,
        "profiles":   profiles_map,
        "node_ids":   node_ids,
        "node_names": [msi.node2name.get(n, str(n)) for n in node_ids],
        "node_types": [msi.node2type.get(n, "?")    for n in node_ids],
        "K":          K,
        "N":          N,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {npz_path}   ({os.path.getsize(npz_path)/1e6:.1f} MB)")
    print(f"Saved {json_path}  ({os.path.getsize(json_path)/1e6:.1f} MB)")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",   required=True,
                        help=f"Disease set name. Built-in: {list(DISEASE_SETS)}")
    parser.add_argument("--config", default=None,
                        help="JSON file with {{label: umls_id}} to override built-in set")
    parser.add_argument("--out-dir", default="results/simplex")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            profiles_map = json.load(f)
    elif args.name in DISEASE_SETS:
        profiles_map = DISEASE_SETS[args.name]
    else:
        raise SystemExit(f"Unknown name '{args.name}'. "
                         f"Built-ins: {list(DISEASE_SETS)}. "
                         f"Use --config to supply a custom set.")

    precompute(profiles_map, args.name, args.out_dir)
