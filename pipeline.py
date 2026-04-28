"""
Pipeline: OpenTargets disease queries → Ensembl→Entrez mapping
         → MSI construction → Diffusion profile computation
"""

import json
import requests

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

DISEASE_MAP = {
    "breast_carcinoma":           "EFO_0000305",
    "non_small_cell_lung_cancer": "EFO_0003060",
    "diabetes_type_2":            "EFO_0001360",
    "alzheimer_disease":          "EFO_0000249",
}

BFS_MAX_DEPTH = 4   # how many ontology levels to descend


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load the Multiscale Interactome (needed to filter disease tree)
# ─────────────────────────────────────────────────────────────────────────────

from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles

msi = MSI()
msi.load()

msi_indications = set(msi.indications_in_graph)
print(f"MSI loaded: {len(msi_indications)} indication nodes available")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BFS over OpenTargets disease ontology, filter to MSI nodes
# ─────────────────────────────────────────────────────────────────────────────

def get_disease_node_info(disease_id):
    """Returns name and dbXRefs for a single disease node."""
    formatted_id = disease_id.replace(":", "_")
    query = """
    query DiseaseInfo($id: String!) {
      disease(efoId: $id) { id  name  dbXRefs }
    }
    """
    response = requests.post(OT_URL, json={"query": query, "variables": {"id": formatted_id}})
    return response.json().get("data", {}).get("disease")


def get_disease_subtypes(disease_id):
    """Returns immediate child disease nodes (with dbXRefs) from OpenTargets."""
    formatted_id = disease_id.replace(":", "_")
    query = """
    query DiseaseSubtypes($id: String!) {
      disease(efoId: $id) {
        children { id  name  dbXRefs }
      }
    }
    """
    response = requests.post(OT_URL, json={"query": query, "variables": {"id": formatted_id}})
    disease_node = response.json().get("data", {}).get("disease")
    return disease_node.get("children", []) if disease_node else []


def get_disease_associated_genes(disease_id, size=50, min_score=0.0):
    """
    Returns genes associated with a disease from OpenTargets, ranked by overall
    association score. Covers genetic, somatic, expression, animal model, and
    pathway evidence — independent of drug targets.

    Args:
        disease_id: EFO/MONDO ID (e.g. 'EFO_0000305')
        size:       max number of gene-disease associations to return
        min_score:  filter out associations below this overall score (0–1)

    Returns:
        list of dicts with keys: ensembl_id, symbol, score, datatype_scores
    """
    formatted_id = disease_id.replace(":", "_")
    query = """
    query DiseaseAssociatedGenes($id: String!, $size: Int!) {
      disease(efoId: $id) {
        associatedTargets(page: {index: 0, size: $size}) {
          rows {
            target {
              id
              approvedSymbol
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    """
    response = requests.post(
        OT_URL,
        json={"query": query, "variables": {"id": formatted_id, "size": size}},
    )
    disease_node = response.json().get("data", {}).get("disease")
    if not disease_node:
        return []

    rows = disease_node.get("associatedTargets", {}).get("rows", [])
    results = []
    for row in rows:
        if row["score"] < min_score:
            continue
        results.append({
            "ensembl_id": row["target"]["id"],
            "symbol":     row["target"]["approvedSymbol"],
            "score":      row["score"],
            "datatype_scores": {
                d["id"]: d["score"]
                for d in row.get("datatypeScores", [])
            },
        })
    return results


def _umls_ids(db_x_refs):
    return [r.split(":")[1] for r in (db_x_refs or []) if r.startswith("UMLS")]


def bfs_disease_tree(root_efo_ids, max_depth=BFS_MAX_DEPTH):
    """
    BFS over the OpenTargets disease ontology.
    Returns {efo_id: {'name', 'umls_ids', 'depth'}} for every discovered node.
    """
    all_nodes = {}
    visited = set()
    queue = []

    # Seed with root nodes (query their own info since they're not returned as children)
    for root_id in root_efo_ids:
        info = get_disease_node_info(root_id)
        if info:
            all_nodes[root_id] = {
                "name": info["name"],
                "umls_ids": _umls_ids(info.get("dbXRefs")),
                "depth": 0,
            }
            queue.append((root_id, 0))

    while queue:
        current_id, depth = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        if depth >= max_depth:
            continue

        children = get_disease_subtypes(current_id)
        for child in children:
            child_id = child["id"]
            if child_id in all_nodes:
                continue
            all_nodes[child_id] = {
                "name": child["name"],
                "umls_ids": _umls_ids(child.get("dbXRefs")),
                "depth": depth + 1,
            }
            queue.append((child_id, depth + 1))

    return all_nodes


print(f"\nTraversing disease ontology (max depth {BFS_MAX_DEPTH})...")
all_disease_nodes = bfs_disease_tree(list(DISEASE_MAP.values()))
print(f"Discovered {len(all_disease_nodes)} disease nodes total")

# Filter to nodes that have a UMLS ID present in the MSI
msi_matched = {}
for efo_id, info in all_disease_nodes.items():
    for umls_id in info["umls_ids"]:
        if umls_id in msi_indications:
            msi_matched[efo_id] = {**info, "umls_id": umls_id}
            break

print(f"Matched {len(msi_matched)} disease nodes to MSI indication codes:")
for efo_id, info in sorted(msi_matched.items(), key=lambda x: x[1]["depth"]):
    print(f"  depth={info['depth']}  {info['umls_id']}  {info['name']}  ({efo_id})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Query drug/target data from OpenTargets for MSI-matched diseases
# ─────────────────────────────────────────────────────────────────────────────

def get_msi_data(disease_id):
    """Returns clinical drug candidates and mechanisms for a disease."""
    formatted_id = disease_id.replace(":", "_")
    query = """
    query DiseaseMsiData($id: String!) {
      disease(efoId: $id) {
        id  name  dbXRefs
        drugAndClinicalCandidates {
          rows {
            drug {
              id  name
              crossReferences { source  ids }
              mechanismsOfAction {
                rows { targets { id  approvedSymbol } }
              }
            }
          }
        }
      }
    }
    """
    response = requests.post(OT_URL, json={"query": query, "variables": {"id": formatted_id}})
    return response.json()


def extract_msi_records(data):
    """Parses an OpenTargets response into an MSI-compatible record."""
    if not data.get("data") or not data["data"].get("disease"):
        return None

    disease_node = data["data"]["disease"]
    umls_ids = [i.split(":")[1] for i in disease_node.get("dbXRefs", []) if i.startswith("UMLS")]
    if not umls_ids:
        return None

    drug_info = {}
    for row in disease_node.get("drugAndClinicalCandidates", {}).get("rows", []):
        drug_entry = row.get("drug")
        if not drug_entry:
            continue
        db_refs = [ref["ids"] for ref in drug_entry.get("crossReferences", []) if ref["source"] == "drugbank"]
        if not db_refs or not db_refs[0]:
            continue
        primary_dbid = db_refs[0][0]

        target_ids = []
        if drug_entry.get("mechanismsOfAction"):
            for mech in drug_entry["mechanismsOfAction"].get("rows", []):
                for target in mech.get("targets", []):
                    target_ids.append(target["id"])

        drug_info[primary_dbid] = {
            "drug_name": drug_entry["name"],
            "target_ensembl_ids": list(set(target_ids)),
        }

    return {
        "umls_id": umls_ids[0],
        "disease_name": disease_node["name"],
        "drugs": drug_info,
    }


msi_disease_data = {}

for efo_id, info in msi_matched.items():
    record = extract_msi_records(get_msi_data(efo_id))
    if record:
        msi_disease_data[record["umls_id"]] = {
            "name": record["disease_name"],
            "efo_id": efo_id,
            "depth": info["depth"],
            "drug_targets": record["drugs"],
        }

with open("msi_disease_data.json", "w") as f:
    json.dump(msi_disease_data, f, indent=4)

print(f"\nSaved {len(msi_disease_data)} disease records to msi_disease_data.json")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Map Ensembl gene IDs to Entrez IDs via BioMart
# ─────────────────────────────────────────────────────────────────────────────

from pybiomart import Dataset


def map_ensembl_to_entrez(ensembl_ids):
    """Maps Ensembl Gene IDs to Entrez IDs using BioMart."""
    dataset = Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
    clean_ids = [str(i).split(".")[0] for i in ensembl_ids]
    results = dataset.query(
        attributes=["ensembl_gene_id", "entrezgene_id"],
        filters={"link_ensembl_gene_id": clean_ids},
    )
    entrez_col = "NCBI gene (formerly Entrezgene) ID"
    results = results.dropna(subset=[entrez_col])
    results = results.rename(columns={"Gene stable ID": "ensembl_gene_id", entrez_col: "entrezgene_id"})
    results["entrezgene_id"] = results["entrezgene_id"].astype(int)
    return results[["ensembl_gene_id", "entrezgene_id"]]


all_ensembl_ids = list({
    eid
    for disease_info in msi_disease_data.values()
    for drug_info in disease_info["drug_targets"].values()
    for eid in drug_info.get("target_ensembl_ids", [])
})

print(f"Mapping {len(all_ensembl_ids)} unique Ensembl IDs to Entrez...")
ensembl_entrez_df = map_ensembl_to_entrez(all_ensembl_ids)
ensembl_to_entrez = dict(zip(ensembl_entrez_df["ensembl_gene_id"], ensembl_entrez_df["entrezgene_id"]))

for disease_info in msi_disease_data.values():
    for drug_info in disease_info["drug_targets"].values():
        drug_info["target_entrez_ids"] = [
            ensembl_to_entrez[e]
            for e in drug_info.get("target_ensembl_ids", [])
            if e in ensembl_to_entrez
        ]

with open("msi_disease_data.json", "w") as f:
    json.dump(msi_disease_data, f, indent=4)

print("Updated msi_disease_data.json with Entrez IDs")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Compute diffusion profiles
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

dp = DiffusionProfiles(
    alpha=0.8595436247434408,
    max_iter=1000,
    tol=1e-06,
    weights={
        "down_biological_function": 4.4863053901688685,
        "indication":               3.541889556309463,
        "biological_function":      6.583155399238509,
        "up_biological_function":   2.09685000906964,
        "protein":                  4.396695660380823,
        "drug":                     3.2071696595616364,
    },
    num_cores=4,
    save_load_file_path="results/",
)

dp.calculate_diffusion_profiles(msi)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Load saved diffusion profiles and extract disease profiles
# ─────────────────────────────────────────────────────────────────────────────

dp_saved = DiffusionProfiles(
    alpha=None, max_iter=None, tol=None, weights=None, num_cores=None,
    save_load_file_path="results/",
)
msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

def get_weighted_diffusion_profile(weighted_nodes, dp_saved):
    """
    Computes a single diffusion profile as a normalised weighted sum of
    pre-computed node profiles.

    Args:
        weighted_nodes: {node_id: weight} — node IDs must exist in
                        dp_saved.drug_or_indication2diffusion_profile
        dp_saved:       loaded DiffusionProfiles object

    Returns:
        np.ndarray of shape (n_nodes,), or None if no node had a profile
    """
    profile = None
    total_weight = 0.0

    for node_id, weight in weighted_nodes.items():
        if node_id not in dp_saved.drug_or_indication2diffusion_profile:
            continue
        node_profile = dp_saved.drug_or_indication2diffusion_profile[node_id]
        profile = weight * node_profile if profile is None else profile + weight * node_profile
        total_weight += abs(weight)

    if profile is None or total_weight == 0:
        return None
    return profile / total_weight


def get_weighted_diffusion_profiles(weighted_node_sets, dp_saved):
    """
    Computes diffusion profiles for multiple weighted node sets.

    Args:
        weighted_node_sets: {label: {node_id: weight}}
        dp_saved:           loaded DiffusionProfiles object

    Returns:
        {label: np.ndarray} — labels with no resolvable nodes are omitted
    """
    results = {}
    for label, weighted_nodes in weighted_node_sets.items():
        covered = [n for n in weighted_nodes if n in dp_saved.drug_or_indication2diffusion_profile]
        missing  = [n for n in weighted_nodes if n not in dp_saved.drug_or_indication2diffusion_profile]
        if missing:
            print(f"  [{label}] skipping {len(missing)} node(s) with no profile: {missing[:5]}")
        profile = get_weighted_diffusion_profile(weighted_nodes, dp_saved)
        if profile is not None:
            results[label] = profile
        else:
            print(f"  [{label}] no diffusion profiles found for any node — skipped")
    return results


def node_relevance_and_specificity(profile_dict, nodelist=None):
    """
    Two complementary information-theoretic measures over a dict of diffusion
    profile vectors {label: np.ndarray}.

    Let M(v) = mean_k P_k(v) be the mixture distribution across all profiles.

    A. Relevance(v) = M(v) · log(N · M(v))
       Node v's contribution to KL(mixture ∥ uniform).
       > 0  →  v is visited more than chance across all profiles (relevant)
       = 0  →  v is at the uniform baseline
       < 0  →  v is suppressed on average

    B. Specificity(v, k) = log(P_k(v) / M(v))
       Pointwise mutual information of node v with label k.
       > 0  →  v is enriched in profile k relative to the mixture
       = 0  →  v behaves the same in k as on average
       < 0  →  v is depleted in profile k

    Complementarity:
       Σ_v P_k(v) · specificity(v, k) = KL(P_k ∥ M)
    The KL divergence of each profile from the mixture equals the
    P_k-weighted mean of per-node specificities — so a node's specificity
    score is its contribution to that profile's "distinctiveness."

    Args:
        profile_dict: {label: np.ndarray of shape (n_nodes,)}
        nodelist:     optional list of node IDs to use as the output index

    Returns:
        relevance:    pd.Series  (n_nodes,)          — measure A
        specificity:  pd.DataFrame (n_nodes, n_labels) — measure B
    """
    labels = list(profile_dict.keys())
    P = np.stack([profile_dict[l] for l in labels])  # (K, N)
    K, N = P.shape

    M = P.mean(axis=0)  # (N,) mixture

    eps = np.finfo(float).tiny

    # A. Relevance — each node's contribution to KL(M ∥ uniform)
    relevance = M * np.log(N * np.maximum(M, eps))

    # B. Specificity — PMI of each (node, label) pair
    spec = np.log(np.maximum(P, eps) / np.maximum(M, eps)[np.newaxis, :])  # (K, N)

    index = nodelist if nodelist is not None else np.arange(N)
    return (
        pd.Series(relevance, index=index, name="relevance"),
        pd.DataFrame(spec.T, index=index, columns=labels),
    )


disease_profiles = {}
for umls_id, info in msi_disease_data.items():
    if umls_id in dp_saved.drug_or_indication2diffusion_profile:
        disease_profiles[umls_id] = dp_saved.drug_or_indication2diffusion_profile[umls_id]
    else:
        print(f"No diffusion profile found for {umls_id} ({info['name']})")

print(f"\nRetrieved diffusion profiles for {len(disease_profiles)} / {len(msi_disease_data)} diseases:")
for umls_id, profile in disease_profiles.items():
    print(f"  {umls_id} — {msi_disease_data[umls_id]['name']}: shape={profile.shape}")

disease_profiles_df = pd.DataFrame(disease_profiles)
print(disease_profiles_df)
