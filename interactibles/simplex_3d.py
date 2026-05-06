#!/usr/bin/env python3
"""
Interactive 3D simplex viewer for MSI diffusion profiles.

Run:
    python interactibles/simplex_3d.py
then open http://127.0.0.1:8050 in a browser.

Controls:
  - Dataset dropdown: switch between precomputed simplex files
  - Drag the 3D plot to rotate
  - Relevance slider: hide nodes below a chosen percentile threshold
  - "New projection" button: sample a fresh random 3D view of the 4-simplex
"""

import sys, os, json, glob, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

PALETTE      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
DEFAULT_STEM = "results/simplex/diverse_k5"
SIMPLEX_DIR  = "results/simplex"


def discover_stems(directory=SIMPLEX_DIR):
    stems = []
    for f in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        stem = f[:-4]
        if os.path.exists(stem + ".json"):
            stems.append(stem)
    return stems

AVAILABLE_STEMS = discover_stems()

# ── Data loading ──────────────────────────────────────────────────────────────

def load_precomputed(stem):
    npz_path  = stem + ".npz"
    json_path = stem + ".json"
    if not (os.path.exists(npz_path) and os.path.exists(json_path)):
        return None
    t0 = time.time()
    arrays = np.load(npz_path)
    with open(json_path) as f:
        meta = json.load(f)
    print(f"Loaded precomputed data from {stem}.*  ({time.time()-t0:.1f}s)")
    return {**meta, **{k: arrays[k] for k in arrays}}


def compute_from_scratch():
    from diff_prof.diffusion_profiles import DiffusionProfiles
    from msi.msi import MSI

    print("No precomputed data found — loading MSI and profiles (slow)...")
    print("Run  python interactibles/precompute_simplex.py  to cache this.")

    msi = MSI()
    msi.load()
    dp = DiffusionProfiles(alpha=None, max_iter=None, tol=None, weights=None,
                           num_cores=None, save_load_file_path="results/")
    msi.load_saved_node_idx_mapping_and_nodelist(dp.save_load_file_path)
    dp.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

    from interactibles.precompute_simplex import PROFILES_MAP
    p        = dp.drug_or_indication2diffusion_profile
    labels   = list(PROFILES_MAP.keys())
    K        = len(labels)
    node_ids = list(msi.nodelist)
    N        = len(node_ids)

    P_mat = np.stack([p[v] for v in PROFILES_MAP.values()], axis=1)
    W     = P_mat / np.maximum(P_mat.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    W_c   = (W - 1.0 / K).astype(np.float32)
    eps   = np.finfo(float).tiny
    M     = P_mat.mean(axis=1)
    return {
        "labels":     labels,
        "node_ids":   node_ids,
        "node_names": [msi.node2name.get(n, str(n)) for n in node_ids],
        "node_types": [msi.node2type.get(n, "?")    for n in node_ids],
        "W_c":        W_c,
        "relevance":  (M * np.log(N * np.maximum(M, eps))).astype(np.float32),
        "dominant":   np.argmax(W, axis=1).astype(np.int8),
        "K": K, "N": N,
    }


_DATA_CACHE = {}

def get_data(stem):
    if stem not in _DATA_CACHE:
        d = load_precomputed(stem)
        if d is None:
            raise FileNotFoundError(f"No precomputed data for {stem}")
        _DATA_CACHE[stem] = d
    return _DATA_CACHE[stem]


# Parse --data stem before Dash sees argv
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data", default=DEFAULT_STEM)
_parser.add_argument("--port", type=int, default=None)
_args, _remaining = _parser.parse_known_args()

# Pre-load default (falls back to compute_from_scratch if needed)
_DATA_CACHE[_args.data] = load_precomputed(_args.data) or compute_from_scratch()

eps = np.finfo(float).tiny

# ── Projection ────────────────────────────────────────────────────────────────

def make_projection(seed, stem):
    d   = get_data(stem)
    W_c = d["W_c"]
    K   = d["K"]
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((K, 3)))
    Q = Q[:, :min(3, K)]
    if Q.shape[1] < 3:
        Q = np.hstack([Q, np.zeros((K, 3 - Q.shape[1]))])
    return (W_c @ Q).astype(np.float32)                    # (N, 3)


# ── Dash app ──────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="MSI Simplex")

SLIDER_MARKS = {i: f"p{i}" for i in range(0, 100, 10)}
STEM_OPTIONS = [{"label": os.path.basename(s), "value": s} for s in AVAILABLE_STEMS]

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "backgroundColor": "#f8f8f8",
           "padding": "16px", "maxWidth": "1100px", "margin": "0 auto"},
    children=[

        html.H3("MSI Diffusion Profile Simplex",
                style={"textAlign": "center", "marginBottom": "4px"}),
        html.P(
            "Nodes embedded in the 4-simplex by barycentric coordinates of their"
            " diffusion profiles, projected to 3-D. Colour = dominant disease."
            " Size = overall relevance.",
            style={"textAlign": "center", "color": "#555", "marginTop": 0,
                   "fontSize": "13px"},
        ),

        # ── Controls row ─────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "24px",
                   "marginBottom": "12px"},
            children=[
                html.Div(
                    style={"minWidth": "180px"},
                    children=[
                        html.Label("Dataset", style={"fontSize": "13px"}),
                        dcc.Dropdown(
                            id="data-stem",
                            options=STEM_OPTIONS,
                            value=_args.data,
                            clearable=False,
                            style={"fontSize": "13px"},
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Minimum relevance (percentile)",
                                   style={"fontSize": "13px"}),
                        dcc.Slider(
                            id="rel-slider",
                            min=0, max=99, step=1, value=90,
                            marks=SLIDER_MARKS,
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                        ),
                    ],
                ),
                html.Button(
                    "New projection",
                    id="new-proj-btn",
                    n_clicks=0,
                    style={"padding": "8px 18px", "cursor": "pointer",
                           "borderRadius": "6px", "border": "1px solid #aaa",
                           "backgroundColor": "#fff", "fontSize": "13px"},
                ),
            ],
        ),

        html.Div(id="status-bar",
                 style={"textAlign": "center", "color": "#666",
                        "fontSize": "12px", "marginBottom": "6px"}),

        # ── Plot ─────────────────────────────────────────────────────────────
        dcc.Graph(
            id="simplex-graph",
            style={"height": "70vh"},
            config={"scrollZoom": True},
        ),

        # ── Legend (dynamic, reflects current dataset labels) ─────────────────
        html.Div(id="legend-div",
                 style={"textAlign": "center", "marginTop": "8px"}),

        # ── Hidden stores ─────────────────────────────────────────────────────
        dcc.Store(id="proj-store",
                  data=make_projection(42, _args.data).tolist()),
    ],
)

# ── Callback: regenerate projection on button click or dataset change ─────────

@app.callback(
    Output("proj-store", "data"),
    Input("new-proj-btn", "n_clicks"),
    Input("data-stem", "value"),
    prevent_initial_call=True,
)
def reshuffle_projection(n_clicks, stem):
    if ctx.triggered_id == "data-stem":
        return make_projection(42, stem).tolist()
    return make_projection(n_clicks, stem).tolist()


# ── Callback: redraw figure when slider, projection, or dataset changes ────────

@app.callback(
    Output("simplex-graph", "figure"),
    Output("status-bar", "children"),
    Output("legend-div", "children"),
    Input("rel-slider", "value"),
    Input("proj-store", "data"),
    State("data-stem", "value"),
)
def update_figure(pct, proj_data, stem):
    d          = get_data(stem)
    labels     = d["labels"]
    K          = d["K"]
    N          = d["N"]
    node_ids   = d["node_ids"]
    node_names = d["node_names"]
    node_types = d["node_types"]
    relevance  = d["relevance"]
    dominant   = d["dominant"]

    proj = np.array(proj_data, dtype=np.float32)

    threshold = np.percentile(relevance, pct)
    mask      = relevance >= threshold

    rel_shown  = relevance[mask]
    dom_shown  = dominant[mask]
    proj_shown = proj[mask]
    idx_shown   = np.where(mask)[0]
    ids_shown   = [node_ids[i]   for i in idx_shown]
    names_shown = [node_names[i] for i in idx_shown]
    types_shown = [node_types[i] for i in idx_shown]

    upper     = np.percentile(rel_shown, 99) if rel_shown.size else 1.0
    rel_norm  = np.clip(rel_shown, 0, upper) / (upper + eps)
    sizes_all = (2 + 12 * rel_norm).astype(np.float32)

    traces = []
    for k, label in enumerate(labels):
        kidx = dom_shown == k
        if not kidx.any():
            continue

        where_k  = np.where(kidx)[0]
        hover    = [
            f"<b>{names_shown[i]}</b>  "
            f"<span style='color:#888'>({ids_shown[i]})</span><br>"
            f"type:      {types_shown[i]}<br>"
            f"relevance: {rel_shown[i]:.5f}<br>"
            f"dominant:  {label}"
            for i in where_k
        ]

        traces.append(go.Scatter3d(
            x=proj_shown[kidx, 0],
            y=proj_shown[kidx, 1],
            z=proj_shown[kidx, 2],
            mode="markers",
            name=label,
            marker=dict(
                size=sizes_all[kidx],
                color=PALETTE[k % len(PALETTE)],
                opacity=0.65,
                line=dict(width=0),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#efefef",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#f8f8f8",
        legend=dict(font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        uirevision=stem,   # reset camera when dataset changes, preserve otherwise
    )

    status = (f"Showing {mask.sum():,} / {N:,} nodes  "
              f"(≥ p{pct} relevance,  threshold = {threshold:.5f})")

    legend = [
        html.Span(
            f"● {label}  ",
            style={"color": PALETTE[k % len(PALETTE)], "fontWeight": "bold",
                   "fontSize": "13px", "marginRight": "8px"},
        )
        for k, label in enumerate(labels)
    ]

    return fig, status, legend


if __name__ == "__main__":
    import socket
    if _args.port:
        port = _args.port
    else:
        port = 8050
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    break
            port += 1

    print(f"Starting on http://127.0.0.1:{port}/")
    app.run(debug=False, port=port)
