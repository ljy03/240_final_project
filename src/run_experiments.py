import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

# Allow running as a script from repo root
import sys

sys.path.append(os.path.dirname(__file__))
from sampled_aspl import sampled_aspl  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    graph_path: str
    output_dir: str
    seed: int
    sampling_sizes: Tuple[int, ...]
    subgraph_sizes: Tuple[int, ...]
    trials: int
    top_k_edges: int
    random_k_edges: int
    brandes_k: int
    brute_force_max_nodes: int


def _ensure_undirected_simple(G: nx.Graph) -> nx.Graph:
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)
    if G.is_directed():
        G = G.to_undirected()
    return G


def _connected_bfs_subgraph(G: nx.Graph, n_nodes: int, rng: random.Random) -> nx.Graph:
    """
    Build a (nearly always) connected induced subgraph by taking the first n_nodes
    visited in a randomized BFS from a random start node.
    """
    if n_nodes >= G.number_of_nodes():
        return G.copy()

    start = rng.choice(list(G.nodes()))
    visited: List[Any] = []
    seen = {start}
    queue = [start]

    while queue and len(visited) < n_nodes:
        u = queue.pop(0)
        visited.append(u)
        nbrs = list(G.neighbors(u))
        rng.shuffle(nbrs)
        for v in nbrs:
            if v not in seen:
                seen.add(v)
                queue.append(v)
            if len(seen) >= n_nodes * 3:
                # Keep the frontier bounded; we only need enough to reach n_nodes.
                break

    # If BFS didn't reach enough nodes (rare in connected graph), top-up randomly.
    if len(visited) < n_nodes:
        remaining = list(set(G.nodes()) - set(visited))
        rng.shuffle(remaining)
        visited.extend(remaining[: max(0, n_nodes - len(visited))])

    H = G.subgraph(visited[:n_nodes]).copy()
    if not nx.is_connected(H):
        # Fall back to largest connected component for ASPL validity.
        lcc_nodes = max(nx.connected_components(H), key=len)
        H = H.subgraph(lcc_nodes).copy()
    return H


def _sample_nodes(G: nx.Graph, k: int, rng: random.Random) -> List[Any]:
    nodes = list(G.nodes())
    if not nodes:
        return []
    k = min(k, len(nodes))
    return rng.sample(nodes, k)


def _sample_edges(G: nx.Graph, k: int, rng: random.Random) -> List[Tuple[Any, Any]]:
    edges = list(G.edges())
    if not edges:
        return []
    k = min(k, len(edges))
    return rng.sample(edges, k)


def _delta_aspl_for_edges(G: nx.Graph, S: Sequence[Any], base_aspl: float, edges: Iterable[Tuple[Any, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (u, v) in edges:
        if not G.has_edge(u, v):
            continue
        G.remove_edge(u, v)
        new_aspl = sampled_aspl(G, S)
        G.add_edge(u, v)
        out.append(
            {
                "u": u,
                "v": v,
                "delta_aspl": float(new_aspl - base_aspl),
                "new_aspl": float(new_aspl),
            }
        )
    return out


def _bruteforce_all_edges_runtime(G: nx.Graph, S: Sequence[Any]) -> Dict[str, Any]:
    """
    Brute force: remove every edge and recompute sampled ASPL.
    Returns runtime stats and max delta (but does NOT store every delta to keep files small).
    """
    t0 = time.perf_counter()
    base_aspl = sampled_aspl(G, S)

    per_edge_times: List[float] = []
    max_delta = float("-inf")
    max_edge: Optional[Tuple[Any, Any]] = None

    for (u, v) in list(G.edges()):
        G.remove_edge(u, v)
        t1 = time.perf_counter()
        new_aspl = sampled_aspl(G, S)
        per_edge_times.append(time.perf_counter() - t1)
        delta = float(new_aspl - base_aspl)
        if delta > max_delta:
            max_delta = delta
            max_edge = (u, v)
        G.add_edge(u, v)

    total = time.perf_counter() - t0
    return {
        "brute_total_runtime_sec": float(total),
        "brute_num_edges_tested": int(len(per_edge_times)),
        "brute_avg_runtime_per_edge_sec": float(sum(per_edge_times) / len(per_edge_times)) if per_edge_times else None,
        "brute_min_runtime_per_edge_sec": float(min(per_edge_times)) if per_edge_times else None,
        "brute_max_runtime_per_edge_sec": float(max(per_edge_times)) if per_edge_times else None,
        "brute_max_delta_aspl": float(max_delta) if max_edge is not None else None,
        "brute_max_delta_edge": list(max_edge) if max_edge is not None else None,
        "baseline_aspl": float(base_aspl),
    }


def _centrality_rank_and_validate(
    G: nx.Graph,
    S: Sequence[Any],
    top_k: int,
    brandes_k: int,
) -> Dict[str, Any]:
    """
    Centrality method: approximate edge betweenness, then validate top_k edges by ΔASPL.
    Returns runtime stats plus per-edge validation deltas for the selected edges.
    """
    t0 = time.perf_counter()
    base_aspl = sampled_aspl(G, S)

    k = min(brandes_k, G.number_of_nodes())
    t1 = time.perf_counter()
    eb = nx.edge_betweenness_centrality(G, k=k, normalized=False)
    centrality_runtime = time.perf_counter() - t1

    ranked = sorted(eb.items(), key=lambda x: x[1], reverse=True)
    chosen = [edge for (edge, _score) in ranked[: min(top_k, len(ranked))]]

    t2 = time.perf_counter()
    deltas = _delta_aspl_for_edges(G, S, base_aspl, chosen)
    validation_runtime = time.perf_counter() - t2

    total = time.perf_counter() - t0
    return {
        "centrality_total_runtime_sec": float(total),
        "centrality_compute_runtime_sec": float(centrality_runtime),
        "centrality_validation_runtime_sec": float(validation_runtime),
        "centrality_brandes_k": int(k),
        "centrality_top_k": int(len(chosen)),
        "baseline_aspl": float(base_aspl),
        "centrality_deltas": deltas,
    }


def run(cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    rng_master = random.Random(cfg.seed)

    G_full = _ensure_undirected_simple(nx.read_graphml(cfg.graph_path))

    meta = {
        "graph_path": cfg.graph_path,
        "full_graph_nodes": int(G_full.number_of_nodes()),
        "full_graph_edges": int(G_full.number_of_edges()),
        "seed": cfg.seed,
        "sampling_sizes": list(cfg.sampling_sizes),
        "subgraph_sizes": list(cfg.subgraph_sizes),
        "trials": cfg.trials,
        "top_k_edges": cfg.top_k_edges,
        "random_k_edges": cfg.random_k_edges,
        "brandes_k": cfg.brandes_k,
        "brute_force_max_nodes": cfg.brute_force_max_nodes,
    }
    with open(os.path.join(cfg.output_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    runtime_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    for n_nodes in cfg.subgraph_sizes:
        for trial in range(cfg.trials):
            # Make each trial reproducible but distinct
            trial_seed = rng_master.randint(0, 2**31 - 1)
            rng = random.Random(trial_seed)

            H = _connected_bfs_subgraph(G_full, n_nodes, rng)
            H = _ensure_undirected_simple(H)

            node_count = int(H.number_of_nodes())
            edge_count = int(H.number_of_edges())

            for s_size in cfg.sampling_sizes:
                S = _sample_nodes(H, s_size, rng)
                if len(S) < 2 or edge_count == 0:
                    continue

                # Centrality method (always, unless graph is tiny/empty)
                cent = _centrality_rank_and_validate(H, S, top_k=cfg.top_k_edges, brandes_k=cfg.brandes_k)

                runtime_rows.append(
                    {
                        "method": "centrality",
                        "subgraph_nodes": node_count,
                        "subgraph_edges": edge_count,
                        "sampling_size": int(len(S)),
                        "trial": int(trial),
                        "trial_seed": int(trial_seed),
                        **{k: v for k, v in cent.items() if k != "centrality_deltas"},
                    }
                )

                for d in cent["centrality_deltas"]:
                    delta_rows.append(
                        {
                            "selection": "centrality_topk",
                            "subgraph_nodes": node_count,
                            "subgraph_edges": edge_count,
                            "sampling_size": int(len(S)),
                            "trial": int(trial),
                            "trial_seed": int(trial_seed),
                            "u": d["u"],
                            "v": d["v"],
                            "delta_aspl": d["delta_aspl"],
                        }
                    )

                # Random edges distribution
                base_aspl = float(cent["baseline_aspl"])
                rand_edges = _sample_edges(H, cfg.random_k_edges, rng)
                rand_deltas = _delta_aspl_for_edges(H, S, base_aspl, rand_edges)
                for d in rand_deltas:
                    delta_rows.append(
                        {
                            "selection": "random_edges",
                            "subgraph_nodes": node_count,
                            "subgraph_edges": edge_count,
                            "sampling_size": int(len(S)),
                            "trial": int(trial),
                            "trial_seed": int(trial_seed),
                            "u": d["u"],
                            "v": d["v"],
                            "delta_aspl": d["delta_aspl"],
                        }
                    )

                # Brute force (only for smaller subgraphs; it's O(E^2) ish)
                if node_count <= cfg.brute_force_max_nodes:
                    brute = _bruteforce_all_edges_runtime(H, S)
                    runtime_rows.append(
                        {
                            "method": "bruteforce_all_edges",
                            "subgraph_nodes": node_count,
                            "subgraph_edges": edge_count,
                            "sampling_size": int(len(S)),
                            "trial": int(trial),
                            "trial_seed": int(trial_seed),
                            **brute,
                        }
                    )

    df_runtime = pd.DataFrame(runtime_rows)
    df_delta = pd.DataFrame(delta_rows)

    df_runtime.to_csv(os.path.join(cfg.output_dir, "runtime_raw.csv"), index=False)
    df_delta.to_csv(os.path.join(cfg.output_dir, "delta_edges_raw.csv"), index=False)

    # Convenience summaries for tables
    if not df_runtime.empty:
        summary_runtime = (
            df_runtime.groupby(["method", "subgraph_nodes", "sampling_size"], dropna=False)
            .agg(
                subgraph_edges=("subgraph_edges", "max"),
                trials=("trial", "nunique"),
                total_runtime_sec=("brute_total_runtime_sec", "mean"),
                centrality_total_runtime_sec=("centrality_total_runtime_sec", "mean"),
                centrality_compute_runtime_sec=("centrality_compute_runtime_sec", "mean"),
                centrality_validation_runtime_sec=("centrality_validation_runtime_sec", "mean"),
                brute_avg_runtime_per_edge_sec=("brute_avg_runtime_per_edge_sec", "mean"),
            )
            .reset_index()
        )
        summary_runtime.to_csv(os.path.join(cfg.output_dir, "runtime_summary.csv"), index=False)

        # Integration-friendly comparison table on overlapping configs
        if {"centrality_total_runtime_sec", "brute_total_runtime_sec"}.issubset(df_runtime.columns):
            d_cent = df_runtime[df_runtime["method"] == "centrality"][
                ["subgraph_nodes", "sampling_size", "trial", "centrality_total_runtime_sec"]
            ]
            d_brut = df_runtime[df_runtime["method"] == "bruteforce_all_edges"][
                ["subgraph_nodes", "sampling_size", "trial", "brute_total_runtime_sec"]
            ]
            merged = d_cent.merge(d_brut, on=["subgraph_nodes", "sampling_size", "trial"], how="inner")
            if not merged.empty:
                merged["brute_over_centrality_ratio"] = (
                    merged["brute_total_runtime_sec"] / merged["centrality_total_runtime_sec"]
                )
                merged.to_csv(os.path.join(cfg.output_dir, "runtime_overlap_comparison.csv"), index=False)

    if not df_delta.empty:
        summary_delta = (
            df_delta.groupby(["selection", "subgraph_nodes", "sampling_size"], dropna=False)
            .agg(
                n_edges=("delta_aspl", "count"),
                delta_mean=("delta_aspl", "mean"),
                delta_median=("delta_aspl", "median"),
                delta_p90=("delta_aspl", lambda s: float(s.quantile(0.90))),
                delta_p99=("delta_aspl", lambda s: float(s.quantile(0.99))),
                delta_max=("delta_aspl", "max"),
            )
            .reset_index()
        )
        summary_delta.to_csv(os.path.join(cfg.output_dir, "delta_summary.csv"), index=False)


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Run ASPL disruption experiments (bruteforce vs centrality).")
    p.add_argument("--graph", default="data/processed/facebook_lcc.graphml")
    p.add_argument("--out", default="outputs/experiments")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sampling-sizes", default="10,30,50,100")
    p.add_argument("--subgraph-sizes", default="100,200,400,800,1200")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--top-k-edges", type=int, default=50)
    p.add_argument("--random-k-edges", type=int, default=50)
    p.add_argument("--brandes-k", type=int, default=400, help="k for approximate Brandes edge betweenness.")
    p.add_argument("--bruteforce-max-nodes", type=int, default=400)
    args = p.parse_args()

    sampling_sizes = tuple(int(x.strip()) for x in args.sampling_sizes.split(",") if x.strip())
    subgraph_sizes = tuple(int(x.strip()) for x in args.subgraph_sizes.split(",") if x.strip())

    return ExperimentConfig(
        graph_path=args.graph,
        output_dir=args.out,
        seed=args.seed,
        sampling_sizes=sampling_sizes,
        subgraph_sizes=subgraph_sizes,
        trials=int(args.trials),
        top_k_edges=int(args.top_k_edges),
        random_k_edges=int(args.random_k_edges),
        brandes_k=int(args.brandes_k),
        brute_force_max_nodes=int(args.bruteforce_max_nodes),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)

