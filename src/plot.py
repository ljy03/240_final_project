import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all(results_dir: str) -> None:
    runtime_path = os.path.join(results_dir, "runtime_raw.csv")
    delta_path = os.path.join(results_dir, "delta_edges_raw.csv")

    if not os.path.exists(runtime_path):
        raise FileNotFoundError(f"Missing {runtime_path}. Run src/run_experiments.py first.")
    if not os.path.exists(delta_path):
        raise FileNotFoundError(f"Missing {delta_path}. Run src/run_experiments.py first.")

    df_rt = pd.read_csv(runtime_path)
    df_de = pd.read_csv(delta_path)

    sns.set_theme(style="whitegrid", context="talk")

    # --- Runtime vs node count (both methods; brute only present for smaller N) ---
    rt_long = []
    if "centrality_total_runtime_sec" in df_rt.columns:
        d = df_rt[df_rt["method"] == "centrality"].copy()
        d["runtime_sec"] = d["centrality_total_runtime_sec"]
        d["method_plot"] = "centrality (betweenness + validate topK)"
        rt_long.append(d[["subgraph_nodes", "sampling_size", "trial", "runtime_sec", "method_plot"]])

    if "brute_total_runtime_sec" in df_rt.columns:
        d = df_rt[df_rt["method"] == "bruteforce_all_edges"].copy()
        d["runtime_sec"] = d["brute_total_runtime_sec"]
        d["method_plot"] = "brute-force (all edges)"
        rt_long.append(d[["subgraph_nodes", "sampling_size", "trial", "runtime_sec", "method_plot"]])

    if rt_long:
        df_rtp = pd.concat(rt_long, ignore_index=True)
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.lineplot(
            data=df_rtp,
            x="subgraph_nodes",
            y="runtime_sec",
            hue="method_plot",
            style="sampling_size",
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set_title("Runtime vs subgraph node count")
        ax.set_xlabel("Subgraph nodes")
        ax.set_ylabel("Runtime (sec)")
        ax.set_yscale("log")
        ax.legend(title="Method / sampling size", bbox_to_anchor=(1.02, 1), loc="upper left")
        _save(fig, os.path.join(results_dir, "fig_runtime_vs_node_count.png"))

    # --- Runtime: brute-force vs centrality (overlapping node counts only) ---
    if {"centrality_total_runtime_sec", "brute_total_runtime_sec"}.issubset(df_rt.columns):
        d_cent = df_rt[df_rt["method"] == "centrality"][["subgraph_nodes", "sampling_size", "trial", "centrality_total_runtime_sec"]]
        d_brut = df_rt[df_rt["method"] == "bruteforce_all_edges"][["subgraph_nodes", "sampling_size", "trial", "brute_total_runtime_sec"]]
        merged = d_cent.merge(d_brut, on=["subgraph_nodes", "sampling_size", "trial"], how="inner")
        if not merged.empty:
            # Side-by-side runtime comparison plot (requested explicitly)
            dcmp = pd.concat(
                [
                    merged.assign(method_plot="centrality", runtime_sec=merged["centrality_total_runtime_sec"])[
                        ["subgraph_nodes", "sampling_size", "trial", "method_plot", "runtime_sec"]
                    ],
                    merged.assign(method_plot="brute-force", runtime_sec=merged["brute_total_runtime_sec"])[
                        ["subgraph_nodes", "sampling_size", "trial", "method_plot", "runtime_sec"]
                    ],
                ],
                ignore_index=True,
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=dcmp,
                x="subgraph_nodes",
                y="runtime_sec",
                hue="method_plot",
                errorbar="sd",
                ax=ax,
            )
            ax.set_title("Runtime: brute-force vs centrality (overlap only, averaged over trials)")
            ax.set_xlabel("Subgraph nodes")
            ax.set_ylabel("Runtime (sec)")
            ax.set_yscale("log")
            ax.legend(title="")
            _save(fig, os.path.join(results_dir, "fig_runtime_bruteforce_vs_centrality.png"))

            merged["speedup_brute_over_centrality"] = merged["brute_total_runtime_sec"] / merged["centrality_total_runtime_sec"]
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.barplot(
                data=merged,
                x="subgraph_nodes",
                y="speedup_brute_over_centrality",
                hue="sampling_size",
                ax=ax,
            )
            ax.set_title("Runtime ratio: brute-force / centrality (overlap only)")
            ax.set_xlabel("Subgraph nodes")
            ax.set_ylabel("Runtime ratio (×)")
            ax.legend(title="Sampling size S", bbox_to_anchor=(1.02, 1), loc="upper left")
            _save(fig, os.path.join(results_dir, "fig_runtime_ratio_brute_over_centrality.png"))

    # --- ΔASPL distribution: random vs centrality-chosen edges ---
    if not df_de.empty:
        d = df_de.copy()
        d = d[d["selection"].isin(["random_edges", "centrality_topk"])].copy()
        d["selection_plot"] = d["selection"].map(
            {"random_edges": "random edges", "centrality_topk": "centrality-chosen (topK)"}
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.violinplot(
            data=d,
            x="subgraph_nodes",
            y="delta_aspl",
            hue="selection_plot",
            split=True,
            inner="quartile",
            cut=0,
            ax=ax,
        )
        ax.set_title("ΔASPL distribution: random edges vs centrality-chosen edges")
        ax.set_xlabel("Subgraph nodes")
        ax.set_ylabel("ΔASPL (ASPL(G\\e) − ASPL(G))")
        ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
        _save(fig, os.path.join(results_dir, "fig_delta_aspl_distribution.png"))

        # Also a simple summary boxplot aggregated over all node counts (often nicer for slides)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=d, x="selection_plot", y="delta_aspl", ax=ax)
        ax.set_title("ΔASPL distribution (all subgraph sizes pooled)")
        ax.set_xlabel("")
        ax.set_ylabel("ΔASPL")
        _save(fig, os.path.join(results_dir, "fig_delta_aspl_distribution_pooled.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot experiment CSV outputs.")
    ap.add_argument("--results", default="outputs/experiments")
    args = ap.parse_args()
    plot_all(args.results)


if __name__ == "__main__":
    main()
