import time
import json
import networkx as nx
from sampled_aspl import sampled_aspl


def edge_delta_aspl(G, S):
    base_aspl = sampled_aspl(G, S)

    delta_results = []
    runtimes = []

    for u, v in list(G.edges()):
        G.remove_edge(u, v)

        start = time.time()
        new_aspl = sampled_aspl(G, S)
        elapsed = time.time() - start

        delta_results.append({
            "edge": [u, v],
            "delta_aspl": new_aspl - base_aspl
        })

        runtimes.append(elapsed)

        G.add_edge(u, v)

    return delta_results, runtimes


if __name__ == "__main__":
    G = nx.read_graphml("data/processed/facebook_subgraph.graphml")

    with open("outputs/sampling_nodes.json") as f:
        S = json.load(f)

    # GraphML nodes are strings
    S = [str(s) for s in S]

    # Filter sampling nodes to subgraph
    G_nodes = set(G.nodes())
    S = [s for s in S if s in G_nodes]

    delta_results, runtimes = edge_delta_aspl(G, S)

    # 1️⃣ Δ(e) values (main deliverable)
    with open("outputs/delta_aspl_subgraph.json", "w") as f:
        json.dump(delta_results, f, indent=2)

    # 2️⃣ Runtime benchmark (summary)
    runtime_summary = {
        "num_edges_tested": len(runtimes),
        "total_runtime_sec": sum(runtimes),
        "avg_runtime_per_edge_sec": sum(runtimes) / len(runtimes),
        "min_runtime_sec": min(runtimes),
        "max_runtime_sec": max(runtimes)
    }

    with open("outputs/runtime_subgraph.json", "w") as f:
        json.dump(runtime_summary, f, indent=2)

    print("Finished edge disruption on subgraph.")