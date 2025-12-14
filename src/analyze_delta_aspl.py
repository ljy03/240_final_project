import json
import os


def analyze(delta_path, runtime_path):
    if not os.path.exists(delta_path):
        raise FileNotFoundError(
            f"{delta_path} not found. Please run edge_disruption.py first."
        )

    if not os.path.exists(runtime_path):
        raise FileNotFoundError(
            f"{runtime_path} not found. Please run edge_disruption.py first."
        )

    # Load delta ASPL data
    with open(delta_path, "r") as f:
        delta_data = json.load(f)

    if len(delta_data) == 0:
        raise ValueError("delta ASPL file is empty")

    most_disruptive = max(delta_data, key=lambda x: x["delta_aspl"])

    # Load runtime summary
    with open(runtime_path, "r") as f:
        runtime_data = json.load(f)

    print("===== Subgraph Edge Disruption Analysis =====\n")

    print("Most Disruptive Edge:")
    print(f"  Edge: {tuple(most_disruptive['edge'])}")
    print(f"  Delta ASPL: {most_disruptive['delta_aspl']:.6f}\n")

    print("Runtime Benchmark:")
    print(f"  Number of edges tested: {runtime_data['num_edges_tested']}")
    print(f"  Total runtime (sec): {runtime_data['total_runtime_sec']:.4f}")
    print(f"  Avg runtime per edge (sec): {runtime_data['avg_runtime_per_edge_sec']:.6f}")
    print(f"  Min runtime (sec): {runtime_data['min_runtime_sec']:.6f}")
    print(f"  Max runtime (sec): {runtime_data['max_runtime_sec']:.6f}")

    # Save most disruptive edge
    summary = {
        "edge": most_disruptive["edge"],
        "delta_aspl": most_disruptive["delta_aspl"]
    }

    with open("outputs/most_disruptive_edge_subgraph.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    analyze(
        delta_path="outputs/delta_aspl_subgraph.json",
        runtime_path="outputs/runtime_subgraph.json")