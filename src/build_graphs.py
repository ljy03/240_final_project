"""
Person 1 - Data + Sampling Lead (Setup Phase)
Toby

This module handles:
- Downloading and loading SNAP ego-Facebook dataset
- Extracting the largest connected component
- Building optional smaller subgraph for testing
- Selecting sampling set S for ASPL computation
- Computing baseline ASPL via BFS
"""

import os
import urllib.request
import gzip
import shutil
from collections import deque
import networkx as nx
from typing import List, Tuple, Set, Dict
import json
import pickle


def download_snap_facebook_dataset(output_dir: str = "data/raw") -> str:
    """
    Download the SNAP ego-Facebook dataset (or use existing file if present).
    
    If the dataset already exists in output_dir, it will be used.
    Otherwise, it will be downloaded from SNAP.
    
    Args:
        output_dir: Directory to save/load the dataset (default: data/raw)
        
    Returns:
        Path to the dataset file (facebook_combined.txt.gz or facebook_combined.txt)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing files (compressed or uncompressed)
    gz_path = os.path.join(output_dir, "facebook_combined.txt.gz")
    txt_path = os.path.join(output_dir, "facebook_combined.txt")
    
    if os.path.exists(txt_path):
        print(f"  - Using existing dataset: {txt_path}")
        return txt_path
    elif os.path.exists(gz_path):
        print(f"  - Using existing compressed dataset: {gz_path}")
        return gz_path
    
    # TODO: Implement automatic download if file doesn't exist
    # Dataset URL: https://snap.stanford.edu/data/facebook_combined.txt.gz
    # If you prefer to manually download, just place facebook_combined.txt.gz in data/raw/
    print(f"  - Dataset not found. Please download from:")
    print(f"    https://snap.stanford.edu/data/facebook_combined.txt.gz")
    print(f"    and place it in: {output_dir}/")
    raise FileNotFoundError(f"Dataset not found in {output_dir}. Please download manually or implement auto-download.")


def load_facebook_graph(dataset_path: str) -> nx.Graph:
    """
    Load the Facebook graph from the downloaded dataset.
    Handles both .gz (compressed) and .txt (uncompressed) files.
    
    Args:
        dataset_path: Path to the dataset file (facebook_combined.txt.gz or facebook_combined.txt)
        
    Returns:
        NetworkX Graph object representing the Facebook network
    """
    # The file format is edge list: node1 node2 (one edge per line)
    if dataset_path.endswith('.gz'):
        # Open compressed file
        with gzip.open(dataset_path, 'rt') as f:
            graph = nx.read_edgelist(f, nodetype=int)
    else:
        # Open uncompressed file
        graph = nx.read_edgelist(dataset_path, nodetype=int)
    
    return graph


def extract_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """
    Extract the largest connected component from the graph.
    
    Args:
        graph: Input NetworkX graph
        
    Returns:
        Largest connected component as a NetworkX graph
    """
    # Get all connected components
    connected_components = list(nx.connected_components(graph))
    
    # Find the largest one
    largest_component = max(connected_components, key=len)
    
    # Create subgraph from largest component
    lcc_graph = graph.subgraph(largest_component).copy()
    
    return lcc_graph


def build_smaller_subgraph(graph: nx.Graph, num_nodes: int = 750) -> nx.Graph:
    """
    Build an optional smaller subgraph (e.g., 500-1000 nodes) for full brute-force testing.
    
    Args:
        graph: Input NetworkX graph
        num_nodes: Target number of nodes (default: 750, between 500-1000)
        
    Returns:
        Smaller subgraph with approximately num_nodes nodes
    """
    import random
    
    # Strategy: Start from a random node and use BFS to collect nodes
    nodes = list(graph.nodes())
    start_node = random.choice(nodes)
    
    # Use BFS to collect nodes until we reach target size
    visited = set()
    queue = [start_node]
    visited.add(start_node)
    
    while queue and len(visited) < num_nodes:
        current = queue.pop(0)
        neighbors = list(graph.neighbors(current))
        random.shuffle(neighbors)  # Randomize neighbor order
        
        for neighbor in neighbors:
            if neighbor not in visited and len(visited) < num_nodes:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Create subgraph from collected nodes
    subgraph = graph.subgraph(visited).copy()
    return subgraph


def select_sampling_set(graph: nx.Graph, num_nodes: int = 40) -> Set[int]:
    """
    Select the sampling set S of 30-50 nodes used for ASPL computation.
    
    Args:
        graph: Input NetworkX graph
        num_nodes: Number of nodes to select (default: 40, between 30-50)
        
    Returns:
        Set of node IDs to use for sampling
    """
    import random
    
    # Strategy: Random sampling from all nodes
    # Alternative: Could use degree-based sampling (select high-degree nodes)
    all_nodes = list(graph.nodes())
    sampling_set = set(random.sample(all_nodes, min(num_nodes, len(all_nodes))))
    
    return sampling_set


def manual_bfs_shortest_paths(graph: nx.Graph, source: int) -> Dict[int, int]:
    """
    MANUAL BFS implementation to compute shortest path lengths from a source node.
    
    This shows how BFS works step-by-step without using NetworkX's built-in function.
    
    BFS Algorithm:
    1. Start at source node (distance = 0)
    2. Visit all neighbors (distance = 1)
    3. Visit neighbors of neighbors (distance = 2)
    4. Continue until all reachable nodes are visited
    
    Args:
        graph: NetworkX graph
        source: Starting node ID
        
    Returns:
        Dictionary mapping {node: shortest_path_length} from source
    """
    
    # Dictionary to store shortest distances: {node: distance}
    distances = {}
    
    # Queue for BFS: (node, distance)
    queue = deque([(source, 0)])
    
    # Mark source as visited with distance 0
    distances[source] = 0
    
    # BFS loop: process nodes level by level
    while queue:
        current_node, current_distance = queue.popleft()
        
        # Visit all neighbors of current node
        for neighbor in graph.neighbors(current_node):
            # If neighbor hasn't been visited yet
            if neighbor not in distances:
                # Distance to neighbor = distance to current + 1
                neighbor_distance = current_distance + 1
                distances[neighbor] = neighbor_distance
                
                # Add neighbor to queue to explore its neighbors later
                queue.append((neighbor, neighbor_distance))
    
    return distances


def compute_baseline_aspl_bfs(graph: nx.Graph, sampling_set: Set[int], use_manual_bfs: bool = False) -> Dict[str, float]:
    """
    Compute baseline ASPL (Average Shortest Path Length) of the network via BFS from S.
    
    Args:
        graph: Input NetworkX graph
        sampling_set: Set of node IDs to use as sources for BFS
        use_manual_bfs: If True, use manual BFS implementation; if False, use NetworkX (default: False)
        
    Returns:
        Dictionary containing:
            - 'aspl': Average shortest path length
            - 'num_paths': Number of paths computed
            - 'max_path_length': Maximum path length found
            - 'min_path_length': Minimum path length found
    """
    all_path_lengths = []
    
    # For each node in sampling_set, compute shortest paths to all other nodes
    for source in sampling_set:
        if use_manual_bfs:
            # Use our manual BFS implementation
            lengths = manual_bfs_shortest_paths(graph, source)
        else:
            # Use NetworkX's optimized BFS (which does the same thing internally)
            lengths = nx.single_source_shortest_path_length(graph, source)
        
        # Add all path lengths (excluding self-loops, i.e., length 0)
        for target, length in lengths.items():
            if source != target:  # Exclude self-paths
                all_path_lengths.append(length)
    
    if not all_path_lengths:
        return {
            'aspl': 0.0,
            'num_paths': 0,
            'max_path_length': 0,
            'min_path_length': 0
        }
    
    # Compute statistics
    aspl = sum(all_path_lengths) / len(all_path_lengths)
    
    return {
        'aspl': aspl,
        'num_paths': len(all_path_lengths),
        'max_path_length': max(all_path_lengths),
        'min_path_length': min(all_path_lengths)
    }


def save_graph(graph: nx.Graph, output_path: str, format: str = "graphml") -> None:
    """
    Save a graph to disk in a clean format.
    
    Args:
        graph: NetworkX graph to save
        output_path: Path to save the graph
        format: Format to save in ('graphml', 'gexf', 'pickle', 'edgelist')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == "graphml":
        nx.write_graphml(graph, output_path)
    elif format == "gexf":
        nx.write_gexf(graph, output_path)
    elif format == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
    elif format == "edgelist":
        nx.write_edgelist(graph, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_baseline_aspl(aspl_data: Dict[str, float], output_path: str) -> None:
    """
    Save baseline ASPL values to a file.
    
    Args:
        aspl_data: Dictionary containing ASPL computation results
        output_path: Path to save the ASPL data (JSON format)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(aspl_data, f, indent=2)


def save_sampling_nodes(sampling_set: Set[int], output_path: str) -> None:
    """
    Save the sampling nodes list S to a file.
    
    Args:
        sampling_set: Set of node IDs
        output_path: Path to save the sampling nodes (JSON format)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert set to sorted list for JSON serialization
    sampling_list = sorted(list(sampling_set))
    
    with open(output_path, 'w') as f:
        json.dump(sampling_list, f, indent=2)


def main():
    """
    Main execution function for Person 1's tasks.
    Orchestrates the entire data loading and sampling pipeline.
    """
    # Step 1: Download and load dataset
    print("Step 1: Downloading SNAP ego-Facebook dataset...")
    dataset_path = download_snap_facebook_dataset("data/raw")
    
    print("Step 2: Loading Facebook graph...")
    graph = load_facebook_graph(dataset_path)
    print(f"  - Original graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Step 3: Extract largest connected component
    print("Step 3: Extracting largest connected component...")
    lcc_graph = extract_largest_connected_component(graph)
    print(f"  - LCC graph: {lcc_graph.number_of_nodes()} nodes, {lcc_graph.number_of_edges()} edges")
    
    # Step 4: Build optional smaller subgraph
    print("Step 4: Building smaller subgraph for testing...")
    subgraph = build_smaller_subgraph(lcc_graph, num_nodes=750)
    print(f"  - Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    # Step 5: Select sampling set
    print("Step 5: Selecting sampling set S...")
    sampling_set = select_sampling_set(lcc_graph, num_nodes=40)
    print(f"  - Sampling set size: {len(sampling_set)} nodes")
    
    # Step 6: Compute baseline ASPL
    print("Step 6: Computing baseline ASPL via BFS...")
    aspl_data = compute_baseline_aspl_bfs(lcc_graph, sampling_set)
    print(f"  - Baseline ASPL: {aspl_data.get('aspl', 'N/A')}")
    
    # Step 7: Save processed graphs
    print("Step 7: Saving processed graphs...")
    os.makedirs("data/processed", exist_ok=True)
    
    save_graph(lcc_graph, "data/processed/facebook_lcc.graphml", format="graphml")
    save_graph(subgraph, "data/processed/facebook_subgraph.graphml", format="graphml")
    
    # Step 8: Save outputs
    print("Step 8: Saving outputs...")
    os.makedirs("outputs", exist_ok=True)
    
    save_baseline_aspl(aspl_data, "outputs/baseline_aspl.json")
    save_sampling_nodes(sampling_set, "outputs/sampling_nodes.json")
    
    print("\n[SUCCESS] All outputs saved")
    print("\nOutputs delivered to team:")
    print("  - Clean graphs: data/processed/facebook_lcc.graphml, data/processed/facebook_subgraph.graphml")
    print("  - Baseline ASPL values: outputs/baseline_aspl.json")
    print("  - Sampling nodes list S: outputs/sampling_nodes.json")


if __name__ == "__main__":
    main()

