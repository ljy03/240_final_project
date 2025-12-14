# 240 Final Project

Network Analysis Project - ASPL Computation

## Project Structure

```
project-root/
├── data/
│   ├── raw/              # Raw downloaded datasets
│   └── processed/        # Processed/cleaned graphs
├── src/
│   ├── build_graphs.py   # Person 1 - Data + Sampling Lead
│   ├── sampled_aspl.py   # Sampled ASPL computation (BFS-based)
│   ├── edge_disruption.py# Brute-force edge disruption analysis (subgraph)
│   └── analyze_delta_aspl.py
│                         # Analysis of ΔASPL results and runtime benchmark
├── outputs/              # Generated outputs (auto-created by scripts)
│   ├── baseline_aspl.json
│   ├── sampling_nodes.json
│   ├── delta_aspl_subgraph.json
│   ├── runtime_subgraph.json
│   └── most_disruptive_edge_subgraph.json
└── README.md

## Person 1 - Data + Sampling Lead (Setup Phase)

**Lead:** Toby

**Location:** `src/build_graphs.py`

### Core Tasks

1. **Download + Load SNAP ego-Facebook dataset**
   - Downloads the dataset from SNAP to `data/raw/`
   - Loads it into a NetworkX graph structure

2. **Extract Largest Connected Component**
   - Identifies and extracts the largest connected component from the graph
   - Ensures all nodes are reachable

3. **Build Optional Smaller Subgraph**
   - Creates a smaller subgraph (500-1000 nodes) for full brute-force testing
   - Useful for algorithm validation on smaller datasets

4. **Select Sampling Set S**
   - Chooses 30-50 nodes to use as sources for ASPL computation
   - These nodes will be used for baseline measurements

5. **Compute Baseline ASPL**
   - Computes Average Shortest Path Length using BFS from sampling set S
   - Provides baseline metrics for comparison

### Outputs Delivered to Team

- **Clean graphs ready for algorithms**
  - `data/processed/facebook_lcc.graphml` - Largest connected component
  - `data/processed/facebook_subgraph.graphml` - Smaller subgraph for testing

- **Baseline ASPL values**
  - `outputs/baseline_aspl.json` - Contains ASPL metrics and statistics

- **Sampling nodes list S**
  - `outputs/sampling_nodes.json` - List of node IDs used for sampling

## Setup

```bash
# Install dependencies
pip install networkx numpy requests

# Run the data sampling pipeline
python src/build_graphs.py
```
## Person 2 - Edge Disruption(Brute force) & ASPL Analysis (Analysis Phase)

**Lead:** Ivo

**Location:** `src/edge_disruption.py`, `src/sampled_aspl.py`, `src/analyze_delta_aspl.py`

### Core Tasks

1. **Compute Sampled ASPL**
   - Computes sampled Average Shortest Path Length (ASPL)
   - Uses BFS from a fixed sampling set of source nodes
   - Sampling set is provided during the setup phase

2. **Brute-Force Edge Disruption (Subgraph)**
   - Iteratively removes each edge in the subgraph
   - Recomputes sampled ASPL after each edge removal
   - Measures the impact of each edge on network efficiency

3. **Compute Edge Disruption Metric**
   - Computes the disruption metric for each edge:
     
     `Δ(e) = ASPL(G \\ e) − ASPL(G)`

4. **Runtime Benchmarking**
   - Measures total and per-edge runtime for brute-force edge removal
   - Reports average, minimum, and maximum runtime statistics

5. **Identify Most Disruptive Edge**
   - Identifies the edge whose removal causes the largest increase in sampled ASPL
   - Summarizes key disruption results for downstream analysis

### Outputs Delivered to Team

- **Edge disruption results**
  - `outputs/delta_aspl_subgraph.json` – Δ(e) values for tested edges

- **Runtime benchmarks**
  - `outputs/runtime_subgraph.json` – Total and average runtime statistics

- **Most disruptive edge**
  - `outputs/most_disruptive_edge_subgraph.json` – Edge with maximum Δ(e)

> **Note:** Files in the `outputs/` directory are generated automatically by running the analysis scripts and are not tracked in the Git repository.

### Setup

```bash
# Run edge disruption analysis on the subgraph
python src/edge_disruption.py

# Analyze results and identify the most disruptive edge
python src/analyze_delta_aspl.py

## Dataset Information

- **Source:** SNAP (Stanford Network Analysis Project)
- **Dataset:** ego-Facebook
- **URL:** https://snap.stanford.edu/data/facebook_combined.txt.gz
- **Format:** Edge list (undirected graph)
