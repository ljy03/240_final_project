# 240 Final Project

Network Analysis Project - ASPL Computation

## Project Structure

```
project-root/
├── data/
│   ├── raw/              # Raw downloaded datasets
│   └── processed/        # Processed/cleaned graphs
├── src/
│   └── build_graphs.py   # Person 1 - Data + Sampling Lead
├── outputs/              # Output files (ASPL values, sampling nodes, etc.)
└── README.md
```

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

## Dataset Information

- **Source:** SNAP (Stanford Network Analysis Project)
- **Dataset:** ego-Facebook
- **URL:** https://snap.stanford.edu/data/facebook_combined.txt.gz
- **Format:** Edge list (undirected graph)
