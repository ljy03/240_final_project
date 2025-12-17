import networkx as nx
import json
import time
import os
from sampled_aspl import sampled_aspl



INPUT_GRAPH = "data/processed/facebook_lcc.graphml"    
INPUT_SAMPLES = "outputs/sampling_nodes.json"  
OUTPUT_DIR = "outputs"
TOP_K = 50                                   
BRANDES_SAMPLES = 400                       

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[1/5] Loading Graph from {INPUT_GRAPH}...")
    if not os.path.exists(INPUT_GRAPH):
        print(f"❌ 找不到文件: {INPUT_GRAPH}")
        return

    G = nx.read_graphml(INPUT_GRAPH).to_undirected()
    print(f"      - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print(f"[2/5] Loading Sampling Nodes from {INPUT_SAMPLES}...")
    with open(INPUT_SAMPLES, "r") as f:
        raw_S = json.load(f)
    S = [str(x) for x in raw_S if str(x) in G]
    print(f"      - Valid Sampling Nodes: {len(S)}")

    print("[3/5] Computing Baseline ASPL...")
    base_aspl = sampled_aspl(G, S)
    print(f"      🎯 Baseline ASPL: {base_aspl:.6f}")

    print(f"[4/5] Running Approximate Brandes (k={BRANDES_SAMPLES})...")
    t0 = time.time()
    
    eb = nx.edge_betweenness_centrality(G, k=BRANDES_SAMPLES, normalized=False)
    

    ranked_edges = sorted(eb.items(), key=lambda x: x[1], reverse=True)
    
    print(f"      ✅ Centrality computed in {time.time()-t0:.2f}s")

    print(f"[5/5] Validating Top {TOP_K} Edges...")
    print(f"{'Rank':<5} | {'Edge':<25} | {'Delta ASPL':<12} | {'New ASPL':<10}")

    validation_results = []
    
    for i in range(TOP_K):
        edge, score = ranked_edges[i]
        u, v = edge
        
        G.remove_edge(u, v)           
        new_aspl = sampled_aspl(G, S) 
        G.add_edge(u, v)              
        
        delta = new_aspl - base_aspl
        
        print(f"{i+1:<5} | {str(edge):<25} | {delta:+.6f}     | {new_aspl:.4f}")
        
        validation_results.append({
            "rank": i+1,
            "edge": edge,
            "score": score,
            "baseline_aspl": base_aspl,
            "new_aspl": new_aspl,
            "delta_aspl": delta
        })

    path_val = os.path.join(OUTPUT_DIR, "method2_top_50_validation.json")
    path_rank = os.path.join(OUTPUT_DIR, "method2_centrality_rankings.json")
    
    with open(path_val, "w") as f:
        json.dump(validation_results, f, indent=2)
        
    simple_rankings = [{"edge": e[0], "score": e[1]} for e in ranked_edges[:2000]]
    with open(path_rank, "w") as f:
        json.dump(simple_rankings, f)


if __name__ == "__main__":
    main()