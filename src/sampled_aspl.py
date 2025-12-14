import networkx as nx

def sampled_aspl(G, S):
    """
    Compute sampled Average Shortest Path Length (ASPL)
    using BFS from nodes in sampling set S.

    Parameters
    ----------
    G : networkx.Graph (assumed connected)
    S : list of nodes (sampling set)

    Returns
    -------
    float
        Sampled ASPL value
    """
    total_dist = 0
    count = 0

    for s in S:
        lengths = nx.single_source_shortest_path_length(G, s)
        for v, d in lengths.items():
            if v != s:
                total_dist += d
                count += 1

    return total_dist / count