import gc
import json
from itertools import combinations
from math import log

import igraph as ig
import leidenalg
import numpy as np
import xgi


def entropy(labels, base=None, norm=False):
    """Computes entropy of label distribution."""

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0

    # Compute entropy
    base = 2 if base is None else base
    for p in probs:
        try:
            ent -= p * log(p, base)
        except Exception as e:
            print(p)
    if norm:
        return ent / (log(n_labels, base))
    else:
        return ent


H = xgi.read_hif("data/deidentified_starterpack_hif.json")
xgi.largest_connected_hypergraph(H, in_place=True)

print("Mapping nodes to unique integers...", flush=True)
node_to_int = {node_id: idx for idx, node_id in enumerate(H.nodes)}
int_to_node = {idx: node_id for node_id, idx in node_to_int.items()}

num_nodes = len(node_to_int)

clique_edges_set = set()

print("Constructing the edge list for the clique expansion...", flush=True)

for edge in H.edges:
    nodes = [node_to_int[node_id] for node_id in H.edges.members(edge)]
    if len(nodes) >= 2:
        for u, v in combinations(sorted(nodes), 2):
            clique_edges_set.add((u, v))

print(
    f"Total number of edges in the clique expansion: {len(clique_edges_set)}",
    flush=True,
)

print("Creating the igraph Graph...", flush=True)
G_ig = ig.Graph()
G_ig.add_vertices(num_nodes)
G_ig.add_edges(clique_edges_set)

del clique_edges_set
gc.collect()

print("Running the Leiden algorithm for community detection...", flush=True)
partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=0)

del G_ig
gc.collect()

print("Mapping nodes to clusters...", flush=True)
node_labels = {
    str(int_to_node[idx]): cluster for idx, cluster in enumerate(partition.membership)
}

del partition
gc.collect()

edge_entropy = {}
for e in H.edges:
    edge_entropy[e] = entropy([node_labels[n] for n in H.edges.members(e)], norm=True)

with open("data/edge_entropy.json", "w") as f:
    f.write(json.dumps(edge_entropy))
