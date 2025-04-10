import gc
import json
from itertools import combinations

import igraph as ig
import leidenalg
import xgi
from tqdm import tqdm

H = xgi.read_hif("data/deidentified_starterpack_hif.json")
xgi.largest_connected_hypergraph(H, in_place=True)

print("Mapping nodes to unique integers...", flush=True)
node_to_int = {node_id: idx for idx, node_id in enumerate(H.nodes)}
int_to_node = {idx: node_id for node_id, idx in node_to_int.items()}

num_nodes = len(node_to_int)

clique_edges_set = set()

print("Constructing the edge list for the clique expansion...", flush=True)

for edge in tqdm(H.edges, desc="Processing hyperedges", unit="edges"):
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

with open("data/node_labels.json", "w") as f:
    f.write(json.dumps(node_labels))
