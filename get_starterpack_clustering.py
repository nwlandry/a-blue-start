import pandas as pd
import xgi

H = xgi.read_hif("data/deidentified_starterpack_hif.json")

import gc
from itertools import combinations

import igraph as ig
import leidenalg
from tqdm import tqdm

print("Mapping nodes to unique integers...")
node_to_int = {node_id: idx for idx, node_id in enumerate(H.nodes)}
int_to_node = {idx: node_id for node_id, idx in node_to_int.items()}

num_nodes = len(node_to_int)

clique_edges_set = set()

print("Constructing the edge list for the clique expansion...")

for edge in tqdm(H.edges, desc="Processing hyperedges", unit="edges"):
    nodes = [node_to_int[node_id] for node_id in H.edges.members(edge)]
    if len(nodes) >= 2:
        for u, v in combinations(nodes, 2):
            clique_edges_set.add((min(u, v), max(u, v)))

print(f"Total number of edges in the clique expansion: {len(clique_edges_set)}")

print("Creating the igraph Graph...")
edge_list = list(clique_edges_set)
del clique_edges_set
gc.collect()

G_ig = ig.Graph()
G_ig.add_vertices(num_nodes)
G_ig.add_edges(edge_list)

del edge_list
gc.collect()

print("Running the Leiden algorithm for community detection...")
partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=0)

del G_ig
gc.collect()

print("Mapping nodes to clusters...")
node_to_cluster = {idx: cluster for idx, cluster in enumerate(partition.membership)}

del partition
gc.collect()

print("calculating uncut hyperedges per size...")
counts_per_size = {}
not_cut_counts_per_size = {}

for edge in tqdm(H.edges, desc="Evaluating hyperedges", unit="edges"):
    nodes = [node_to_int[node_id] for node_id in H.edges.members(edge)]
    clusters = set(node_to_cluster[node] for node in nodes)
    size = len(nodes)
    counts_per_size[size] = counts_per_size.get(size, 0) + 1
    if len(clusters) == 1:
        not_cut_counts_per_size[size] = not_cut_counts_per_size.get(size, 0) + 1

del node_to_int
del int_to_node
gc.collect()

sizes = sorted(counts_per_size.keys())
fractions = [
    not_cut_counts_per_size.get(size, 0) / counts_per_size[size] for size in sizes
]
df = pd.DataFrame({"sizes": sizes[1:], "fractions": fractions[1:]})

df.to_csv("data/starterpack_clustering.csv.gz")
