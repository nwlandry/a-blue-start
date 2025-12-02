import gzip
import json
from collections import defaultdict
from datetime import datetime
from os import makedirs
from os.path import join

import numpy as np
import xgi

# base_dir = "/scratch/yyu8dx/Research/bluesky-graph/postprocessed_data/SOMAR"
# base_dir = "data"
base_dir = "/Users/a404/OneDrive - University of Virginia/SOMAR_v2/"
starterpack_file = "deidentified_starterpack_hif.json.gz"

with gzip.open(join(base_dir, starterpack_file), "rt", encoding="utf-8") as f:
    hif_dict = json.load(f)
H = xgi.from_hif_dict(hif_dict)

print("Loaded hypergraph!")

cc = [len(c) for c in xgi.connected_components(H)]
component_sizes, component_numbers = np.unique(cc, return_counts=True)

print("Calculated component sizes!")

edge_sizes = H.edges.size.aslist()
degrees = H.nodes.degree.aslist()

num_nodes = H.num_nodes
num_edges = H.num_edges

min_degree = H.nodes.degree.min()
max_degree = H.nodes.degree.max()
mean_degree = H.nodes.degree.mean()
median_degree = H.nodes.degree.median()
mode_degree = H.nodes.degree.mode()

min_edge_size = H.edges.size.min()
max_edge_size = H.edges.size.max()
mean_edge_size = H.edges.size.mean()
median_edge_size = H.edges.size.median()
mode_edge_size = H.edges.size.mode()

number_created = defaultdict(lambda: 0)
account_age_at_creation = []
starterpack_date_created = []

for e in H.edges:
    edge_attrs = H.edges[e]
    n = edge_attrs["creator-id"]
    number_created[n] += 1

    date_created = datetime.fromisoformat(edge_attrs["date-created"])

    try:
        node_attrs = H.nodes[n]
        creator_date_created = datetime.fromisoformat(node_attrs["date-created"])
    except KeyError as e:
        creator_date_created = datetime(1, 1, 1, 0, 0, 0)

    if date_created > datetime(2020, 1, 1, 0, 0, 0):
        starterpack_date_created.append(edge_attrs["date-created"])

    if creator_date_created > datetime(2020, 1, 1, 0, 0, 0):
        # filtering out nans
        account_age_at_creation.append((date_created - creator_date_created).days)
        if account_age_at_creation[-1] < 0:
            print(edge_attrs["date-created"], node_attrs["date-created"])


print("Calculated temporal information")

data = {}
data["num-nodes"] = num_nodes
data["num-edges"] = num_edges
data["degrees"] = degrees
data["edge-sizes"] = edge_sizes
data["min-degree"] = min_degree
data["max-degree"] = max_degree
data["mean-degree"] = mean_degree
data["median-degree"] = median_degree
data["mode-degree"] = mode_degree
data["min-edge-size"] = min_edge_size
data["max-edge-size"] = max_edge_size
data["mean-edge-size"] = mean_edge_size
data["median-edge-size"] = median_edge_size
data["mode-edge-size"] = mode_edge_size
data["components"] = [component_sizes.tolist(), component_numbers.tolist()]
data["date-created"] = starterpack_date_created
data["number-created"] = list(number_created.values())
data["account-age-at-creation"] = account_age_at_creation

datastring = json.dumps(data, indent=2)
makedirs("data", exist_ok=True)
with open("data/starterpack-stats.json", "w") as f:
    f.write(datastring)
