import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import xgi

H = xgi.read_hif("data/deidentified_starterpack_hif.json")

cc = [len(c) for c in xgi.connected_components(H)]
component_sizes, component_numbers = np.unique(cc, return_counts=True)

edge_sizes = H.edges.size.aslist()
degrees = H.nodes.degree.aslist()
num_nodes = H.num_nodes
num_edges = H.num_edges

number_created = defaultdict(lambda: 0)
account_age_at_creation = []
for e in H.edges:
    attrs = H.edges[e]
    number_created[attrs["creator-id"]] += 1
    date_created = datetime.fromisoformat(attrs["date-created"])
    creator_date_created = datetime.fromisoformat(attrs["creator-date-created"])
    if creator_date_created != datetime(1, 1, 1, 0, 0, 0):
        # filtering out nans
        account_age_at_creation.append((date_created - creator_date_created).days)

date_created = H.edges.attrs("date-created").aslist()

data = {}
data["num-nodes"] = num_nodes
data["num-edges"] = num_edges
data["degrees"] = degrees
data["edge-sizes"] = edge_sizes
data["components"] = [component_sizes.tolist(), component_numbers.tolist()]
data["date-created"] = date_created
data["number-created"] = list(number_created.values())
data["account-age-at-creation"] = account_age_at_creation

datastring = json.dumps(data, indent=2)
with open("data/starterpack-stats.json", "w") as f:
    f.write(datastring)
