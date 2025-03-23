import numpy as np
import json

import xgi

H = xgi.read_hif("data/deidentified_starterpack_hif.json")

smin = 1
smax = 400

idx1 = []
idx2 = []
weights = []
edges = H.edges.members(dtype=dict)
for i, e1 in edges.items():
    for j in H.edges.neighbors(i):
        if i < j:
            idx1.append(np.uint32(i))
            idx2.append(np.uint32(j))
            e2 = H.edges.members(j)
            weights.append(np.uint16(len(e1.intersection(e2))))
            print(i, j, flush=True)

num_nodes = {}
num_edges = {}
for s in range(smin, smax + 1):
    nodes = set()
    num_edges = 0

    for i, j, w in zip(idx1, idx2, weights):
        nodes.add(i)
        nodes.add(j)
        if w >= s:
            num_edges += 1
    print(s, flush=True)
    num_nodes[s] = len(nodes)
    num_edges[s] = num_edges

data = {}
data["num-nodes"] = num_nodes
data['num-edges'] = num_edges

datastring = json.dumps(data, indent=2)

with open("data/s-line-stats.json", "w") as f:
    f.write(datastring)