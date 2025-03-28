import numpy as np
from collections import defaultdict
from scipy import sparse
from tarjan import tarjan
import pandas as pd

df = pd.read_csv("data/deidentified_follows_edgelist.csv", header=None)
print("Loaded edgelist!", flush=True)
n_nodes = len(set(df[0].unique()).union(df[1].unique()))

out_degree = df.groupby(df[0]).agg("count")
in_degree = df.groupby(df[1]).agg("count")
print("Computed degrees!", flush=True)

source = df[0].to_numpy()
target = df[1].to_numpy()

gd = defaultdict(list)
for i, j in zip(source, target):
    gd[i].append(j)

print("Directed graph created!", flush=True)

scc = tarjan(gd)
sccs = [len(c) for c in scc]
print("Strongly connected component sizes computed!", flush=True)

gu = defaultdict(list)
for i, j in zip(np.concatenate([source, target]), np.concatenate([target, source])):
    gu[i].append(j)

print("Undirected graph created!", flush=True)

wcc = tarjan(gu)
wccs = [len(c) for c in wcc]
print("Weakly connected component sizes computed!", flush=True)

np.savetxt("data/follows_sccs.csv.gz", sccs, fmt="%d")
np.savetxt("data/follows_wccs.csv.gz", wccs, fmt="%d")
np.savetxt("data/follows_in-degree.csv.gz", in_degree, fmt="%d")
np.savetxt("data/follows_out-degree.csv.gz", out_degree, fmt="%d")
