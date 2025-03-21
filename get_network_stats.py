import numpy as np
from dask import dataframe
from scipy import sparse
from tarjan import tarjan

df = dataframe.read_csv("data/deidentified_follows_edgelist.csv", header=None)
print("Loaded edgelist!")
n_nodes = len(set(df[0].unique()).union(df[1].unique()))

out_degree = df.groupby(df[0]).agg("count")
in_degree = df.groupby(df[1]).agg("count")
print("Computed degrees!")

source = df[0].to_dask_array(lengths=True)
target = df[1].to_dask_array(lengths=True)
mtx = sparse.coo_array((np.ones(len(df)), (source, target)))
mtx = mtx.tocsr()
print(mtx.shape)

gd = {}
for node in range(n_nodes):
    gd[node] = mtx[[node], :].nonzero()[1]

print("Directed graph created!")

scc = tarjan(gd)
sccs = [len(c) for c in scc]


mtx = sparse.coo_array(
    (np.ones(2 * len(df)), (source.concatenate(target), target.concatenate(source)))
)
mtx = mtx.tocsr()
print(mtx.shape)

gu = {}
for node in range(n_nodes):
    gu[node] = mtx[[node], :].nonzero()[1]

print("Undirected graph created!")

wcc = tarjan(gu)
wccs = [len(c) for c in wcc]

np.savetxt("data/follows_sccs.csv.gz", sccs, fmt="%d")
np.savetxt("data/follows_wccs.csv.gz", wccs, fmt="%d")
np.savetxt("data/follows_in-degree.csv.gz", in_degree, fmt="%d")
np.savetxt("data/follows_out-degree.csv.gz", out_degree, fmt="%d")
