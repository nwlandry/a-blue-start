import xgi
import gzip
import numpy as np
import scipy.stats as ss
import json
from collections import defaultdict


def count_node_degrees(file_path):
    node_degrees = defaultdict(int)

    with gzip.open(file_path, 'rt') as f:
        for i, row in enumerate(f):
            node1, node2 = row.strip().split(",")
            if i % 10000000 == 0:
                print(f"Processed {i} rows")
            node_degrees[int(node1)] += 1
            node_degrees[int(node2)] += 1
    return dict(node_degrees)

def compare_top_k(d_f, d_s, k):
    top_g = sorted(d_f.items(), key=lambda x: x[1], reverse=True)[:k]
    top_h = sorted(d_s.items(), key=lambda x: x[1], reverse=True)[:k]
    
    top_g_nodes = {node for node, _ in top_g}
    top_h_nodes = {node for node, _ in top_h}
    
    common_nodes = top_g_nodes.intersection(top_h_nodes)
    
    common_nodes_list = sorted(common_nodes)
    
    cent1 = np.array([d_f[node] for node in common_nodes_list])
    cent2 = np.array([d_s[node] for node in common_nodes_list])
    
    return common_nodes_list, cent1, cent2

def _log_bin_stats(x, y, nbins=30, reducer=np.mean):
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])          # geometric mid‑points
    y_binned = np.full(nbins, np.nan)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= lo) & (x < hi)
        if mask.any():
            y_binned[i] = reducer(y[mask])

    ok = ~np.isnan(y_binned)
    return centres[ok], y_binned[ok]

d_f = count_node_degrees('data/deidentified_follows_edgelist.csv.gz')
H = xgi.read_hif('data/deidentified_starterpack_hif.json')
d_s = H.degree()

nbins = 30
common_nodes_list, cent1, cent2 = compare_top_k(d_f, d_s, k=1000000)

index_start=2

ran = np.logspace(np.log10(index_start - 1), np.log10(len(cent1) - 1), 500, dtype=int)

sp = np.argsort(cent1)
c1, c2 = cent1[sp], cent2[sp]
tau_12 = np.array([ss.kendalltau(c1[-s:], c2[-s:], variant="b")[0] for s in ran])

sp = np.argsort(cent2)
c1, c2 = cent1[sp], cent2[sp]
tau_21 = np.array([ss.kendalltau(c2[-s:], c1[-s:], variant="b")[0] for s in ran])

xs = ran + 1

x1, y1 = _log_bin_stats(xs, tau_12, nbins=nbins)
x2, y2 = _log_bin_stats(xs, tau_21, nbins=nbins)


data = {}
data["x1"] = x1.tolist()
data["x2"] = x2.tolist()
data["y1"] = y1.tolist()
data["y2"] = y2.tolist()

with open("data/starterpack_kendall_tau.json", "w") as f:
    f.write(json.dumps(data))