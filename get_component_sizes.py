from collections import defaultdict
import gzip
from os.path import join

import numpy as np
from tarjan import tarjan

base_dir = '/home/smith.alyss/'

gd = defaultdict(list)
gu = defaultdict(list)
with gzip.open(join(base_dir, 'deidentified_follows_edgelist.csv.gz'), 'r') as f:
    for row in f.readlines():
        spl = row.decode("utf-8").split(',')
        i = int(spl[0])
        j = int(spl[1])
        gd[i].append(j)
        gu[i].append(j)
        gu[j].append(i)

print("Directed and undirected graphs created!", flush=True)

scc = tarjan(gd)
sccs = [len(c) for c in scc]
print("Strongly connected component sizes computed!", flush=True)

wcc = tarjan(gu)
wccs = [len(c) for c in wcc]
print("Weakly connected component sizes computed!", flush=True)

np.savetxt("data/follows_sccs.csv.gz", sccs, fmt="%d")
np.savetxt("data/follows_wccs.csv.gz", wccs, fmt="%d")
