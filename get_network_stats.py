from collections import defaultdict
from os.path import join

import numpy as np
import polars as pl
from tarjan import tarjan

base_dir = "/Users/yyu8dx/Library/CloudStorage/OneDrive-UniversityofVirginia/Research/bluesky/SOMAR"

df = pl.read_csv(
    join(base_dir, "deidentified_follows_edgelist.csv"),
    new_columns=["from", "to", "date_followed"],
    has_header=False,
)
print("Loaded edgelist!", flush=True)

out_degree = df["from"].value_counts()["count"].to_numpy()
in_degree = df["to"].value_counts()["count"].to_numpy()
follow_volume = df["date_followed"].value_counts().to_numpy()

print("Computed degrees!", flush=True)

gd = defaultdict(list)
gu = defaultdict(list)
for i, j, _ in df.iter_rows():
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
np.savetxt("data/follows_volume.csv", follow_volume, fmt=["%s", "%d"], delimiter=",")
np.savetxt("data/follows_in-degree.csv.gz", in_degree, fmt="%d")
np.savetxt("data/follows_out-degree.csv.gz", out_degree, fmt="%d")
