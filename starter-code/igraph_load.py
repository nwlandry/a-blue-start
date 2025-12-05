from datetime import datetime
import gzip
import json
from os.path import join

import igraph as ig
import polars as pl
import xgi

base_dir = "/home/smith.alyss/"
EPOCH = datetime(1970, 1, 1)
print("starting; time is: ", flush=True)
print(datetime.now(), flush=True)

# load node list
nodes_path = join(base_dir, "deidentified_nodes.jsonl.gz")
nodes = pl.read_ndjson(nodes_path, schema={"date-created": pl.Date, "id": pl.Int64, "active": pl.String, "status": pl.String})
nodes = nodes.with_columns((pl.col("date-created") - EPOCH).dt.total_days().alias("created_days_since_epoch"))
print("nodes loaded; time is: ", flush=True)
print(datetime.now(), flush=True)

# load hypergraph
sp_path = join(base_dir, "deidentified_starterpack_hif.json.gz")
with gzip.open(sp_path, "rt", encoding="utf-8") as f:
    hif_dict = json.load(f)
H = xgi.from_hif_dict(hif_dict)
for e in H.edges:
    edge_attrs = H.edges[e]
    days_since_epoch = (datetime.fromisoformat(edge_attrs["date-created"]) - EPOCH).days
    H.edges[e]['created_days_since_epoch'] = days_since_epoch
print("hypergraph loaded; time is: ", flush=True)
print(datetime.now(), flush=True)

G = ig.Graph(n=len(nodes))
with gzip.open(join(base_dir, 'deidentified_follows_edgelist.csv.gz'), 'rt', encoding="utf-8") as f:
    tmp_edges = []
    tmp_ts = {}
    for ix, row in enumerate(f.readlines()):
        spl = row.strip().split(',')
        i = int(spl[0])
        j = int(spl[1])
        days_since_epoch = -1
        try:
            days_since_epoch = (datetime.fromisoformat(spl[2]) - EPOCH).days
        except Exception as e:
            print(e, flush=True)
            print(spl, flush=True)
        tmp_edges.append((i, j))
        tmp_ts[ix] = days_since_epoch

        if (ix + 1) % 250000000 == 0: # note that this will slow down each iteration. adding edges in roughly 10% batched increments limits RAM usage.
            G.add_edges(tmp_edges)
            tmp_edges = []
            for ix, ts in tmp_ts.items():
                G.es[ix]['created_days_since_epoch'] = ts
            tmp_ts = {}
            print('edges loaded successfully', ix, flush=True)
            print(datetime.now(), flush=True)


G.add_edges(tmp_edges)
for ix, ts in tmp_ts.items():
    G.es[ix]['created_days_since_epoch'] = ts
print("follow graph loaded; time is: ", flush=True)
print(datetime.now(), flush=True)
