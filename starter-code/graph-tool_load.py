import gzip
import json
import random
from datetime import datetime
from os.path import join

import polars as pl
import xgi
from graph_tool.all import *

base_dir = "/home/smith.alyss/"
EPOCH = datetime(1970, 1, 1)
print("starting; time is: ", flush=True)
print(datetime.now(), flush=True)

# load node list
nodes_path = join(base_dir, "deidentified_nodes.jsonl.gz")
nodes = pl.read_ndjson(
    nodes_path,
    schema={
        "date-created": pl.Date,
        "id": pl.Int64,
        "active": pl.String,
        "status": pl.String,
    },
)
nodes = nodes.with_columns(
    (pl.col("date-created") - EPOCH).dt.total_days().alias("created_days_since_epoch")
)
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
    H.edges[e]["created_days_since_epoch"] = days_since_epoch
print("hypergraph loaded; time is: ")
print(datetime.now(), flush=True)

# load following graph
G = Graph(directed=True)
print("new GT graph", flush=True)
created_days_since_epoch = G.new_edge_property("int")
with gzip.open(
    join(base_dir, "deidentified_follows_edgelist.csv.gz"), "rt", encoding="utf-8"
) as f:
    tmp_edges = []
    for ix, row in enumerate(f.readlines()):
        try:
            spl = row.strip().split(",")
            i = int(spl[0])
            j = int(spl[1])
            ts = (datetime.fromisoformat(spl[2]) - EPOCH).days
            tmp_edges.append((i, j, ts))
        except Exception as e:
            print(e, flush=True)
            print(spl, flush=True)
        if (
            ix + 1
        ) % 100000000 == 0:  # adjustable to suit your RAM availability; 100 million = ~6 minutes of loading pretty consistently
            G.add_edge_list(tmp_edges, eprops=[created_days_since_epoch])
            tmp_edges = []
            print("processed edges up till", ix, flush=True)
            print(datetime.now(), flush=True)

G.add_edge_list(tmp_edges, eprops=[created_days_since_epoch])
G.ep["created_days_since_epoch"] = created_days_since_epoch
print("built graph")
print("following loaded; time is: ", flush=True)
print(datetime.now(), flush=True)
