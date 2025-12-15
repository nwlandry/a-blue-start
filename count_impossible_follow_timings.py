from collections import defaultdict
from os.path import join

import numpy as np
import polars as pl

base_dir = "/home/smith.alyss/"

df = pl.read_csv(
    join(base_dir, "deidentified_follows_edgelist.csv.gz"),
    schema={"from": pl.Int64, "to": pl.Int64, "date_followed": pl.Date},
    has_header=False,
)
print("Loaded edgelist; there are this many following events", len(df), flush=True)

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

ts_incorrect = (
    df.join(
        nodes.select(["id", "date-created"]), left_on="to", right_on="id", how="left"
    )
    .join(
        nodes.select(["id", "date-created"]),
        left_on="from",
        right_on="id",
        suffix="_from",
        how="left",
    )
    .filter(
        (pl.col("date_followed") < pl.col("date-created_from"))
        | (pl.col("date_followed") < pl.col("date-created"))
    )
)
print("there are this many wrong timestamps:", len(ts_incorrect), flush=True)
