from collections import defaultdict
from os.path import join

import numpy as np
import polars as pl

base_dir = "/home/smith.alyss/"

df = pl.read_csv(
    join(base_dir, "deidentified_follows_edgelist.csv.gz"),
    new_columns=["from", "to", "date_followed"],
    has_header=False,
)
print("Loaded edgelist!", flush=True)

out_degree = df["from"].value_counts()["count"].to_numpy()
in_degree = df["to"].value_counts()["count"].to_numpy()
follow_volume = df["date_followed"].value_counts().to_numpy()

df = df.with_columns(
    pl.col("date_followed").str.to_datetime().dt.epoch("s").alias("datetime")
)

SECONDS_PER_DAY = 86_400

in_degree_std = (
    df.group_by("to")
    .agg(
        (pl.col("datetime").std() / SECONDS_PER_DAY).alias("in_degree_std")
    )  # std in seconds  # seconds → days
    .select("in_degree_std")
    .to_series()
    .to_list()
)

out_degree_std = (
    df.group_by("from")
    .agg(
        (pl.col("datetime").std() / SECONDS_PER_DAY).alias("out_degree_std")
    )  # std in seconds  # seconds → days
    .select("out_degree_std")
    .to_series()
    .to_list()
)

in_degree_std = [np.nan if x is None else x for x in in_degree_std]
out_degree_std = [np.nan if x is None else x for x in out_degree_std]
print("Computed degrees!", flush=True)

np.savetxt("data/in_degree_std.csv.gz", in_degree_std, fmt="%f")
np.savetxt("data/out_degree_std.csv.gz", out_degree_std, fmt="%f")
np.savetxt("data/follows_volume.csv", follow_volume, fmt=["%s", "%d"], delimiter=",")
np.savetxt("data/follows_in-degree.csv.gz", in_degree, fmt="%d")
np.savetxt("data/follows_out-degree.csv.gz", out_degree, fmt="%d")
