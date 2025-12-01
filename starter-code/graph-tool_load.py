from graph_tool.all import *
from os.path import join
import datetime as dt
import gzip

base_dir = './data/'
G = Graph(directed=True)
print("new GT graph")
timestamp = G.new_edge_property("object")
with gzip.open(join(base_dir, 'deidentified_follows_edgelist.csv.gz'), 'r') as f:
    for row in f.readlines():
        try:
            spl = row.decode("utf-8").strip().split(',')
            i = int(spl[0])
            j = int(spl[1])
            e = G.add_edge(i, j)
            timestamp[e] = None
            ts = dt.datetime.strptime(spl[2], '%Y-%m-%d')
            timestamp[e] = ts
        except Exception as e:
            print(e, flush=True)
            print(spl, flush=True)
G.ep['date_followed'] = timestamp
print("built graph")
