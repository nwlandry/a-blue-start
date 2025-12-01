import datetime as dt
import gzip
from os.path import join

import networkx as nx

base_dir = "./data/"
G = nx.DiGraph()
with gzip.open(join(base_dir, 'deidentified_follows_edgelist.csv.gz'), 'r') as f:
    for row in f.readlines():       
            spl = row.decode("utf-8").strip().split(',')
            i = int(spl[0])
            j = int(spl[1])
            date_followed = None
            try:
                date_followed = dt.datetime.strptime(spl[2], '%Y-%m-%d')
            except Exception as e:
                print(e)
                print(spl)
            G.add_edge(i, j, date_followed=date_followed)
