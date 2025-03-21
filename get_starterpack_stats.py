import numpy as np
import xgi

H = xgi.read_hif("data/SOMAR/deidentified_starterpack_hif.json")

cc = [len(c) for c in xgi.connected_components(H)]
component_sizes, component_numbers = np.unique(cc, return_counts=True)

edge_sizes = H.edges.size.asnumpy()
degrees = H.nodes.degree.asnumpy()
