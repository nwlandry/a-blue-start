# a-blue-start

This repository accompanies the preprint ["A Blue Start: A large-scale pairwise and higher-order social network dataset"](https://arxiv.org/abs/2505.11608) by Alyssa Smith, Ilya Amburg, Sagar Kumar, Brooke Foucault Welles, and Nicholas W. Landry. It provides all codes for reproducing the analyses and figures in the manuscript.

## Repository structure:
* `data` contains data on the network statistics plotted in Figs. 1-6 in the paper.
* `figures` contains the Figs. 1-6 in the paper.
* `starter-code` contains example code for loading the datasets into [igraph](https://python.igraph.org/en/stable/) and [graph-tool](https://graph-tool.skewed.de/).

## Scripts
The following scripts are used to generate statistics from the network data:
* `get_degree_sequences.py`: This script is used to get the in- and out-degree sequences for the following network; it saves them as .csv or .csv.gz files. 
* `get_component_sizes.py`: This script obtains the sizes of the weakly and strongly connected components for the following network and saves them as gzipped CSV files.
* `get_starterpack_clustering`: This script is used to get node cluster labels based on the Leiden algorithm and then compute the normalized entropies of each edge.
* `get_starterpack_k_core.py`: This script performs a k-core decomposition of the starter packs.
* `get_starterpack_kendall_tau.py`: This script compares nodal rankings from the starter pack network and compares to the following network ranking using the Kendall Tau measure.
* `get_starterpack_pair_cooccurrence.py`: This script returns the distribution of two-node co-occurrence frequencies in the starter pack network.
```python
python get_pair_co-occurrence.py --input_filepath "deidentified_starterpack_hif.json.gz" --max_pack_size 4069 --num_workers 10
```
* `get_starterpack_pair_s_line_counts.py`: This script returns the number of nodes and edges in the s-line graph for $s=1,2,\dots,345$. It can be run as follows:
```python
python s_line_count.py --input_filepath "deidentified_starterpack_hif.json.gz" --smin 1 --smax 345 --output data/s_count.csv
```
* `get_starterpack_stats.py`: This script returns basic statistics of the starter pack network as a JSON file.


## Plotting

* `plot_network_stats.ipynb`: This notebook plots Fig. 4 and also prints the basic network statistics in a readable way.
* `plot_starterpack_stats.ipynb`: This notebook plots Figs. 1-3, 5 and also prints the basic starter pack statistics in a readable way.

## Starter Code
* `starter-code/graph-tool_load.py` loads the node dataset as a [polars](https://pola.rs/) dataframe, the starter pack dataset in [XGI](https://xgi.readthedocs.io/en/stable/), and the following network in [graph-tool](https://graph-tool.skewed.de/). This code uses about 310 GB of RAM and takes about 2.75 hours to run.
* `starter-code/igraph_load.py` loads the node dataset as a [polars](https://pola.rs/) dataframe, the starter pack dataset in [XGI](https://xgi.readthedocs.io/en/stable/)    , and the following network in [igraph](https://python.igraph.org/en/stable/). This code uses about 460 GB of RAM and took about 5.5 hours to run.

## Getting started

Start by downloading the [data](https://doi.org/10.3886/kf6e-zq94) from the Social Media Archive @ ICPSR (SOMAR).
