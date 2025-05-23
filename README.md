# a-blue-start

This repository accompanies the preprint ["A Blue Start: A large-scale pairwise and higher-order social network dataset"](https://arxiv.org/abs/2505.11608) by Alyssa Smith, Ilya Amburg, Sagar Kumar, Brooke Foucault Welles, and Nicholas W. Landry. It provides all codes for reproducing the analyses and figures in the manuscript.

## Repository structure:
* `data` contains data on the network statistics plotted in Figs. 1-6 in the paper.
* `figures` contains the Figs. 1-6 in the paper.

## Scripts
The following scripts are used to generate statistics from the network data:
* `get_network_stats.py`: This script is used to get the in- and out-degree sequences and the sizes of the weakly and strongly connected components for the following network and save them as a JSON file.
* `get_starterpack_clustering`: This script is used to get node cluster labels based on the Leiden algorithm and then compute the normalized entropies of each edge.
* `get_starterpack_k-core.py`: This script performs a k-core decomposition of the starter packs.
* `get_starterpack_kendall_tau.py`: This script compares nodal rankings from the starter pack network and compares to the following network ranking using the Kendall Tau measure.
* `get_starterpack_pair_co-occurrence.py`: This script returns the distribution of two-node co-occurrence frequencies in the starter pack network.
```python
python get_pair_co-occurrence.py --input_filepath "data/deidentified_starterpack_hif.json" --max_pack_size 4070 --num_workers 10
```
* `get_starterpack_pair_s-line_count.py`: This script returns the number of nodes and edges in the s-line graph for $s=1,2,\dots,345$. It can be run as follows:
```python
python s_line_count.py data/deidentified_starterpack_hif.json --smin 1 --smax 345 --output data/s_count.txt
```
* `get_starterpack_stats.py`: This script returns basic statistics of the starter pack network as a JSON file.


## Plotting

* `plot_network_stats.ipynb`: This notebook plots Fig. 5 and also prints the basic network statistics in a readable way.
* `plot_starterpack_stats.ipynb`: This notebook plots Figs. 1-4, 6 and also prints the basic starter pack statistics in a readable way.

## Getting started

Start by downloading the [data](https://socialmediaarchive.org/record/78) from the Social Media Archive @ ICPSR (SOMAR).
