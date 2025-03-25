#!/usr/bin/env python3
import xgi
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
import argparse
import os
import seaborn as sns

def process_chunk(hyperedge_list, max_pack_size=100, sample_size=1000):
    """
    Function to process a chunk of hyperedges (starter packs).
    Counts user pair co-occurrences within these hyperedges.
    """
    local_counter = collections.Counter()

    for users_in_hedge in hyperedge_list:
        if len(users_in_hedge) < 2:
            continue

        if len(users_in_hedge) <= max_pack_size:
            for i in range(len(users_in_hedge)):
                user_i = users_in_hedge[i]
                for j in range(i+1, len(users_in_hedge)):
                    user_j = users_in_hedge[j]
                    sorted_pair = (min(user_i, user_j), max(user_i, user_j))
                    local_counter[sorted_pair] += 1
        else:
            # For large starter packs, apply an approximation method
            total_pairs = len(users_in_hedge) * (len(users_in_hedge) - 1) // 2
            if total_pairs <= sample_size:
                user_pairs = list(itertools.combinations(users_in_hedge, 2))
                sampled_pairs = user_pairs
            else:
                sampled_pairs = set()
                while len(sampled_pairs) < sample_size:
                    user_i, user_j = random.sample(users_in_hedge, 2)
                    sorted_pair = (min(user_i, user_j), max(user_i, user_j))
                    sampled_pairs.add(sorted_pair)
            scaling_factor = total_pairs / len(sampled_pairs)
            for pair in sampled_pairs:
                local_counter[pair] += scaling_factor

    return local_counter

def main():
    parser = argparse.ArgumentParser(description='Compute and plot the probability distribution of user pair co-occurrences in starter packs.')
    parser.add_argument('input_filepath', type=str, help='Path to the input hypergraph file (HIF format).')
    parser.add_argument('output_figure_path', type=str, help='Path to save the output figure.')
    parser.add_argument('--max_pack_size', type=int, default=100, help='Maximum starter pack size to process fully.')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of user pairs to sample in large starter packs.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for parallel processing.')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Number of hyperedges per processing chunk.')
    args = parser.parse_args()

    if not os.path.isfile(args.input_filepath):
        print(f"Input file {args.input_filepath} does not exist.")
        return

    print("Reading the hypergraph...")
    H = xgi.read_hif(args.input_filepath)
    print("Hypergraph loaded.")

    all_hyperedges_data = [list(H.edges.members(hedge)) for hedge in H.edges]

    hyperedge_chunks = [all_hyperedges_data[i:i + args.chunk_size] for i in range(0, len(all_hyperedges_data), args.chunk_size)]

    cooccurrence_counter = collections.Counter()

    total_chunks = len(hyperedge_chunks)
    print(f"Processing {len(all_hyperedges_data)} starter packs in {total_chunks} chunks using {args.num_workers} workers...")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_chunk, chunk, args.max_pack_size, args.sample_size): i for i, chunk in enumerate(hyperedge_chunks)}
        for future in tqdm(as_completed(futures), total=total_chunks):
            local_counter = future.result()
            cooccurrence_counter.update(local_counter)

    cooccurrence_counts_list = list(cooccurrence_counter.values())
    cooccurrence_counts_list = [int(round(count)) for count in cooccurrence_counts_list]

    cooccurrence_counts_list = [count for count in cooccurrence_counts_list if count > 0]

    min_count = min(cooccurrence_counts_list)
    max_count = max(cooccurrence_counts_list)

    num_bins = 100
    bins = np.logspace(np.log10(min_count), np.log10(max_count), num=num_bins)

    hist, bin_edges = np.histogram(cooccurrence_counts_list, bins=bins, density=False)

    # normalize the histogram to get probabilities
    probabilities = hist / hist.sum()

    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    plt.loglog(bin_centers, probabilities, 'ko', markersize=2)
    sns.despine()
    plt.xlabel('Number of co-occurrences')
    plt.ylabel('Probability')
    plt.title('User Pair Co-Occurrence Distribution')
    plt.tight_layout()
    plt.savefig(args.output_figure_path)
    print(f"Figure saved to {args.output_figure_path}")

if __name__ == '__main__':
    main()