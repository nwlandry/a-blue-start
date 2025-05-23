#!/usr/bin/env python3
import argparse
import collections
import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgi
from tqdm import tqdm


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
                for j in range(i + 1, len(users_in_hedge)):
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
    parser = argparse.ArgumentParser(
        description="Compute and plot the probability distribution of user pair co-occurrences in starter packs."
    )
    parser.add_argument(
        "--input_filepath",
        type=str,
        help="Path to the input hypergraph file (HIF format).",
    )
    parser.add_argument(
        "--max_pack_size",
        type=int,
        default=100,
        help="Maximum starter pack size to process fully.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of user pairs to sample in large starter packs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of hyperedges per processing chunk.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_filepath):
        print(f"Input file {args.input_filepath} does not exist.")
        return

    print("Reading the hypergraph...")
    H = xgi.read_hif(args.input_filepath)
    print("Hypergraph loaded.")

    all_hyperedges_data = [list(H.edges.members(hedge)) for hedge in H.edges]

    hyperedge_chunks = [
        all_hyperedges_data[i : i + args.chunk_size]
        for i in range(0, len(all_hyperedges_data), args.chunk_size)
    ]

    cooccurrence_counter = collections.Counter()

    total_chunks = len(hyperedge_chunks)
    print(
        f"Processing {len(all_hyperedges_data)} starter packs in {total_chunks} chunks using {args.num_workers} workers..."
    )

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_chunk, chunk, args.max_pack_size, args.sample_size
            ): i
            for i, chunk in enumerate(hyperedge_chunks)
        }
        for future in tqdm(as_completed(futures), total=total_chunks):
            local_counter = future.result()
            cooccurrence_counter.update(local_counter)

    cooccurrence_counts = [
        round(count) for count in cooccurrence_counter.values() if round(count) > 0
    ]

    np.savetxt(
        "data/starterpack_pair_co-occurrence.csv.gz", cooccurrence_counts, fmt="%d"
    )


if __name__ == "__main__":
    main()
