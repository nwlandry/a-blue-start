import argparse
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import xgi
from tqdm import tqdm


def compute_s_line_graph_counts(
    H: xgi.Hypergraph,
    s_min: int,
    s_max: int,
    max_hyperedge_size=None,
    num_processes=None,
):
    if not num_processes:
        num_processes = cpu_count()
    print("Filtering hyperedges based on s_min and max_hyperedge_size...", flush=True)
    if max_hyperedge_size is not None:
        filtered_edges = [
            e for e in H.edges if s_min <= len(H.edges.members(e)) <= max_hyperedge_size
        ]
    else:
        filtered_edges = [e for e in H.edges if len(H.edges.members(e)) >= s_min]
    print(f"Number of hyperedges after filtering: {len(filtered_edges)}", Flush=True)

    filtered_nodes = set()
    for e in filtered_edges:
        filtered_nodes.update(H.edges.members(e))
    filtered_nodes = list(filtered_nodes)
    print(f"Number of nodes after filtering: {len(filtered_nodes)}", flush=True)

    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(filtered_nodes)}
    num_nodes = len(node_id_to_idx)

    edge_id_to_idx = {edge_id: idx for idx, edge_id in enumerate(filtered_edges)}
    num_edges = len(edge_id_to_idx)

    print("Creating edge memberships...", flush=True)
    edge_memberships = []
    for edge_id in filtered_edges:
        members = [node_id_to_idx[node] for node in H.edges.members(edge_id)]
        edge_memberships.append(np.array(members, dtype=np.uint32))

    del H

    print("Creating node to edges mapping...", flush=True)
    node_to_edges_list = [[] for _ in range(num_nodes)]
    for edge_idx, nodes in enumerate(edge_memberships):
        for node_idx in nodes:
            node_to_edges_list[node_idx].append(edge_idx)
    node_to_edges = [
        np.array(edge_indices, dtype=np.uint32) for edge_indices in node_to_edges_list
    ]
    del node_to_edges_list

    s_values = list(range(s_min, s_max + 1))

    edge_indices = np.arange(num_edges)
    edge_batches = np.array_split(edge_indices, num_processes)

    worker_args = []
    for batch_edges in edge_batches:
        worker_args.append((batch_edges, edge_memberships, node_to_edges, s_min))
    with Pool(processes=num_processes) as pool:
        print("Processing hyperedge blocks in parallel...", flush=True)
        results = list(
            tqdm(
                pool.imap_unordered(process_edges_batch, worker_args),
                total=num_processes,
            )
        )
    print("Combining results...", flush=True)
    row_inds = []
    col_inds = []
    data_vals = []
    for row_ind, col_ind, data in results:
        row_inds.append(row_ind)
        col_inds.append(col_ind)
        data_vals.append(data)

    row_inds = np.concatenate(row_inds)
    col_inds = np.concatenate(col_inds)
    data_vals = np.concatenate(data_vals)
    del results

    print("Computing s-line graph statistics...", flush=True)
    stats = {}
    for s in s_values:
        valid_idx = np.where(data_vals >= s)[0]
        if valid_idx.size > 0:
            edges_row = row_inds[valid_idx]
            edges_col = col_inds[valid_idx]
            all_edges = np.concatenate([edges_row, edges_col])
            unique_nodes = np.unique(all_edges)
            num_nodes_s = unique_nodes.size
            num_edges_s = valid_idx.size
            stats[s] = {"num_nodes": int(num_nodes_s), "num_edges": int(num_edges_s)}
    return stats


def process_edges_batch(args):
    batch_edges, edge_memberships, node_to_edges, s_min = args
    row_inds = []
    col_inds = []
    data_vals = []
    for e_i in batch_edges:
        nodes_ei = edge_memberships[e_i]
        nodes_ei_set = set(nodes_ei)
        # Create a set of edges that share nodes with e_i
        neighbor_edges = set()
        for node_idx in nodes_ei:
            neighbor_edges.update(node_to_edges[node_idx])
        # For e_j > e_i to avoid duplicate pairs
        neighbor_edges = [e_j for e_j in neighbor_edges if e_j > e_i]
        for e_j in neighbor_edges:
            nodes_ej = edge_memberships[e_j]
            if len(nodes_ej) < s_min:
                continue
            overlap_count = len(nodes_ei_set.intersection(nodes_ej))
            if overlap_count >= s_min:
                row_inds.append(e_i)
                col_inds.append(e_j)
                data_vals.append(overlap_count)
    return (
        np.array(row_inds, dtype=np.uint32),
        np.array(col_inds, dtype=np.uint32),
        np.array(data_vals, dtype=np.uint16),
    )


def save_graph_stats(stats, stats_output_path):
    if stats:
        with open(stats_output_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Saved graph statistics to {stats_output_path}", flush=True)
    else:
        print("No statistics to save.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute s-line graph statistics from a hypergraph."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input hypergraph file."
    )
    parser.add_argument(
        "--stats_output",
        type=str,
        required=True,
        help="File path to save the number of nodes and edges in each s-line graph.",
    )
    parser.add_argument(
        "--s_min", type=int, required=True, help="Minimum s value (inclusive)."
    )
    parser.add_argument(
        "--s_max", type=int, required=True, help="Maximum s value (inclusive)."
    )
    parser.add_argument(
        "--max_hyperedge_size",
        type=int,
        required=False,
        default=None,
        help="Maximum hyperedge size to consider (inclusive).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        required=False,
        default=None,
        help="Number of processes to use in the computation.",
    )
    args = parser.parse_args()

    input_path = args.input
    stats_output_path = args.stats_output
    s_min = args.s_min
    s_max = args.s_max
    max_hyperedge_size = args.max_hyperedge_size
    num_processes = args.num_processes

    print("Reading hypergraph...", flush=True)
    H = xgi.read_hif(input_path)
    # H = xgi.load_xgi_data("diseasome")
    print(
        f"Read hypergraph with {H.num_nodes} nodes and {H.num_edges} hyperedges.",
        flush=True,
    )

    print("Computing s-line graph statistics...", flush=True)
    stats = compute_s_line_graph_counts(
        H, s_min, s_max, max_hyperedge_size, num_processes
    )

    save_graph_stats(stats, stats_output_path)


if __name__ == "__main__":
    main()
