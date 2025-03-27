import xgi
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import argparse
import gc


def relabel_and_extract_hyperedges(H, smin):
    """
    Relabels nodes to integers starting from 0 and returns a list of hyperedges as sets,
    excluding hyperedges smaller than smin.
    """
    xgi.convert_labels_to_integers(H, in_place=True)
    hyperedges = [set(H.edges.members(e)) for e in H.edges if len(H.edges.members(e)) >= smin]
    del H
    gc.collect()
    return hyperedges


def build_inverted_index(hyperedges):
    """
    Builds an inverted index: node -> set of hyperedge indices.
    """
    index = defaultdict(set)
    for i, hedge in enumerate(hyperedges):
        for node in hedge:
            index[node].add(i)
    return index


def worker_task(args):
    """
    Worker task to process a chunk of hyperedges.
    Returns a dict of s -> edge_count and a dict of s -> set of active node indices.
    """
    start, end, hyperedges, inverted_index, smin, smax = args
    edge_counts = defaultdict(int)
    active_nodes = defaultdict(set)

    for i in range(start, end):
        hedge = hyperedges[i]
        candidates = defaultdict(int)
        for node in hedge:
            for j in inverted_index[node]:
                if j > i:
                    candidates[j] += 1
        for j, overlap in candidates.items():
            if overlap >= smin:
                for s in range(smin, min(overlap, smax) + 1):
                    edge_counts[s] += 1
                    active_nodes[s].update([i, j])
    return edge_counts, active_nodes


def parallel_count_s_line_graph(hyperedges, inverted_index, smin, smax, num_workers=None):
    """
    parallel computation of s-line graph stats for all s in [smin, smax].
    """
    num_edges = len(hyperedges)
    num_workers = num_workers or mp.cpu_count()
    chunk_size = (num_edges + num_workers - 1) // num_workers
    chunks = [(i, min(i + chunk_size, num_edges), hyperedges, inverted_index, smin, smax)
              for i in range(0, num_edges, chunk_size)]

    edge_counts_total = defaultdict(int)
    active_nodes_total = defaultdict(set)

    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for edge_counts, active_nodes in pool.imap_unordered(worker_task, chunks):
                for s in edge_counts:
                    edge_counts_total[s] += edge_counts[s]
                    active_nodes_total[s].update(active_nodes[s])
                pbar.update(1)

    result = {s: (len(active_nodes_total[s]), edge_counts_total[s]) for s in range(smin, smax + 1)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute s-line graph summary statistics.")
    parser.add_argument("hypergraph_file", help="Path to hypergraph edge list")
    parser.add_argument("--smin", type=int, default=1, help="Minimum s value (inclusive)")
    parser.add_argument("--smax", type=int, default=3, help="Maximum s value (inclusive)")
    parser.add_argument("--output", default="sline_summary.txt", help="Output file for results")

    args = parser.parse_args()

    print(f"reading hypergraph from: {args.hypergraph_file}")

    H = xgi.read_hif(args.hypergraph_file)

    hyperedges = relabel_and_extract_hyperedges(H, args.smin)
    inverted_index = build_inverted_index(hyperedges)

    print(f"Computing s-line graph statistics for s in [{args.smin}, {args.smax}]...")
    results = parallel_count_s_line_graph(hyperedges, inverted_index, args.smin, args.smax)

    with open(args.output, 'w') as out:
        out.write("s\tnodes\tedges\n")
        for s in range(args.smin, args.smax + 1):
            nodes, edges = results[s]
            out.write(f"{s}\t{nodes}\t{edges}\n")

    print(f"\nSummary written to {args.output}")
