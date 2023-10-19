#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""
import argparse
import os
import sys
import networkx as nx
import matplotlib
from operator import itemgetter
import random
random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
matplotlib.use("Agg")

__author__ = "Souad YOUJIL ABADI"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Souad YOUJIL ABADI"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Souad YOUJIL ABADI"
__email__ = "souad_youjil@hotmail.com"
__status__ = "Developpement"


def isfile(path):  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file doesn't exist

    :return: (str) Path
    """
    if not os.path.isfile(path):
        if os.path.isdir(path):
            msg = "{0} is a directory".format(path)
        else:
            msg = "{0} does not exist.".format(path)
        raise argparse.ArgumentTypeError(msg)
    return path


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', dest='fastq_file', type=isfile,
                        required=True, help="Fastq file")
    parser.add_argument('-k', dest='kmer_size', type=int,
                        default=22, help="k-mer size (default 22)")
    parser.add_argument('-o', dest='output_file', type=str,
                        default=os.curdir + os.sep + "contigs.fasta",
                        help="Output contigs in fasta file (default contigs.fasta)")
    parser.add_argument('-f', dest='graphimg_file', type=str,
                        help="Save graph as an image (png)")
    return parser.parse_args()


def read_fastq(fastq_file):
    """Extract reads from fastq files.

    :param fastq_file: (str) Path to the fastq file.
    :return: A generator object that iterate the read sequences. 
    """
    with open(fastq_file, "r") as file:
        for line in file:
            # Yielding the sequence which is the second line in every
            # set of 4 lines in the FASTQ format
            yield next(file).strip()
            # Skip the next two lines (the '+' and quality lines)
            next(file)
            next(file)


def cut_kmer(read, kmer_size):
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that iterate the kmers of of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i+kmer_size]


def build_kmer_dict(fastq_file, kmer_size):
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}

    for read in read_fastq(fastq_file):
        for kmer in cut_kmer(read, kmer_size):
            if kmer not in kmer_dict:
                kmer_dict[kmer] = 1
            else:
                kmer_dict[kmer] += 1

    return kmer_dict


def build_graph(kmer_dict):
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = nx.DiGraph()

    for kmer, weight in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]
        # If the edge between the prefix and the suffix already exists, 
        # update the weight
        if graph.has_edge(prefix, suffix):
            graph[prefix][suffix]['weight'] += weight
        else:
            graph.add_edge(prefix, suffix, weight=weight)

    return graph


def remove_paths(graph, path_list, delete_entry_node, delete_sink_node):
    """Remove a list of paths from a graph. A path is a set of connected nodes in the graph.

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of paths
    :param delete_entry_node: (bool) True to remove the first node of a path
    :param delete_sink_node: (bool) True to remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # If delete_entry_node is True, then the first node in the path should be removed, so the starting index should be 0.
    # If delete_entry_node is False, then the first node in the path should be kept, so the starting index should be 1.
    start_index = 1 - int(delete_entry_node)

    for path in path_list:
        if delete_sink_node:
            graph.remove_nodes_from(path[start_index:])
        else:
            graph.remove_nodes_from(path[start_index:-1])

    return graph


def select_best_path(graph, path_list, path_length, weight_avg_list, 
                     delete_entry_node=False, delete_sink_node=False):
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path 
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # Calculate standard deviation of weight averages
    weight_stddev = statistics.stdev(weight_avg_list)

    if weight_stddev > 0:
        # Select path with highest weight average
        best_path_index = weight_avg_list.index(max(weight_avg_list))
    else:
        # Calculate standard deviation of path lengths
        length_stddev = statistics.stdev(path_length)

        if length_stddev > 0:
            # Select longest path
            best_path_index = path_length.index(max(path_length))
        else:
            # Randomly select path
            best_path_index = random.randint(0, len(path_list) - 1)

    # Remove nodes from all paths except the best one
    for i, path in enumerate(path_list):
        if i != best_path_index:
            nodes_to_remove = path.copy() # Make a copy of the path to avoid modifying the original path
            if not delete_entry_node:
                nodes_to_remove.pop(0)
            if not delete_sink_node:
                nodes_to_remove.pop()
            graph.remove_nodes_from(nodes_to_remove)

    return graph


def path_average_weight(graph, path):
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean([d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)])


def solve_bubble(graph, ancestor_node, descendant_node):
    """Remove all but the best path between ancestor and descendant nodes in a graph.

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph 
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Get all simple paths between ancestor and descendant nodes
    simple_paths = list(nx.all_simple_paths(graph, ancestor_node, descendant_node))

    # Calculate length and weight of each simple path
    path_lengths = []
    path_weights = []
    for path in simple_paths:
        path_lengths.append(len(path) - 1)
        path_weights.append(path_average_weight(graph, path))

    # Use select_best_path to choose the best path and remove other paths
    graph = select_best_path(graph, simple_paths, path_lengths, path_weights)

    return graph


def simplify_bubbles(graph):
    """Simplify all bubbles in a `networkx` network.

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph.nodes():
        # Get the list of predecessors for the current node
        predecessors = list(graph.predecessors(node))
        # Check if the node has more than one predecessor
        if len(predecessors) > 1:
            # Iterate over each unique pair of predecessors
            for i, pred_i in enumerate(predecessors):
                for pred_j in predecessors[i+1:]:
                    # Find the lowest common ancestor between the pair of predecessors
                    ancestor_node = nx.lowest_common_ancestor(graph, pred_i, pred_j)

                    if ancestor_node is not None and ancestor_node != node:
                        # If a common ancestor is found, solve the bubble 
                        # and simplify the resulting graph recursively
                        graph = solve_bubble(graph, ancestor_node, node)
                        graph = simplify_bubbles(graph)
                        # Return the graph after simplifying all bubbles
                        return graph
    # If no bubbles were found, return the original graph
    return graph


def solve_entry_tips(graph, starting_nodes):
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Determine paths based on starting nodes
    if len(starting_nodes) == 2:
        start_node1, start_node2 = starting_nodes
        common_ancestor = nx.lowest_common_ancestor(graph.reverse(), start_node1, start_node2)
        found_paths = [
            list(nx.all_simple_paths(graph, start_node1, common_ancestor))[0],
            list(nx.all_simple_paths(graph, start_node2, common_ancestor))[0]
        ]
    else:
        found_paths = []
        for i, start_node1 in enumerate(starting_nodes[:-1]):
            for start_node2 in starting_nodes[i+1:]:
                common_ancestor = nx.lowest_common_ancestor(graph.reverse(), start_node1, start_node2)
                found_paths.extend([
                    list(nx.all_simple_paths(graph, start_node1, common_ancestor))[0],
                    list(nx.all_simple_paths(graph, start_node2, common_ancestor))[0]
                ])

    # Check if any paths were found
    if not found_paths:
        return graph

    # Compute path lengths and weights
    path_lengths = [len(path) for path in found_paths]
    path_weights = [path_average_weight(graph, path) for path in found_paths]

    # Select the best path and update the graph
    graph = select_best_path(graph, found_paths, path_lengths, path_weights, delete_entry_node=True)

    return graph


def solve_out_tips(graph, ending_nodes):
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Generate node pairs for the ending nodes
    if len(ending_nodes) == 2:
        node_pairs = [tuple(ending_nodes)]
    else:
        node_pairs = [(node_i, node_j) for node_i in ending_nodes for node_j in ending_nodes if node_i != node_j]

    # Compute lowest common ancestor for each node pair
    common_ancestors = [nx.lowest_common_ancestor(graph, pair[0], pair[1]) for pair in node_pairs]

    # Initialize lists for paths, path lengths, and weights
    paths, path_lengths, weights = [], [], []

    # Populate the paths, path lengths, and weights lists
    for i, pair in enumerate(node_pairs):
        for node in pair:
            current_path = list(nx.all_simple_paths(graph, common_ancestors[i], node))[0]
            paths.append(current_path)
            path_lengths.append(len(current_path))
            weights.append(path_average_weight(graph, current_path))

    # Use select_best_path to refine the graph based on the best path
    graph = select_best_path(graph, paths, path_lengths, weights, delete_sink_node=True)

    for node in ending_nodes:
        if node not in graph:
            continue
        predecessors = list(graph.predecessors(node))
        if not predecessors:
            graph.remove_node(node)

    return graph


def get_starting_nodes(graph):
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node in graph.nodes() if not list(graph.predecessors(node))]


def get_sink_nodes(graph):
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node in graph.nodes() if not list(graph.successors(node))]


def get_contigs(graph, starting_nodes, ending_nodes):
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object 
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            # Check if start and end nodes exist in the graph
            if start in graph and end in graph:
                if nx.has_path(graph, start, end):
                    path = nx.shortest_path(graph, start, end)
                    # Construct the contig from the path
                    contig = path[0]
                    for i in range(1, len(path)):
                        contig += path[i][-1]
                    contigs.append([contig, len(contig)])
    return contigs


def save_contigs(contigs_list, output_file):
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (str) Path to the output file
    """
    with open(output_file, 'w') as out_file:
        for idx, (contig, length) in enumerate(contigs_list):
            # Write contig in FASTA format
            out_file.write(f">contig_{idx} len={length}\n")
            out_file.write(f"{textwrap.fill(contig, width=80)}\n")


def draw_graph(graph, graphimg_file):  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (str) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    pos = nx.spring_layout(graph)
    # pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=6, alpha=0.5,
                           edge_color='b', style='dashed')
    nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file)


#==============================================================
# Main program
#==============================================================
def main():  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Step 1: Set up
    print("Step 1: Set up")
    kmer_data = build_kmer_dict(args.fastq_file, args.kmer_size)
    current_graph = build_graph(kmer_data)

    # Step 2: Graph simplification
    print("Step 2: Graph simplification")
    current_graph = simplify_bubbles(current_graph)

    # Step 3: Resolving tips
    print("Step 3: Resolving tips")
    nodes_without_predecessors = get_starting_nodes(current_graph)
    current_graph = solve_entry_tips(current_graph, nodes_without_predecessors)
    nodes_without_successors = get_sink_nodes(current_graph)
    current_graph = solve_out_tips(current_graph, nodes_without_successors)

    # Step 4: Extracting contigs
    print("Step 4: Extracting contigs")
    derived_contigs = get_contigs(current_graph, nodes_without_predecessors, nodes_without_successors)
    save_contigs(derived_contigs, args.output_file)

    # Optional: Visualization
    print("Optional: Visualization")
    # if args.graphimg_file:
    #     draw_graph(current_graph, args.graphimg_file)


if __name__ == '__main__':  # pragma: no cover
    main()