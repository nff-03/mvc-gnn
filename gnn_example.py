# Combinatorial Optimization with Physics-Inspired Graph Neural Networks

# This notebook demonstrates how to solve combinatorial optimization problems using 
# physics-inspired graph neural networks, as outlined in the paper:
# M. J. A. Schuetz, J. K. Brubaker, H. G. Katzgraber, 
# _Combinatorial Optimization with Physics-Inspired Graph Neural Networks_, 
# [arXiv:2107.01188](https://arxiv.org/abs/2107.01188).
#
# While the original paper focuses on the Maximum Independent Set (MIS) problem, 
# here I apply the methodology to the Minimum Vertex Cover (MVC) problem. 
#
# The implementation uses the open-source `dgl` library for building the graph neural network.
#
# A `requirements.txt` file is provided to help create the required Python environment.
# Some packages may not be available via default conda channels on macOS, 
# so suggested channels are included. You can create the environment using:
#
# > conda create -n <environment_name> python=3 --file requirements.txt -c conda-forge -c dglteam -c pytorch
#
# Code adapted from the official Amazon Science implementation:
# https://github.com/amazon-science/co-with-gnns-example
# 
# Original implementation by Amazon Science was designed for the Maximum Independent Set (MIS) problem.
# This version has been modified by Nour Fayadh for the Minimum Vertex Cover (MVC) problem.
# 
# This file includes reused components under the MIT-0 license.
# See LICENSE-SAMPLECODE.txt for details.


import os
import random
from collections import OrderedDict, defaultdict
from itertools import chain, combinations, islice
from time import time

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from docplex.mp.model import Model
from networkx.algorithms.approximation.independent_set import \
    maximum_independent_set as mis

# MacOS can have issues with MKL. For more details, see
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# fix seed to ensure consistent results
seed_value = 1
random.seed(seed_value)        # seed python RNG
np.random.seed(seed_value)     # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')


# We first load a few general utility functions from ```utils.py``` before defining some helper functions specific to the MIS problem. 

# General utilities

from utils import (gen_combinations, generate_graph, get_gnn, loss_func,
                   qubo_dict_to_torch, run_gnn_training)

# Problem-specific (MVC) utilities

# helper function to generate Q matrix for Minimum Vertex Cover (MVC)
def gen_q_dict_mvc(nx_G, penalty=6):
    """
    Helper function to generate QUBO matrix for MVC.
    
    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Assign 1 to the diagonal terms in the Q matrix
    for u in nx_G.nodes:
        Q_dic[(u, u)] = 1

    # Update Q matrix for every edge in the graph
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty
        Q_dic[(v, u)] = penalty
        Q_dic[(u, u)] += -penalty
        Q_dic[(v,v)] += -penalty


    return Q_dic

def greedy_postprocess(bit_string, edges):
    """
    Post-process the GNN bit-string output by ensuring every edge is covered.
    
    For each edge (u, v) in the graph, if neither vertex is in the cover,
    the algorithm adds both u and v (greedily) to cover the edge.
    
    Args:
        bit_string (list or np.array): Binary list where 1 indicates the vertex is in the cover.
        edges (list of tuple): List of edges, with vertices indexed from 0.
    
    Returns:
        list: Updated bit string representing a valid vertex cover.
    """
    # Convert bit_string into a set of vertices in the cover.
    cover = {i for i, bit in enumerate(bit_string) if bit}
    
    # Greedily ensure that each edge is covered.
    changed = True
    while changed:
        changed = False
        for u, v in edges:
            if u not in cover and v not in cover:
                # If the edge is uncovered, add both endpoints.
                cover.add(u)
                cover.add(v)
                changed = True
    
    # Reconstruct a bit string of the same length.
    new_bit_string = [1 if i in cover else 0 for i in range(len(bit_string))]
    return new_bit_string

def minimize_postprocessed_cover(bit_string, edges):
    """
    Given a valid cover (as a bit string), remove redundant vertices while keeping all edges covered.
    
    For each vertex in the cover, the function checks if its removal still leaves every edge covered.
    
    Args:
        bit_string (list or np.array): Binary list representing the vertex cover.
        edges (list of tuple): List of edges in the graph.
    
    Returns:
        list: A minimized binary cover (bit string) with redundant vertices removed.
    """
    # Convert the bit string into a set for easier manipulation.
    cover = {i for i, bit in enumerate(bit_string) if bit}
    
    # Try to remove each vertex and verify that every edge remains covered.
    for vertex in sorted(cover):
        temp_cover = cover - {vertex}
        if all((u in temp_cover or v in temp_cover) for u, v in edges):
            cover = temp_cover  # Removal successful
    # Reconstruct the bit string.
    new_bit_string = [1 if i in cover else 0 for i in range(len(bit_string))]
    return new_bit_string

# functions used for different testing methods to compare to gnn solution

def solve_mvc_ilp(G):
    """
    Solves the Minimum Vertex Cover (MVC) problem using an ILP formulation with DOcplex.
    
    Parameters:
        G (networkx.Graph): Input graph.
    
    Returns:
        cover (list): List of nodes representing the minimum vertex cover.
        ilp_time (float): Time taken to solve the ILP.
    """
    model = Model(name="Min_Vertex_Cover")

    # Create binary decision variables for each node
    x_vars = {node: model.binary_var(name=f"x_{node}") for node in G.nodes()}

    # Objective: minimize total number of selected vertices
    model.minimize(model.sum(x_vars[node] for node in G.nodes()))

    # Constraints: at least one endpoint of each edge must be selected
    for i, j in G.edges():
        model.add_constraint(x_vars[i] + x_vars[j] >= 1, f"edge_{i}_{j}")

    # Auto-scale time limit based on graph size
    n = G.number_of_nodes()
    if n <= 5000:
        t = 120
    elif n <= 50000:
        t = 600
    elif n <= 200000:
        t = 1800
    else:
        t = 7200

    # Note: This may return suboptimal solutions if the time expires before convergence.
    # Check the MIP gap in the logs to assess how close the result is to optimal.
    model.set_time_limit(t)

    # Solve the model
    start_time = time()
    solution = model.solve(log_output=False)  # Set log_output=True for debug
    ilp_time = time() - start_time

    # Extract the vertex cover from the solution
    cover = [node for node in G.nodes() if x_vars[node].solution_value >= 0.99]

    # Get MIP gap
    mip_gap = model.solve_details.mip_relative_gap

    return cover, ilp_time, mip_gap

def heuristic_vertex_cover(graph):
    """
    Finds an approximate minimum vertex cover using the heuristic described in the paper.
    
    Parameters:
        graph (networkx.Graph): An undirected graph.
    
    Returns:
        set: A vertex cover set.
    """
    start = time()
    # Convert the graph into a dictionary for faster access
    G = graph.copy()
    C = set()  # Vertex cover set
    I = set()  # Set of visited vertices
    
    # Step 1: Start with a vertex of minimum degree
    v = min(G.nodes, key=lambda node: G.degree[node])
    C.update(set(G.neighbors(v)))  # Add its neighbors to the cover
    I.add(v)  # Mark v as considered

    # Step 2: Iteratively add new vertices that minimize |N(w) ∪ C|
    while set(G.nodes) - (C | I):
        remaining_nodes = set(G.nodes) - (C | I)
        w = min(remaining_nodes, key=lambda node: len(set(G.neighbors(node)) - C))
        C.update(set(G.neighbors(w)))  # Add the neighborhood of w
        I.add(w)  # Mark w as considered

    total_time = time() - start
    
    return C, total_time


def approx_vertex_cover(graph):
    """
    Finds an approximate vertex cover using a simple greedy algorithm.

    Parameters:
        graph (networkx.Graph): An undirected graph.

    Returns:
        set: A vertex cover set.
    """

    start = time()
    G = graph.copy()
    C = set()  # Vertex cover set

    while G.edges:  # While there are edges in the graph
        # Choose an arbitrary edge (u, v)
        u, v = next(iter(G.edges))
        
        # Choose either u or v randomly (you can also choose based on a heuristic)
        chosen_vertex = random.choice([u, v])
        C.add(chosen_vertex)

        # Remove all edges incident on the chosen vertex
        G.remove_node(chosen_vertex)

    total_time = time() - start

    return C, total_time


# Setting hyperparameters

# NN learning hypers #
number_epochs = int(1e5)
learning_rate = 1e-3
PROB_THRESHOLD = 0.5

# Early stopping to allow NN to train to near-completion
tol = 1e-4          # loss must change by more than tol, or trigger
patience = 100    # number early stopping triggers before breaking loop

# Running the solution

# Function to calculate approximation ratio
def calculate_approx_ratio(solution_size, optimal_size):
    return (solution_size / optimal_size)

# Define graph sizes for experiments
graph_sizes = list(range(10, 101, 10)) + list(range(120, 981, 20)) + list(range(1000, 9001, 1000)) + list(range(10000, 90001, 10000)) + list(100000)

# Initialize result storage for comparison and GNN experiments
cmp_results = []
gnn_results = []

# Set parameters for random graph generation
d = 3 # change according to the kind of graph you are testing (either 3-degree or 5-degree)
p = None
graph_type = 'reg'

# Loop over different graph sizes
for size in graph_sizes:

    G = generate_graph(n=size, d=d, p=p, graph_type=graph_type, random_seed=seed_value)   # Constructs a random d-regular graph

    num_edges = G.number_of_edges()
    graph_id = f"G{size}"

    # For each graph, run multiple trials
    for run_id in range (1, 4):
        # Run ILP solver
        ilp_cover, ilp_time, mip_gap = solve_mvc_ilp(G)
        optimal_size = len(ilp_cover)

        # Run heuristic solver
        heuristic_cover, heuristic_time = heuristic_vertex_cover(G)
        
        # Run approximation solver
        approx_cover, approx_time= approx_vertex_cover(G)

        # Store comparison results
        cmp_results.append({
            "Graph ID": graph_id,
            "Run #": run_id,
            "Num Nodes": size,
            "Num Edges": num_edges,
            "Solution Method": "ILP",
            "Solution Size": optimal_size,
            "Runtime (s)": ilp_time,
            "MIP Gap": mip_gap
        })

        cmp_results.append({
            "Graph ID": graph_id,
            "Run #": run_id,
            "Num Nodes": size,
            "Num Edges": num_edges,
            "Solution Method": "Heuristic",
            "Solution Size": len(heuristic_cover),
            "Runtime (s)": heuristic_time,
            "Approximation Ratio": calculate_approx_ratio(len(heuristic_cover), optimal_size)
        })
        cmp_results.append({
            "Graph ID": graph_id,
            "Run #": run_id,
            "Num Nodes": size,
            "Num Edges": num_edges,
            "Solution Method": "Approximate",
            "Solution Size": len(approx_cover),
            "Runtime (s)": approx_time,
            "Approximation Ratio": calculate_approx_ratio(len(approx_cover), optimal_size)
        })

    # get DGL graph from networkx graph, load onto device
    graphDGL = dgl.from_networkx(nx_graph=G)
    graphDGL = graphDGL.to(TORCH_DEVICE)

    # Construct Q matrix for graph for GNN input
    q_matrix = qubo_dict_to_torch(G, gen_q_dict_mvc(G), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

    # GNN Training - repeat over multiple runs
    for run_id in range(1, 11):

        # Set GNN hyperparameters based on graph size
        # Initialize GNN model, embedding layer, and optimizer

        dim_embedding = int(np.sqrt(size))    # e.g. 10
        hidden_dim = int(dim_embedding/2)  # e.g. 5

        opt_params = {'lr': learning_rate}
        gnn_hypers = {
            'dim_embedding': dim_embedding,
            'hidden_dim': hidden_dim,
            'dropout': 0.0,
            'number_classes': 1,
            'prob_threshold': PROB_THRESHOLD,
            'number_epochs': number_epochs,
            'tolerance': tol,
            'patience': patience
        }

        net, embed, optimizer = get_gnn(size, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)

        # For tracking hyperparameters in results object
        gnn_hypers.update(opt_params)

        # Train the GNN model
        print('Running GNN...')
        gnn_start = time()

        _, epoch, final_bitstring, best_bitstring = run_gnn_training(
            q_matrix, graphDGL, net, embed, optimizer, gnn_hypers['number_epochs'],
            gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'])

        gnn_time = time() - gnn_start

        # Calculate loss and begin post-processing to convert GNN output into a valid cover
        final_loss = loss_func(final_bitstring.float(), q_matrix)
        final_bitstring_str = ','.join([str(x) for x in final_bitstring])

        # Run greedy post-processing and then minimize the cover
        postprocessing_start = time()
        postprocessed_cover = greedy_postprocess(final_bitstring, G.edges())
        minimized_cover = minimize_postprocessed_cover(postprocessed_cover, G.edges())
        postprocessing_time = time() - postprocessing_start

        total_time = gnn_time + postprocessing_time

        # Store GNN results
        gnn_results.append({
        "Graph ID": graph_id,
        "Run #": run_id,
        "Num Nodes": size,
        "Num Edges": num_edges,
        "Solution Size": sum(minimized_cover),
        "Runtime (s)": total_time,
        "Approximation Ratio": calculate_approx_ratio(sum(minimized_cover), optimal_size)
        })

# Convert to DataFrames and save to Excel
cmp_df = pd.DataFrame(cmp_results)
gnn_df = pd.DataFrame(gnn_results)

# Save results to an Excel file
output_path1 = "Test.xlsx"
output_path2 = "Test2.xlsx"
with pd.ExcelWriter(output_path1) as writer:
    cmp_df.to_excel(writer, sheet_name="CMP results", index=False)

with pd.ExcelWriter(output_path2) as writer:
    gnn_df.to_excel(writer, sheet_name="GNN results", index=False)

print(f"Results saved to {output_path1}")
print(f"Results saved to {output_path2}")

