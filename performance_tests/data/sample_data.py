import polars as pl
import numpy as np
from pathlib import Path
import time
import itertools
import random



def create_sample_data(num_groups: int = 1_000):
    # Parameters for graph size
    NUM_GROUPS = num_groups  # Number of unique groups
    NODES_PER_GROUP = 10  # Number of nodes per group
    MAX_POSSIBLE_EDGES = NODES_PER_GROUP * (NODES_PER_GROUP - 1) // 2
    ADDITIONAL_EDGES_PER_GROUP = min(50000, MAX_POSSIBLE_EDGES - (NODES_PER_GROUP - 1))

    start_time = time.time()

    edges_list = []  # List to hold DataFrames per group

    print("Starting to generate nodes and edges for each group...")
    for group_idx in range(NUM_GROUPS):
        group_nodes = np.array([f"group{group_idx}_node{i}" for i in range(NODES_PER_GROUP)])

        # Create a connected component within the group by linking nodes in a chain
        chain_edges_from = group_nodes[:-1]
        chain_edges_to = group_nodes[1:]

        # Generate all possible edges excluding self-loops
        possible_edges = list(itertools.combinations(group_nodes, 2))

        # Remove chain edges from possible edges
        chain_edges = list(zip(chain_edges_from, chain_edges_to))
        remaining_edges = list(set(possible_edges) - set(chain_edges))

        # Sample additional edges without replacement
        num_additional_edges = min(ADDITIONAL_EDGES_PER_GROUP, len(remaining_edges))
        sampled_edges = random.sample(remaining_edges, num_additional_edges)

        # Combine chain edges and sampled edges
        edges_from = np.concatenate([chain_edges_from, [edge[0] for edge in sampled_edges]])
        edges_to = np.concatenate([chain_edges_to, [edge[1] for edge in sampled_edges]])

        # Create DataFrame for this group
        group_df = pl.DataFrame({
            "from": edges_from,
            "to": edges_to
        })

        edges_list.append(group_df)

        if (group_idx + 1) % 1000 == 0:
            print(f"Processed {group_idx + 1} groups out of {NUM_GROUPS}")

    mid_time = time.time()
    print(f"Finished generating edges for {NUM_GROUPS} groups.")
    print(f"Time taken for edge generation: {mid_time - start_time:.2f} seconds")

    # Concatenate all group DataFrames
    print("Concatenating all group DataFrames...")
    df = pl.concat(edges_list)

    after_df_time = time.time()
    print(f"DataFrame created with {df.height} edges.")
    print(f"Time taken to create DataFrame: {after_df_time - mid_time:.2f} seconds")

    # Save the DataFrame to Parquet
    print("Saving DataFrame to Parquet file...")
    output_file = Path(f"performance_tests/data/data_{NUM_GROUPS}.parquet")
    df.write_parquet(str(output_file))

    end_time = time.time()
    print(f"Dataset saved to {output_file}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Time taken to save DataFrame to Parquet: {end_time - after_df_time:.2f} seconds")



def create_sample_data_complex(num_groups: int = 1_000, cross_group_edges: int = 500):
    # Parameters for graph size
    NUM_GROUPS = num_groups  # Number of unique groups
    NODES_PER_GROUP = 10  # Number of nodes per group
    MAX_POSSIBLE_EDGES = NODES_PER_GROUP * (NODES_PER_GROUP - 1) // 2
    ADDITIONAL_EDGES_PER_GROUP = min(50000, MAX_POSSIBLE_EDGES - (NODES_PER_GROUP - 1))

    start_time = time.time()

    edges_list = []  # List to hold DataFrames per group

    print("Starting to generate nodes and edges for each group...")
    all_group_nodes = []  # To store nodes across all groups for cross-group edges

    for group_idx in range(NUM_GROUPS):
        group_nodes = np.array([f"group{group_idx}_node{i}" for i in range(NODES_PER_GROUP)])
        all_group_nodes.extend(group_nodes)

        # Create a connected component within the group by linking nodes in a chain
        chain_edges_from = group_nodes[:-1]
        chain_edges_to = group_nodes[1:]

        # Generate all possible edges excluding self-loops
        possible_edges = list(itertools.combinations(group_nodes, 2))

        # Remove chain edges from possible edges
        chain_edges = list(zip(chain_edges_from, chain_edges_to))
        remaining_edges = list(set(possible_edges) - set(chain_edges))

        # Sample additional edges without replacement
        num_additional_edges = min(ADDITIONAL_EDGES_PER_GROUP, len(remaining_edges))
        sampled_edges = random.sample(remaining_edges, num_additional_edges)

        # Combine chain edges and sampled edges
        edges_from = np.concatenate([chain_edges_from, [edge[0] for edge in sampled_edges]])
        edges_to = np.concatenate([chain_edges_to, [edge[1] for edge in sampled_edges]])

        # Create DataFrame for this group
        group_df = pl.DataFrame({
            "from": edges_from,
            "to": edges_to
        })

        edges_list.append(group_df)

        if (group_idx + 1) % 1000 == 0:
            print(f"Processed {group_idx + 1} groups out of {NUM_GROUPS}")

    # Add cross-group edges to create large connected components
    print("Adding cross-group edges...")
    cross_group_edges_list = []
    for _ in range(cross_group_edges):
        node1, node2 = random.sample(all_group_nodes, 2)
        cross_group_edges_list.append((node1, node2))

    cross_group_df = pl.DataFrame({
        "from": [edge[0] for edge in cross_group_edges_list],
        "to": [edge[1] for edge in cross_group_edges_list]
    })

    # Add cross-group edges to the list of edges
    edges_list.append(cross_group_df)

    mid_time = time.time()
    print(f"Finished generating edges for {NUM_GROUPS} groups.")
    print(f"Time taken for edge generation: {mid_time - start_time:.2f} seconds")

    # Concatenate all group DataFrames
    print("Concatenating all group DataFrames...")
    df = pl.concat(edges_list)

    after_df_time = time.time()
    print(f"DataFrame created with {df.height} edges.")
    print(f"Time taken to create DataFrame: {after_df_time - mid_time:.2f} seconds")

    # Save the DataFrame to Parquet
    print("Saving DataFrame to Parquet file...")
    output_file = Path(f"performance_tests/data/data_{NUM_GROUPS}.parquet")
    df.write_parquet(str(output_file))

    end_time = time.time()
    print(f"Dataset saved to {output_file}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Time taken to save DataFrame to Parquet: {end_time - after_df_time:.2f} seconds")

for i in range(1, 7):
    print(10**i)
    create_sample_data_complex(10**i)
