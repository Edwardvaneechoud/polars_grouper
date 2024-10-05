import polars as pl
import numpy as np


def super_merger_simple(df: pl.DataFrame, from_col_name: str, to_col_name: str) -> pl.DataFrame:
    """
    Optimized version to find connected components from a Polars DataFrame of edges and add the group information.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with columns 'from' and 'to', representing the edges.
    from_col_name : str
        The name of the column containing the source nodes.
    to_col_name : str
        The name of the column containing the target nodes.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional 'group' column representing the connected component group.
    """
    # Step 1: Convert DataFrame columns to NumPy arrays for fast processing
    from_nodes = df[from_col_name].to_numpy()
    to_nodes = df[to_col_name].to_numpy()

    # Get unique nodes and assign them unique integer IDs
    unique_nodes, node_indices = np.unique(np.concatenate((from_nodes, to_nodes)), return_inverse=True)
    num_nodes = len(unique_nodes)

    # Reshape node_indices to get corresponding edges
    edges = node_indices.reshape(-1, 2)

    # Step 2: Initialize Union-Find data structures
    parent = np.arange(num_nodes)
    rank = np.zeros(num_nodes, dtype=int)

    # Path compression for 'find' operation
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    # Union by rank
    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    # Step 3: Process each edge with union
    for edge in edges:
        union(edge[0], edge[1])

    # Step 4: Assign unique group IDs based on root representative
    group_mapping = {}
    group_counter = 1

    # Find the root of each node
    root_ids = np.array([find(node) for node in range(num_nodes)])
    groups = np.zeros(len(from_nodes), dtype=int)

    for i, from_node in enumerate(from_nodes):
        root = root_ids[node_indices[i]]
        if root not in group_mapping:
            group_mapping[root] = group_counter
            group_counter += 1
        groups[i] = group_mapping[root]

    # Step 5: Add the group information back to the DataFrame
    df = df.with_columns(pl.Series("group", groups))

    return df
