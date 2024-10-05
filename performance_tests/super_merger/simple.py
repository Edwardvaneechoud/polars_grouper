import polars as pl

def super_merger_simple(df: pl.DataFrame, from_col_name: str, to_col_name: str) -> pl.DataFrame:
    """
    Find connected components from a Polars DataFrame of edges and add the group information.

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
    # Convert the DataFrame into a list of edges
    edges = df.select([pl.col(from_col_name), pl.col(to_col_name)]).to_numpy().tolist()

    # Step 1: Map nodes to unique integer IDs
    node_to_id = {}
    id_counter = 0
    for from_node, to_node in edges:
        if from_node not in node_to_id:
            node_to_id[from_node] = id_counter
            id_counter += 1
        if to_node not in node_to_id:
            node_to_id[to_node] = id_counter
            id_counter += 1

    num_nodes = id_counter

    # Step 2: Create closures for Union-Find operations
    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    # Closure for `find` with path compression
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])  # Path compression
        return parent[node]

    # Closure for `union` with union by rank
    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)

        if root1 != root2:
            # Union by rank
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    # Step 3: Process each edge with the union operation
    for from_node, to_node in edges:
        node1 = node_to_id[from_node]
        node2 = node_to_id[to_node]
        union(node1, node2)

    # Step 4: Assign unique group IDs based on the root representative of each node
    group_mapping = {}
    group_counter = 1

    # Find root for each node in the original DataFrame and assign a group ID
    groups = []
    for from_node in df[from_col_name].to_list():
        root = find(node_to_id[from_node])
        if root not in group_mapping:
            group_mapping[root] = group_counter
            group_counter += 1
        groups.append(group_mapping[root])

    # Step 5: Add the group information back to the DataFrame
    df = df.with_columns(group=pl.Series("group", groups))

    return df

