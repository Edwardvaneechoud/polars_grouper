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

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "from": ["A", "B", "C", "E", "F", "G"],
    ...     "to": ["B", "C", "D", "F", "G", "H"]
    ... })
    >>> result_df = super_merger_simple(df, "from", "to")
    >>> print(result_df)

    Notes
    -----
    - This implementation finds connected components by iterating through each edge.
    - It assigns group numbers in a greedy manner based on whether nodes have already been assigned to a group.
    - If neither node has been assigned, it creates a new group.
    - If one of the nodes is in a group, it assigns the other to that same group.
    - This version is the most stright forward and simple implementation.
    """
    # Convert the DataFrame into a list of edges
    edges = df.select([pl.col(from_col_name), pl.col(to_col_name)]).to_numpy().tolist()

    # Initialize the group tracking
    node_to_group = {}
    group_idx = 1

    # Iterate over the edges to assign group numbers
    for from_node, to_node in edges:
        if from_node not in node_to_group and to_node not in node_to_group:
            # Create a new group for both nodes
            node_to_group[from_node] = group_idx
            node_to_group[to_node] = group_idx
            group_idx += 1
        elif from_node in node_to_group and to_node not in node_to_group:
            # Assign the group of `from_node` to `to_node`
            node_to_group[to_node] = node_to_group[from_node]
        elif to_node in node_to_group and from_node not in node_to_group:
            # Assign the group of `to_node` to `from_node`
            node_to_group[from_node] = node_to_group[to_node]

    # Create the group column based on the assignments
    groups = [node_to_group.get(node, None) for node in df["from"].to_list()]

    # Add the group information back to the DataFrame
    df = df.with_columns(group=pl.Series("group", groups))

    return df
