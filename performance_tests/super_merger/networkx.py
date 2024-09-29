import networkx as nx
import polars as pl


def super_merger_networkx(df: pl.DataFrame, from_col_name: str, to_col_name: str ) -> pl.DataFrame:
    """
    Find connected components from a Polars DataFrame of edges and add the group information using NetworkX.

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
    >>> result_df = super_merger_networkx(df, "from", "to")
    >>> print(result_df)

    Notes
    -----
    - This implementation uses the NetworkX library to find connected components in the graph.
    - NetworkX provides an easy and efficient way to work with graph structures and allows us to determine all connected components.
    - This approach is especially beneficial for its readability and the powerful graph operations NetworkX provides.

    """
    # Convert the DataFrame into a list of edges
    edges = df.select([pl.col(from_col_name), pl.col(to_col_name)]).to_numpy().tolist()

    # Create the graph using NetworkX
    G = nx.Graph()
    G.add_edges_from(edges)

    # Find connected components
    components = list(nx.connected_components(G))

    # Assign group indices to nodes
    node_to_group = {}
    for group_idx, component in enumerate(components, start=1):
        for node in component:
            node_to_group[node] = group_idx

    # Add the group information back to the DataFrame
    groups = [node_to_group.get(node, None) for node in df["from"].to_list()]
    df = df.with_columns(group=pl.Series("group", groups))

    return df
