from collections import defaultdict, deque
import polars as pl


def super_merger_bfs(df: pl.DataFrame, from_col_name: str, to_col_name: str) -> pl.DataFrame:
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
    ...     "from": ["A", "B", "C", "E", "F", "G", "I"],
    ...     "to": ["B", "C", "D", "F", "G", "J", "K"]
    ... })
    >>> result_df = super_merger_bfs(df, "from", "to")
    >>> print(result_df)

    Notes
    -----
    - The function first converts the input DataFrame into a list of edges.
    - Then, a graph is created using an adjacency list representation with the help of `defaultdict`.
    - The function uses BFS to find all connected components and assigns a unique group index to each component.
    - The result is a DataFrame with an additional 'group' column, where nodes in the same component share the same group number.

    """
    # Convert the DataFrame into a list of edges
    edges = df.select([pl.col(from_col_name), pl.col(to_col_name)]).to_numpy().tolist()

    # Create the graph representation using adjacency list
    graph = defaultdict(set)
    for from_node, to_node in edges:
        graph[from_node].add(to_node)
        graph[to_node].add(from_node)

    # Find connected components using BFS
    visited = set()
    node_to_group = {}
    group_idx = 0

    for node in graph:
        if node not in visited:
            # Perform BFS to find all nodes in the current component
            queue = deque([node])
            component = []

            while queue:
                current_node = queue.popleft()
                if current_node not in visited:
                    visited.add(current_node)
                    component.append(current_node)
                    queue.extend(graph[current_node] - visited)

            # Assign a group number to each node in this component
            for component_node in component:
                node_to_group[component_node] = group_idx + 1  # Group numbers start from 1

            group_idx += 1

    # Add the group information back to the DataFrame
    groups = [node_to_group.get(node, None) for node in df["from"].to_list()]
    return df.with_columns(group=pl.Series("group", groups))
