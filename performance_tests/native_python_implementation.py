from collections import defaultdict, deque
import polars as pl


def add_connected_components(df):
    """
    Find connected components from a Polars DataFrame of edges and add the group information.

    Args:
    df (pl.DataFrame): Polars DataFrame with columns 'from' and 'to', representing the edges.

    Returns:
    pl.DataFrame: The input DataFrame with an additional 'group' column representing the connected component group.
    """
    # Convert the DataFrame into a list of edges
    edges = df.select([pl.col("from"), pl.col("to")]).to_numpy().tolist()

    # Create the graph representation
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
    df = df.with_columns(group=pl.Series("group", groups))

    return df
