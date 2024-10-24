from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars_grouper._internal import __version__ as __version__
from typing import TypeVar

DF = TypeVar("DF", pl.DataFrame, pl.LazyFrame)

if TYPE_CHECKING:
    from polars_grouper.typing import IntoExpr

LIB = Path(__file__).parent


def graph_solver(expr_from: IntoExpr, expr_to: IntoExpr) -> pl.Expr:
    """
    Identify connected components in a graph represented by edges.

    This function processes a graph structure represented as edges between nodes and identifies
    all connected components, assigning each node to its respective component group.

    Parameters
    ----------
    expr_from : IntoExpr
        Expression representing the source nodes of the edges.
    expr_to : IntoExpr
        Expression representing the destination nodes of the edges.

    Returns
    -------
    pl.Expr
        A Polars expression that resolves to a Series containing component group assignments.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "source": ["A", "B", "C", "D", "E"],
    ...     "target": ["B", "C", "A", "E", "F"]
    ... })
    >>>
    >>> # Find connected components
    >>> result = df.with_columns(
    ...     graph_solver(pl.col("source"), pl.col("target")).alias("component")
    ... )
    >>> print(result)
    shape: (5, 3)
    ┌────────┬────────┬───────────┐
    │ source ┆ target ┆ component │
    │ str    ┆ str    ┆ i64      │
    ╞════════╪════════╪═══════════╡
    │ A      ┆ B      ┆ 1        │
    │ B      ┆ C      ┆ 1        │
    │ C      ┆ A      ┆ 1        │
    │ D      ┆ E      ┆ 2        │
    │ E      ┆ F      ┆ 2        │
    └────────┴────────┴───────────┘

    Notes
    -----
    - The function treats the graph as undirected by default
    - Component groups are assigned arbitrary but consistent numeric identifiers
    - Isolated nodes will be assigned their own unique component

    """
    return register_plugin_function(
        args=[expr_from, expr_to],
        plugin_path=LIB,
        function_name="graph_solver",
        is_elementwise=False,
    )


def calculate_shortest_path(
    expr_from: IntoExpr,
    expr_to: IntoExpr,
    weights: IntoExpr,
    directed: bool = False
) -> pl.Expr:
    """
    Calculate the shortest paths between all pairs of nodes in a weighted graph.

    Implements Dijkstra's algorithm to find the shortest paths between all pairs of nodes
    in a graph. The function can handle both directed and undirected graphs.

    Parameters
    ----------
    expr_from : IntoExpr
        Expression representing the source nodes of the edges.
    expr_to : IntoExpr
        Expression representing the destination nodes of the edges.
    weights : IntoExpr
        Expression representing the edge weights. Must be non-negative values.
    directed : bool, default False
        If True, treats the graph as directed. If False, treats edges as bidirectional.

    Returns
    -------
    pl.Expr
        A Polars expression that resolves to a struct containing three fields:
        - "from": source node of the path
        - "to": destination node of the path
        - "distance": total distance (sum of weights) of the shortest path

    Examples
    --------
    >>> import polars as pl
    >>> # Create a sample graph
    >>> df = pl.DataFrame({
    ...     "from": ["A", "A", "B", "C"],
    ...     "to": ["B", "C", "C", "D"],
    ...     "weight": [1.0, 2.0, 1.0, 1.5]
    ... })
    >>>
    >>> # Calculate shortest paths
    >>> result = df.select(
    ...     calculate_shortest_path(
    ...         pl.col("from"),
    ...         pl.col("to"),
    ...         pl.col("weight"),
    ...         directed=True
    ...     ).alias("paths")
    ... ).unnest("paths")
    >>>
    >>> print(result)
    shape: (6, 3)
    ┌──────┬──────┬──────────┐
    │ from ┆ to   ┆ distance │
    │ str  ┆ str  ┆ f64     │
    ╞══════╪══════╪══════════╡
    │ A    ┆ B    ┆ 1.0     │
    │ A    ┆ C    ┆ 2.0     │
    │ A    ┆ D    ┆ 3.5     │
    │ B    ┆ C    ┆ 1.0     │
    │ B    ┆ D    ┆ 2.5     │
    │ C    ┆ D    ┆ 1.5     │
    └──────┴──────┴──────────┘

    Notes
    -----
    - Returns only existing paths (unreachable pairs are excluded)
    - Weights must be non-negative
    - For undirected graphs, paths A→B and B→A will have the same distance
    - Memory usage scales with O(V²) where V is the number of vertices

    """
    return register_plugin_function(
        args=[expr_from, expr_to, weights],
        plugin_path=LIB,
        function_name="graph_find_shortest_path",
        is_elementwise=False,
        changes_length=True,
        kwargs={"directed": directed}
    )


def page_rank(
    expr_from: IntoExpr,
    expr_to: IntoExpr,
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6
) -> pl.Expr:
    """
    Calculate PageRank scores for nodes in a graph.

    Implements the PageRank algorithm to compute importance scores for nodes in a graph
    based on the link structure. The algorithm was originally developed by Google"s founders
    to rank web pages.

    Parameters
    ----------
    expr_from : IntoExpr
        Expression representing the source nodes of the edges.
    expr_to : IntoExpr
        Expression representing the destination nodes of the edges.
    damping_factor : float, default 0.85
        Probability of following a link versus random jumping (range 0-1).
        Higher values give more weight to the link structure.
    max_iterations : int, default 100
        Maximum number of iterations for the algorithm to converge.
    convergence_threshold : float, default 1e-6
        Minimum change in scores between iterations to consider the algorithm converged.

    Returns
    -------
    pl.Expr
        A Polars expression that resolves to a struct containing:
        - "node": node identifier
        - "score": PageRank score for the node
        - "iterations": number of iterations until convergence

    Examples
    --------
    >>> import polars as pl
    >>> # Create a sample web graph
    >>> df = pl.DataFrame({
    ...     "from": ["A", "A", "B", "C", "D"],
    ...     "to": ["B", "C", "C", "A", "B"]
    ... })
    >>>
    >>> # Calculate PageRank scores
    >>> result = df.select(
    ...     page_rank(
    ...         pl.col("from"),
    ...         pl.col("to"),
    ...         damping_factor=0.85,
    ...         max_iterations=50
    ...     ).alias("pagerank")
    ... ).unnest("pagerank")
    >>>
    >>> print(result)
    shape: (4, 2)
    ┌──────┬───────────┐
    │ node ┆ score     │
    │ str  ┆ f64      │
    ╞══════╪═══════════╡
    │ A    ┆ 0.283019 │
    │ B    ┆ 0.330189 │
    │ C    ┆ 0.254717 │
    │ D    ┆ 0.132075 │
    └──────┴───────────┘

    Notes
    -----
    - The sum of all PageRank scores will be approximately 1.0
    - Isolated nodes receive a minimum base score
    - Higher damping factors may require more iterations to converge
    - The algorithm may not converge if max_iterations is too low

    """
    return register_plugin_function(
        args=[expr_from, expr_to],
        plugin_path=LIB,
        function_name="page_rank",
        is_elementwise=False,
        kwargs={"damping_factor": damping_factor,
                "max_iterations": max_iterations,
                "convergence_threshold": convergence_threshold}
    )


def super_merger(df: DF, from_col_name: str, to_col_name: str) -> DF:
    """
    Group nodes into connected components based on edge relationships.

    A high-level wrapper function that identifies connected components in a graph
    and adds group assignments to the input DataFrame.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input DataFrame containing edge information.
    from_col_name : str
        Name of the column containing source nodes.
    to_col_name : str
        Name of the column containing destination nodes.

    Returns
    -------
    DF
        Input DataFrame with an additional "group" column containing component assignments.

    Examples
    --------
    >>> import polars as pl
    >>> # Create sample data with multiple components
    >>> df = pl.DataFrame({
    ...     "from": ["A", "B", "C", "D", "E", "F"],
    ...     "to": ["B", "C", "A", "E", "F", "D"],
    ...     "value": [1, 2, 3, 4, 5, 6]
    ... })
    >>>
    >>> # Find connected components
    >>> result = super_merger(df, "from", "to")
    >>> print(result)
    shape: (6, 4)
    ┌──────┬──────┬───────┬───────┐
    │ from ┆ to   ┆ value ┆ group │
    │ str  ┆ str  ┆ i64   ┆ i64   │
    ╞══════╪══════╪═══════╪═══════╡
    │ A    ┆ B    ┆ 1     ┆ 1     │
    │ B    ┆ C    ┆ 2     ┆ 1     │
    │ C    ┆ A    ┆ 3     ┆ 1     │
    │ D    ┆ E    ┆ 4     ┆ 2     │
    │ E    ┆ F    ┆ 5     ┆ 2     │
    │ F    ┆ D    ┆ 6     ┆ 2     │
    └──────┴──────┴───────┴───────┘

    Notes
    -----
    - Preserves all columns from the input DataFrame
    - Works with both eager (DataFrame) and lazy (LazyFrame) evaluation
    - Group assignments are arbitrary but consistent integers
    - The function treats the graph as undirected

    """
    return df.with_columns(graph_solver(pl.col(from_col_name), pl.col(to_col_name)).alias("group"))


def super_merger_weighted(
    df: DF,
    from_col_name: str,
    to_col_name: str,
    weighted_col_name: str,
    weight_threshold: float = 0.1
) -> DF:
    """
    Group nodes into connected components considering edge weights.

    Similar to super_merger, but only considers edges with weights above a specified threshold
    when identifying connected components.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input DataFrame containing edge information and weights.
    from_col_name : str
        Name of the column containing source nodes.
    to_col_name : str
        Name of the column containing destination nodes.
    weighted_col_name : str
        Name of the column containing edge weights.
    weight_threshold : float, default 0.1
        Minimum weight value for an edge to be considered in component identification.

    Returns
    -------
    DF
        Filtered DataFrame containing only edges above the weight threshold,
        with an additional "group" column containing component assignments.

    Examples
    --------
    >>> import polars as pl
    >>> # Create sample weighted graph
    >>> df = pl.DataFrame({
    ...     "from": ["A", "B", "C", "D", "E"],
    ...     "to": ["B", "C", "D", "E", "A"],
    ...     "weight": [0.9, 0.2, 0.05, 0.8, 0.3]
    ... })
    >>>
    >>> # Find components with weight threshold
    >>> result = super_merger_weighted(
    ...     df,
    ...     "from",
    ...     "to",
    ...     "weight",
    ...     weight_threshold=0.3
    ... )
    >>> print(result)
    shape: (3, 4)
    ┌──────┬──────┬────────┬───────┐
    │ from ┆ to   ┆ weight ┆ group │
    │ str  ┆ str  ┆ f64    ┆ i64   │
    ╞══════╪══════╪════════╪═══════╡
    │ A    ┆ B    ┆ 0.9    ┆ 1     │
    │ D    ┆ E    ┆ 0.8    ┆ 2     │
    │ E    ┆ A    ┆ 0.3    ┆ 1     │
    └──────┴──────┴────────┴───────┘

    Notes
    -----
    - Filters out edges below the weight threshold before component identification
    - Returns only rows corresponding to edges above the threshold
    - Works with both eager (DataFrame) and lazy (LazyFrame) evaluation
    - The function treats the graph as undirected
    - Weight threshold should be chosen based on the scale of your weight values

    """
    return (df.filter(pl.col(weighted_col_name) >= weight_threshold)
            .with_columns(graph_solver(pl.col(from_col_name), pl.col(to_col_name)).alias("group")))


def betweenness_centrality(
    expr_from: IntoExpr,
    expr_to: IntoExpr,
    normalized: bool = True,
    directed: bool = False
) -> pl.Expr:
    """
    Calculate betweenness centrality for all nodes in a graph.

    Betweenness centrality measures the extent to which a node lies on paths between other nodes.
    Nodes with high betweenness centrality are important bridges between different parts of a network.

    Parameters
    ----------
    expr_from : IntoExpr
        Expression representing the source nodes of the edges.
    expr_to : IntoExpr
        Expression representing the destination nodes of the edges.
    normalized : bool, default True
        If True, the betweenness values are normalized by `2/((n-1)(n-2))` for undirected graphs
        and `1/((n-1)(n-2))` for directed graphs, where n is the number of nodes.
    directed : bool, default False
        If True, treats the graph as directed. If False, treats edges as bidirectional.

    Returns
    -------
    pl.Expr
        A Polars expression that resolves to a struct containing:
        - "node": node identifier
        - "centrality": betweenness centrality score

    Examples
    --------
    >>> import polars as pl
    >>> # Create a sample graph
    >>> df = pl.DataFrame({
    ...     "from": ["A", "A", "B", "C", "D", "E"],
    ...     "to": ["B", "C", "C", "D", "E", "A"]
    ... })
    >>>
    >>> # Calculate betweenness centrality
    >>> result = df.select(
    ...     betweenness_centrality(
    ...         pl.col("from"),
    ...         pl.col("to"),
    ...         normalized=True,
    ...         directed=False
    ...     ).alias("centrality")
    ... ).unnest("centrality")
    >>>
    >>> print(result)
    shape: (5, 2)
    ┌──────┬────────────┐
    │ node ┆ centrality │
    │ str  ┆ f64        │
    ╞══════╪════════════╡
    │ A    ┆ 0.5       │
    │ B    ┆ 0.1       │
    │ C    ┆ 0.7       │
    │ D    ┆ 0.3       │
    │ E    ┆ 0.1       │
    └──────┴────────────┘

    Notes
    -----
    - Higher values indicate nodes that act as important bridges in the network
    - Values range from 0 to 1 when normalized
    - Computation time is O(|V||E|) for unweighted graphs
    - Memory usage is O(|V| + |E|)
    - For large graphs, consider using approximate algorithms
    - Isolated nodes will have centrality of 0

    """
    return register_plugin_function(
        args=[expr_from, expr_to],
        plugin_path=LIB,
        function_name="graph_betweenness_centrality",
        is_elementwise=False,
        changes_length=True,
        kwargs={"normalized": normalized, "directed": directed}
    )


def graph_association_rules(
    transaction_id: IntoExpr,
    item_id: IntoExpr,
    frequency: IntoExpr | None = None,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    max_itemset_size: int = 50,
    weighted: bool = False
) -> pl.Expr:
    """
    Perform association rule mining to identify item relationships, frequent patterns,
    and lift scores using a graph-based approach.

    Parameters
    ----------
    transaction_id : IntoExpr
        Expression identifying unique transactions
    item_id : IntoExpr
        Expression identifying items in each transaction
    frequency : IntoExpr, optional
        Expression containing item frequencies in transactions
    min_support : float, default 0.01
        Minimum support threshold (proportion of transactions containing the item)
    min_confidence : float, default 0.1
        Minimum confidence threshold for association rules
    max_itemset_size : int, default 50
        Maximum itemset size to consider (prevents combinatorial explosion)
    weighted : bool, default False
        If True, uses frequencies to weight association rules

    Returns
    -------
    pl.Expr
        A Polars expression that resolves to a struct containing:
        - "item": item identifier
        - "support": support count in transactions
        - "lift_score": item"s importance in association network
        - "pattern": frequent pattern identifier
        - "consequents": list of top 5 frequently associated items
        - "confidence_scores": confidence scores for associations

    Examples
    --------
    >>> import polars as pl
    >>> # Create sample transaction data
    >>> df = pl.DataFrame({
    ...     "transaction_id": [1, 1, 1, 2, 2, 3],
    ...     "item_id": ["A", "B", "C", "B", "D", "A"],
    ...     "frequency": [1, 2, 1, 1, 1, 1]
    ... })
    >>>
    >>> # Mine association rules
    >>> result = df.select(
    ...     graph_association_rules(
    ...         pl.col("transaction_id"),
    ...         pl.col("item_id"),
    ...         pl.col("frequency"),
    ...         min_support=0.1,
    ...         weighted=True
    ...     ).alias("rules")
    ... ).unnest("rules")
    >>>
    >>> print(result)
    shape: (4, 6)
    ┌──────┬─────────┬───────────┬─────────┬─────────────┬──────────────────┐
    │ item ┆ support ┆ lift_score┆ pattern ┆ consequents ┆ confidence_scores │
    │ str  ┆ f64     ┆ f64      ┆ u32     ┆ list[str]   ┆ list[f64]        │
    ╞══════╪═════════╪═══════════╪═════════╪═════════════╪══════════════════╡
    │ A    ┆ 2.0     ┆ 0.667    ┆ 1       ┆ ["B", "C"]  ┆ [0.5, 0.5]      │
    │ B    ┆ 3.0     ┆ 1.0      ┆ 1       ┆ ["D", "A"]  ┆ [0.33, 0.33]    │
    │ C    ┆ 1.0     ┆ 0.333    ┆ 1       ┆ ["A", "B"]  ┆ [1.0, 1.0]      │
    │ D    ┆ 1.0     ┆ 0.333    ┆ 1       ┆ ["B"]       ┆ [1.0]           │
    └──────┴─────────┴───────────┴─────────┴─────────────┴──────────────────┘

    Notes
    -----
    - Items must appear in min_support proportion of transactions to be included
    - Confidence scores indicate strength of association rules
    - Patterns identify groups of frequently associated items
    - Lift score indicates item"s importance in association networks
    - Large itemsets (> max_itemset_size) are filtered to prevent performance issues

    """
    return register_plugin_function(
        args=[transaction_id, item_id] + ([frequency] if frequency is not None else []),
        plugin_path=LIB,
        function_name="graph_association_rules",
        is_elementwise=False,
        changes_length=True,
        kwargs={
            "min_support": min_support,
            "min_confidence": min_confidence,
            "max_itemset_size": max_itemset_size,
            "weighted": weighted
        }
    )