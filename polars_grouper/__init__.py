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


# Register the graph_solver function
def graph_solver(expr_from: IntoExpr, expr_to: IntoExpr) -> pl.Expr:
    """
    This function registers a plugin function to solve graph components.

    Parameters
    ----------
    expr_from : IntoExpr
        The column representing the source nodes of the edges.
    expr_to : IntoExpr
        The column representing the destination nodes of the edges.

    Returns
    -------
    pl.Expr
        An expression that can be used in Polars transformations to represent the graph component solution.

    Notes
    -----
    This function registers a custom plugin for Polars that performs graph operations
    by processing columns that represent edges between nodes.
    """
    return register_plugin_function(
        args=[expr_from, expr_to],
        plugin_path=LIB,
        function_name="graph_solver",
        is_elementwise=False
    )


def super_merger(df: DF, from_col_name: str, to_col_name: str) -> DF:
    """
    Find connected components from a Polars DataFrame of edges and add group information.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Polars DataFrame with columns representing the edges.
    from_col_name : str
        The name of the column containing the source nodes.
    to_col_name : str
        The name of the column containing the target nodes.

    Returns
    -------
    DF
        The input DataFrame with an additional 'group' column representing the connected component group.

    Examples
    --------
    >>> lf = pl.LazyFrame({
    >>> "from": ["A", "B", "C", "E", "F", "G", "I"],
    >>> "to": ["B", "C", "D", "F", "G", "J", "K"] })
    >>> result_df = super_merger(lf, "from", "to")
    >>> print(result_df.collect())
    """
    return df.with_columns(graph_solver(pl.col(from_col_name), pl.col(to_col_name)).alias('group'))
