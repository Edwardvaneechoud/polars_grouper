from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_graph._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_graph.typing import IntoExpr

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
