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
    return register_plugin_function(
        args=[expr_from, expr_to],
        plugin_path=LIB,
        function_name="graph_solver",
        is_elementwise=False,
        changes_length=True
    )
