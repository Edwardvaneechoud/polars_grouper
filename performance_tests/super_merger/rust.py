from polars_grouper import graph_solver
import polars as pl
from typing import TypeVar


DF = TypeVar("DF", pl.DataFrame, pl.LazyFrame)


def super_merger_rust(df: DF, from_col_name: str, to_col_name: str) -> DF:
    """
    This function finds connected components from a Polars DataFrame of edges and adds the group information.

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
    >>> lf = pl.LazyFrame({
    >>> "from": ["A", "B", "C", "E", "F", "G", "I"],
    >>> "to": ["B", "C", "D", "F", "G", "J", "K"] })
    >>> result_df = super_merger_rust(lf, "from", "to")
    >>> print(result_df.collect())
    """

    return df.with_columns(graph_solver(pl.col(from_col_name), pl.col(to_col_name)).alias('group'))

