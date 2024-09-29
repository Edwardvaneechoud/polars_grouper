import polars as pl
from polars_graph import graph_solver


def test_piglatinnify():
    df = pl.DataFrame(
        {
            "from": ["A", "B", "C", "E", "F", "G", "I"],
            "to": ["B", "C", "D", "F", "G", "J", "K"]
        }
    )
    result = df.select(graph_solver(pl.col("from"), pl.col("to")).alias('group'))

    expected_df = pl.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 2, 3]
        }
    )

    assert result.equals(expected_df)
