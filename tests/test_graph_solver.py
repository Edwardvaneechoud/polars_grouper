import polars as pl
from polars_graph import graph_solver


def test_piglatinnify():
    df = pl.DataFrame(
        {
            "from": ["A", "B", "C", "E", "F", "G", "I"],
            "to": ["B", "C", "D", "F", "G", "J", "K"]
        }
    )
    result = df.select(graph_solver(pl.col("from"), pl.col("to")))

    expected_df = pl.DataFrame(
        {
            "english": ["this", "is", "not", "pig", "latin"],
            "pig_latin": ["histay", "siay", "otnay", "igpay", "atinlay"],
        }
    )

    assert result.equals(expected_df)


