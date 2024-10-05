import polars as pl
import pytest
from polars_grouper import graph_solver, super_merger


def test_graph_solver():
    """
    Test that the graph_solver correctly assigns group IDs to connected components.
    """
    df = pl.DataFrame(
        {
            "from": ["A", "B", "C", "E", "F", "G", "I", "I", 'AA'],
            "to": ["B", "C", "D", "F", "G", "J", "K", "J", 'Z']
        }
    )
    result_df = df.select(graph_solver(pl.col("from"), pl.col("to")).alias('group'))
    expected_df = pl.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 2, 2, 2, 3]
        }
    )

    assert result_df.equals(expected_df), "The graph_solver did not assign the expected group IDs."


def test_super_merger():
    """
    Test that the supermerger function correctly adds group IDs to a DataFrame.
    """
    df = pl.DataFrame(
        {
            "from": ["A", "B", "C", "E", "F", "G", "I"],
            "to": ["B", "C", "D", "F", "G", "J", "K"]
        }
    )

    result_df = super_merger(df, "from", "to")
    expected_df = pl.DataFrame(
        {
            "from": ["A", "B", "C", "E", "F", "G", "I"],
            "to": ["B", "C", "D", "F", "G", "J", "K"],
            "group": [1, 1, 1, 2, 2, 2, 3]
        }
    )

    assert result_df.equals(expected_df), "The supermerger did not assign the expected group IDs."


def test_supermerger_with_empty_df():
    """
    Test that the supermerger function works correctly with an empty DataFrame.
    """
    df = pl.DataFrame(
        {
            "from": [],
            "to": []
        }
    )

    result_df = super_merger(df, "from", "to")
    expected_df = pl.DataFrame(
        {
            "from": [],
            "to": [],
            "group": []
        }
    )

    assert result_df.equals(expected_df), "The supermerger did not handle an empty DataFrame as expected."


def test_supermerger_with_single_component():
    """
    Test that the supermerger function works correctly with a single connected component.
    """
    df = pl.DataFrame(
        {
            "from": ["A", "B", "C"],
            "to": ["B", "C", "A"]
        }
    )

    result_df = super_merger(df, "from", "to")
    expected_df = pl.DataFrame(
        {
            "from": ["A", "B", "C"],
            "to": ["B", "C", "A"],
            "group": [1, 1, 1]
        }
    )

    assert result_df.equals(expected_df), "The supermerger did not correctly identify a single connected component."


if __name__ == '__main__':
    pytest.main()