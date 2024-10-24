import polars as pl
import pytest
from polars_grouper import graph_solver, super_merger, page_rank, calculate_shortest_path, betweenness_centrality, graph_association_rules
import math


def test_page_rank():
    df = pl.DataFrame(
        {
            "from": ["A", "B", "C", "E", "F", "G", "I", "I", 'AA'],
            "to": ["B", "C", "D", "F", "G", "J", "K", "J", 'Z']
        }
    )
    result_df = df.select(page_rank(pl.col("from"), pl.col("to")).alias('rank'))
    expected_df = pl.DataFrame(
        {
            "rank": [0.012500000000000002, 0.023125000000000007, 0.032156250000000004, 0.012500000000000002,
                     0.023125000000000007, 0.032156250000000004, 0.012500000000000002, 0.012500000000000002,
                     0.012500000000000002]
        }
    )
    assert result_df.equals(expected_df), "The rank values were not calculated as expected."


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


def test_basic_betweenness_centrality():
    """
    Test betweenness centrality on a simple line graph: A -- B -- C
    B should have the highest centrality as it's on all paths.
    """
    df = pl.DataFrame({
        "from": ["A", "B"],
        "to": ["B", "C"]
    })

    result = df.select(
        betweenness_centrality(
            pl.col("from"),
            pl.col("to"),
            normalized=True,
            directed=False
        ).alias("centrality")
    ).unnest("centrality")

    # Get centrality for node B
    b_centrality = result.filter(pl.col("node") == "B")["centrality"][0]
    # Get centrality for end nodes
    end_centrality = result.filter(pl.col("node").is_in(["A", "C"]))["centrality"].mean()

    assert b_centrality > end_centrality
    assert math.isclose(b_centrality, 1.0, rel_tol=1e-5)
    assert math.isclose(end_centrality, 0.0, rel_tol=1e-5)


def test_star_graph_betweenness():
    """
    Test betweenness centrality on a star graph:
          B
          |
    C --- A --- D
          |
          E
    Central node A should have highest centrality.
    """
    df = pl.DataFrame({
        "from": ["A", "A", "A", "A"],
        "to": ["B", "C", "D", "E"]
    })

    result = df.select(
        betweenness_centrality(
            pl.col("from"),
            pl.col("to"),
            normalized=True,
            directed=False
        ).alias("centrality")
    ).unnest("centrality")

    # Get centrality for center node A
    center_centrality = result.filter(pl.col("node") == "A")["centrality"][0]
    # Get centrality for peripheral nodes
    peripheral_centrality = result.filter(pl.col("node") != "A")["centrality"].mean()

    assert center_centrality > peripheral_centrality
    assert math.isclose(peripheral_centrality, 0.0, rel_tol=1e-5)


def test_directed_vs_undirected():
    """
    Test that directed and undirected graphs give different results
    for the same edge set.
    """
    df = pl.DataFrame({
        "from": ["A", "B", "C"],
        "to": ["B", "C", "A"]
    })

    directed_result = df.select(
        betweenness_centrality(
            pl.col("from"),
            pl.col("to"),
            normalized=True,
            directed=True
        ).alias("centrality")
    ).unnest("centrality")

    undirected_result = df.select(
        betweenness_centrality(
            pl.col("from"),
            pl.col("to"),
            normalized=True,
            directed=False
        ).alias("centrality")
    ).unnest("centrality")

    # Results should be different for directed vs undirected
    assert not directed_result.equals(undirected_result)


def test_disconnected_components():
    """
    Test betweenness centrality with disconnected components:
    A -- B -- C   D -- E
    """
    df = pl.DataFrame({
        "from": ["A", "B", "D"],
        "to": ["B", "C", "E"]
    })

    result = df.select(
        betweenness_centrality(
            pl.col("from"),
            pl.col("to"),
            normalized=True,
            directed=False
        ).alias("centrality")
    ).unnest("centrality")

    # Node B should have highest centrality in its component
    b_centrality = result.filter(pl.col("node") == "B")["centrality"][0]
    assert b_centrality > 0

    # End nodes should have zero centrality
    end_nodes = result.filter(pl.col("node").is_in(["A", "C", "D", "E"]))
    assert all(math.isclose(c, 0.0, rel_tol=1e-5) for c in end_nodes["centrality"])


def test_betweenness_empty_graph():
    """
    Test betweenness centrality with an empty graph.
    """
    df = pl.DataFrame({
        "from": [],
        "to": []
    })

    result = df.select(
        betweenness_centrality(
            pl.col("from"),
            pl.col("to")
        ).alias("centrality")
    ).unnest("centrality")

    assert len(result) == 0


def test_basic_association_rules():
    # Create sample transaction data
    df = pl.DataFrame({
        "transaction_id": [1, 1, 1, 2, 2, 3],
        "item_id": ["A", "B", "C", "B", "D", "A"],
        "frequency": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    })

    # Apply association rules
    result = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency"),
            min_support=0.1,
            min_confidence=0.1,
            weighted=True
        ).alias("rules")
    ).unnest("rules")

    # Check basic properties
    assert len(result) > 0
    assert all(col in result.columns for col in [
        "item", "support", "lift_score", "pattern",
        "consequents", "confidence_scores"
    ])

    # Check data types
    assert result.schema["item"] == pl.String
    assert result.schema["support"] == pl.Float64
    assert result.schema["lift_score"] == pl.Float64
    assert result.schema["pattern"] == pl.UInt32
    assert result.schema["consequents"].inner == pl.String
    assert result.schema["confidence_scores"].inner == pl.Float64


def test_empty_transactions():
    df = pl.DataFrame({
        "transaction_id": [],
        "item_id": [],
        "frequency": []
    })

    result = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency")
        ).alias("rules")
    ).unnest("rules")

    assert len(result) == 0


def test_single_item_transactions():
    df = pl.DataFrame({
        "transaction_id": [1, 2, 3],
        "item_id": ["A", "A", "A"],
        "frequency": [1.0, 1.0, 1.0]
    })

    result = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency")
        ).alias("rules")
    ).unnest("rules")

    # Should have one item with no associations
    assert len(result) == 1
    assert result["item"][0] == "A"
    assert len(result["consequents"][0]) == 0
    assert len(result["confidence_scores"][0]) == 0


def test_min_support_threshold():
    df = pl.DataFrame({
        "transaction_id": [1, 1, 2, 3, 4],
        "item_id": ["A", "B", "B", "C", "C"],
        "frequency": [1.0, 1.0, 1.0, 1.0, 1.0]
    })

    # Set high min_support to filter out rare items
    result = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency"),
            min_support=0.5  # 50% of transactions
        ).alias("rules")
    ).unnest("rules")
    # Only items appearing in ≥50% of transactions should remain
    items = result["item"].to_list()
    assert "B" in items  # Appears in 2/4 transactions
    assert "C" in items  # Appears in 2/4 transactions
    assert "A" not in items  # Appears in only 1/4 transactions


def test_weighted_vs_unweighted():
    df = pl.DataFrame({
        "transaction_id": [1, 1, 2, 2],
        "item_id": ["A", "B", "A", "B"],
        "frequency": [1.0, 2.0, 2.0, 1.0]
    })

    # Get results with and without weighting
    weighted = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency"),
            weighted=True
        ).alias("rules")
    ).unnest("rules")

    unweighted = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency"),
            weighted=False
        ).alias("rules")
    ).unnest("rules")

    # Support values should differ between weighted and unweighted
    assert not all(w == u for w, u in zip(weighted["support"], unweighted["support"]))


def test_max_itemset_size():
    # Create data with large transactions
    transaction = list(range(1, 52))  # 51 items
    df = pl.DataFrame({
        "transaction_id": [1] * 51,
        "item_id": [f"item_{i}" for i in transaction],
        "frequency": [1.0] * 51
    })

    result = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency"),
            max_itemset_size=50
        ).alias("rules")
    ).unnest("rules")

    # Should still process the data without error
    assert len(result) > 0


def test_null_handling():
    df = pl.DataFrame({
        "transaction_id": [1, 1, None, 2, 2],
        "item_id": ["A", "B", "C", None, "D"],
        "frequency": [1.0, None, 1.0, 1.0, 1.0]
    })

    # Should handle nulls without error
    result = df.select(
        graph_association_rules(
            pl.col("transaction_id"),
            pl.col("item_id"),
            pl.col("frequency")
        ).alias("rules")
    ).unnest("rules")

    assert len(result) > 0


def test_calculate_shortest_path():
    df = pl.DataFrame({
        "from": ["A", "A", "B", "C"],
        "to": ["B", "C", "C", "D"],
        "weight": [1.0, 2.0, 1.0, 1.5]
    })

    result = df.select(
        calculate_shortest_path(
            pl.col("from"),
            pl.col("to"),
            pl.col("weight"),
            directed=False
        ).alias("paths")
    ).unnest("paths")

    # Check if all paths are present
    expected_paths = {
        ('A', 'B'): 1.0,
        ('A', 'C'): 2.0,  # Direct path
        ('A', 'D'): 3.5,  # Path through C
        ('B', 'C'): 1.0,
        ('B', 'D'): 2.5,  # Path through C
        ('C', 'D'): 1.5
    }

    # Convert result to dictionary for easier comparison
    actual_paths = {(row['from'], row['to']): row['distance']
                    for row in result.to_dicts()}
    print(actual_paths)
    assert len(result) == len(expected_paths)
    for (start, end), distance in expected_paths.items():
        assert abs(actual_paths[(start, end)] - distance) < 1e-6


def test_directed_path():
    df = pl.DataFrame({
        "from": ["A", "B", "B", "C"],
        "to": ["B", "C", "A", "A"],
        "weight": [1.0, 2.0, 3.0, 4.0]
    })

    result = df.select(
        calculate_shortest_path(
            pl.col("from"),
            pl.col("to"),
            pl.col("weight"),
            directed=True
        ).alias("paths")
    ).unnest("paths")

    # Verify asymmetric paths
    paths_dict = {(row['from'], row['to']): row['distance']
                  for row in result.to_dicts()}

    # A → B should be 1.0
    assert abs(paths_dict[('A', 'B')] - 1.0) < 1e-6
    # B → A should be 3.0 (direct path)
    assert abs(paths_dict[('B', 'A')] - 3.0) < 1e-6


def test_cycle_path():
    df = pl.DataFrame({
        "from": ["A", "B", "C", "A"],
        "to": ["B", "C", "A", "C"],
        "weight": [1.0, 1.0, 3.0, 2.0]
    })

    result = df.select(
        calculate_shortest_path(
            pl.col("from"),
            pl.col("to"),
            pl.col("weight"),
            directed=True
        ).alias("paths")
    ).unnest("paths")

    paths_dict = {(row['from'], row['to']): row['distance']
                  for row in result.to_dicts()}

    # A → C should choose shorter path (A → B → C = 2.0) over direct path (3.0)
    assert abs(paths_dict[('A', 'C')] - 2.0) < 1e-6


def test_calculate_path_empty_graph():
    df = pl.DataFrame({
        "from": [],
        "to": [],
        "weight": []
    })

    result = df.select(
        calculate_shortest_path(
            pl.col("from"),
            pl.col("to"),
            pl.col("weight")
        ).alias("paths")
    ).unnest("paths")

    assert len(result) == 0



if __name__ == '__main__':
    pytest.main()