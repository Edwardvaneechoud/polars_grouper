# PolarsGrouper

PolarsGrouper is a Rust-based extension for Polars that provides efficient graph analysis capabilities, with a focus on component grouping and network analysis.

## Core Features

### Component Grouping
- `super_merger`: Easy-to-use wrapper for grouping connected components
- `super_merger_weighted`: Component grouping with weight thresholds
- Efficient implementation using Rust and Polars
- Works with both eager and lazy Polars DataFrames

### Additional Graph Analytics
- **Shortest Path Analysis**: Find shortest paths between nodes
- **PageRank**: Calculate node importance scores
- **Betweenness Centrality**: Identify key bridge nodes
- **Association Rules**: Discover item relationships and patterns

## Installation

```sh
pip install polars-grouper

# For development:
python -m venv .venv
source .venv/bin/activate
maturin develop
```

## Usage Examples

### Basic Component Grouping
The core functionality uses `super_merger` to identify connected components:

```python
import polars as pl
from polars_grouper import super_merger

df = pl.DataFrame({
    "from": ["A", "B", "C", "D", "E", "F"],
    "to": ["B", "C", "A", "E", "F", "D"],
    "value": [1, 2, 3, 4, 5, 6]
})

result = super_merger(df, "from", "to")
print(result)
```

### Weighted Component Grouping
For cases where edge weights matter:

```python
from polars_grouper import super_merger_weighted

df = pl.DataFrame({
    "from": ["A", "B", "C", "D", "E"],
    "to": ["B", "C", "D", "E", "A"],
    "weight": [0.9, 0.2, 0.05, 0.8, 0.3]
})

result = super_merger_weighted(
    df, 
    "from", 
    "to", 
    "weight",
    weight_threshold=0.3
)
print(result)
```

### Additional Graph Analytics

#### Shortest Path Analysis
Find shortest paths between nodes:

```python
from polars_grouper import calculate_shortest_path

df = pl.DataFrame({
    "from": ["A", "A", "B", "C"],
    "to": ["B", "C", "C", "D"],
    "weight": [1.0, 2.0, 1.0, 1.5]
})

paths = df.select(
    calculate_shortest_path(
        pl.col("from"),
        pl.col("to"),
        pl.col("weight"),
        directed=False
    ).alias("paths")
).unnest("paths")
```

#### PageRank Calculation
Calculate node importance:

```python
from polars_grouper import page_rank

df = pl.DataFrame({
    "from": ["A", "A", "B", "C", "D"],
    "to": ["B", "C", "C", "A", "B"]
})

rankings = df.select(
    page_rank(
        pl.col("from"),
        pl.col("to"),
        damping_factor=0.85
    ).alias("pagerank")
).unnest("pagerank")
```

#### Association Rule Mining
Discover item relationships:

```python
from polars_grouper import graph_association_rules

transactions = pl.DataFrame({
    "transaction_id": [1, 1, 1, 2, 2, 3],
    "item_id": ["A", "B", "C", "B", "D", "A"],
    "frequency": [1, 2, 1, 1, 1, 1]
})

rules = transactions.select(
    graph_association_rules(
        pl.col("transaction_id"),
        pl.col("item_id"),
        pl.col("frequency"),
        min_support=0.1
    ).alias("rules")
).unnest("rules")
```

#### Betweenness Centrality
Identify bridge nodes:

```python
from polars_grouper import betweenness_centrality

df = pl.DataFrame({
    "from": ["A", "A", "B", "C", "D", "E"],
    "to": ["B", "C", "C", "D", "E", "A"]
})

centrality = df.select(
    betweenness_centrality(
        pl.col("from"),
        pl.col("to"),
        normalized=True
    ).alias("centrality")
).unnest("centrality")
```

## Performance

The library is implemented in Rust for high performance:
- Efficient memory usage
- Fast computation for large graphs
- Seamless integration with Polars' lazy evaluation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
