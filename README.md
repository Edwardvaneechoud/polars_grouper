# PolarsGrouper

PolarsGrouper is a Rust-based extension for Polars that efficiently groups connected components in dataframes for graph analysis.

## Features
- Group connected components in dataframes.
- High performance using Rust and Polars.
- Works with both eager and lazy Polars DataFrames.

## Installation
To install PolarsGrouper, use `pip`:

```sh
pip install polars-grouper
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Build and install using maturin
maturin develop
```

## Usage
graph_solver
Register the graph_solver function:

```python
import polars as pl
from polars_grouper import graph_solver

df = pl.DataFrame({
    "from": ["A", "B", "C"],
    "to": ["B", "C", "D"]
})

result_df = df.with_columns(
    graph_solver(pl.col("from"), pl.col("to")).alias("group")
)
print(result_df)

```

# Super_merger
Use super_merger to add connected component group information to a dataframe:

```python
from polars_grouper import super_merger
import polars as pl

df = pl.LazyFrame({
    "from": ["A", "B", "C"],
    "to": ["B", "C", "D"]
})

result_df = super_merger(df, "from", "to")
print(result_df.collect())

```
# Performance
Comparing the rust implementation with the native python options, the rust version outperforms the Python implementations significantly:

![image](https://github.com/user-attachments/assets/fcb7df04-ccd3-48ad-b488-912d4abad0df)

