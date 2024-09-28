import polars as pl
import polars_graph as pg
import random
import string

import polars as pl
from performance_tests.native_python_implementation import add_connected_components
from time import time

# df = pl.read_parquet('~/downloads/artist_match.parquet')


# Load the Parquet file
input_file = "graph_edges_large.parquet"
df = pl.read_parquet(input_file)


# Convert Polars DataFrame into a list of edges using the utility function
print(add_connected_components(df).select('group').unique())
print(df.select(group=pg.graph_solver(pl.col("from"), pl.col("to"))))