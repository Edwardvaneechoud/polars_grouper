import polars as pl
import polars_graph as pg

import polars as pl
from performance_tests.native_python_implementation import add_connected_components
from time import time

# df = pl.read_parquet('~/downloads/artist_match.parquet')


data = {
    "from": ["A", "B", "C", "E", "F", "G", "I"],
    "to": ["B", "C", "D", "F", "G", "J", "K"]
}

df = pl.LazyFrame(data)

# Load the Parquet file
input_file = "performance_tests/data/graph_edges_large.parquet"
df = pl.scan_parquet(input_file)


start_time = time()
result_add_connected_components = add_connected_components(df.collect()).select('group').unique()
end_time = time()
print(f"Time taken by add_connected_components: {end_time - start_time:.4f} seconds")

# Measure time for graph_solver function
start_time = time()
result_graph_solver = df.with_columns(group=pg.graph_solver(pl.col("from"), pl.col("to")).alias('group')).select('group').unique().collect()
end_time = time()
print(f"Time taken by graph_solver: {end_time - start_time:.4f} seconds")
