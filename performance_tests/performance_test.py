from performance_tests.super_merger.simple import super_merger_simple
from performance_tests.super_merger.networkx import super_merger_networkx
from performance_tests.super_merger.bfs import super_merger_bfs
from performance_tests.super_merger.rust import super_merger_rust
import polars as pl
import timeit

df = pl.read_parquet('performance_tests/data/data_10000.parquet')


# Define a wrapper function for each implementation to use in timeit
def test_simple():
    super_merger_simple(df, 'from', 'to')


def test_networkx():
    super_merger_networkx(df, 'from', 'to')


def test_bfs():
    super_merger_bfs(df, 'from', 'to')


def test_rust():
    r = super_merger_rust(df, 'from', 'to').select('group').unique()


# Run the performance tests
num_runs = 10  # Set the number of times to run each function

# Time the simple implementation
simple_time = timeit.timeit("test_simple()", globals=globals(), number=num_runs)
print(f"Average time for super_merger_simple: {simple_time / num_runs:.4f} seconds per run")

# Time the NetworkX implementation
networkx_time = timeit.timeit("test_networkx()", globals=globals(), number=num_runs)
print(f"Average time for super_merger_networkx: {networkx_time / num_runs:.4f} seconds per run")

# Time the BFS implementation
bfs_time = timeit.timeit("test_bfs()", globals=globals(), number=num_runs)
print(f"Average time for super_merger_bfs: {bfs_time / num_runs:.4f} seconds per run")
#
# # Time the Rust implementation
rust_time = timeit.timeit("test_rust()", globals=globals(), number=num_runs)
print(f"Average time for super_merger_rust: {rust_time / num_runs:.4f} seconds per run")
