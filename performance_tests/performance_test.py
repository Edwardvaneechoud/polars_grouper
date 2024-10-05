from performance_tests.super_merger.simple import super_merger_simple
from performance_tests.super_merger.rust import super_merger_rust
import polars as pl
import time


# Define a helper function to measure execution time
def measure_time(func, df, name, num_runs=10):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        func(df, 'from', 'to').count()
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time = total_time / num_runs
    print(f"Average time for {name}: {avg_time:.4f} seconds per run")
    return avg_time


# Loop through different sizes of data
for size in [100, 1000, 10000, 100000, 1000000]:
    print(f"Running tests for dataset size: {size}")

    # Load the dataset
    df = pl.read_parquet(f'performance_tests/data/data_{size}.parquet')
    # ndf = df.rename({'from': 'to', 'to': 'from'})
    #df = pl.concat([df, ndf], how='diagonal_relaxed')
    if size == 1000000:
        n_runs = 2
    elif size == 100000:
        n_runs = 5
    else:
        n_runs = 10
    # Measure time for each implementation
    simple_time = measure_time(super_merger_simple, df, 'super_merger_simple', n_runs)
    rust_time = measure_time(super_merger_rust, df, 'super_merger_rust', n_runs)

    # Print the performance ratio
    print(f"Performance ratio (simple/rust) for size {size}: {simple_time / rust_time:.4f}")
    print("-" * 50)


