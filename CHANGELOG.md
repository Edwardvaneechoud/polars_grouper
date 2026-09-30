# Changelog

All notable changes to `polars-grouper` are documented here.

## 0.6.1

### Added

- **Linux aarch64 (arm64) wheel.** Releases now include a
  `cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64` wheel, with the same abi3 tag and
  glibc baseline as the x86_64 one. `pip install polars-grouper` on linux/arm64 (Docker on
  Apple Silicon, AWS Graviton) no longer falls back to the sdist, so it no longer needs a Rust
  toolchain and a C compiler. Before publishing, CI installs this wheel in `python:3.12-slim`
  on an arm64 runner and runs the test suite against it.

### Fixed

- `polars_grouper.__version__` reported `0.5.1` in the 0.6.0 release. It now matches the
  package version.

No API changes. The supported Polars range is unchanged.

## 0.6.0

### Added

- **Hierarchy / bill-of-materials explosion**: `hierarchy_totals`, `hierarchy_levels` and
  `hierarchy_paths` resolve the transitive closure of a `parent -> child` edge list in a single
  (lazy) expression, replacing `WITH RECURSIVE` queries and for-loops. Quantities multiply along
  each path and add up across paths. The three functions return the same explosion at
  increasing detail: one row per ancestor/descendant pair, per pair and level, or per path
  (an indented BOM with the full `path` list). Options: `top_level_only`, `include_self` and
  `max_depth`. A cycle raises an error that names it, and integer node ids keep their dtype.

## 0.5.1

### Fixed

- **`calculate_shortest_path(..., directed=True)` returned every reachable pair twice.**
  Results were duplicated by an exact factor of 2 for all directed graphs, with no error
  raised. Anything aggregating a directed result — a sum, a count, a mean weighted by row
  count — was wrong by that factor, so **directed results computed with 0.5.0 or earlier
  should be recomputed**. Undirected results were unaffected. The cause was the
  result-collection loop: the inner loop already ran over every *ordered* pair, and a
  second reverse-direction pass then emitted each pair's mirror image again.
- `calculate_shortest_path` now broadcasts a length-1 weights argument across every edge,
  so `pl.lit(1.0)` means "all edges weigh the same". Previously the scalar was zipped
  against the node columns, truncating the graph to a single edge and silently returning
  a result computed from it. Any other length mismatch now raises a `ShapeMismatch`
  naming the offending argument and the expected length, instead of truncating.
- `calculate_shortest_path` no longer panics on an empty edge list (an unsigned subtraction
  overflowed when reserving result capacity for a zero-node graph).

### Changed

- `calculate_shortest_path` now runs one single-source Dijkstra per node instead of one
  per node *pair*, reducing the number of traversals from O(V²) to O(V). Output row
  ordering is unchanged.
