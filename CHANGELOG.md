# Changelog

All notable changes to `polars-grouper` are documented here.

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
