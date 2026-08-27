use crate::graph_utils::{
    broadcast_to_len, edge_count, to_float64_chunked, to_string_chunked, usize_to_t, AsUsize,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::convert::TryFrom;

// Type aliases to reduce complexity
type NodeMap<T> = HashMap<String, T>;
type Edge<T> = (T, T, i64);
type EdgeList<T> = Vec<Edge<T>>;
type ProcessResult<T> = PolarsResult<(NodeMap<T>, T, EdgeList<T>)>;

#[derive(Deserialize)]
struct ShortestPathKwargs {
    directed: bool,
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: i64,
    position: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn process_edges_with_weights<T>(
    from: StringChunked,
    to: StringChunked,
    weights: Float64Chunked,
) -> ProcessResult<T>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let estimated_size = from.len();
    let mut node_to_id = HashMap::with_capacity(estimated_size);
    let mut edges = Vec::with_capacity(estimated_size);
    let mut id_counter: T = usize_to_t(0);

    let mut process_node = |node: &str| -> T {
        if let Some(&id) = node_to_id.get(node) {
            id
        } else {
            let id = id_counter;
            node_to_id.insert(node.to_string(), id);
            id_counter = usize_to_t(id_counter.as_usize() + 1);
            id
        }
    };

    from.iter()
        .zip(to.iter())
        .zip(weights.iter())
        .try_for_each(|((from_node, to_node), weight)| -> PolarsResult<()> {
            if let (Some(f), Some(t), Some(w)) = (from_node, to_node, weight) {
                let f_id = process_node(f);
                let t_id = process_node(t);
                edges.push((f_id, t_id, (w * 1000.0) as i64));
            }
            Ok(())
        })?;

    Ok((node_to_id, id_counter, edges))
}

/// Single-source Dijkstra.
///
/// Returns the cost of the cheapest path from `start_id` to every node, with
/// `i64::MAX` marking nodes that are unreachable. Resolving every target in one
/// traversal is what lets the caller emit exactly one row per (source, target)
/// pair from a single loop.
fn shortest_paths_from(start_id: usize, adj_list: &[Vec<(usize, i64)>]) -> Vec<i64> {
    let num_nodes = adj_list.len();
    let mut dist = vec![i64::MAX; num_nodes];
    dist[start_id] = 0;

    let mut heap = BinaryHeap::with_capacity(num_nodes);
    heap.push(State {
        cost: 0,
        position: start_id,
    });

    while let Some(State { cost, position }) = heap.pop() {
        if cost > dist[position] {
            continue;
        }

        for &(neighbor, weight) in &adj_list[position] {
            let next_cost = cost + weight;
            if next_cost < dist[neighbor] {
                dist[neighbor] = next_cost;
                heap.push(State {
                    cost: next_cost,
                    position: neighbor,
                });
            }
        }
    }

    dist
}

fn shortest_path_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "shortest_paths".into(),
        DataType::Struct(vec![
            Field::new("from".into(), DataType::String),
            Field::new("to".into(), DataType::String),
            Field::new("distance".into(), DataType::Float64),
        ]),
    ))
}

#[polars_expr(output_type_func=shortest_path_output)]
fn graph_find_shortest_path(inputs: &[Series], kwargs: ShortestPathKwargs) -> PolarsResult<Series> {
    // `from`/`to` define the edges; a scalar weight is then stretched to match them.
    let num_edges = edge_count(&inputs[..2]);
    let from = to_string_chunked(&broadcast_to_len(&inputs[0], num_edges, "from")?)?;
    let to = to_string_chunked(&broadcast_to_len(&inputs[1], num_edges, "to")?)?;
    let weights = to_float64_chunked(&broadcast_to_len(&inputs[2], num_edges, "weights")?)?;

    type NodeId = u32;

    let (node_to_id, id_counter, edges) = process_edges_with_weights::<NodeId>(from, to, weights)?;
    let num_nodes = id_counter.as_usize();

    let mut adj_list = vec![Vec::new(); num_nodes];
    for (from_id, to_id, weight) in edges {
        adj_list[from_id.as_usize()].push((to_id.as_usize(), weight));
        if !kwargs.directed {
            adj_list[to_id.as_usize()].push((from_id.as_usize(), weight));
        }
    }

    let ordered_pairs = num_nodes * num_nodes.saturating_sub(1);
    let estimated_pairs = if kwargs.directed {
        ordered_pairs
    } else {
        ordered_pairs / 2
    };

    let mut from_nodes = Vec::with_capacity(estimated_pairs);
    let mut to_nodes = Vec::with_capacity(estimated_pairs);
    let mut distances = Vec::with_capacity(estimated_pairs);

    let mut node_ids: Vec<(&String, usize)> = node_to_id
        .iter()
        .map(|(name, &id)| (name, id.as_usize()))
        .collect();
    node_ids.sort_by(|a, b| a.0.cmp(b.0));

    // A directed graph emits every ordered pair, so each source scans every target.
    // An undirected pair is emitted once; since node_ids is sorted by name, starting
    // at i + 1 also puts the lexicographically smaller node in `from`.
    for (i, &(start_name, start_id)) in node_ids.iter().enumerate() {
        let dist = shortest_paths_from(start_id, &adj_list);
        let first_target = if kwargs.directed { 0 } else { i + 1 };

        for (j, &(target_name, target_id)) in node_ids.iter().enumerate().skip(first_target) {
            if i == j || dist[target_id] == i64::MAX {
                continue;
            }

            from_nodes.push(start_name.clone());
            to_nodes.push(target_name.clone());
            distances.push(dist[target_id] as f64 / 1000.0);
        }
    }

    let fields = [
        Series::new(PlSmallStr::from("from"), from_nodes),
        Series::new(PlSmallStr::from("to"), to_nodes),
        Series::new(PlSmallStr::from("distance"), distances),
    ];

    let length = fields.first().map(|s| s.len()).unwrap_or(0);
    StructChunked::from_series(PlSmallStr::from("shortest_paths"), length, fields.iter())
        .map(|ca| ca.into_series())
}
