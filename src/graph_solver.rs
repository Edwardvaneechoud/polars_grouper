use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::convert::TryFrom;

use crate::graph_utils::{process_edges, to_string_chunked, usize_to_t, AsUsize};

struct UnionFind<T>
where
    T: Copy + PartialEq + AsUsize,
{
    nodes: Vec<T>,
}

impl<T> UnionFind<T>
where
    T: Copy + PartialEq + AsUsize + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    fn new(size: usize) -> Self {
        UnionFind {
            nodes: (0..size).map(|i| usize_to_t(i)).collect(),
        }
    }

    #[inline(always)]
    fn find(&mut self, mut x: T) -> T {
        while x != self.nodes[x.as_usize()] {
            let parent = self.nodes[x.as_usize()];
            self.nodes[x.as_usize()] = self.nodes[parent.as_usize()];
            x = parent;
        }
        x
    }

    #[inline(always)]
    fn union(&mut self, x: T, y: T) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x != root_y {
            self.nodes[root_y.as_usize()] = root_x;
        }
    }
}

#[polars_expr(output_type = UInt64)]
fn graph_solver(inputs: &[Series]) -> PolarsResult<Series> {
    let from = to_string_chunked(&inputs[0])?;
    let to = to_string_chunked(&inputs[1])?;

    let len = from.len();

    if len <= u16::MAX as usize {
        process_graph::<u16>(&from, &to)
    } else if len <= u32::MAX as usize {
        process_graph::<u32>(&from, &to)
    } else {
        process_graph::<u64>(&from, &to)
    }
}

fn process_graph<T>(from: &StringChunked, to: &StringChunked) -> PolarsResult<Series>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize + Into<u64>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let (node_to_id, id_counter, edges) = process_edges::<T>(from, to)?;

    // Initialize the UnionFind structure with the number of nodes
    let num_nodes = id_counter.as_usize();
    let mut uf = UnionFind::new(num_nodes);

    // Process the edges with union-find
    edges.iter().for_each(|&(f_id, t_id)| {
        uf.union(f_id, t_id);
    });

    // Initialize group IDs and counters
    let mut group_ids = vec![usize_to_t(0); num_nodes];
    let mut group_counter: T = usize_to_t(1);

    // Assign group IDs
    for id in (0..num_nodes).map(|i| usize_to_t(i)) {
        let root = uf.find(id);
        if group_ids[root.as_usize()] == usize_to_t(0) {
            group_ids[root.as_usize()] = group_counter;
            group_counter = usize_to_t(group_counter.as_usize() + 1);
        }
        group_ids[id.as_usize()] = group_ids[root.as_usize()];
    }

    // Map the group IDs to the original nodes
    let groups: Vec<u64> = from
        .iter()
        .map(|from_node| {
            from_node
                .and_then(|node| node_to_id.get(node))
                .map(|&id| group_ids[id.as_usize()].into())
                .unwrap_or(0)
        })
        .collect();

    Ok(UInt64Chunked::from_vec("group".into(), groups).into_series())
}
