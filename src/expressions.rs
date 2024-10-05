use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rustc_hash::{FxHashMap, FxBuildHasher};
use smallvec::SmallVec;
use std::convert::TryFrom;

trait AsUsize {
    fn as_usize(&self) -> usize;
}

impl AsUsize for u16 {
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl AsUsize for u32 {
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl AsUsize for u64 {
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

fn usize_to_t<T>(value: usize) -> T
where
    T: TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    T::try_from(value).expect("Invalid conversion from usize")
}

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
            nodes: (0..size)
                .map(|i| usize_to_t(i))
                .collect(),
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
    let from = if inputs[0].dtype() == &DataType::String {
        inputs[0].str()?.clone()
    } else {
        inputs[0].cast(&DataType::String)?.str()?.clone()
    };

    let to = if inputs[1].dtype() == &DataType::String {
        inputs[1].str()?.clone()
    } else {
        inputs[1].cast(&DataType::String)?.str()?.clone()
    };

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
    let mut node_to_id: FxHashMap<&str, T> =
        FxHashMap::with_capacity_and_hasher(from.len(), FxBuildHasher);
    let mut id_counter: T = usize_to_t(0);
    let mut edges = SmallVec::<[(T, T); 1024]>::with_capacity(from.len());

    from.iter().zip(to.iter()).try_for_each(|(from_node, to_node)| -> PolarsResult<()> {
        if let (Some(f), Some(t)) = (from_node, to_node) {
            let f_id = *node_to_id.entry(f).or_insert_with(|| {
                let id = id_counter;
                id_counter = usize_to_t(id_counter.as_usize() + 1);
                id
            });
            let t_id = *node_to_id.entry(t).or_insert_with(|| {
                let id = id_counter;
                id_counter = usize_to_t(id_counter.as_usize() + 1);
                id
            });
            edges.push((f_id, t_id));
        }
        Ok(())
    })?;

    let num_nodes = id_counter.as_usize();
    let mut uf = UnionFind::new(num_nodes);

    edges.iter().for_each(|&(f_id, t_id)| {
        uf.union(f_id, t_id);
    });

    let mut group_ids = vec![usize_to_t(0); num_nodes];
    let mut group_counter: T = usize_to_t(1); // Explicitly specify type T

    for id in (0..num_nodes).map(|i| usize_to_t(i)) {
        let root = uf.find(id);
        if group_ids[root.as_usize()] == usize_to_t(0) {
            group_ids[root.as_usize()] = group_counter;
            group_counter = usize_to_t(group_counter.as_usize() + 1);
        }
        group_ids[id.as_usize()] = group_ids[root.as_usize()];
    }

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
