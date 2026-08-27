use polars::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::convert::TryFrom;

// Type aliases to simplify complex types
type NodeMap<T> = FxHashMap<String, T>;
type EdgeList<T> = SmallVec<[(T, T); 1024]>;
type ProcessResult<T> = PolarsResult<(NodeMap<T>, T, EdgeList<T>)>;

// Rest of the traits and implementations remain the same...
pub trait AsUsize {
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

pub fn usize_to_t<T>(value: usize) -> T
where
    T: TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    T::try_from(value).expect("Invalid conversion from usize")
}

pub fn to_string_chunked(series: &Series) -> PolarsResult<StringChunked> {
    if series.dtype() == &DataType::String {
        Ok(series.str()?.clone())
    } else {
        Ok(series.cast(&DataType::String)?.str()?.clone())
    }
}

pub fn to_int64_chunked(series: &Series) -> PolarsResult<Int64Chunked> {
    if series.dtype() == &DataType::Int64 {
        Ok(series.i64()?.clone())
    } else {
        Ok(series.cast(&DataType::Int64)?.i64()?.clone())
    }
}

pub fn to_float64_chunked(series: &Series) -> PolarsResult<Float64Chunked> {
    if series.dtype() == &DataType::Float64 {
        Ok(series.f64()?.clone())
    } else {
        Ok(series.cast(&DataType::Float64)?.f64()?.clone())
    }
}

fn get_or_insert_id<T>(node: &str, node_to_id: &mut NodeMap<T>, id_counter: &mut T) -> T
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    *node_to_id.entry(node.to_string()).or_insert_with(|| {
        let id = *id_counter;
        *id_counter = usize_to_t(id_counter.as_usize() + 1);
        id
    })
}

pub fn process_edges<T>(from: &StringChunked, to: &StringChunked) -> ProcessResult<T>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut node_to_id: NodeMap<T> = FxHashMap::default();
    let mut id_counter: T = usize_to_t(0);
    let mut edges = EdgeList::with_capacity(from.len());

    // Process the edges
    from.iter()
        .zip(to.iter())
        .try_for_each(|(from_node, to_node)| -> PolarsResult<()> {
            if let (Some(f), Some(t)) = (from_node, to_node) {
                let f_id = get_or_insert_id(f, &mut node_to_id, &mut id_counter);
                let t_id = get_or_insert_id(t, &mut node_to_id, &mut id_counter);
                edges.push((f_id, t_id));
            }
            Ok(())
        })?;

    Ok((node_to_id, id_counter, edges))
}

/// Align an input column with the number of edges in the graph.
///
/// A length-1 column is treated as a scalar and repeated, so `pl.lit(1.0)` means
/// "every edge weighs the same". Any other length mismatch is an error: zipping
/// columns of unequal length silently truncates to the shortest one and builds a
/// graph that is missing most of its edges.
pub fn broadcast_to_len(series: &Series, len: usize, role: &str) -> PolarsResult<Series> {
    if series.len() == len {
        Ok(series.clone())
    } else if series.len() == 1 {
        Ok(series.new_from_index(0, len))
    } else {
        polars_bail!(
            ShapeMismatch:
            "`{}` has length {}, expected {} (one value per edge) or 1 (a scalar broadcast to every edge)",
            role, series.len(), len
        )
    }
}

/// The number of edges implied by the node columns of a graph.
///
/// Columns of length 1 are scalars and do not constrain the edge count; if every
/// column is a scalar the graph has a single edge.
pub fn edge_count(inputs: &[Series]) -> usize {
    inputs
        .iter()
        .map(|s| s.len())
        .filter(|&l| l != 1)
        .max()
        .unwrap_or(1)
}
