use polars::prelude::*;
use rustc_hash::FxHashMap;
use std::convert::TryFrom;
use smallvec::SmallVec;

// Implement AsUsize trait to convert to usize
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

fn get_or_insert_id<T>(
    node: &str,
    node_to_id: &mut FxHashMap<String, T>,
    id_counter: &mut T,
) -> T
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
pub fn process_edges<T>(
    from: &StringChunked,
    to: &StringChunked,
) -> PolarsResult<(FxHashMap<String, T>, T, SmallVec<[(T, T); 1024]>)>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut node_to_id: FxHashMap<String, T> = FxHashMap::default();
    let mut id_counter: T = usize_to_t(0);
    let mut edges = SmallVec::<[(T, T); 1024]>::with_capacity(from.len());

    // Process the edges
    from.iter().zip(to.iter()).try_for_each(|(from_node, to_node)| -> PolarsResult<()> {
        if let (Some(f), Some(t)) = (from_node, to_node) {
            let f_id = get_or_insert_id(f, &mut node_to_id, &mut id_counter);
            let t_id = get_or_insert_id(t, &mut node_to_id, &mut id_counter);
            edges.push((f_id, t_id));
        }
        Ok(())
    })?;

    Ok((node_to_id, id_counter, edges))
}
