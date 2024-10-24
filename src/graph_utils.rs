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

    // Closure to insert nodes into node_to_id and update id_counter
    let mut get_or_insert_id = |node: &str| -> T {
        *node_to_id.entry(node.to_string()).or_insert_with(|| {
            let id = id_counter;
            id_counter = usize_to_t(id_counter.as_usize() + 1);
            id
        })
    };

    // Process the edges
    from.iter().zip(to.iter()).try_for_each(|(from_node, to_node)| -> PolarsResult<()> {
        if let (Some(f), Some(t)) = (from_node, to_node) {
            let f_id = get_or_insert_id(f);
            let t_id = get_or_insert_id(t);
            edges.push((f_id, t_id));
        }
        Ok(())
    })?;

    Ok((node_to_id, id_counter, edges))
}
