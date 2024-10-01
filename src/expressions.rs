use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use hashbrown::HashMap;
use petgraph::unionfind::UnionFind;
use smallvec::SmallVec;

#[polars_expr(output_type = UInt32)]
fn graph_solver(inputs: &[Series]) -> PolarsResult<Series> {
    let from = inputs[0].str()?;
    let to = inputs[1].str()?;

    // Map strings to integer IDs using hashbrown for optimized hashing
    let mut node_to_id: HashMap<&str, u32> = HashMap::with_capacity(from.len() + to.len());
    let mut id_counter: u32 = 0;

    // Assign unique IDs to nodes in a single iteration
    from.into_iter().chain(to.into_iter()).for_each(|val| {
        if let Some(node) = val {
            node_to_id.entry(node).or_insert_with(|| {
                let id = id_counter;
                id_counter += 1;
                id
            });
        }
    });

    let num_nodes = id_counter as usize;
    let mut uf = UnionFind::new(num_nodes);

    // Use a SmallVec to store edges to minimize heap allocations
    let mut edges: SmallVec<[(u32, u32); 1024]> = SmallVec::new();

    // Collect edges from input and map nodes to their IDs in a single pass
    from.into_iter()
        .zip(to.into_iter())
        .for_each(|(f_opt, t_opt)| {
            if let (Some(f), Some(t)) = (f_opt, t_opt) {
                let f_id = *node_to_id.get(f).unwrap();
                let t_id = *node_to_id.get(t).unwrap();
                edges.push((f_id, t_id));
                // Perform union operation directly during edge collection to avoid extra iteration
                uf.union(f_id as usize, t_id as usize);
            }
        });

    // Find the connected component for each node
    let group_ids: Vec<u32> = (0..num_nodes)
        .map(|id| uf.find(id) as u32)
        .collect();

    // Generate group values for the original series
    let result_values: Vec<Option<u32>> = from
        .into_iter()
        .zip(to.into_iter())
        .map(|(f_opt, t_opt)| {
            f_opt.or(t_opt).map(|node| {
                let node_id = *node_to_id.get(node).unwrap();
                group_ids[node_id as usize]
            })
        })
        .collect();

    let result = UInt32Chunked::new("group".into(), result_values);

    Ok(result.into_series())
}
