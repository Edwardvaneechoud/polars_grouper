use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use ahash::AHashMap;

#[polars_expr(output_type = UInt32)]
fn graph_solver(inputs: &[Series]) -> PolarsResult<Series> {
    let from = inputs[0].str()?;
    let to = inputs[1].str()?;

    // Map strings to integer IDs
    let mut node_to_id = AHashMap::with_capacity(from.len() + to.len());
    let mut id_counter = 0usize;

    from.into_iter().chain(to.into_iter()).for_each(|val| {
        if let Some(node) = val {
            node_to_id.entry(node).or_insert_with(|| {
                let id = id_counter;
                id_counter += 1;
                id
            });
        }
    });

    let num_nodes = id_counter;
    let mut adj_list = vec![Vec::new(); num_nodes];

    // Build the adjacency list
    inputs[0]
        .str()?
        .into_iter()
        .zip(inputs[1].str()?.into_iter())
        .for_each(|(f_opt, t_opt)| {
            if let (Some(f), Some(t)) = (f_opt, t_opt) {
                let f_id = node_to_id[&f];
                let t_id = node_to_id[&t];
                adj_list[f_id].push(t_id);
                adj_list[t_id].push(f_id);
            }
        });

    // Initialize visited vector and group IDs
    let mut visited = vec![false; num_nodes];
    let mut group_ids = vec![0u32; num_nodes];
    let mut group_counter = 1u32;

    // Perform BFS to find connected components
    for node_id in 0..num_nodes {
        if !visited[node_id] {
            let mut queue = Vec::new(); // Using Vec instead of VecDeque
            queue.push(node_id);
            visited[node_id] = true;
            group_ids[node_id] = group_counter;

            while let Some(current) = queue.pop() {
                for &neighbor in &adj_list[current] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        group_ids[neighbor] = group_counter;
                        queue.push(neighbor);
                    }
                }
            }

            group_counter += 1;
        }
    }

    // Map back to the original data
    let mut result_values = Vec::with_capacity(inputs[0].len());

    inputs[0]
        .str()?
        .into_iter()
        .zip(inputs[1].str()?.into_iter())
        .for_each(|(f_opt, t_opt)| {
            let node_opt = f_opt.or(t_opt);
            if let Some(node) = node_opt {
                let node_id = node_to_id[&node];
                let group_id = group_ids[node_id];
                result_values.push(Some(group_id));
            } else {
                result_values.push(None);
            }
        });

    let result = UInt32Chunked::new("group".into(), result_values);

    Ok(result.into_series())
}
