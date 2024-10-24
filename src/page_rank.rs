use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::convert::TryFrom;
use serde::Deserialize;

use crate::graph_utils::{process_edges, to_string_chunked, AsUsize};


#[derive(Deserialize)]
struct PageRankKwargs {
    damping_factor: f64,
    max_iterations: u16,
    convergence_threshold: f64,
}


#[polars_expr(output_type = Float64)]
fn page_rank(inputs: &[Series], kwargs: PageRankKwargs) -> PolarsResult<Series> {
    let from = to_string_chunked(&inputs[0])?;
    let to = to_string_chunked(&inputs[1])?;
    let damping_factor = kwargs.damping_factor;
    let max_iterations = kwargs.max_iterations;
    let convergence_threshold = kwargs.convergence_threshold;

    let len = from.len();

    if len <= u16::MAX as usize {
        calculate_pagerank::<u16>(&from, &to, damping_factor, max_iterations, convergence_threshold)
    } else if len <= u32::MAX as usize {
        calculate_pagerank::<u32>(&from, &to, damping_factor, max_iterations, convergence_threshold)
    } else {
        calculate_pagerank::<u64>(&from, &to, damping_factor, max_iterations, convergence_threshold)
    }
}

fn calculate_pagerank<T>(
    from: &StringChunked,
    to: &StringChunked,
    damping_factor: f64,
    max_iterations: u16,
    convergence_threshold: f64,
) -> PolarsResult<Series>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize + Into<u64>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    // Use process_edges from graph_utils
    let (node_to_id, id_counter, edges) = process_edges::<T>(from, to)?;
    let num_nodes = id_counter.as_usize();

    // Create adjacency list representation
    let mut outgoing_edges: Vec<Vec<T>> = vec![Vec::new(); num_nodes];
    let mut incoming_edges: Vec<Vec<T>> = vec![Vec::new(); num_nodes];

    // Process edges using the SmallVec from process_edges
    for &(from_id, to_id) in edges.iter() {
        outgoing_edges[from_id.as_usize()].push(to_id);
        incoming_edges[to_id.as_usize()].push(from_id);
    }

    // Initialize PageRank scores
    let mut page_ranks: Vec<f64> = vec![1.0 / num_nodes as f64; num_nodes];
    let mut new_ranks: Vec<f64> = vec![0.0; num_nodes];

    // PageRank iteration
    for _ in 0..max_iterations {
        let mut total_diff = 0.0;
        new_ranks.fill(0.0);

        // Calculate new rank for each node
        for node in 0..num_nodes {
            let base_rank = (1.0 - damping_factor) / num_nodes as f64;

            let incoming_rank: f64 = incoming_edges[node].iter()
                .map(|&from| {
                    let out_count = outgoing_edges[from.as_usize()].len() as f64;
                    if out_count > 0.0 {
                        page_ranks[from.as_usize()] / out_count
                    } else {
                        0.0
                    }
                })
                .sum();

            new_ranks[node] = base_rank + damping_factor * incoming_rank;
        }

        // Calculate total difference for convergence check
        for i in 0..num_nodes {
            total_diff += (new_ranks[i] - page_ranks[i]).abs();
        }

        page_ranks.copy_from_slice(&new_ranks);

        if total_diff < convergence_threshold {
            break;
        }
    }

    // Map the PageRank scores back to the original nodes using node_to_id from process_edges
    let scores: Vec<f64> = from
        .iter()
        .map(|from_node| {
            from_node
                .and_then(|node| node_to_id.get(node))
                .map(|&id| page_ranks[id.as_usize()])
                .unwrap_or(0.0)
        })
        .collect();

    Ok(Float64Chunked::from_vec("pagerank".into(), scores).into_series())
}