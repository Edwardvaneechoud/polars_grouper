#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::collections::{HashMap, HashSet, VecDeque};

type Graph = HashMap<String, HashSet<String>>;

fn bfs(start_node: &str, graph: &Graph, visited: &mut HashSet<String>, component: &mut Vec<String>) {
    let mut queue = VecDeque::new();
    queue.push_back(start_node.to_string());
    visited.insert(start_node.to_string());

    while let Some(node) = queue.pop_front() {
        component.push(node.clone());

        if let Some(neighbors) = graph.get(&node) {
            for neighbor in neighbors {
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor.clone());
                }
            }
        }
    }
}

fn find_connected_components(graph: &Graph) -> Vec<Vec<String>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for node in graph.keys() {
        if !visited.contains(node) {
            let mut component = Vec::new();
            bfs(node, graph, &mut visited, &mut component);
            components.push(component);
        }
    }

    components
}

#[polars_expr(output_type=String)]
fn graph_solver(inputs: &[Series]) -> PolarsResult<Series> {
    let from: &StringChunked = inputs[0].str()?;
    let to: &StringChunked = inputs[1].str()?;

    let mut graph: Graph = HashMap::with_capacity(from.len());

    // Construct the graph with less cloning
    for (f, t) in from.iter().zip(to.iter()) {
        if let (Some(f), Some(t)) = (f, t) {
            graph.entry(f.to_string()).or_default().insert(t.to_string());
            graph.entry(t.to_string()).or_default().insert(f.to_string());
        } else if let Some(f) = f {
            graph.entry(f.to_string()).or_default();
        } else if let Some(t) = t {
            graph.entry(t.to_string()).or_default();
        }
    }

    let components = find_connected_components(&graph);
    let mut node_to_group = HashMap::with_capacity(graph.len());

    for (i, component) in components.iter().enumerate() {
        let group = (i + 1).to_string(); // Group numbers start from 1
        for node in component {
            node_to_group.insert(node.as_str(), group.clone());
        }
    }

    // Map nodes back to their group numbers, avoiding extra allocations
    let mut result = Vec::with_capacity(from.len());
    for (f, t) in from.iter().zip(to.iter()) {
        if let Some(node) = f.or(t) {
            result.push(node_to_group.get(node).cloned());
        } else {
            result.push(None);
        }
    }

    Ok(StringChunked::from_iter_options("group".into(), result.into_iter()).into_series())
}