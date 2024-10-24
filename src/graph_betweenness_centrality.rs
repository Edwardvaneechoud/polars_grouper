use polars::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::convert::TryFrom;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use crate::graph_utils::{AsUsize, usize_to_t, to_string_chunked};

#[derive(Deserialize)]
struct BetweennessCentralityKwargs {
    normalized: bool,
    directed: bool,
}

fn process_edges<T>(
    from: &StringChunked,
    to: &StringChunked,
) -> PolarsResult<(HashMap<String, T>, T, Vec<(T, T)>)>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{

    let estimated_size = from.len();
    let mut node_to_id = HashMap::with_capacity(estimated_size);
    let mut edges = Vec::with_capacity(estimated_size);
    let mut id_counter: T = usize_to_t(0);

    let mut get_or_insert_id = |node: &str| -> T {
        *node_to_id.entry(node.to_string()).or_insert_with(|| {
            let id = id_counter;
            id_counter = usize_to_t(id_counter.as_usize() + 1);
            id
        })
    };

    from.iter()
        .zip(to.iter())
        .try_for_each(|(from_node, to_node)| -> PolarsResult<()> {
            if let (Some(f), Some(t)) = (from_node, to_node) {
                let f_id = get_or_insert_id(f);
                let t_id = get_or_insert_id(t);
                edges.push((f_id, t_id));
            }
            Ok(())
        })?;

    Ok((node_to_id, id_counter, edges))
}
fn calculate_betweenness<T>(
    adj_list: &[Vec<usize>],
    num_nodes: usize,
    normalized: bool,
    directed: bool,
) -> Vec<f64>
where
    T: TryFrom<usize> + Copy + PartialEq + AsUsize + Into<u64>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut centrality = vec![0.0; num_nodes];

    // For each node as source
    for source in 0..num_nodes {
        let mut stack = Vec::new();
        let mut paths = vec![0; num_nodes];
        let mut distances = vec![-1i32; num_nodes];
        let mut queue = VecDeque::new();
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];

        // BFS initialization
        paths[source] = 1;
        distances[source] = 0;
        queue.push_back(source);

        // BFS to find shortest paths
        while let Some(v) = queue.pop_front() {
            stack.push(v);

            for &w in &adj_list[v] {
                // Path discovery
                if distances[w] < 0 {
                    queue.push_back(w);
                    distances[w] = distances[v] + 1;
                }

                // Path counting
                if distances[w] == distances[v] + 1 {
                    paths[w] += paths[v];
                    predecessors[w].push(v);
                }
            }
        }

        // Accumulation
        let mut delta = vec![0.0; num_nodes];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                let coeff = (paths[v] as f64 / paths[w] as f64) * (1.0 + delta[w]);
                delta[v] += coeff;
            }
            if w != source {
                centrality[w] += delta[w];
            }
        }
    }

    // For undirected graphs, divide by 2 since each path is counted twice
    if !directed {
        for c in &mut centrality {
            *c /= 2.0;
        }
    }

    // Normalize if requested
    if normalized {
        let n = num_nodes as f64;
        let norm = if directed {
            1.0 / ((n - 1.0) * (n - 2.0))
        } else {
            2.0 / ((n - 1.0) * (n - 2.0))
        };

        for c in &mut centrality {
            *c *= norm;
        }
    }

    centrality
}


fn betweenness_centrality_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "betweenness_centrality".into(),
        DataType::Struct(vec![
            Field::new("node".into(), DataType::String),
            Field::new("centrality".into(), DataType::Float64),
        ])
    ))
}

#[polars_expr(output_type_func=betweenness_centrality_output)]
fn graph_betweenness_centrality(inputs: &[Series], kwargs: BetweennessCentralityKwargs) -> PolarsResult<Series> {
    let from = to_string_chunked(&inputs[0])?;
    let to = to_string_chunked(&inputs[1])?;
    type NodeId = u32;

    let (node_to_id, id_counter, edges) = process_edges::<NodeId>(&from, &to)?;
    let num_nodes = id_counter.as_usize();

    // Create adjacency list
    let mut adj_list = vec![Vec::new(); num_nodes];
    for (from_id, to_id) in edges {
        adj_list[from_id.as_usize()].push(to_id.as_usize());
        if !kwargs.directed {
            adj_list[to_id.as_usize()].push(from_id.as_usize());
        }
    }

    // Calculate centrality
    let centrality = calculate_betweenness::<NodeId>(
        &adj_list,
        num_nodes,
        kwargs.normalized,
        kwargs.directed
    );

    // Create result series
    let mut nodes = Vec::with_capacity(num_nodes);
    let mut centrality_values = Vec::with_capacity(num_nodes);

    // Create reverse mapping for node IDs to names
    let id_to_node: HashMap<_, _> = node_to_id
        .iter()
        .map(|(k, &v)| (v.as_usize(), k.clone()))
        .collect();

    // Build result vectors
    for i in 0..num_nodes {
        if let Some(node_name) = id_to_node.get(&i) {
            nodes.push(node_name.clone());
            centrality_values.push(centrality[i]);
        }
    }

    let fields = vec![
        Series::new("node".into(), nodes),
        Series::new("centrality".into(), centrality_values),
    ];

    StructChunked::from_series("betweenness_centrality".into(), &fields)
        .map(|ca| ca.into_series())
}