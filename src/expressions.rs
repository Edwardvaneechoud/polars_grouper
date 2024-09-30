use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use ahash::AHashMap;
use rayon::prelude::*;

// Union-Find structure
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![1; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            // Union by rank
            if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }
}

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
    let mut uf = UnionFind::new(num_nodes);

    // Collect edges into batches
    let edges: Vec<_> = inputs[0]
        .str()?
        .into_iter()
        .zip(inputs[1].str()?.into_iter())
        .filter_map(|(f_opt, t_opt)| {
            if let (Some(f), Some(t)) = (f_opt, t_opt) {
                Some((f, t))
            } else {
                None
            }
        })
        .collect();

    let chunk_size = (edges.len() / num_nodes.max(1)).max(1);
    let edge_chunks: Vec<_> = edges.chunks(chunk_size).collect();

    // Perform union operations for each edge chunk in parallel
    let mut union_finds: Vec<UnionFind> = edge_chunks
        .into_par_iter()
        .map(|chunk| {
            let mut local_uf = UnionFind::new(num_nodes);
            chunk.iter().for_each(|&(f, t)| {
                let f_id = node_to_id[&f];
                let t_id = node_to_id[&t];
                local_uf.union(f_id, t_id);
            });
            local_uf
        })
        .collect();

    // Merge all local UnionFind structures into a single UnionFind structure
    for local_uf in union_finds.iter_mut() {
        for i in 0..num_nodes {
            uf.union(i, local_uf.find(i));
        }
    }

    // Find the connected component for each node
    let mut group_ids = vec![0u32; num_nodes];
    for (node, &id) in node_to_id.iter() {
        let root_id = uf.find(id);
        group_ids[id] = root_id as u32;
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
