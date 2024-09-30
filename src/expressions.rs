use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use ahash::AHashMap;

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    #[inline]
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![1; size],
        }
    }

    #[inline]
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    #[inline]
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

    // Map strings to integer IDs with minimal overhead
    let mut node_to_id = AHashMap::with_capacity(from.len() + to.len());
    let mut id_counter = 0usize;

    // Assign unique IDs to nodes
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

    // Perform union operations for each edge with fewer lookups
    inputs[0]
        .str()?
        .into_iter()
        .zip(inputs[1].str()?.into_iter())
        .for_each(|(f_opt, t_opt)| {
            if let (Some(f), Some(t)) = (f_opt, t_opt) {
                let f_id = *node_to_id.get(f).unwrap();
                let t_id = *node_to_id.get(t).unwrap();
                uf.union(f_id, t_id);
            }
        });

    // Find the connected component for each node (direct assignment)
    let group_ids: Vec<u32> = (0..num_nodes)
        .map(|id| uf.find(id) as u32)
        .collect();

    // Generate group values for the original series
    let result_values: Vec<Option<u32>> = from
        .into_iter()
        .zip(to.into_iter())
        .map(|(f_opt, t_opt)| {
            let node_opt = f_opt.or(t_opt);
            node_opt.map(|node| {
                let node_id = *node_to_id.get(node).unwrap();
                group_ids[node_id]
            })
        })
        .collect();

    let result = UInt32Chunked::new("group".into(), result_values);

    Ok(result.into_series())
}
