use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rustc_hash::FxHashMap; // FxHashMap for faster hashing
use smallvec::SmallVec;

struct UnionFind {
    nodes: SmallVec<[u64; 1024]>, // Combined representation of (parent, rank) packed in an u64
}

impl UnionFind {
    fn new(size: usize) -> Self {
        UnionFind {
            nodes: (0..size).map(|i| i as u64).collect(), // Initialize parent as self, rank as 0
        }
    }

    #[inline(always)]
    fn extract_parent(val: u64) -> usize {
        (val & 0xFFFFFFFF) as usize
    }

    #[inline(always)]
    fn extract_rank(val: u64) -> usize {
        (val >> 32) as usize
    }

    #[inline(always)]
    fn set_rank_parent(&mut self, idx: usize, parent: usize, rank: usize) {
        self.nodes[idx] = ((rank as u64) << 32) | (parent as u64);
    }

    #[inline(always)]
    fn find(&mut self, mut x: usize) -> usize {
        let mut root = x;
        while root != Self::extract_parent(self.nodes[root]) {
            root = Self::extract_parent(self.nodes[root]);
        }
        // Two-step compression for efficiency
        while x != root {
            let parent = Self::extract_parent(self.nodes[x]);
            self.set_rank_parent(x, root, Self::extract_rank(self.nodes[x]));
            x = parent;
        }
        root
    }

    #[inline(always)]
    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            let rank_x = Self::extract_rank(self.nodes[root_x]);
            let rank_y = Self::extract_rank(self.nodes[root_y]);
            if rank_x > rank_y {
                self.set_rank_parent(root_y, root_x, rank_y);
            } else if rank_x < rank_y {
                self.set_rank_parent(root_x, root_y, rank_x);
            } else {
                self.set_rank_parent(root_y, root_x, rank_y);
                self.set_rank_parent(root_x, root_x, rank_x + 1);
            }
        }
    }
}

#[polars_expr(output_type = UInt32)]
fn graph_solver(inputs: &[Series]) -> PolarsResult<Series> {
    let from = inputs[0].str()?;
    let to = inputs[1].str()?;

    // Step 1: Map nodes to unique IDs during edge processing to avoid multiple passes
    let mut node_to_id: FxHashMap<&str, usize> = FxHashMap::default();
    let mut id_counter: usize = 0;
    let mut edges = SmallVec::<[(usize, usize); 1024]>::new();

    from.into_iter()
        .zip(to.into_iter())
        .for_each(|(from_node, to_node)| {
            if let (Some(f), Some(t)) = (from_node, to_node) {
                let f_id = *node_to_id.entry(f).or_insert_with(|| {
                    let id = id_counter;
                    id_counter += 1;
                    id
                });
                let t_id = *node_to_id.entry(t).or_insert_with(|| {
                    let id = id_counter;
                    id_counter += 1;
                    id
                });
                edges.push((f_id, t_id));
            }
        });

    let num_nodes = id_counter;
    let mut uf = UnionFind::new(num_nodes);

    // Step 2: Perform union operations on edges
    edges.iter().for_each(|&(f_id, t_id)| {
        uf.union(f_id, t_id);
    });

    // Step 3: Map each node to its connected component group
    let mut root_to_group = vec![0; num_nodes]; // Using a Vec instead of FxHashMap for faster lookup
    let mut group_counter: u32 = 1;

    let mut group_ids = Vec::with_capacity(num_nodes);
    for id in 0..num_nodes {
        let root = uf.find(id);
        if root_to_group[root] == 0 {
            root_to_group[root] = group_counter;
            group_counter += 1;
        }
        group_ids.push(root_to_group[root]);
    }

    // Step 4: Create the group column without using Option<>
    let groups: Vec<u32> = from
        .into_iter()
        .map(|from_node| {
            let node = from_node.unwrap(); // Assuming we know these are all `Some`, as we filtered `None` earlier
            let node_id = *node_to_id.get(node).unwrap();
            group_ids[node_id]
        })
        .collect();

    let result = UInt32Chunked::from_slice("group".into(), &groups);

    Ok(result.into_series())
}
