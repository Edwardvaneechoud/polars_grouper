//! Hierarchy (bill of materials) explosion: the transitive closure of a parent -> child
//! edge list, with quantities multiplied along each path.
//!
//! Three expressions share the graph and the traversals here and differ only in how much
//! path detail they keep: `hierarchy_totals` (one row per ancestor/descendant pair),
//! `hierarchy_levels` (one row per pair and level) and `hierarchy_paths` (one row per path).

use crate::graph_utils::{
    broadcast_to_len, edge_count, process_weighted_edges, to_float64_chunked, to_string_chunked,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

type Children = Vec<Vec<(u32, f64)>>;

/// Longest cycle printed in full in the error raised for a cyclic hierarchy.
const MAX_CYCLE_NODES_SHOWN: usize = 10;

#[derive(Deserialize)]
struct HierarchyKwargs {
    top_level_only: bool,
    include_self: bool,
    max_depth: Option<u32>,
}

/// An acyclic parent -> child graph, ready to explode.
struct Hierarchy {
    /// `(child, quantity)` per parent, in input order; duplicate edges are kept.
    children: Children,
    /// Node labels indexed by node id, already in the output node dtype.
    names: Series,
    /// Nodes to explode from, in order of first appearance in the input.
    roots: Vec<u32>,
    include_self: bool,
    max_depth: u32,
}

impl Hierarchy {
    fn build(inputs: &[Series], kwargs: &HierarchyKwargs) -> PolarsResult<Self> {
        // `parent`/`child` define the edges; a scalar quantity is then stretched to match them.
        let num_edges = edge_count(&inputs[..2]);
        let parent = to_string_chunked(&broadcast_to_len(&inputs[0], num_edges, "parent")?)?;
        let child = to_string_chunked(&broadcast_to_len(&inputs[1], num_edges, "child")?)?;
        let quantity = to_float64_chunked(&broadcast_to_len(&inputs[2], num_edges, "quantity")?)?;

        let (node_to_id, num_nodes, edges) =
            process_weighted_edges::<u32>(&parent, &child, &quantity, "quantity")?;
        let num_nodes = num_nodes as usize;

        let mut labels = vec![""; num_nodes];
        for (label, &id) in &node_to_id {
            labels[id as usize] = label.as_str();
        }

        let mut children: Children = vec![Vec::new(); num_nodes];
        let mut in_degree = vec![0u32; num_nodes];
        for (p, c, q) in edges {
            children[p as usize].push((c, q));
            in_degree[c as usize] += 1;
        }

        check_acyclic(&children, &in_degree, &labels)?;

        let roots = (0..num_nodes)
            .filter(|&v| {
                (!kwargs.top_level_only || in_degree[v] == 0)
                    && (kwargs.include_self || !children[v].is_empty())
            })
            .map(|v| v as u32)
            .collect();

        let names = Series::new(PlSmallStr::EMPTY, labels)
            .strict_cast(&node_dtype(inputs[0].dtype(), inputs[1].dtype()))?;

        Ok(Hierarchy {
            children,
            names,
            roots,
            include_self: kwargs.include_self,
            max_depth: kwargs.max_depth.unwrap_or(u32::MAX),
        })
    }

    fn num_nodes(&self) -> usize {
        self.children.len()
    }

    fn is_leaf(&self, node: IdxSize) -> bool {
        self.children[node as usize].is_empty()
    }

    fn node_column(&self, name: &str, ids: &[IdxSize]) -> PolarsResult<Series> {
        Ok(self.names.take_slice(ids)?.with_name(name.into()))
    }

    /// Visit every node below `root`, one level at a time, as `visit(level, node, quantity)`.
    ///
    /// All paths reaching a node at the same level are merged into a single visit with their
    /// quantities summed. The work per level is therefore bounded by the edges leaving the
    /// frontier rather than by the number of paths, which is what keeps a shared
    /// subassembly from being re-exploded once per path the way a recursive CTE would.
    fn walk_levels(
        &self,
        root: u32,
        scratch: &mut LevelScratch,
        mut visit: impl FnMut(u32, u32, f64),
    ) {
        scratch.frontier.clear();
        scratch.frontier.push((root, 1.0));
        let mut level = 0;

        while !scratch.frontier.is_empty() && level < self.max_depth {
            level += 1;
            for &(node, quantity) in &scratch.frontier {
                for &(child, per) in &self.children[node as usize] {
                    let c = child as usize;
                    if !scratch.in_next[c] {
                        scratch.in_next[c] = true;
                        scratch.next_quantity[c] = 0.0;
                        scratch.next.push(child);
                    }
                    scratch.next_quantity[c] += quantity * per;
                }
            }

            scratch.frontier.clear();
            for &child in &scratch.next {
                let c = child as usize;
                scratch.in_next[c] = false;
                visit(level, child, scratch.next_quantity[c]);
                scratch.frontier.push((child, scratch.next_quantity[c]));
            }
            scratch.next.clear();
        }
    }
}

/// Buffers for `Hierarchy::walk_levels`, reused across roots so a walk allocates nothing.
struct LevelScratch {
    frontier: Vec<(u32, f64)>,
    next: Vec<u32>,
    next_quantity: Vec<f64>,
    in_next: Vec<bool>,
}

impl LevelScratch {
    fn new(num_nodes: usize) -> Self {
        LevelScratch {
            frontier: Vec::new(),
            next: Vec::new(),
            next_quantity: vec![0.0; num_nodes],
            in_next: vec![false; num_nodes],
        }
    }
}

/// Output dtype of the node columns.
///
/// Nodes are interned as strings, which round-trips losslessly for integer and string ids,
/// so those keep their input dtype when parent and child agree; anything else is String.
fn node_dtype(parent: &DataType, child: &DataType) -> DataType {
    if parent == child && (parent.is_integer() || parent.is_string()) {
        parent.clone()
    } else {
        DataType::String
    }
}

/// Fail with one offending cycle if the graph is not a DAG.
///
/// Uses Kahn's algorithm. A cycle means infinitely many paths and therefore infinite
/// quantities; in a bill of materials it is a data error (an item listed as its own
/// component somewhere below itself), so it is reported rather than worked around.
fn check_acyclic(children: &Children, in_degree: &[u32], labels: &[&str]) -> PolarsResult<()> {
    let mut unresolved = in_degree.to_vec();
    let mut ready: Vec<u32> = (0..children.len())
        .filter(|&v| unresolved[v] == 0)
        .map(|v| v as u32)
        .collect();
    let mut resolved = 0;

    while let Some(node) = ready.pop() {
        resolved += 1;
        for &(child, _) in &children[node as usize] {
            unresolved[child as usize] -= 1;
            if unresolved[child as usize] == 0 {
                ready.push(child);
            }
        }
    }

    if resolved == children.len() {
        return Ok(());
    }

    let cycle = find_cycle(children, &unresolved);
    let mut description = cycle
        .iter()
        .take(MAX_CYCLE_NODES_SHOWN)
        .map(|&v| labels[v as usize])
        .collect::<Vec<_>>()
        .join(" -> ");
    if cycle.len() > MAX_CYCLE_NODES_SHOWN {
        description.push_str(&format!(" -> ... ({} nodes)", cycle.len()));
    } else {
        description.push_str(&format!(" -> {}", labels[cycle[0] as usize]));
    }
    polars_bail!(
        ComputeError:
        "the hierarchy contains a cycle, so its quantities are unbounded: {}", description
    )
}

/// Extract one cycle, in parent -> child order, from the nodes Kahn's algorithm left unresolved.
///
/// Every unresolved node still has an unresolved parent, so walking up the parents from
/// any of them must revisit a node; the stretch of the walk since that node is a cycle.
fn find_cycle(children: &Children, unresolved: &[u32]) -> Vec<u32> {
    let mut parents = vec![Vec::new(); children.len()];
    for (parent, edges) in children.iter().enumerate() {
        if unresolved[parent] > 0 {
            for &(child, _) in edges {
                parents[child as usize].push(parent as u32);
            }
        }
    }

    let mut position = vec![usize::MAX; children.len()];
    let mut walk = Vec::new();
    let mut node = unresolved
        .iter()
        .position(|&d| d > 0)
        .expect("an unresolved node exists") as u32;
    while position[node as usize] == usize::MAX {
        position[node as usize] = walk.len();
        walk.push(node);
        node = parents[node as usize][0];
    }

    let mut cycle = walk.split_off(position[node as usize]);
    cycle.reverse();
    // The walk ran child -> parent; reversed, it ends on its starting node, which reads best first.
    cycle.rotate_right(1);
    cycle
}

fn pair_fields(input_fields: &[Field]) -> Vec<Field> {
    let node = node_dtype(input_fields[0].dtype(), input_fields[1].dtype());
    vec![
        Field::new("ancestor".into(), node.clone()),
        Field::new("descendant".into(), node),
        Field::new("level".into(), DataType::UInt32),
        Field::new("quantity".into(), DataType::Float64),
        Field::new("is_leaf".into(), DataType::Boolean),
    ]
}

fn totals_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "hierarchy_totals".into(),
        DataType::Struct(pair_fields(input_fields)),
    ))
}

fn levels_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "hierarchy_levels".into(),
        DataType::Struct(pair_fields(input_fields)),
    ))
}

fn paths_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let node = node_dtype(input_fields[0].dtype(), input_fields[1].dtype());
    Ok(Field::new(
        "hierarchy_paths".into(),
        DataType::Struct(vec![
            Field::new("ancestor".into(), node.clone()),
            Field::new("descendant".into(), node.clone()),
            Field::new("level".into(), DataType::UInt32),
            Field::new("parent".into(), node.clone()),
            Field::new("quantity_per".into(), DataType::Float64),
            Field::new("quantity".into(), DataType::Float64),
            Field::new("is_leaf".into(), DataType::Boolean),
            Field::new("path".into(), DataType::List(Box::new(node))),
        ]),
    ))
}

fn into_struct(name: &str, fields: &[Series]) -> PolarsResult<Series> {
    let length = fields.first().map(|s| s.len()).unwrap_or(0);
    StructChunked::from_series(name.into(), length, fields.iter()).map(|ca| ca.into_series())
}

/// Output rows keyed by (ancestor, descendant), shared by the totals and levels expressions.
#[derive(Default)]
struct PairRows {
    ancestor: Vec<IdxSize>,
    descendant: Vec<IdxSize>,
    level: Vec<u32>,
    quantity: Vec<f64>,
}

impl PairRows {
    fn push(&mut self, ancestor: u32, descendant: u32, level: u32, quantity: f64) {
        self.ancestor.push(ancestor as IdxSize);
        self.descendant.push(descendant as IdxSize);
        self.level.push(level);
        self.quantity.push(quantity);
    }

    fn finish(self, hierarchy: &Hierarchy, name: &str) -> PolarsResult<Series> {
        let is_leaf: Vec<bool> = self
            .descendant
            .iter()
            .map(|&d| hierarchy.is_leaf(d))
            .collect();
        into_struct(
            name,
            &[
                hierarchy.node_column("ancestor", &self.ancestor)?,
                hierarchy.node_column("descendant", &self.descendant)?,
                Series::new("level".into(), self.level),
                Series::new("quantity".into(), self.quantity),
                Series::new("is_leaf".into(), is_leaf),
            ],
        )
    }
}

#[polars_expr(output_type_func=totals_output)]
fn hierarchy_totals(inputs: &[Series], kwargs: HierarchyKwargs) -> PolarsResult<Series> {
    let hierarchy = Hierarchy::build(inputs, &kwargs)?;
    let num_nodes = hierarchy.num_nodes();
    let mut scratch = LevelScratch::new(num_nodes);
    let mut rows = PairRows::default();

    let mut total = vec![0.0; num_nodes];
    let mut shallowest = vec![0u32; num_nodes];
    let mut seen = vec![false; num_nodes];
    let mut seen_order = Vec::new();

    for &root in &hierarchy.roots {
        if hierarchy.include_self {
            rows.push(root, root, 0, 1.0);
        }
        // Levels arrive in increasing order, so the first visit of a node is its shallowest.
        hierarchy.walk_levels(root, &mut scratch, |level, node, quantity| {
            let v = node as usize;
            if !seen[v] {
                seen[v] = true;
                total[v] = 0.0;
                shallowest[v] = level;
                seen_order.push(node);
            }
            total[v] += quantity;
        });
        for node in seen_order.drain(..) {
            let v = node as usize;
            seen[v] = false;
            rows.push(root, node, shallowest[v], total[v]);
        }
    }

    rows.finish(&hierarchy, "hierarchy_totals")
}

#[polars_expr(output_type_func=levels_output)]
fn hierarchy_levels(inputs: &[Series], kwargs: HierarchyKwargs) -> PolarsResult<Series> {
    let hierarchy = Hierarchy::build(inputs, &kwargs)?;
    let mut scratch = LevelScratch::new(hierarchy.num_nodes());
    let mut rows = PairRows::default();

    for &root in &hierarchy.roots {
        if hierarchy.include_self {
            rows.push(root, root, 0, 1.0);
        }
        hierarchy.walk_levels(root, &mut scratch, |level, node, quantity| {
            rows.push(root, node, level, quantity);
        });
    }

    rows.finish(&hierarchy, "hierarchy_levels")
}

/// A node on the current DFS path and the index of its next child edge to follow.
struct Frame {
    node: u32,
    next_edge: usize,
    quantity: f64,
}

struct PathRows {
    ancestor: Vec<IdxSize>,
    descendant: Vec<IdxSize>,
    level: Vec<u32>,
    parent: Vec<Option<IdxSize>>,
    quantity_per: Vec<Option<f64>>,
    quantity: Vec<f64>,
    path: ListPrimitiveChunkedBuilder<IdxType>,
}

impl PathRows {
    fn new() -> Self {
        PathRows {
            ancestor: Vec::new(),
            descendant: Vec::new(),
            level: Vec::new(),
            parent: Vec::new(),
            quantity_per: Vec::new(),
            quantity: Vec::new(),
            path: ListPrimitiveChunkedBuilder::new("path".into(), 0, 0, IDX_DTYPE),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        ancestor: u32,
        descendant: u32,
        level: u32,
        parent: Option<u32>,
        quantity_per: Option<f64>,
        quantity: f64,
        path: &[IdxSize],
    ) {
        self.ancestor.push(ancestor as IdxSize);
        self.descendant.push(descendant as IdxSize);
        self.level.push(level);
        self.parent.push(parent.map(|p| p as IdxSize));
        self.quantity_per.push(quantity_per);
        self.quantity.push(quantity);
        self.path.append_slice(path);
    }

    fn finish(mut self, hierarchy: &Hierarchy) -> PolarsResult<Series> {
        let is_leaf: Vec<bool> = self
            .descendant
            .iter()
            .map(|&d| hierarchy.is_leaf(d))
            .collect();
        let parent_ids: IdxCa = self.parent.into_iter().collect();
        let path = self
            .path
            .finish()
            .apply_to_inner(&|ids| hierarchy.names.take(ids.idx()?))?
            .into_series();

        into_struct(
            "hierarchy_paths",
            &[
                hierarchy.node_column("ancestor", &self.ancestor)?,
                hierarchy.node_column("descendant", &self.descendant)?,
                Series::new("level".into(), self.level),
                hierarchy
                    .names
                    .take(&parent_ids)?
                    .with_name("parent".into()),
                Series::new("quantity_per".into(), self.quantity_per),
                Series::new("quantity".into(), self.quantity),
                Series::new("is_leaf".into(), is_leaf),
                path.with_name("path".into()),
            ],
        )
    }
}

#[polars_expr(output_type_func=paths_output)]
fn hierarchy_paths(inputs: &[Series], kwargs: HierarchyKwargs) -> PolarsResult<Series> {
    let hierarchy = Hierarchy::build(inputs, &kwargs)?;
    let mut rows = PathRows::new();
    let mut stack: Vec<Frame> = Vec::new();
    // Mirrors the nodes on `stack`, in the id type the `path` column is built from.
    let mut path: Vec<IdxSize> = Vec::new();

    for &root in &hierarchy.roots {
        if hierarchy.include_self {
            rows.push(root, root, 0, None, None, 1.0, &[root as IdxSize]);
        }
        stack.push(Frame {
            node: root,
            next_edge: 0,
            quantity: 1.0,
        });
        path.push(root as IdxSize);

        // Depth-first, so rows come out in indented-BOM order.
        loop {
            let child_level = stack.len() as u32;
            let Some(frame) = stack.last_mut() else {
                break;
            };
            let edges = &hierarchy.children[frame.node as usize];
            if child_level > hierarchy.max_depth || frame.next_edge == edges.len() {
                stack.pop();
                path.pop();
                continue;
            }

            let (child, per) = edges[frame.next_edge];
            frame.next_edge += 1;
            let parent = frame.node;
            let quantity = frame.quantity * per;

            path.push(child as IdxSize);
            rows.push(
                root,
                child,
                child_level,
                Some(parent),
                Some(per),
                quantity,
                &path,
            );
            stack.push(Frame {
                node: child,
                next_edge: 0,
                quantity,
            });
        }
    }

    rows.finish(&hierarchy)
}
