//! One discovery-tree/path kernel for current topology and immutable selections.
use crate::{EdgeId, GraphError, GraphResult, NodeId};
use std::collections::{HashMap, HashSet, VecDeque};

pub(crate) trait Source {
    fn contains_node(&self, node: NodeId) -> bool;
    fn outgoing(&self, node: NodeId) -> GraphResult<impl Iterator<Item = (NodeId, EdgeId)> + '_>;
}
pub(crate) type Parents = HashMap<NodeId, (NodeId, EdgeId)>;
pub(crate) type Membership = (HashSet<NodeId>, HashSet<EdgeId>);

pub(crate) fn breadth_tree(
    source: &impl Source,
    start: NodeId,
    max_depth: Option<usize>,
    goal: Option<NodeId>,
) -> GraphResult<(HashSet<NodeId>, Parents)> {
    check_start(source, start)?;
    let mut visited = HashSet::from([start]);
    let mut parents = Parents::new();
    let mut queue = VecDeque::from([(start, 0usize)]);
    while let Some((node, depth)) = queue.pop_front() {
        if Some(node) == goal {
            break;
        }
        if max_depth.is_some_and(|limit| depth >= limit) {
            continue;
        }
        for (neighbor, edge) in source.outgoing(node)? {
            if visited.insert(neighbor) {
                parents.insert(neighbor, (node, edge));
                queue.push_back((neighbor, depth + 1));
            }
        }
    }
    Ok((visited, parents))
}
fn check_start(source: &impl Source, start: NodeId) -> GraphResult<()> {
    if source.contains_node(start) {
        Ok(())
    } else {
        Err(GraphError::InvalidInput(format!(
            "Traversal start {start} does not belong to this view"
        )))
    }
}
pub(crate) fn bfs(
    source: &impl Source,
    start: NodeId,
    max_depth: Option<usize>,
) -> GraphResult<Membership> {
    let (nodes, parents) = breadth_tree(source, start, max_depth, None)?;
    Ok((nodes, parents.into_values().map(|(_, edge)| edge).collect()))
}
pub(crate) fn dfs(
    source: &impl Source,
    start: NodeId,
    max_depth: Option<usize>,
) -> GraphResult<Membership> {
    check_start(source, start)?;
    let mut nodes = HashSet::from([start]);
    let mut edges = HashSet::new();
    let mut stack = vec![(0usize, source.outgoing(start)?)];
    while let Some((depth, neighbors)) = stack.last_mut() {
        if max_depth.is_some_and(|limit| *depth >= limit) {
            stack.pop();
            continue;
        }
        let depth = *depth;
        if let Some((neighbor, edge)) = neighbors.next() {
            if nodes.insert(neighbor) {
                edges.insert(edge);
                stack.push((depth + 1, source.outgoing(neighbor)?));
            }
        } else {
            stack.pop();
        }
    }
    Ok((nodes, edges))
}
pub(crate) fn has_path(source: &impl Source, start: NodeId, target: NodeId) -> GraphResult<bool> {
    if !source.contains_node(start) || !source.contains_node(target) {
        return Ok(false);
    }
    if start == target {
        return Ok(true);
    }
    Ok(breadth_tree(source, start, None, Some(target))?
        .0
        .contains(&target))
}
pub(crate) fn shortest_path(
    source: &impl Source,
    start: NodeId,
    target: NodeId,
) -> GraphResult<Option<Membership>> {
    if !source.contains_node(start) || !source.contains_node(target) {
        return Ok(None);
    }
    let (visited, parents) = breadth_tree(source, start, None, Some(target))?;
    if !visited.contains(&target) {
        return Ok(None);
    }
    let mut nodes = HashSet::from([target]);
    let mut edges = HashSet::new();
    let mut current = target;
    while current != start {
        let &(parent, edge) = parents
            .get(&current)
            .ok_or_else(|| GraphError::InvalidInput("Incomplete path predecessor chain".into()))?;
        nodes.insert(parent);
        edges.insert(edge);
        current = parent;
    }
    Ok(Some((nodes, edges)))
}

impl Source for crate::GraphView {
    fn contains_node(&self, node: NodeId) -> bool {
        self.contains_node(node)
    }
    fn outgoing(&self, node: NodeId) -> GraphResult<impl Iterator<Item = (NodeId, EdgeId)> + '_> {
        self.outgoing_occurrences(node)
    }
}

/// Borrow the owner's topology snapshot; selection validation belongs to its adapter.
/// No property payloads, frozen graph, history or writable owner are copied.
pub(crate) struct Live<'a> {
    nodes: &'a HashSet<NodeId>,
    edges: &'a HashSet<EdgeId>,
    adjacency: std::sync::Arc<HashMap<NodeId, Vec<(NodeId, EdgeId)>>>,
}
impl<'a> Live<'a> {
    pub(crate) fn new(
        graph: &crate::Graph,
        nodes: &'a HashSet<NodeId>,
        edges: &'a HashSet<EdgeId>,
    ) -> Self {
        let adjacency = graph.space().snapshot(&graph.pool()).3;
        Self {
            nodes,
            edges,
            adjacency,
        }
    }
}
impl Source for Live<'_> {
    fn contains_node(&self, node: NodeId) -> bool {
        self.nodes.contains(&node)
    }
    fn outgoing(&self, node: NodeId) -> GraphResult<impl Iterator<Item = (NodeId, EdgeId)> + '_> {
        let mut row: Vec<_> = self
            .adjacency
            .get(&node)
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .copied()
            .filter(|(neighbor, edge)| self.nodes.contains(neighbor) && self.edges.contains(edge))
            .collect();
        row.sort_unstable();
        Ok(row.into_iter())
    }
}
