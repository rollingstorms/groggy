//! Finite conjunctive graph patterns over immutable captured state.
use super::{
    endpoints::EndpointIndex, EdgeFilter, GraphQueryIndex, NodeFilter, TraversalDirection,
};
use crate::{EdgeId, GraphError, GraphResult, GraphView, NodeId};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

/// A named node binding with an optional existing native predicate.
#[derive(Debug, Clone)]
pub struct PatternNode {
    pub variable: String,
    pub filter: Option<NodeFilter>,
}
/// A named edge instance connecting declared node variables.
#[derive(Debug, Clone)]
pub struct PatternEdge {
    pub variable: String,
    pub source: String,
    pub target: String,
    pub direction: TraversalDirection,
    pub filter: Option<EdgeFilter>,
}
/// A finite graph-shaped conjunction, including cycles, branches and disconnected
/// components. Variables are unique across both kinds; endpoints name nodes.
#[derive(Debug, Clone, Default)]
pub struct GraphPattern {
    pub nodes: Vec<PatternNode>,
    pub edges: Vec<PatternEdge>,
}
/// Explicit search/output controls. Default binding semantics permit reuse;
/// distinctness is caller-selected, never implicit fact deduplication.
#[derive(Debug, Clone, Default)]
pub struct PatternOptions {
    pub distinct_nodes: bool,
    pub distinct_edges: bool,
    pub offset: usize,
    pub limit: Option<usize>,
    /// Maximum attempted variable assignments, excluding predicate, topology and
    /// candidate preparation. This is not a wall-time or memory bound.
    /// Exhaustion returns an error, never an apparently complete partial result.
    pub max_expansions: Option<usize>,
}
/// Binding IDs follow the pattern's declared node/edge variable order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternMatch {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
}
/// Immutable binding rows and their exact source capture. Results preserve source
/// write eligibility and remain readable after owner mutation/closure or pattern drop.
#[derive(Debug, Clone)]
pub struct PatternMatches {
    source: GraphView,
    node_variables: Arc<[String]>,
    edge_variables: Arc<[String]>,
    rows: Vec<PatternMatch>,
    truncated: bool,
    expansions: usize,
}
impl PatternMatches {
    /// Original captured graph selection and clocks.
    pub fn source_view(&self) -> &GraphView {
        &self.source
    }
    /// Node-variable names in declaration order.
    pub fn node_variables(&self) -> &[String] {
        &self.node_variables
    }
    /// Edge-variable names in declaration order.
    pub fn edge_variables(&self) -> &[String] {
        &self.edge_variables
    }
    /// Matching rows, ordered lexicographically by declared node IDs then edge IDs.
    pub fn rows(&self) -> &[PatternMatch] {
        &self.rows
    }
    /// True only after observing an additional complete match beyond the page.
    pub fn truncated(&self) -> bool {
        self.truncated
    }
    /// Attempted variable assignments; this excludes filter and topology preparation.
    pub fn expansions(&self) -> usize {
        self.expansions
    }
    /// Graph containing exactly a row's bound nodes/edge instances. It preserves
    /// isolated bindings and source clocks; it does not induce unrelated edges.
    pub fn match_view(&self, row: usize) -> GraphResult<GraphView> {
        let binding = self
            .rows
            .get(row)
            .ok_or_else(|| invalid("Pattern match row is out of bounds"))?;
        GraphView::from_membership(
            self.source.read_state().clone(),
            binding.nodes.iter().copied().collect(),
            binding.edges.iter().copied().collect(),
        )
    }
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
struct Constraint {
    source: usize,
    target: usize,
    direction: TraversalDirection,
    allowed: Option<HashSet<EdgeId>>,
}
enum Topology<'a> {
    Indexed(&'a EndpointIndex),
    Scanned(EndpointIndex),
}
impl Topology<'_> {
    fn get(&self) -> &EndpointIndex {
        match self {
            Self::Indexed(v) => v,
            Self::Scanned(v) => v,
        }
    }
}
struct Frame {
    candidates: Arc<[usize]>,
    next: usize,
}
impl Frame {
    fn new(candidates: Arc<[usize]>) -> Self {
        Self {
            candidates,
            next: 0,
        }
    }
}
struct Search<'a> {
    index: &'a GraphQueryIndex,
    topology: Option<Topology<'a>>,
    nodes: Vec<Arc<[NodeId]>>,
    edges: Vec<Constraint>,
    incident_constraints: Vec<Vec<usize>>,
    options: &'a PatternOptions,
    expansions: usize,
}
impl GraphPattern {
    /// Validate names and endpoint declarations without consulting graph state.
    pub fn validate(&self) -> GraphResult<()> {
        self.positions().map(|_| ())
    }
    fn positions(&self) -> GraphResult<HashMap<&str, usize>> {
        if self.nodes.is_empty() {
            return Err(invalid(
                "A graph pattern requires at least one node variable",
            ));
        }
        let mut names = HashSet::new();
        let mut nodes = HashMap::new();
        for (position, node) in self.nodes.iter().enumerate() {
            if node.variable.trim().is_empty() || !names.insert(node.variable.as_str()) {
                return Err(invalid(format!(
                    "Pattern variable must be nonempty and unique: {}",
                    node.variable
                )));
            }
            nodes.insert(node.variable.as_str(), position);
        }
        for edge in &self.edges {
            if edge.variable.trim().is_empty() || !names.insert(edge.variable.as_str()) {
                return Err(invalid(format!(
                    "Pattern variable must be nonempty and unique: {}",
                    edge.variable
                )));
            }
            if !nodes.contains_key(edge.source.as_str())
                || !nodes.contains_key(edge.target.as_str())
            {
                return Err(invalid(format!(
                    "Pattern edge {} must reference declared node variables",
                    edge.variable
                )));
            }
        }
        Ok(nodes)
    }
}
impl GraphQueryIndex {
    /// Execute a graph-shaped pattern using captured indexes or explicit scan
    /// permission. Reuse/distinctness, page bounds and search budget are explicit.
    pub fn match_pattern(
        &self,
        pattern: &GraphPattern,
        options: &PatternOptions,
        allow_scan: bool,
    ) -> GraphResult<PatternMatches> {
        let positions = pattern.positions()?;
        let width = pattern
            .nodes
            .len()
            .checked_add(pattern.edges.len())
            .ok_or_else(|| invalid("Pattern variable count overflows"))?;
        let topology = if pattern.edges.is_empty() {
            None
        } else if let Some(index) = self.endpoint_index() {
            Some(Topology::Indexed(index))
        } else if allow_scan {
            Some(Topology::Scanned(EndpointIndex::scan(self.view())?))
        } else {
            return Err(invalid(
                "No topology index for pattern matching; build it explicitly or request allow_scan",
            ));
        };
        let all_nodes: Arc<[NodeId]> = self.view().node_ids().into();
        let mut nodes = Vec::with_capacity(pattern.nodes.len());
        // Prepare every filter before search, including filters behind empty candidates.
        for node in &pattern.nodes {
            nodes.push(match &node.filter {
                Some(filter) => self.find_nodes(filter, allow_scan)?.into(),
                None => Arc::clone(&all_nodes),
            });
        }
        let mut edges = Vec::with_capacity(pattern.edges.len());
        for edge in &pattern.edges {
            edges.push(Constraint {
                source: positions[edge.source.as_str()],
                target: positions[edge.target.as_str()],
                direction: edge.direction,
                allowed: edge
                    .filter
                    .as_ref()
                    .map(|filter| {
                        self.find_edges(filter, allow_scan)
                            .map(|ids| ids.into_iter().collect())
                    })
                    .transpose()?,
            });
        }
        let mut incident_constraints = vec![Vec::new(); pattern.nodes.len()];
        for (position, edge) in edges.iter().enumerate() {
            incident_constraints[edge.source].push(position);
            if edge.target != edge.source {
                incident_constraints[edge.target].push(position);
            }
        }
        let mut search = Search {
            index: self,
            topology,
            nodes,
            edges,
            incident_constraints,
            options,
            expansions: 0,
        };
        let (rows, truncated) = search.run(width)?;
        Ok(PatternMatches {
            source: self.view().clone(),
            node_variables: pattern
                .nodes
                .iter()
                .map(|v| v.variable.clone())
                .collect::<Vec<_>>()
                .into(),
            edge_variables: pattern
                .edges
                .iter()
                .map(|v| v.variable.clone())
                .collect::<Vec<_>>()
                .into(),
            rows,
            truncated,
            expansions: search.expansions,
        })
    }
}
impl Search<'_> {
    fn topology(&self) -> GraphResult<&EndpointIndex> {
        self.topology
            .as_ref()
            .map(Topology::get)
            .ok_or_else(|| invalid("Pattern has no prepared endpoint source"))
    }
    fn edge_candidates(&self, edge: &Constraint, values: &[usize]) -> GraphResult<Vec<EdgeId>> {
        let mut ids =
            self.topology()?
                .between(values[edge.source], values[edge.target], edge.direction);
        if let Some(allowed) = &edge.allowed {
            ids.retain(|id| allowed.contains(id));
        }
        Ok(ids)
    }
    fn candidates(&self, depth: usize, values: &[usize]) -> GraphResult<Frame> {
        if depth >= self.nodes.len() {
            return Ok(Frame::new(
                self.edge_candidates(&self.edges[depth - self.nodes.len()], values)?
                    .into(),
            ));
        }
        let base = &self.nodes[depth];
        let mut joined: Option<BTreeSet<NodeId>> = None;
        for &position in &self.incident_constraints[depth] {
            let edge = &self.edges[position];
            let neighbor = if edge.source == depth && edge.target < depth {
                Some((edge.target, false))
            } else if edge.target == depth && edge.source < depth {
                Some((edge.source, true))
            } else {
                None
            };
            let Some((bound, is_source)) = neighbor else {
                continue;
            };
            let bound = values[bound];
            let mut candidates = BTreeSet::new();
            for &id in self.topology()?.incident(bound) {
                if edge
                    .allowed
                    .as_ref()
                    .is_some_and(|allowed| !allowed.contains(&id))
                {
                    continue;
                }
                let (s, t) = self.index.view().edge_endpoints(id)?;
                let other = if s == bound { t } else { s };
                let (source, target) = if is_source {
                    (bound, other)
                } else {
                    (other, bound)
                };
                if compatible(
                    self.index.view().is_directed(),
                    edge.direction,
                    source,
                    target,
                    s,
                    t,
                ) && base.binary_search(&other).is_ok()
                {
                    candidates.insert(other);
                }
            }
            match &mut joined {
                Some(current) => current.retain(|id| candidates.contains(id)),
                None => joined = Some(candidates),
            }
        }
        Ok(Frame::new(joined.map_or_else(
            || Arc::clone(base),
            |ids| ids.into_iter().collect::<Vec<_>>().into(),
        )))
    }
    fn consistent(&self, depth: usize, values: &[usize]) -> GraphResult<bool> {
        for &position in &self.incident_constraints[depth] {
            let edge = &self.edges[position];
            if edge.source <= depth
                && edge.target <= depth
                && self.edge_candidates(edge, values)?.is_empty()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn tick(&mut self) -> GraphResult<()> {
        if self
            .options
            .max_expansions
            .is_some_and(|max| self.expansions >= max)
        {
            return Err(invalid(
                "Pattern expansion budget exhausted; no partial result was returned",
            ));
        }
        self.expansions = self
            .expansions
            .checked_add(1)
            .ok_or_else(|| invalid("Pattern expansion counter overflow"))?;
        Ok(())
    }
    fn run(&mut self, width: usize) -> GraphResult<(Vec<PatternMatch>, bool)> {
        let mut values = vec![0; width];
        let mut frames = vec![self.candidates(0, &values)?];
        let mut rows = Vec::new();
        let mut skipped = self.options.offset;
        while !frames.is_empty() {
            let depth = frames.len() - 1;
            let frame = &mut frames[depth];
            let Some(&candidate) = frame.candidates.get(frame.next) else {
                frames.pop();
                continue;
            };
            frame.next += 1;
            self.tick()?;
            values[depth] = candidate;
            if depth < self.nodes.len() {
                if self.options.distinct_nodes && values[..depth].contains(&candidate) {
                    continue;
                }
                if !self.consistent(depth, &values)? {
                    continue;
                }
            } else if self.options.distinct_edges
                && values[self.nodes.len()..depth].contains(&candidate)
            {
                continue;
            }
            if depth + 1 < width {
                frames.push(self.candidates(depth + 1, &values)?);
                continue;
            }
            if skipped > 0 {
                skipped -= 1;
                continue;
            }
            if self.options.limit.is_some_and(|limit| rows.len() >= limit) {
                return Ok((rows, true));
            }
            rows.push(PatternMatch {
                nodes: values[..self.nodes.len()].to_vec(),
                edges: values[self.nodes.len()..].to_vec(),
            });
        }
        Ok((rows, false))
    }
}
fn compatible(
    directed: bool,
    direction: TraversalDirection,
    source: NodeId,
    target: NodeId,
    s: NodeId,
    t: NodeId,
) -> bool {
    if !directed || direction == TraversalDirection::Both {
        (source == s && target == t) || (source == t && target == s)
    } else if direction == TraversalDirection::Incoming {
        source == t && target == s
    } else {
        source == s && target == t
    }
}
