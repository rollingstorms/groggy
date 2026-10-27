//! Explicit immutable indexes over a captured selection. Keys borrow frozen
//! values during construction; retained postings store IDs, never payload copies.
use super::{AttributeFilter, EdgeFilter, NodeFilter};
use crate::storage::numeric::scalar::{attribute_number, compare_inputs};
use crate::{subgraphs::GraphView, AttrName, AttrValue, GraphError, GraphResult};
use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

/// Reusable secondary indexes bound to one immutable graph selection.
/// Building/replacing an index is explicit. Mutations, branch switches and
/// recovery never change this capture or silently rebuild its indexes.
#[derive(Debug, Clone)]
pub struct GraphQueryIndex {
    view: GraphView,
    data: Arc<IndexData>,
}
#[derive(Debug)]
struct IndexData {
    nodes: HashMap<AttrName, ColumnIndex>,
    edges: HashMap<AttrName, ColumnIndex>,
    topology: Option<TopologyIndex>,
    labels: Option<HashMap<String, Vec<usize>>>,
}
#[derive(Debug)]
struct ColumnIndex {
    present: Vec<usize>,
    equality: Vec<usize>,
    ranges: [Vec<usize>; 4],
}
#[derive(Debug, Default)]
struct TopologyIndex {
    endpoints: HashMap<(usize, usize), Vec<usize>>,
    incident: HashMap<usize, Vec<usize>>,
    neighbors: HashMap<usize, BTreeSet<usize>>,
    degrees: Vec<(usize, usize)>,
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn value<'a>(view: &'a GraphView, node: bool, id: usize, name: &AttrName) -> Option<&'a AttrValue> {
    let snapshot = view.read_state().snapshot();
    (if node {
        &snapshot.node_attributes
    } else {
        &snapshot.edge_attributes
    })
    .get(&id)
    .and_then(|attrs| attrs.get(name))
}
fn range_domain(value: &AttrValue) -> Option<usize> {
    if let Some(number) = attribute_number(value) {
        return compare_inputs(number, number).map(|_| 0);
    }
    match value {
        AttrValue::UtcInstant(_) => Some(1),
        AttrValue::Date(_) => Some(2),
        AttrValue::Duration(_) => Some(3),
        _ => None,
    }
}
fn range_cmp(a: &AttrValue, b: &AttrValue) -> Option<Ordering> {
    attribute_number(a)
        .zip(attribute_number(b))
        .and_then(|(a, b)| compare_inputs(a, b))
        .or_else(|| a.temporal_cmp(b))
}
impl ColumnIndex {
    fn build(view: &GraphView, node: bool, ids: &[usize], name: &AttrName) -> Self {
        let rows: Vec<_> = ids
            .iter()
            .filter_map(|&id| value(view, node, id, name).map(|v| (id, v)))
            .collect();
        let present = rows.iter().map(|(id, _)| *id).collect();
        let mut equality = rows.clone();
        equality.sort_unstable_by(|(a, av), (b, bv)| av.cmp(bv).then(a.cmp(b)));
        let mut ranges = std::array::from_fn(|_| Vec::new());
        for (id, value) in &rows {
            if let Some(domain) = range_domain(value) {
                ranges[domain].push((*id, *value));
            }
        }
        for rows in &mut ranges {
            // All rows in a domain are comparable; NaNs never enter range indexes.
            rows.sort_unstable_by(|(a, av), (b, bv)| {
                range_cmp(av, bv).unwrap_or(Ordering::Equal).then(a.cmp(b))
            });
        }
        Self {
            present,
            equality: equality.into_iter().map(|(id, _)| id).collect(),
            ranges: ranges.map(|rows| rows.into_iter().map(|(id, _)| id).collect()),
        }
    }
    fn query(
        &self,
        view: &GraphView,
        node: bool,
        name: &AttrName,
        filter: Option<&AttributeFilter>,
    ) -> Vec<usize> {
        let Some(filter) = filter else {
            return self.present.clone();
        };
        let (target, equal, negate, greater, inclusive) = match filter {
            AttributeFilter::Equals(v) => (v, true, false, false, false),
            AttributeFilter::NotEquals(v) => (v, true, true, false, false),
            AttributeFilter::GreaterThan(v) => (v, false, false, true, false),
            AttributeFilter::GreaterThanOrEqual(v) => (v, false, false, true, true),
            AttributeFilter::LessThan(v) => (v, false, false, false, false),
            AttributeFilter::LessThanOrEqual(v) => (v, false, false, false, true),
        };
        let rows = if equal {
            &self.equality
        } else if let Some(domain) = range_domain(target) {
            &self.ranges[domain]
        } else {
            return Vec::new();
        };
        let compare = |id: &usize| {
            value(view, node, *id, name).map_or(Ordering::Equal, |v| {
                if equal {
                    v.cmp(target)
                } else {
                    range_cmp(v, target).unwrap_or(Ordering::Equal)
                }
            })
        };
        let lower = rows.partition_point(|id| compare(id) == Ordering::Less);
        let upper = rows.partition_point(|id| compare(id) != Ordering::Greater);
        let mut result = if equal {
            if negate {
                rows[..lower]
                    .iter()
                    .chain(&rows[upper..])
                    .copied()
                    .collect()
            } else {
                rows[lower..upper].to_vec()
            }
        } else if greater {
            rows[if inclusive { lower } else { upper }..].to_vec()
        } else {
            rows[..if inclusive { upper } else { lower }].to_vec()
        };
        result.sort_unstable();
        result
    }
}
impl GraphQueryIndex {
    /// Explicitly index named columns and optional selected topology. Empty names
    /// and duplicates reject; a named absent column is a valid empty index.
    pub fn build(
        view: GraphView,
        node_properties: &[AttrName],
        edge_properties: &[AttrName],
        topology: bool,
    ) -> GraphResult<Self> {
        fn columns(
            view: &GraphView,
            node: bool,
            ids: &[usize],
            names: &[AttrName],
        ) -> GraphResult<HashMap<AttrName, ColumnIndex>> {
            let mut result = HashMap::new();
            for name in names {
                if name.trim().is_empty() || result.contains_key(name) {
                    return Err(invalid(format!(
                        "Index property names must be nonempty and unique: {name}"
                    )));
                }
                result.insert(name.clone(), ColumnIndex::build(view, node, ids, name));
            }
            Ok(result)
        }
        let nodes = columns(&view, true, &view.node_ids(), node_properties)?;
        let edges = columns(&view, false, &view.edge_ids(), edge_properties)?;
        let labels = if nodes.contains_key("labels") {
            let mut labels: HashMap<String, Vec<usize>> = HashMap::new();
            for id in view.node_ids() {
                match value(&view, true, id, &"labels".into()) {
                    None => {}
                    Some(AttrValue::TextVec(names)) => {
                        for name in names {
                            labels.entry(name.clone()).or_default().push(id);
                        }
                    }
                    Some(_) => return Err(invalid("Indexed labels must be a text vector")),
                }
            }
            Some(labels)
        } else {
            None
        };
        let topology = if topology {
            let mut index = TopologyIndex::default();
            for edge in view.edge_ids() {
                let (s, t) = view.edge_endpoints(edge)?;
                let pair = if !view.is_directed() && s > t {
                    (t, s)
                } else {
                    (s, t)
                };
                index.endpoints.entry(pair).or_default().push(edge);
                index.incident.entry(s).or_default().push(edge);
                if s != t {
                    index.incident.entry(t).or_default().push(edge);
                }
                index.neighbors.entry(s).or_default().insert(t);
                index.neighbors.entry(t).or_default().insert(s);
            }
            for node in view.node_ids() {
                index.degrees.push((view.degree(node)?, node));
            }
            index.degrees.sort_unstable();
            Some(index)
        } else {
            None
        };
        Ok(Self {
            view,
            data: Arc::new(IndexData {
                nodes,
                edges,
                topology,
                labels,
            }),
        })
    }
    /// The exact capture and selection indexed; clones share frozen values/postings.
    pub fn view(&self) -> &GraphView {
        &self.view
    }
    /// Explicitly build the same requested indexes for a new immutable selection.
    /// The old index and every result derived from it remain unchanged.
    pub fn rebuild(&self, view: GraphView) -> GraphResult<Self> {
        Self::build(
            view,
            &self.node_properties(),
            &self.edge_properties(),
            self.has_topology(),
        )
    }
    /// Sorted explicitly indexed node property names, including absent columns.
    pub fn node_properties(&self) -> Vec<AttrName> {
        let mut names: Vec<_> = self.data.nodes.keys().cloned().collect();
        names.sort_unstable();
        names
    }
    /// Sorted explicitly indexed edge property names.
    pub fn edge_properties(&self) -> Vec<AttrName> {
        let mut names: Vec<_> = self.data.edges.keys().cloned().collect();
        names.sort_unstable();
        names
    }
    /// Whether selected incidence, endpoints and degrees were explicitly indexed.
    pub fn has_topology(&self) -> bool {
        self.data.topology.is_some()
    }
    fn require_column(&self, node: bool, name: &AttrName) -> GraphResult<&ColumnIndex> {
        (if node {
            &self.data.nodes
        } else {
            &self.data.edges
        })
        .get(name)
        .ok_or_else(|| {
            invalid(format!(
                "No {} property index for {name}; build it explicitly or request allow_scan",
                if node { "node" } else { "edge" }
            ))
        })
    }
    fn topology(&self) -> GraphResult<&TopologyIndex> {
        self.data
            .topology
            .as_ref()
            .ok_or_else(|| invalid("No topology index; build it explicitly or request allow_scan"))
    }
    fn check_node(&self, filter: &NodeFilter) -> GraphResult<()> {
        match filter {
            NodeFilter::HasAttribute { name }
            | NodeFilter::AttributeEquals { name, .. }
            | NodeFilter::AttributeFilter { name, .. } => {
                self.require_column(true, name)?;
            }
            NodeFilter::DegreeRange { .. } | NodeFilter::HasNeighbor { .. } => {
                self.topology()?;
            }
            NodeFilter::And(filters) | NodeFilter::Or(filters) => {
                for f in filters {
                    self.check_node(f)?;
                }
            }
            NodeFilter::Not(f) => self.check_node(f)?,
            NodeFilter::NodeSet(_) => {}
        }
        Ok(())
    }
    fn check_edge(&self, filter: &EdgeFilter) -> GraphResult<()> {
        match filter {
            EdgeFilter::HasAttribute { name }
            | EdgeFilter::AttributeEquals { name, .. }
            | EdgeFilter::AttributeFilter { name, .. } => {
                self.require_column(false, name)?;
            }
            EdgeFilter::ConnectsNodes { .. } | EdgeFilter::ConnectsAny(_) => {
                self.topology()?;
            }
            EdgeFilter::And(filters) | EdgeFilter::Or(filters) => {
                for f in filters {
                    self.check_edge(f)?;
                }
            }
            EdgeFilter::Not(f) => self.check_edge(f)?,
        }
        Ok(())
    }
    /// Execute entirely against captured indexes; missing indexes fail unless
    /// the caller explicitly permits the existing column-scan executor.
    pub fn filter_nodes(&self, filter: &NodeFilter, allow_scan: bool) -> GraphResult<GraphView> {
        self.view
            .select_nodes(&self.find_nodes(filter, allow_scan)?)
    }
    /// Sorted matching IDs, avoiding graph-view materialization and its induced
    /// edge selection cost. Missing indexes never trigger an implicit scan.
    pub fn find_nodes(&self, filter: &NodeFilter, allow_scan: bool) -> GraphResult<Vec<usize>> {
        if let Err(error) = self.check_node(filter) {
            return if allow_scan {
                super::filtering::nodes(
                    &super::filtering::Frozen(&self.view),
                    &self.view.node_ids(),
                    filter,
                )
            } else {
                Err(error)
            };
        }
        Ok(self.node_ids(filter)?.into_iter().collect())
    }
    /// Preserve parallel edge identities and the captured selected endpoint closure.
    pub fn filter_edges(&self, filter: &EdgeFilter, allow_scan: bool) -> GraphResult<GraphView> {
        self.view
            .select_edges(&self.find_edges(filter, allow_scan)?)
    }
    /// Sorted matching edge IDs without copying values or constructing a view.
    pub fn find_edges(&self, filter: &EdgeFilter, allow_scan: bool) -> GraphResult<Vec<usize>> {
        if let Err(error) = self.check_edge(filter) {
            return if allow_scan {
                super::filtering::edges(
                    &super::filtering::Frozen(&self.view),
                    &self.view.edge_ids(),
                    filter,
                )
            } else {
                Err(error)
            };
        }
        Ok(self.edge_ids(filter)?.into_iter().collect())
    }
    /// Select a semantic node label through an explicitly requested labels index.
    pub fn nodes_with_label(&self, label: &str) -> GraphResult<GraphView> {
        if label.trim().is_empty() {
            return Err(invalid("Label must be nonempty"));
        }
        let labels = self
            .data
            .labels
            .as_ref()
            .ok_or_else(|| invalid("No node labels index; build it explicitly"))?;
        self.view
            .select_nodes(labels.get(label).map_or(&[], Vec::as_slice))
    }
    fn property(
        &self,
        node: bool,
        name: &AttrName,
        filter: Option<&AttributeFilter>,
    ) -> GraphResult<BTreeSet<usize>> {
        Ok(self
            .require_column(node, name)?
            .query(&self.view, node, name, filter)
            .into_iter()
            .collect())
    }
    fn node_ids(&self, filter: &NodeFilter) -> GraphResult<BTreeSet<usize>> {
        Ok(match filter {
            NodeFilter::HasAttribute { name } => self.property(true, name, None)?,
            NodeFilter::AttributeEquals { name, value } => {
                self.property(true, name, Some(&AttributeFilter::Equals(value.clone())))?
            }
            NodeFilter::AttributeFilter { name, filter } => {
                self.property(true, name, Some(filter))?
            }
            NodeFilter::NodeSet(ids) => ids.intersection(self.view.nodes()).copied().collect(),
            NodeFilter::DegreeRange { min, max } => {
                let rows = &self.topology()?.degrees;
                if min > max {
                    BTreeSet::new()
                } else {
                    let begin = rows.partition_point(|(d, _)| d < min);
                    let end = rows.partition_point(|(d, _)| d <= max);
                    rows[begin..end].iter().map(|(_, id)| *id).collect()
                }
            }
            NodeFilter::HasNeighbor { neighbor_id } => self
                .topology()?
                .neighbors
                .get(neighbor_id)
                .cloned()
                .unwrap_or_default(),
            NodeFilter::And(filters) => {
                let Some((first, rest)) = filters.split_first() else {
                    return Ok(BTreeSet::new());
                };
                let mut result = self.node_ids(first)?;
                for f in rest {
                    let next = self.node_ids(f)?;
                    result.retain(|id| next.contains(id));
                }
                result
            }
            NodeFilter::Or(filters) => {
                let mut result = BTreeSet::new();
                for f in filters {
                    result.extend(self.node_ids(f)?);
                }
                result
            }
            NodeFilter::Not(f) => {
                let excluded = self.node_ids(f)?;
                self.view
                    .nodes()
                    .iter()
                    .filter(|id| !excluded.contains(id))
                    .copied()
                    .collect()
            }
        })
    }
    fn edge_ids(&self, filter: &EdgeFilter) -> GraphResult<BTreeSet<usize>> {
        Ok(match filter {
            EdgeFilter::HasAttribute { name } => self.property(false, name, None)?,
            EdgeFilter::AttributeEquals { name, value } => {
                self.property(false, name, Some(&AttributeFilter::Equals(value.clone())))?
            }
            EdgeFilter::AttributeFilter { name, filter } => {
                self.property(false, name, Some(filter))?
            }
            EdgeFilter::ConnectsNodes { source, target } => {
                let pair = if !self.view.is_directed() && source > target {
                    (*target, *source)
                } else {
                    (*source, *target)
                };
                self.topology()?
                    .endpoints
                    .get(&pair)
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect()
            }
            EdgeFilter::ConnectsAny(nodes) => {
                let topology = self.topology()?;
                nodes
                    .iter()
                    .flat_map(|n| topology.incident.get(n).into_iter().flatten())
                    .copied()
                    .collect()
            }
            EdgeFilter::And(filters) => {
                let Some((first, rest)) = filters.split_first() else {
                    return Ok(self.view.edges().iter().copied().collect());
                };
                let mut result = self.edge_ids(first)?;
                for f in rest {
                    let next = self.edge_ids(f)?;
                    result.retain(|id| next.contains(id));
                }
                result
            }
            EdgeFilter::Or(filters) => {
                let mut result = BTreeSet::new();
                for f in filters {
                    result.extend(self.edge_ids(f)?);
                }
                result
            }
            EdgeFilter::Not(f) => {
                let excluded = self.edge_ids(f)?;
                self.view
                    .edges()
                    .iter()
                    .filter(|id| !excluded.contains(id))
                    .copied()
                    .collect()
            }
        })
    }
}
