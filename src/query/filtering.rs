//! Shared column-oriented predicate execution over live or immutable read sources.
use super::{AttributeFilter, EdgeFilter, NodeFilter};
use crate::{
    state::space::GraphSpace, storage::pool::GraphPool, subgraphs::GraphView, AttrName, AttrValue,
    EdgeId, GraphError, GraphResult, NodeId,
};
use rayon::prelude::*;
use std::{
    cell::OnceCell,
    collections::{HashMap, HashSet},
};

pub(crate) trait Source {
    fn attribute(&self, id: usize, name: &AttrName, node: bool) -> GraphResult<Option<&AttrValue>>;
    fn degree(&self, node: NodeId) -> GraphResult<usize>;
    fn incident_neighbor(&self, node: NodeId, neighbor: NodeId) -> GraphResult<bool>;
    fn endpoints(&self, edge: EdgeId) -> GraphResult<(NodeId, NodeId)>;
    fn directed(&self) -> bool;
}
fn attributes(
    source: &impl Source,
    ids: &[usize],
    name: &AttrName,
    node: bool,
    predicate: Option<&AttributeFilter>,
) -> GraphResult<Vec<usize>> {
    // Resolve the complete requested column before matching; do not clone payloads.
    let values = ids
        .iter()
        .map(|&id| source.attribute(id, name, node).map(|value| (id, value)))
        .collect::<GraphResult<Vec<_>>>()?;
    let matches = |(id, value): (usize, Option<&AttrValue>)| {
        value
            .filter(|value| predicate.is_none_or(|p| p.matches(value)))
            .map(|_| id)
    };
    Ok(if values.len() > 1000 {
        values.into_par_iter().filter_map(matches).collect()
    } else {
        values.into_iter().filter_map(matches).collect()
    })
}
fn ordered_members(ids: &[usize], selected: &HashSet<usize>) -> Vec<usize> {
    ids.iter()
        .copied()
        .filter(|id| selected.contains(id))
        .collect()
}
pub(crate) fn nodes(
    source: &impl Source,
    ids: &[NodeId],
    filter: &NodeFilter,
) -> GraphResult<Vec<NodeId>> {
    match filter {
        NodeFilter::HasAttribute { name } => attributes(source, ids, name, true, None),
        NodeFilter::AttributeEquals { name, value } => attributes(
            source,
            ids,
            name,
            true,
            Some(&AttributeFilter::Equals(value.clone())),
        ),
        NodeFilter::AttributeFilter { name, filter } => {
            attributes(source, ids, name, true, Some(filter))
        }
        NodeFilter::NodeSet(selected) => Ok(ordered_members(ids, selected)),
        NodeFilter::DegreeRange { min, max } => ids
            .iter()
            .copied()
            .filter_map(|id| match source.degree(id) {
                Ok(n) if n >= *min && n <= *max => Some(Ok(id)),
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            })
            .collect(),
        NodeFilter::HasNeighbor { neighbor_id } => ids
            .iter()
            .copied()
            .filter_map(|id| match source.incident_neighbor(id, *neighbor_id) {
                Ok(true) => Some(Ok(id)),
                Ok(false) => None,
                Err(e) => Some(Err(e)),
            })
            .collect(),
        NodeFilter::And(filters) => {
            // Preserve established node-And([]) behavior (matches nothing).
            if filters.is_empty() {
                return Ok(Vec::new());
            }
            let mut result = ids.to_vec();
            for filter in filters {
                result = nodes(source, &result, filter)?;
                if result.is_empty() {
                    break;
                }
            }
            Ok(result)
        }
        NodeFilter::Or(filters) => {
            let mut selected = HashSet::new();
            for filter in filters {
                selected.extend(nodes(source, ids, filter)?);
            }
            Ok(ordered_members(ids, &selected))
        }
        NodeFilter::Not(filter) => {
            let selected: HashSet<_> = nodes(source, ids, filter)?.into_iter().collect();
            Ok(ids
                .iter()
                .copied()
                .filter(|id| !selected.contains(id))
                .collect())
        }
    }
}
pub(crate) fn edges(
    source: &impl Source,
    ids: &[EdgeId],
    filter: &EdgeFilter,
) -> GraphResult<Vec<EdgeId>> {
    match filter {
        EdgeFilter::HasAttribute { name } => attributes(source, ids, name, false, None),
        EdgeFilter::AttributeEquals { name, value } => attributes(
            source,
            ids,
            name,
            false,
            Some(&AttributeFilter::Equals(value.clone())),
        ),
        EdgeFilter::AttributeFilter { name, filter } => {
            attributes(source, ids, name, false, Some(filter))
        }
        EdgeFilter::ConnectsNodes {
            source: a,
            target: b,
        } => ids
            .iter()
            .copied()
            .filter_map(|id| match source.endpoints(id) {
                Ok((s, t))
                    if (s == *a && t == *b) || (!source.directed() && s == *b && t == *a) =>
                {
                    Some(Ok(id))
                }
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            })
            .collect(),
        EdgeFilter::ConnectsAny(selected) => {
            let selected: HashSet<_> = selected.iter().copied().collect();
            ids.iter()
                .copied()
                .filter_map(|id| match source.endpoints(id) {
                    Ok((s, t)) if selected.contains(&s) || selected.contains(&t) => Some(Ok(id)),
                    Ok(_) => None,
                    Err(e) => Some(Err(e)),
                })
                .collect()
        }
        EdgeFilter::And(filters) => {
            // Edge-And([]) historically matches all candidate instances.
            let mut result = ids.to_vec();
            for filter in filters {
                result = edges(source, &result, filter)?;
                if result.is_empty() {
                    break;
                }
            }
            Ok(result)
        }
        EdgeFilter::Or(filters) => {
            let mut selected = HashSet::new();
            for filter in filters {
                selected.extend(edges(source, ids, filter)?);
            }
            Ok(ordered_members(ids, &selected))
        }
        EdgeFilter::Not(filter) => {
            let selected: HashSet<_> = edges(source, ids, filter)?.into_iter().collect();
            Ok(ids
                .iter()
                .copied()
                .filter(|id| !selected.contains(id))
                .collect())
        }
    }
}

pub(crate) struct Live<'a> {
    pool: &'a GraphPool,
    space: &'a GraphSpace,
    scope: Option<&'a HashSet<EdgeId>>,
    incidence: OnceCell<GraphResult<HashMap<NodeId, Vec<NodeId>>>>,
}
impl<'a> Live<'a> {
    pub(crate) fn new(
        pool: &'a GraphPool,
        space: &'a GraphSpace,
        scope: Option<&'a HashSet<EdgeId>>,
    ) -> Self {
        Self {
            pool,
            space,
            scope,
            incidence: OnceCell::new(),
        }
    }
    fn incidence(&self) -> GraphResult<&HashMap<NodeId, Vec<NodeId>>> {
        self.incidence
            .get_or_init(|| {
                let mut result: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
                let edges: Vec<_> = match self.scope {
                    Some(ids) => ids.iter().copied().collect(),
                    None => self.space.edge_ids(),
                };
                for edge in edges {
                    let (s, t) = self.endpoints(edge)?;
                    result.entry(s).or_default().push(t);
                    if s != t {
                        result.entry(t).or_default().push(s);
                    }
                }
                Ok(result)
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}
impl Source for Live<'_> {
    fn attribute(&self, id: usize, name: &AttrName, node: bool) -> GraphResult<Option<&AttrValue>> {
        let index = if node {
            self.space.get_node_attr_index(id, name)
        } else {
            self.space.get_edge_attr_index(id, name)
        };
        index
            .map(|index| {
                self.pool
                    .get_attr_by_index(name, index, node)
                    .ok_or_else(|| {
                        GraphError::InvalidInput(format!(
                            "Invalid query value reference for entity {id}, attribute {name}"
                        ))
                    })
            })
            .transpose()
    }
    fn degree(&self, node: NodeId) -> GraphResult<usize> {
        Ok(self.incidence()?.get(&node).map_or(0, Vec::len))
    }
    fn incident_neighbor(&self, node: NodeId, neighbor: NodeId) -> GraphResult<bool> {
        Ok(self
            .incidence()?
            .get(&node)
            .is_some_and(|ids| ids.contains(&neighbor)))
    }
    fn endpoints(&self, edge: EdgeId) -> GraphResult<(NodeId, NodeId)> {
        if !self.space.contains_edge(edge) {
            return Err(GraphError::InvalidInput(format!(
                "Inactive query edge {edge}"
            )));
        }
        let pair = self.pool.get_edge_endpoints(edge).ok_or_else(|| {
            GraphError::InvalidInput(format!("Missing query endpoints for edge {edge}"))
        })?;
        if !self.space.contains_node(pair.0) || !self.space.contains_node(pair.1) {
            return Err(GraphError::InvalidInput(format!(
                "Inactive query endpoint for edge {edge}"
            )));
        }
        Ok(pair)
    }
    fn directed(&self) -> bool {
        self.pool.graph_type() == crate::types::GraphType::Directed
    }
}
pub(crate) struct Frozen<'a>(pub(crate) &'a GraphView);
impl Source for Frozen<'_> {
    fn attribute(&self, id: usize, name: &AttrName, node: bool) -> GraphResult<Option<&AttrValue>> {
        let state = self.0.read_state().snapshot();
        let attrs = if node {
            &state.node_attributes
        } else {
            &state.edge_attributes
        };
        Ok(attrs.get(&id).and_then(|attrs| attrs.get(name)))
    }
    fn degree(&self, node: NodeId) -> GraphResult<usize> {
        self.0.degree(node)
    }
    fn incident_neighbor(&self, node: NodeId, neighbor: NodeId) -> GraphResult<bool> {
        for edge in self.0.selected_incident(node)? {
            let (s, t) = self.0.edge_endpoints(edge)?;
            if (s == node && t == neighbor) || (t == node && s == neighbor) {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn endpoints(&self, edge: EdgeId) -> GraphResult<(NodeId, NodeId)> {
        self.0.edge_endpoints(edge)
    }
    fn directed(&self) -> bool {
        self.0.is_directed()
    }
}
