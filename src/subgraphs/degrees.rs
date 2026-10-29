//! Immutable topology-only inputs for connected degree adapters.
use super::Subgraph;
use crate::{errors::GraphResult, state::topology::Csr, GraphError, NodeId};
use std::sync::Arc;

/// Edge-instance counting direction. Undirected inputs use incident counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegreeMode {
    /// Every incident instance, including a self-loop once.
    Incident,
    /// Incoming directed instances.
    Incoming,
    /// Outgoing directed instances.
    Outgoing,
}

/// Prepared degree rows without an owner, properties, or writable graph handle.
/// This object can be computed on a worker after preparation on the owner thread.
#[derive(Clone)]
pub struct SubgraphDegreeRead {
    allowed: Arc<[NodeId]>,
    nodes: Arc<[NodeId]>,
    primary: Arc<Csr>,
    outgoing: Option<Arc<Csr>>,
}

impl SubgraphDegreeRead {
    /// Count requested IDs in caller order, preserving repetitions. Rejects the
    /// whole request when an ID is outside the original selected membership.
    pub fn counts(&self, nodes: &[NodeId]) -> GraphResult<Vec<usize>> {
        nodes
            .iter()
            .map(|&node| {
                if self.allowed.binary_search(&node).is_err() {
                    return Err(GraphError::NodeNotFound {
                        node_id: node,
                        operation: "selected degree".into(),
                        suggestion: "Node must belong to the selected graph".into(),
                    });
                }
                let index = self.nodes.binary_search(&node).map_err(|_| {
                    GraphError::InvalidInput("Degree input has inconsistent membership".into())
                })?;
                let incident = self.primary.neighbors(index).len();
                Ok(if let Some(outgoing) = &self.outgoing {
                    let row = outgoing.neighbors(index);
                    // Incident = incoming + outgoing - loops; loops count once
                    // in each directional result as well as in total degree.
                    incident - row.len() + row.iter().filter(|&&target| target == index).count()
                } else {
                    incident
                })
            })
            .collect()
    }
}

impl Subgraph {
    /// Prepare immutable degree inputs from a validated current selection.
    /// `full_graph` expands the counted edge domain to the current owner while
    /// retaining the selection as the set of addressable result IDs. It never
    /// resurrects a stale selection or captures properties/history.
    pub fn degree_read(
        &self,
        mode: DegreeMode,
        full_graph: bool,
    ) -> GraphResult<SubgraphDegreeRead> {
        let owner = self.graph();
        let graph = owner.borrow();
        self.validate_query_scope(&graph)?;
        let undirected = graph.is_undirected();
        let allowed = self.ordered_nodes();
        let full = full_graph.then(|| {
            Self::new(
                owner.clone(),
                graph.node_ids().into_iter().collect(),
                graph.edge_ids().into_iter().collect(),
                "degree_owner_scope".into(),
            )
        });
        drop(graph);
        let source = full.as_ref().unwrap_or(self);
        let incoming = !undirected && mode == DegreeMode::Incoming;
        let primary =
            source.get_or_build_csr_internal(undirected || mode != DegreeMode::Outgoing)?;
        let outgoing = if incoming {
            Some(source.get_or_build_csr_internal(false)?)
        } else {
            None
        };
        Ok(SubgraphDegreeRead {
            allowed,
            nodes: source.ordered_nodes(),
            primary,
            outgoing,
        })
    }
}
