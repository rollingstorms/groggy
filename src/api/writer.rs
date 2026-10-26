//! Opt-in checked direct writes over the single batch publication boundary.
use crate::{
    AttrName, AttrValue, BatchEdgeRef, BatchNodeRef, BatchResult, EdgeId, Graph, GraphBatch,
    GraphResult, GraphSchema, NodeId, StateId,
};
use std::collections::HashMap;

/// A checked direct-write session. Each successful call publishes one version.
/// Rust exclusively borrows the graph for this session; legacy APIs remain an
/// explicit compatibility path after the session is dropped.
pub struct GraphWriter<'a> {
    graph: &'a mut Graph,
    expected_state: StateId,
    author: String,
    schema: Option<&'a GraphSchema>,
    effective_at: Option<crate::TimestampMicros>,
}
impl Graph {
    /// Checked, automatically published alternative to the legacy infallible constructor.
    pub fn try_add_node(&mut self) -> GraphResult<NodeId> {
        self.writer("groggy").add_node(HashMap::new())
    }
    /// Add all nodes in one checked publication; an empty input is a no-op.
    pub fn try_add_nodes(&mut self, count: usize) -> GraphResult<Vec<NodeId>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        self.writer("groggy").add_nodes(count)
    }

    /// Start a publishing write session at the current head. No state is changed.
    pub fn writer(&mut self, author: impl Into<String>) -> GraphWriter<'_> {
        let expected_state = self.current_state_id();
        GraphWriter {
            graph: self,
            expected_state,
            author: author.into(),
            schema: None,
            effective_at: None,
        }
    }
}
impl<'a> GraphWriter<'a> {
    /// Mark this session's selected published state without publishing another delta.
    pub fn checkpoint(&mut self, message: impl Into<String>) -> GraphResult<StateId> {
        self.graph.checkpoint_with_schema(
            self.expected_state,
            message.into(),
            self.author.clone(),
            self.schema,
        )
    }

    /// Require each resulting graph to satisfy this policy during the session.
    pub fn with_schema(mut self, schema: &'a GraphSchema) -> Self {
        self.schema = Some(schema);
        self
    }
    /// Use an explicit effective-time default for session operations unless a batch supplies its own.
    pub fn with_effective_at(mut self, effective_at: Option<crate::TimestampMicros>) -> Self {
        self.effective_at = effective_at;
        self
    }
    /// Select the optimistic head expected for the next operation.
    pub fn expect_state(mut self, state: StateId) -> Self {
        self.expected_state = state;
        self
    }
    /// Last successfully published state (or the session's initial expected head).
    pub fn state_id(&self) -> StateId {
        self.expected_state
    }
    /// Read the graph without routing a mutation around the publication boundary.
    pub fn graph(&self) -> &Graph {
        self.graph
    }
    /// Publish a staged mixed operation, advancing the session only on success.
    pub fn apply_batch(
        &mut self,
        mut batch: GraphBatch,
        message: impl Into<String>,
    ) -> GraphResult<BatchResult> {
        if batch.effective_at.is_none() {
            batch.effective_at = self.effective_at;
        }
        let result = if let Some(schema) = self.schema {
            self.graph.apply_batch_with_schema(
                batch,
                self.expected_state,
                message.into(),
                self.author.clone(),
                schema,
            )
        } else {
            self.graph.apply_batch(
                batch,
                self.expected_state,
                message.into(),
                self.author.clone(),
            )
        }?;
        self.expected_state = result.state_id;
        Ok(result)
    }
    /// Create one node with all its initial properties in the same version.
    pub fn add_node(&mut self, attrs: HashMap<AttrName, AttrValue>) -> GraphResult<NodeId> {
        Ok(self.add_nodes_with_attrs(vec![attrs])?[0])
    }
    /// Create unlabelled nodes using the existing bulk allocator.
    pub fn add_nodes(&mut self, count: usize) -> GraphResult<Vec<NodeId>> {
        Ok(self
            .apply_batch(
                GraphBatch {
                    add_nodes: count,
                    ..Default::default()
                },
                "add_nodes",
            )?
            .added_nodes)
    }
    /// Create rows in one version; a rejected row consumes no IDs.
    pub fn add_nodes_with_attrs(
        &mut self,
        rows: Vec<HashMap<AttrName, AttrValue>>,
    ) -> GraphResult<Vec<NodeId>> {
        let mut batch = GraphBatch {
            add_nodes: rows.len(),
            ..Default::default()
        };
        for (position, attrs) in rows.into_iter().enumerate() {
            for (name, value) in attrs {
                batch
                    .node_attrs
                    .entry(name)
                    .or_default()
                    .push((BatchNodeRef::Added(position), value));
            }
        }
        Ok(self.apply_batch(batch, "add_nodes_with_attrs")?.added_nodes)
    }
    /// Create an edge and its properties in one version.
    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        attrs: HashMap<AttrName, AttrValue>,
    ) -> GraphResult<EdgeId> {
        Ok(self.add_edges_with_attrs(vec![(source, target, attrs)])?[0])
    }
    /// Create edge rows in one version, retaining parallel instance identities.
    pub fn add_edges_with_attrs(
        &mut self,
        rows: Vec<(NodeId, NodeId, HashMap<AttrName, AttrValue>)>,
    ) -> GraphResult<Vec<EdgeId>> {
        let mut batch = GraphBatch::default();
        for (position, (source, target, attrs)) in rows.into_iter().enumerate() {
            batch.add_edges.push((
                BatchNodeRef::Existing(source),
                BatchNodeRef::Existing(target),
            ));
            for (name, value) in attrs {
                batch
                    .edge_attrs
                    .entry(name)
                    .or_default()
                    .push((BatchEdgeRef::Added(position), value));
            }
        }
        Ok(self.apply_batch(batch, "add_edges_with_attrs")?.added_edges)
    }
    /// Publish one scalar property assignment.
    pub fn set_node_attr(
        &mut self,
        node: NodeId,
        name: AttrName,
        value: AttrValue,
    ) -> GraphResult<StateId> {
        self.set_node_attrs(HashMap::from([(name, vec![(node, value)])]))
    }
    /// Publish all node columns atomically in one version.
    pub fn set_node_attrs(
        &mut self,
        columns: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> GraphResult<StateId> {
        let node_attrs = columns
            .into_iter()
            .map(|(name, values)| {
                (
                    name,
                    values
                        .into_iter()
                        .map(|(id, value)| (BatchNodeRef::Existing(id), value))
                        .collect(),
                )
            })
            .collect();
        Ok(self
            .apply_batch(
                GraphBatch {
                    node_attrs,
                    ..Default::default()
                },
                "set_node_attrs",
            )?
            .state_id)
    }
    /// Publish one scalar edge property assignment.
    pub fn set_edge_attr(
        &mut self,
        edge: EdgeId,
        name: AttrName,
        value: AttrValue,
    ) -> GraphResult<StateId> {
        self.set_edge_attrs(HashMap::from([(name, vec![(edge, value)])]))
    }
    /// Publish all edge columns atomically in one version.
    pub fn set_edge_attrs(
        &mut self,
        columns: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> GraphResult<StateId> {
        let edge_attrs = columns
            .into_iter()
            .map(|(name, values)| {
                (
                    name,
                    values
                        .into_iter()
                        .map(|(id, value)| (BatchEdgeRef::Existing(id), value))
                        .collect(),
                )
            })
            .collect();
        Ok(self
            .apply_batch(
                GraphBatch {
                    edge_attrs,
                    ..Default::default()
                },
                "set_edge_attrs",
            )?
            .state_id)
    }
    /// Publish explicit absence of one node property.
    pub fn unset_node_attr(&mut self, id: NodeId, name: AttrName) -> GraphResult<StateId> {
        self.unset_node_attrs(HashMap::from([(name, vec![id])]))
    }
    /// Publish node property removals as one checked columnar batch.
    pub fn unset_node_attrs(
        &mut self,
        columns: HashMap<AttrName, Vec<NodeId>>,
    ) -> GraphResult<StateId> {
        let unset_node_attrs = columns
            .into_iter()
            .map(|(name, ids)| (name, ids.into_iter().map(BatchNodeRef::Existing).collect()))
            .collect();
        Ok(self
            .apply_batch(
                GraphBatch {
                    unset_node_attrs,
                    ..Default::default()
                },
                "unset_node_attrs",
            )?
            .state_id)
    }
    /// Publish explicit absence of one edge property.
    pub fn unset_edge_attr(&mut self, id: EdgeId, name: AttrName) -> GraphResult<StateId> {
        self.unset_edge_attrs(HashMap::from([(name, vec![id])]))
    }
    /// Publish edge property removals as one checked columnar batch.
    pub fn unset_edge_attrs(
        &mut self,
        columns: HashMap<AttrName, Vec<EdgeId>>,
    ) -> GraphResult<StateId> {
        let unset_edge_attrs = columns
            .into_iter()
            .map(|(name, ids)| (name, ids.into_iter().map(BatchEdgeRef::Existing).collect()))
            .collect();
        Ok(self
            .apply_batch(
                GraphBatch {
                    unset_edge_attrs,
                    ..Default::default()
                },
                "unset_edge_attrs",
            )?
            .state_id)
    }
    /// Publish node removals and their incident edge removals together.
    pub fn remove_nodes(&mut self, nodes: Vec<NodeId>) -> GraphResult<StateId> {
        Ok(self
            .apply_batch(
                GraphBatch {
                    remove_nodes: nodes.into_iter().map(BatchNodeRef::Existing).collect(),
                    ..Default::default()
                },
                "remove_nodes",
            )?
            .state_id)
    }
    /// Publish explicit edge removals together.
    pub fn remove_edges(&mut self, edges: Vec<EdgeId>) -> GraphResult<StateId> {
        Ok(self
            .apply_batch(
                GraphBatch {
                    remove_edges: edges.into_iter().map(BatchEdgeRef::Existing).collect(),
                    ..Default::default()
                },
                "remove_edges",
            )?
            .state_id)
    }
    /// Explicitly assign or rename a node ID and publish its index/property change together.
    pub fn set_node_external_id(&mut self, node: NodeId, name: String) -> GraphResult<StateId> {
        Ok(self
            .apply_batch(
                GraphBatch {
                    node_external_ids: vec![(BatchNodeRef::Existing(node), name)],
                    ..Default::default()
                },
                "set_node_external_id",
            )?
            .state_id)
    }
    /// Explicitly assign or rename an edge ID in one version.
    pub fn set_edge_external_id(&mut self, edge: EdgeId, name: String) -> GraphResult<StateId> {
        Ok(self
            .apply_batch(
                GraphBatch {
                    edge_external_ids: vec![(BatchEdgeRef::Existing(edge), name)],
                    ..Default::default()
                },
                "set_edge_external_id",
            )?
            .state_id)
    }
    fn publish_upsert_plan(&mut self, batch: GraphBatch, message: &str) -> GraphResult<()> {
        self.graph.validate_write_head(self.expected_state)?;
        if batch.has_operations() {
            self.apply_batch(batch, message)?;
        } else if let Some(schema) = self.schema {
            self.graph.check_schema_policy(schema)?;
        }
        Ok(())
    }
    /// Idempotent managed node upserts through the checked publication boundary.
    pub fn upsert_nodes(&mut self, rows: Vec<crate::NodeUpsert>) -> GraphResult<Vec<NodeId>> {
        self.graph.validate_write_head(self.expected_state)?;
        let (batch, ids) = self.graph.published_node_upserts(rows)?;
        self.publish_upsert_plan(batch, "upsert_nodes")?;
        Ok(ids)
    }
    /// Idempotent managed edge upserts, preserving endpoints and parallel identities.
    pub fn upsert_edges(&mut self, rows: Vec<crate::EdgeUpsert>) -> GraphResult<Vec<EdgeId>> {
        self.graph.validate_write_head(self.expected_state)?;
        let (batch, ids) = self.graph.published_edge_upserts(rows)?;
        self.publish_upsert_plan(batch, "upsert_edges")?;
        Ok(ids)
    }
}
