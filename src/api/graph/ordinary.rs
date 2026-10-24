//! Publication policy for fallible ordinary operations and compatibility imports.
use super::*;

impl Graph {
    /// Create an empty publishing graph. Use the fallible insert_nodes or
    /// try_add_node constructors; existing infallible constructors remain manual.
    pub fn new_publishing() -> Self {
        let mut graph = Self::new();
        graph.automatic_publication = true;
        graph
    }

    /// Enable checked publication for fallible ordinary operations. Infallible
    /// legacy Rust constructors remain manual; use insert_nodes/try_add_node.
    /// Changing policy requires a clean state, preventing silent mixed writers.
    pub fn set_automatic_publication(&mut self, enabled: bool) -> GraphResult<()> {
        self.validate_write_head(self.current_state_id())?;
        self.automatic_publication = enabled;
        Ok(())
    }
    /// Whether fallible ordinary mutations publish through the checked coordinator.
    pub fn automatic_publication(&self) -> bool {
        self.automatic_publication
    }

    /// Fallible node-count constructor respecting this graph's publication policy.
    pub fn insert_nodes(&mut self, count: usize) -> GraphResult<Vec<NodeId>> {
        if self.automatic_publication {
            self.try_add_nodes(count)
        } else {
            Ok(self.add_nodes(count))
        }
    }

    /// Create edges with all initial properties in one checked operation.
    pub fn add_edges_with_attrs(
        &mut self,
        rows: Vec<(NodeId, NodeId, HashMap<AttrName, AttrValue>)>,
    ) -> GraphResult<Vec<EdgeId>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        self.validate_new_edge_attr_rows(rows.iter().map(|(_, _, attrs)| attrs))?;
        if self.automatic_publication {
            return self.writer("groggy").add_edges_with_attrs(rows);
        }
        let pairs: Vec<_> = rows
            .iter()
            .map(|&(source, target, _)| (source, target))
            .collect();
        let edges = self.add_edges(&pairs)?;
        let mut columns: HashMap<AttrName, Vec<(EdgeId, AttrValue)>> = HashMap::new();
        for (&edge, (_, _, attrs)) in edges.iter().zip(rows) {
            for (name, value) in attrs {
                columns.entry(name).or_default().push((edge, value));
            }
        }
        self.set_edge_attrs(columns)?;
        Ok(edges)
    }

    /// Load an active export as one newly published imported state. This creates
    /// new acceptance clocks; it does not recover the source graph's history.
    pub fn load_publishing_from_path(path: &Path) -> GraphResult<Self> {
        Self::load_from_path(path)?.into_publishing()
    }

    /// Adopt a privately assembled graph as publishing, accepting its pending
    /// import as one state. This is an explicit conversion, not history recovery.
    pub fn into_publishing(mut self) -> GraphResult<Self> {
        let graph = &mut self;
        if graph.has_uncommitted_changes() {
            let now = crate::state::publication::now_micros()?;
            graph.commit_internal(
                "import active graph".into(),
                "groggy".into(),
                Some(crate::PublicationMetadata {
                    recorded_at: now,
                    effective_at: now,
                    change_times: Default::default(),
                }),
            )?;
        }
        graph.set_automatic_publication(true)?;
        Ok(self)
    }

    pub(super) fn publish_graph_copy(&mut self, other: &Graph) -> GraphResult<()> {
        let mut nodes = other.node_ids();
        nodes.sort_unstable();
        if nodes.is_empty() {
            return Ok(());
        }
        let positions: HashMap<_, _> = nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut batch = GraphBatch {
            add_nodes: nodes.len(),
            ..Default::default()
        };
        for (position, &node) in nodes.iter().enumerate() {
            for (name, value) in other.get_node_attrs(node)? {
                if name == EXTERNAL_ID {
                    return Err(GraphError::InvalidInput(
                        "add_graph cannot copy managed external IDs; use explicit upserts".into(),
                    ));
                }
                if name == "entity_type" {
                    if value.get_text().as_deref() != Some("base") {
                        return Err(GraphError::InvalidInput(
                            "add_graph requires explicit copying of pooled/meta references".into(),
                        ));
                    }
                    continue;
                }
                if matches!(value, AttrValue::SubgraphRef(_)) {
                    return Err(GraphError::InvalidInput(
                        "add_graph requires explicit copying of pooled/meta references".into(),
                    ));
                }
                batch
                    .node_attrs
                    .entry(name)
                    .or_default()
                    .push((crate::BatchNodeRef::Added(position), value));
            }
        }
        let mut edges = other.edge_ids();
        edges.sort_unstable();
        for (position, edge) in edges.into_iter().enumerate() {
            let (source, target) = other.edge_endpoints(edge)?;
            let source = *positions.get(&source).ok_or_else(|| {
                GraphError::InvalidInput("Copied edge has an inactive source".into())
            })?;
            let target = *positions.get(&target).ok_or_else(|| {
                GraphError::InvalidInput("Copied edge has an inactive target".into())
            })?;
            batch.add_edges.push((
                crate::BatchNodeRef::Added(source),
                crate::BatchNodeRef::Added(target),
            ));
            for (name, value) in other.get_edge_attrs(edge)? {
                if name == EXTERNAL_ID {
                    return Err(GraphError::InvalidInput(
                        "add_graph cannot copy managed external IDs; use explicit upserts".into(),
                    ));
                }
                if matches!(value, AttrValue::SubgraphRef(_)) {
                    return Err(GraphError::InvalidInput(
                        "add_graph requires explicit copying of pooled/meta references".into(),
                    ));
                }
                batch
                    .edge_attrs
                    .entry(name)
                    .or_default()
                    .push((crate::BatchEdgeRef::Added(position), value));
            }
        }
        self.apply_batch(
            batch,
            self.current_state_id(),
            "add_graph".into(),
            "groggy".into(),
        )
        .map(|_| ())
    }

    pub(super) fn publish_meta_node(&mut self, subgraph_id: SubgraphId) -> GraphResult<NodeId> {
        self.pool.borrow().get_subgraph(subgraph_id)?;
        let mut batch = GraphBatch {
            add_nodes: 1,
            ..Default::default()
        };
        batch.node_attrs.insert(
            "contains_subgraph".into(),
            vec![(
                crate::BatchNodeRef::Added(0),
                AttrValue::SubgraphRef(subgraph_id),
            )],
        );
        let mut prepared = self.prepare_publication(batch, self.current_state_id(), None)?;
        let node = self.pool.borrow().statistics().node_count;
        // Trusted system metadata, never accepted from user batch assignments.
        prepared.batch.node_attrs.insert(
            "entity_type".into(),
            vec![(node, AttrValue::Text("meta".into()))],
        );
        Ok(self
            .execute_publication(prepared, "create_meta_node".into(), "groggy".into())?
            .added_nodes[0])
    }
}
