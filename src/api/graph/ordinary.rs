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

    pub(super) fn copy_graph_active(&mut self, other: &Graph) -> GraphResult<()> {
        // Validate every active index and endpoint before allocating destination
        // IDs. Unlike permissive legacy row readers, this cannot drop corruption.
        let snapshot = other
            .space
            .capture_read_snapshot(&other.pool.borrow(), other.current_state_id())?;
        let mut nodes = snapshot.active_nodes;
        nodes.sort_unstable();
        if nodes.is_empty() {
            return Ok(());
        }
        let mut edges: Vec<_> = snapshot.edges.into_iter().collect();
        edges.sort_unstable_by_key(|&(id, _)| id);
        let counts = self.pool.borrow().statistics();
        if counts.node_count.checked_add(nodes.len()).is_none()
            || counts.edge_count.checked_add(edges.len()).is_none()
        {
            return Err(GraphError::InvalidInput(
                "Graph copy ID allocation would overflow".into(),
            ));
        }
        let node_ids: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, counts.node_count + i))
            .collect();
        let edge_ids: HashMap<_, _> = edges
            .iter()
            .enumerate()
            .map(|(i, &(id, _))| (id, counts.edge_count + i))
            .collect();
        let mapped = |ids: &HashMap<usize, usize>, id: usize| {
            ids.get(&id).copied().ok_or_else(|| {
                GraphError::InvalidInput(format!("Graph copy lost active entity {id}"))
            })
        };
        let mut batch = GraphBatch {
            add_nodes: nodes.len(),
            ..Default::default()
        };
        for &(_, (source, target)) in &edges {
            // capture_read_snapshot already verified both active endpoints.
            batch.add_edges.push((
                crate::BatchNodeRef::Added(mapped(&node_ids, source)? - counts.node_count),
                crate::BatchNodeRef::Added(mapped(&node_ids, target)? - counts.node_count),
            ));
        }
        for (is_node, attributes) in [
            (true, snapshot.node_attributes),
            (false, snapshot.edge_attributes),
        ] {
            for (id, attrs) in attributes {
                for (name, value) in attrs {
                    if name == EXTERNAL_ID {
                        return Err(GraphError::InvalidInput(
                            "add_graph cannot copy managed external IDs; use explicit upserts"
                                .into(),
                        ));
                    }
                    if is_node && name == "entity_type" {
                        if value.get_text().as_deref() != Some("base") {
                            return Err(GraphError::InvalidInput(
                                "add_graph requires explicit copying of pooled/meta references"
                                    .into(),
                            ));
                        }
                        continue;
                    }
                    if name == "contains_subgraph" && !matches!(value, AttrValue::Null) {
                        return Err(GraphError::InvalidInput(
                            "add_graph requires explicit copying of pooled/meta references".into(),
                        ));
                    }
                    let value = crate::storage::value_references::remap(
                        value,
                        &node_ids,
                        &edge_ids,
                        "copied active graph",
                    )
                    .map_err(|error| {
                        GraphError::InvalidInput(format!(
                            "add_graph {} {id}.{name}: {error}",
                            if is_node { "node" } else { "edge" },
                        ))
                    })?;
                    if is_node {
                        batch.node_attrs.entry(name).or_default().push((
                            crate::BatchNodeRef::Added(mapped(&node_ids, id)? - counts.node_count),
                            value,
                        ));
                    } else {
                        batch.edge_attrs.entry(name).or_default().push((
                            crate::BatchEdgeRef::Added(mapped(&edge_ids, id)? - counts.edge_count),
                            value,
                        ));
                    }
                }
            }
        }
        if self.automatic_publication {
            self.apply_batch(
                batch,
                self.current_state_id(),
                "add_graph".into(),
                "groggy".into(),
            )
            .map(|_| ())
        } else {
            let prepared = batch.prepare(
                |id| self.has_node(id),
                |id| self.has_edge(id),
                |id| self.pool.borrow().get_edge_endpoints(id),
                counts.node_count,
                counts.edge_count,
            )?;
            self.execute_prepared_mutation(prepared).map(|_| ())
        }
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
