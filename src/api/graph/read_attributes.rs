//! Current property readers share liveness and retained-value validation.
use super::*;

type AttributeRows = HashMap<usize, HashMap<AttrName, AttrValue>>;

fn invalid_value(id: usize, name: &str, index: usize, is_node: bool) -> GraphError {
    GraphError::InvalidInput(format!(
        "Missing retained value for {} {id}, attribute {name}, index {index}",
        if is_node { "node" } else { "edge" },
    ))
}

fn value_at<'a>(
    pool: &'a GraphPool,
    id: usize,
    name: &AttrName,
    index: usize,
    is_node: bool,
) -> GraphResult<&'a AttrValue> {
    pool.get_attr_by_index(name, index, is_node)
        .ok_or_else(|| invalid_value(id, name, index, is_node))
}

impl Graph {
    fn attribute_entity_active(&self, id: usize, is_node: bool) -> bool {
        if is_node {
            self.space.contains_node(id)
        } else {
            self.space.contains_edge(id)
        }
    }

    fn validate_attribute_entity(
        &self,
        id: usize,
        is_node: bool,
        operation: &str,
    ) -> GraphResult<()> {
        if self.attribute_entity_active(id, is_node) {
            return Ok(());
        }
        let plural = operation == "bulk get attributes";
        if is_node {
            Err(GraphError::NodeNotFound {
                node_id: id,
                operation: operation.into(),
                suggestion: if plural {
                    "Check if all nodes exist with contains_node()"
                } else {
                    "Check if node exists with contains_node()"
                }
                .into(),
            })
        } else {
            Err(GraphError::EdgeNotFound {
                edge_id: id,
                operation: operation.into(),
                suggestion: if plural {
                    "Check if all edges exist with contains_edge()"
                } else {
                    "Check if edge exists with contains_edge()"
                }
                .into(),
            })
        }
    }

    fn current_attribute(
        &self,
        id: usize,
        name: &AttrName,
        is_node: bool,
    ) -> GraphResult<Option<AttrValue>> {
        self.validate_attribute_entity(id, is_node, "get attribute")?;
        let pool = self.pool.borrow();
        self.space
            .get_attr_index(id, name, is_node)
            .map(|index| value_at(&pool, id, name, index, is_node).cloned())
            .transpose()
    }

    fn current_attribute_row(
        &self,
        pool: &GraphPool,
        id: usize,
        is_node: bool,
        names: Option<&[AttrName]>,
    ) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let mut output = HashMap::new();
        if let Some(names) = names {
            // Requested bulk projections do not scan/clone unrelated column indices.
            for name in names {
                if let Some(index) = self.space.get_attr_index(id, name, is_node) {
                    output.insert(
                        name.clone(),
                        value_at(pool, id, name, index, is_node)?.clone(),
                    );
                }
            }
        } else {
            let indices = if is_node {
                self.space.get_node_attr_indices(id)
            } else {
                self.space.get_edge_attr_indices(id)
            };
            output.reserve(indices.len());
            for (name, index) in indices {
                let value = value_at(pool, id, &name, index, is_node)?.clone();
                output.insert(name, value);
            }
        }
        Ok(output)
    }

    fn current_attribute_rows(
        &self,
        ids: Vec<usize>,
        names: &[AttrName],
        is_node: bool,
    ) -> GraphResult<AttributeRows> {
        for &id in &ids {
            self.validate_attribute_entity(id, is_node, "bulk get attributes")?;
        }
        let pool = self.pool.borrow();
        ids.into_iter()
            .map(|id| {
                self.current_attribute_row(&pool, id, is_node, Some(names))
                    .map(|row| (id, row))
            })
            .collect()
    }

    fn current_attribute_column(
        &self,
        name: &AttrName,
        ids: &[usize],
        is_node: bool,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        let pool = self.pool.borrow();
        // Resolve the storage column once, preserving the attribute-first bulk path.
        let column = if is_node {
            pool.get_node_attribute_column(name)
        } else {
            pool.get_edge_attribute_column(name)
        };
        ids.iter()
            .map(|&id| {
                // Aligned column APIs retain their existing None-for-inactive policy.
                if !self.attribute_entity_active(id, is_node) {
                    return Ok(None);
                }
                self.space
                    .get_attr_index(id, name, is_node)
                    .map(|index| {
                        column
                            .and_then(|values| values.get(index))
                            .cloned()
                            .ok_or_else(|| invalid_value(id, name, index, is_node))
                    })
                    .transpose()
            })
            .collect()
    }

    /// Read one active node's property. Absence is None; a corrupt retained index is an error.
    pub fn get_node_attr(&self, node: NodeId, attr: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.current_attribute(node, attr, true)
    }

    /// Read one active edge's property. Absence is None; a corrupt retained index is an error.
    pub fn get_edge_attr(&self, edge: EdgeId, attr: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.current_attribute(edge, attr, false)
    }

    /// Read all current properties of an active node, validating every retained index.
    pub fn get_node_attrs(&self, node: NodeId) -> GraphResult<HashMap<AttrName, AttrValue>> {
        self.validate_attribute_entity(node, true, "get attributes")?;
        self.current_attribute_row(&self.pool.borrow(), node, true, None)
    }

    /// Read all current properties of an active edge, validating every retained index.
    pub fn get_edge_attrs(&self, edge: EdgeId) -> GraphResult<HashMap<AttrName, AttrValue>> {
        self.validate_attribute_entity(edge, false, "get attributes")?;
        self.current_attribute_row(&self.pool.borrow(), edge, false, None)
    }

    /// Project requested properties for active nodes. Missing keys are omitted; Null is retained.
    pub fn get_node_attrs_bulk(
        &self,
        nodes: Vec<NodeId>,
        attrs: Vec<AttrName>,
    ) -> GraphResult<HashMap<NodeId, HashMap<AttrName, AttrValue>>> {
        self.current_attribute_rows(nodes, &attrs, true)
    }

    /// Project requested properties for active edges. Missing keys are omitted; Null is retained.
    pub fn get_edge_attrs_bulk(
        &self,
        edges: Vec<EdgeId>,
        attrs: Vec<AttrName>,
    ) -> GraphResult<HashMap<EdgeId, HashMap<AttrName, AttrValue>>> {
        self.current_attribute_rows(edges, &attrs, false)
    }

    /// Read one node column in requested order, retaining duplicates and None for inactive IDs.
    pub fn get_nodes_attrs(
        &self,
        attr: &AttrName,
        requested_nodes: &[NodeId],
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        self.current_attribute_column(attr, requested_nodes, true)
    }

    /// Read one edge column in requested order, retaining duplicates and None for inactive IDs.
    pub fn get_edges_attrs(
        &self,
        attr: &AttrName,
        requested_edges: &[EdgeId],
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        self.current_attribute_column(attr, requested_edges, false)
    }

    /// Current property column for all active nodes in GraphSpace's ID order.
    pub fn _get_node_attribute_column(
        &self,
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        self.get_nodes_attrs(attr, &self.space.node_ids())
    }

    /// Current property column for all active edges in GraphSpace's ID order.
    pub fn _get_edge_attribute_column(
        &self,
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        self.get_edges_attrs(attr, &self.space.edge_ids())
    }

    /// Compatibility name for the aligned node column reader.
    pub fn _get_node_attributes_for_nodes(
        &self,
        node_ids: &[NodeId],
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        self.get_nodes_attrs(attr, node_ids)
    }

    /// Compatibility name for the aligned edge column reader.
    pub fn _get_edge_attributes_for_edges(
        &self,
        edge_ids: &[EdgeId],
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        self.get_edges_attrs(attr, edge_ids)
    }
}
