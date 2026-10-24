//! Semantic kinds are versioned columnar properties, separate from storage identity.
use super::*;

/// Optional semantic labels on nodes (independent of base/meta entity_type).
pub const LABELS: &str = "labels";
/// Optional semantic relationship kind on edges.
pub const RELATIONSHIP_TYPE: &str = "relationship_type";

fn valid_name(value: &str) -> bool {
    !value.trim().is_empty()
}

pub(crate) fn validate_kind_property(
    name: &str,
    value: &AttrValue,
    is_node: bool,
) -> GraphResult<()> {
    if name == LABELS {
        if !is_node {
            return Err(GraphError::InvalidInput(
                "labels belongs to nodes; use relationship_type for edges".into(),
            ));
        }
        if matches!(value, AttrValue::FloatVec(values) if values.is_empty())
            || matches!(value, AttrValue::Float64Vec(values) if values.is_empty())
        {
            return Ok(());
        }
        let AttrValue::TextVec(labels) = value else {
            return Err(GraphError::InvalidInput(
                "labels must be a text vector; use an empty text vector to clear labels".into(),
            ));
        };
        let mut seen = HashSet::new();
        for label in labels {
            if !valid_name(label) || !seen.insert(label) {
                return Err(GraphError::InvalidInput(
                    "labels must contain distinct nonempty names".into(),
                ));
            }
        }
    }
    if name == RELATIONSHIP_TYPE {
        if is_node {
            return Err(GraphError::InvalidInput(
                "relationship_type belongs to edges; use labels for nodes".into(),
            ));
        }
        if !matches!(value, AttrValue::Null) && !value.get_text().is_some_and(|s| valid_name(&s)) {
            return Err(GraphError::InvalidInput(
                "relationship_type must be a nonempty text value or null to clear it".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn normalize_kind_property(
    name: &str,
    mut value: AttrValue,
    is_node: bool,
) -> GraphResult<AttrValue> {
    normalize_kind_property_in_place(name, &mut value, is_node)?;
    Ok(value)
}

pub(crate) fn normalize_kind_property_in_place(
    name: &str,
    value: &mut AttrValue,
    is_node: bool,
) -> GraphResult<()> {
    validate_kind_property(name, value, is_node)?;
    if name == LABELS {
        match value {
            AttrValue::TextVec(labels) => labels.sort_unstable(),
            AttrValue::FloatVec(_) | AttrValue::Float64Vec(_) => {
                *value = AttrValue::TextVec(Vec::new())
            }
            _ => {}
        }
    }
    Ok(())
}

impl Graph {
    /// Replace a node's semantic labels. Names are case sensitive and stored sorted.
    pub fn set_node_labels(&mut self, node: NodeId, labels: Vec<String>) -> GraphResult<()> {
        self.set_node_attr(node, LABELS.into(), AttrValue::TextVec(labels))
    }

    /// Read semantic labels; absence means no labels. Inactive IDs are errors.
    pub fn node_labels(&self, node: NodeId) -> GraphResult<Vec<String>> {
        match self.get_node_attr(node, &LABELS.into())? {
            None => Ok(Vec::new()),
            Some(AttrValue::TextVec(labels)) => Ok(labels),
            Some(_) => Err(GraphError::InvalidInput(
                "Stored labels are not a text vector".into(),
            )),
        }
    }

    /// Replace an edge's relationship kind; None explicitly clears its kind.
    pub fn set_edge_relationship_type(
        &mut self,
        edge: EdgeId,
        kind: Option<String>,
    ) -> GraphResult<()> {
        self.set_edge_attr(
            edge,
            RELATIONSHIP_TYPE.into(),
            kind.map(AttrValue::Text).unwrap_or(AttrValue::Null),
        )
    }

    /// Read the optional semantic relationship kind of an active edge.
    pub fn edge_relationship_type(&self, edge: EdgeId) -> GraphResult<Option<String>> {
        match self.get_edge_attr(edge, &RELATIONSHIP_TYPE.into())? {
            None | Some(AttrValue::Null) => Ok(None),
            Some(value) => value.get_text().map(Some).ok_or_else(|| {
                GraphError::InvalidInput("Stored relationship_type is not text".into())
            }),
        }
    }

    /// Find active nodes carrying a label. Linear column scan until item 18 indexes.
    pub fn nodes_with_label(&self, label: &str) -> GraphResult<Vec<NodeId>> {
        if !valid_name(label) {
            return Err(GraphError::InvalidInput("Label must be nonempty".into()));
        }
        let ids = self.node_ids();
        let values = self.get_nodes_attrs(&LABELS.into(), &ids)?;
        let mut found: Vec<_> = ids
            .into_iter()
            .zip(values)
            .filter_map(|(id, value)| match value {
                Some(AttrValue::TextVec(labels)) if labels.iter().any(|name| name == label) => {
                    Some(id)
                }
                _ => None,
            })
            .collect();
        found.sort_unstable();
        Ok(found)
    }

    /// Find active edges with this relationship type, preserving parallel IDs.
    pub fn edges_with_relationship_type(&self, kind: &str) -> GraphResult<Vec<EdgeId>> {
        if !valid_name(kind) {
            return Err(GraphError::InvalidInput(
                "Relationship type must be nonempty".into(),
            ));
        }
        let ids = self.edge_ids();
        let values = self.get_edges_attrs(&RELATIONSHIP_TYPE.into(), &ids)?;
        let mut found: Vec<_> = ids
            .into_iter()
            .zip(values)
            .filter_map(|(id, value)| {
                value
                    .filter(|v| v.get_text().as_deref() == Some(kind))
                    .map(|_| id)
            })
            .collect();
        found.sort_unstable();
        Ok(found)
    }
}
