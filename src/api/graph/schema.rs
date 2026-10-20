//! Shared schema validation for active graphs and proposed batch states.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Logical property families; compact/compressed representations share a family.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyKind {
    Any,
    Int,
    Float,
    Bool,
    Text,
    IntVec,
    FloatVec,
    BoolVec,
    TextVec,
    Bytes,
    Json,
    NodeArray,
    EdgeArray,
    SubgraphRef,
}
impl PropertyKind {
    fn matches(self, value: &AttrValue) -> bool {
        use AttrValue::*;
        match self {
            Self::Any => true,
            Self::Int => matches!(value, Int(_) | SmallInt(_)),
            Self::Float => matches!(value, Float(_)),
            Self::Bool => matches!(value, Bool(_)),
            Self::Text => matches!(value, Text(_) | CompactText(_) | CompressedText(_)),
            Self::IntVec => matches!(value, IntVec(_)),
            Self::FloatVec => matches!(value, FloatVec(_) | CompressedFloatVec(_)),
            Self::BoolVec => matches!(value, BoolVec(_)),
            Self::TextVec => matches!(value, TextVec(_)),
            Self::Bytes => matches!(value, Bytes(_)),
            Self::Json => matches!(value, Json(_)),
            Self::NodeArray => matches!(value, NodeArray(_)),
            Self::EdgeArray => matches!(value, EdgeArray(_)),
            Self::SubgraphRef => matches!(value, SubgraphRef(_)),
        }
    }
}
fn required_default() -> bool {
    true
}

/// Required means present; nullable independently controls explicit Null values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropertyConstraint {
    /// Logical family accepted for a non-null value.
    #[serde(rename = "type")]
    pub value_type: PropertyKind,
    /// Missing properties violate the rule when true (default).
    #[serde(default = "required_default")]
    pub required: bool,
    /// Explicit Null is accepted only when true (default false).
    #[serde(default)]
    pub nullable: bool,
}

/// Rules apply to each matching label or relationship type. Other properties and
/// unlabeled/untyped entities remain flexible. Multiple labels apply all rules.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSchema {
    /// Semantic node label -> property name -> constraint.
    #[serde(default)]
    pub node_labels: BTreeMap<String, BTreeMap<AttrName, PropertyConstraint>>,
    /// Relationship type -> property name -> constraint.
    #[serde(default)]
    pub relationship_types: BTreeMap<String, BTreeMap<AttrName, PropertyConstraint>>,
}

/// One deterministic, structured schema finding; auditing never mutates the graph.
#[derive(Debug, Clone)]
pub struct SchemaViolation {
    /// Active or proposed node/edge carrying the offending value.
    /// Proposed IDs are not allocated when validation rejects the batch.
    pub entity: ExternalEntityRef,
    /// Node label or relationship type selecting this rule.
    pub selector: String,
    /// Property being checked.
    pub property: AttrName,
    /// Missing, null, or type mismatch detail.
    pub message: String,
}

impl GraphSchema {
    /// Parse a strict schema definition; unknown fields/types fail explicitly.
    pub fn from_json(json: &str) -> GraphResult<Self> {
        let schema: Self = serde_json::from_str(json)
            .map_err(|error| GraphError::InvalidInput(format!("Invalid schema: {error}")))?;
        schema.validate_definition()?;
        Ok(schema)
    }

    fn validate_definition(&self) -> GraphResult<()> {
        if self
            .node_labels
            .keys()
            .chain(self.relationship_types.keys())
            .any(|name| name.trim().is_empty())
        {
            return Err(GraphError::InvalidInput(
                "Schema label/type selectors must be nonempty".into(),
            ));
        }
        Ok(())
    }

    /// Audit active graph properties against all applicable rules, in stable order.
    pub fn validate_graph(&self, graph: &Graph) -> GraphResult<Vec<SchemaViolation>> {
        self.validate_state(graph.node_ids(), graph.edge_ids(), |is_node, id, name| {
            if is_node {
                graph.get_node_attr(id, &name.into())
            } else {
                graph.get_edge_attr(id, &name.into())
            }
        })
    }

    // One validator serves audit and pre-publication enforcement, so absence,
    // nullability and logical families have identical semantics in both paths.
    fn validate_state<F>(
        &self,
        mut nodes: Vec<NodeId>,
        mut edges: Vec<EdgeId>,
        mut property: F,
    ) -> GraphResult<Vec<SchemaViolation>>
    where
        F: FnMut(bool, usize, &str) -> GraphResult<Option<AttrValue>>,
    {
        self.validate_definition()?;
        let mut violations = Vec::new();
        nodes.sort_unstable();
        for id in nodes {
            let labels = match property(true, id, LABELS)? {
                None => Vec::new(),
                Some(AttrValue::TextVec(labels)) => labels,
                Some(_) => {
                    return Err(GraphError::InvalidInput(
                        "Stored labels are not a text vector".into(),
                    ))
                }
            };
            for label in labels {
                if let Some(rules) = self.node_labels.get(&label) {
                    for (name, rule) in rules {
                        check(
                            rule,
                            property(true, id, name)?,
                            ExternalEntityRef::Node(id),
                            &label,
                            name,
                            &mut violations,
                        );
                    }
                }
            }
        }
        edges.sort_unstable();
        for id in edges {
            let kind = match property(false, id, RELATIONSHIP_TYPE)? {
                None | Some(AttrValue::Null) => None,
                Some(value) => Some(value.get_text().ok_or_else(|| {
                    GraphError::InvalidInput("Stored relationship_type is not text".into())
                })?),
            };
            if let Some(kind) = kind {
                if let Some(rules) = self.relationship_types.get(&kind) {
                    for (name, rule) in rules {
                        check(
                            rule,
                            property(false, id, name)?,
                            ExternalEntityRef::Edge(id),
                            &kind,
                            name,
                            &mut violations,
                        );
                    }
                }
            }
        }
        Ok(violations)
    }

    /// Validate a prepared overlay without allocating entities, touching property
    /// pools, recording changes or reconstructing/cloning the entire graph.
    pub(super) fn validate_proposed(
        &self,
        graph: &Graph,
        batch: &crate::api::batch::PreparedBatch,
        first_node: NodeId,
        first_edge: EdgeId,
    ) -> GraphResult<Vec<SchemaViolation>> {
        let removed_nodes: HashSet<_> = batch.remove_nodes.iter().copied().collect();
        let removed_edges: HashSet<_> = batch.remove_edges.iter().copied().collect();
        let added_nodes = first_node..first_node + batch.add_nodes;
        let added_edges = first_edge..first_edge + batch.add_edges.len();
        let mut nodes = graph.node_ids();
        nodes.extend(added_nodes.clone());
        nodes.retain(|id| !removed_nodes.contains(id));
        let mut edges = Vec::new();
        for id in graph.edge_ids() {
            if removed_edges.contains(&id) {
                continue;
            }
            let (source, target) = graph.edge_endpoints(id)?;
            if !removed_nodes.contains(&source) && !removed_nodes.contains(&target) {
                edges.push(id);
            }
        }
        for (offset, &(source, target)) in batch.add_edges.iter().enumerate() {
            let id = first_edge + offset;
            if !removed_edges.contains(&id)
                && !removed_nodes.contains(&source)
                && !removed_nodes.contains(&target)
            {
                edges.push(id);
            }
        }
        let node_updates = assignment_index(&batch.node_attrs);
        let edge_updates = assignment_index(&batch.edge_attrs);
        self.validate_state(nodes, edges, |is_node, id, name| {
            let updates = if is_node {
                &node_updates
            } else {
                &edge_updates
            };
            if let Some(value) = updates.get(&id).and_then(|row| row.get(name)) {
                return Ok(Some((**value).clone()));
            }
            let added = if is_node {
                added_nodes.contains(&id)
            } else {
                added_edges.contains(&id)
            };
            if added {
                return Ok(if is_node && name == "entity_type" {
                    Some(AttrValue::Text("base".into()))
                } else {
                    None
                });
            }
            if is_node {
                graph.get_node_attr(id, &name.into())
            } else {
                graph.get_edge_attr(id, &name.into())
            }
        })
    }
}
// Index only staged assignments; existing values remain in their current columns.
fn assignment_index(
    columns: &HashMap<AttrName, Vec<(usize, AttrValue)>>,
) -> HashMap<usize, HashMap<&str, &AttrValue>> {
    let mut by_entity: HashMap<_, HashMap<_, _>> = HashMap::new();
    for (name, values) in columns {
        for (id, value) in values {
            by_entity
                .entry(*id)
                .or_default()
                .insert(name.as_str(), value);
        }
    }
    by_entity
}

fn check(
    rule: &PropertyConstraint,
    value: Option<AttrValue>,
    entity: ExternalEntityRef,
    selector: &str,
    name: &str,
    violations: &mut Vec<SchemaViolation>,
) {
    let message = match value {
        None if rule.required => Some("required property is absent".into()),
        None => None,
        Some(AttrValue::Null) if !rule.nullable => Some("explicit null is not allowed".into()),
        Some(AttrValue::Null) => None,
        Some(value) if !rule.value_type.matches(&value) => Some(format!(
            "expected {:?}, found {}",
            rule.value_type,
            value.type_name()
        )),
        Some(_) => None,
    };
    if let Some(message) = message {
        violations.push(SchemaViolation {
            entity,
            selector: selector.into(),
            property: name.into(),
            message,
        });
    }
}
impl Graph {
    /// Audit an explicit schema without installing a write policy or changing state.
    pub fn validate_schema(&self, schema: &GraphSchema) -> GraphResult<Vec<SchemaViolation>> {
        schema.validate_graph(self)
    }
}

pub(super) fn reject_violations(violations: Vec<SchemaViolation>) -> GraphResult<()> {
    if violations.is_empty() {
        return Ok(());
    }
    let details: Vec<_> = violations
        .into_iter()
        .map(|violation| {
            let (kind, id) = match violation.entity {
                ExternalEntityRef::Node(id) => ("node", id),
                ExternalEntityRef::Edge(id) => ("edge", id),
            };
            format!(
                "{kind} {id} [{}].{}: {}",
                violation.selector, violation.property, violation.message
            )
        })
        .collect();
    Err(GraphError::InvalidInput(format!(
        "Schema rejected batch: {}",
        details.join("; ")
    )))
}
impl Graph {
    pub(crate) fn check_schema_at(&self, schema: &GraphSchema, state: StateId) -> GraphResult<()> {
        let snapshot = self.history.reconstruct_state_at(state)?;
        let violations = schema.validate_state(
            snapshot.active_nodes.clone(),
            snapshot.edges.keys().copied().collect(),
            |is_node, id, name| {
                let attributes = if is_node {
                    &snapshot.node_attributes
                } else {
                    &snapshot.edge_attributes
                };
                Ok(attributes.get(&id).and_then(|row| row.get(name)).cloned())
            },
        )?;
        reject_violations(violations)
    }

    pub(crate) fn check_schema_policy(&self, schema: &GraphSchema) -> GraphResult<()> {
        reject_violations(self.validate_schema(schema)?)
    }
}
