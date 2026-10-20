//! Explicit schema auditing. Installation/enforcement across every write is item 9.
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
    /// Active node or edge carrying the offending value.
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
        self.validate_definition()?;
        let mut violations = Vec::new();
        let mut nodes = graph.node_ids();
        nodes.sort_unstable();
        for id in nodes {
            for label in graph.node_labels(id)? {
                if let Some(rules) = self.node_labels.get(&label) {
                    for (name, rule) in rules {
                        check(
                            rule,
                            graph.get_node_attr(id, name)?,
                            ExternalEntityRef::Node(id),
                            &label,
                            name,
                            &mut violations,
                        );
                    }
                }
            }
        }
        let mut edges = graph.edge_ids();
        edges.sort_unstable();
        for id in edges {
            if let Some(kind) = graph.edge_relationship_type(id)? {
                if let Some(rules) = self.relationship_types.get(&kind) {
                    for (name, rule) in rules {
                        check(
                            rule,
                            graph.get_edge_attr(id, name)?,
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
