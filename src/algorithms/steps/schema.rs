//! Step schema system for validation and type checking.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::algorithms::CostHint;

/// Describes the complete signature of a step primitive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepSchema {
    /// Unique step identifier (e.g., "core.add").
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Performance complexity hint.
    pub cost_hint: CostHint,
    /// Required and optional input variables.
    pub inputs: Vec<ParameterSchema>,
    /// Generated output variables.
    pub outputs: Vec<ParameterSchema>,
    /// Configuration parameters.
    pub params: Vec<ParameterSchema>,
    /// Tags for categorization (e.g., ["arithmetic", "binary"]).
    pub tags: Vec<String>,
}

/// Schema for a single parameter, input, or output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter name.
    pub name: String,
    /// Description of the parameter's purpose.
    pub description: String,
    /// Expected type(s).
    pub param_type: ParameterType,
    /// Whether this parameter is required.
    pub required: bool,
    /// Default value if omitted (JSON-serialized).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// Validation constraints.
    #[serde(default)]
    pub constraints: Vec<Constraint>,
}

/// Type descriptor for parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParameterType {
    /// Integer value.
    Int,
    /// Floating-point value.
    Float,
    /// Text string.
    Text,
    /// Boolean value.
    Bool,
    /// Node map variable (NodeId -> value).
    NodeMap { value_type: Box<ParameterType> },
    /// Edge map variable (EdgeId -> value).
    EdgeMap { value_type: Box<ParameterType> },
    /// Columnar node storage.
    NodeColumn { value_type: Box<ParameterType> },
    /// Scalar variable.
    Scalar { value_type: Box<ParameterType> },
    /// Temporal snapshot.
    Snapshot,
    /// Temporal index.
    TemporalIndex,
    /// Union of multiple types.
    Union { types: Vec<ParameterType> },
    /// Any type accepted.
    Any,
}

impl ParameterType {
    pub fn is_compatible(&self, other: &ParameterType) -> bool {
        match (self, other) {
            (ParameterType::Any, _) | (_, ParameterType::Any) => true,
            (ParameterType::Int, ParameterType::Int) => true,
            (ParameterType::Float, ParameterType::Float | ParameterType::Int) => true,
            (ParameterType::Text, ParameterType::Text) => true,
            (ParameterType::Bool, ParameterType::Bool) => true,
            (ParameterType::NodeMap { .. }, ParameterType::NodeMap { .. }) => true,
            (ParameterType::NodeColumn { .. }, ParameterType::NodeColumn { .. }) => true,
            (ParameterType::EdgeMap { .. }, ParameterType::EdgeMap { .. }) => true,
            (ParameterType::Scalar { .. }, ParameterType::Scalar { .. }) => true,
            (ParameterType::Snapshot, ParameterType::Snapshot) => true,
            (ParameterType::TemporalIndex, ParameterType::TemporalIndex) => true,
            (ParameterType::Union { types }, other) => types.iter().any(|t| t.is_compatible(other)),
            (other, ParameterType::Union { types }) => types.iter().any(|t| other.is_compatible(t)),
            _ => false,
        }
    }
}

impl fmt::Display for ParameterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParameterType::Int => write!(f, "int"),
            ParameterType::Float => write!(f, "float"),
            ParameterType::Text => write!(f, "text"),
            ParameterType::Bool => write!(f, "bool"),
            ParameterType::NodeMap { value_type } => write!(f, "NodeMap<{}>", value_type),
            ParameterType::EdgeMap { value_type } => write!(f, "EdgeMap<{}>", value_type),
            ParameterType::NodeColumn { value_type } => write!(f, "NodeColumn<{}>", value_type),
            ParameterType::Scalar { value_type } => write!(f, "Scalar<{}>", value_type),
            ParameterType::Snapshot => write!(f, "Snapshot"),
            ParameterType::TemporalIndex => write!(f, "TemporalIndex"),
            ParameterType::Union { types } => {
                write!(f, "Union<")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ">")
            }
            ParameterType::Any => write!(f, "any"),
        }
    }
}

/// Validation constraint for parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Constraint {
    /// Value must be within range.
    Range { min: f64, max: f64 },
    /// Value must be positive.
    Positive,
    /// Value must be non-negative.
    NonNegative,
    /// Text must match regex pattern.
    Pattern { pattern: String },
    /// Value must be one of the listed options.
    Enum { options: Vec<String> },
    /// Custom validation message.
    Custom { message: String },
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::Range { min, max } => write!(f, "range [{}, {}]", min, max),
            Constraint::Positive => write!(f, "must be positive"),
            Constraint::NonNegative => write!(f, "must be non-negative"),
            Constraint::Pattern { pattern } => write!(f, "must match pattern '{}'", pattern),
            Constraint::Enum { options } => write!(f, "must be one of: {}", options.join(", ")),
            Constraint::Custom { message } => write!(f, "{}", message),
        }
    }
}

/// Registry storing schemas for all registered steps.
#[derive(Default, Debug)]
pub struct SchemaRegistry {
    schemas: HashMap<String, StepSchema>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a schema for a step.
    pub fn register(&mut self, schema: StepSchema) {
        self.schemas.insert(schema.id.clone(), schema);
    }

    /// Get schema for a step by ID.
    pub fn get(&self, id: &str) -> Option<&StepSchema> {
        self.schemas.get(id)
    }

    /// Check if a step has a registered schema.
    pub fn contains(&self, id: &str) -> bool {
        self.schemas.contains_key(id)
    }

    /// List all registered step IDs.
    pub fn step_ids(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    /// Find steps by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&StepSchema> {
        self.schemas
            .values()
            .filter(|schema| schema.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Export all schemas as JSON for documentation.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.schemas)
    }
}

/// Builder for constructing step schemas with fluent API.
pub struct StepSchemaBuilder {
    id: String,
    description: String,
    cost_hint: CostHint,
    inputs: Vec<ParameterSchema>,
    outputs: Vec<ParameterSchema>,
    params: Vec<ParameterSchema>,
    tags: Vec<String>,
}

impl StepSchemaBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: String::new(),
            cost_hint: CostHint::Unknown,
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn cost_hint(mut self, hint: CostHint) -> Self {
        self.cost_hint = hint;
        self
    }

    pub fn input(mut self, param: ParameterSchema) -> Self {
        self.inputs.push(param);
        self
    }

    pub fn output(mut self, param: ParameterSchema) -> Self {
        self.outputs.push(param);
        self
    }

    pub fn param(mut self, param: ParameterSchema) -> Self {
        self.params.push(param);
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn build(self) -> StepSchema {
        StepSchema {
            id: self.id,
            description: self.description,
            cost_hint: self.cost_hint,
            inputs: self.inputs,
            outputs: self.outputs,
            params: self.params,
            tags: self.tags,
        }
    }
}

/// Builder for parameter schemas.
pub struct ParameterSchemaBuilder {
    name: String,
    description: String,
    param_type: ParameterType,
    required: bool,
    default: Option<serde_json::Value>,
    constraints: Vec<Constraint>,
}

impl ParameterSchemaBuilder {
    pub fn new(name: impl Into<String>, param_type: ParameterType) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            param_type,
            required: false,
            default: None,
            constraints: Vec::new(),
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn default(mut self, value: serde_json::Value) -> Self {
        self.default = Some(value);
        self.required = false;
        self
    }

    pub fn constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn build(self) -> ParameterSchema {
        ParameterSchema {
            name: self.name,
            description: self.description,
            param_type: self.param_type,
            required: self.required,
            default: self.default,
            constraints: self.constraints,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_type_compatibility() {
        assert!(ParameterType::Float.is_compatible(&ParameterType::Int));
        assert!(!ParameterType::Int.is_compatible(&ParameterType::Float));
        assert!(ParameterType::Any.is_compatible(&ParameterType::Int));
        assert!(ParameterType::Int.is_compatible(&ParameterType::Any));
    }

    #[test]
    fn build_schema_with_fluent_api() {
        let schema = StepSchemaBuilder::new("core.add")
            .description("Add two values")
            .cost_hint(CostHint::Linear)
            .input(
                ParameterSchemaBuilder::new("left", ParameterType::Float)
                    .required()
                    .build(),
            )
            .input(
                ParameterSchemaBuilder::new("right", ParameterType::Float)
                    .required()
                    .build(),
            )
            .output(
                ParameterSchemaBuilder::new("target", ParameterType::Float)
                    .required()
                    .build(),
            )
            .tag("arithmetic")
            .build();

        assert_eq!(schema.id, "core.add");
        assert_eq!(schema.inputs.len(), 2);
        assert_eq!(schema.outputs.len(), 1);
        assert_eq!(schema.tags, vec!["arithmetic"]);
    }

    #[test]
    fn schema_registry_stores_and_retrieves() {
        let mut registry = SchemaRegistry::new();
        let schema = StepSchemaBuilder::new("test.step")
            .description("Test step")
            .build();

        registry.register(schema);
        assert!(registry.contains("test.step"));
        assert_eq!(registry.get("test.step").unwrap().id, "test.step");
    }
}
