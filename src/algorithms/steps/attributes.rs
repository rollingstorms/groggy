//! Attribute loading and attaching step primitives.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::traits::SubgraphOperations;
use crate::types::{AttrName, EdgeId, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{AtomicStepEffect, Step, StepMetadata, StepRegistry, StepScope};

/// Load an existing node attribute into a step variable.
pub struct LoadNodeAttrStep {
    target: String,
    attr: AttrName,
    default: AlgorithmParamValue,
}

impl LoadNodeAttrStep {
    pub fn new(target: impl Into<String>, attr: AttrName, default: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            attr,
            default,
        }
    }
}

impl Step for LoadNodeAttrStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.load_node_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Load a node attribute into a variable".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().node_set().len());
        for &node in scope.node_ids() {
            let value = match scope
                .node_attribute(ctx, node, &self.attr)?
                .and_then(AlgorithmParamValue::from_attr_value)
            {
                Some(attr_value) => attr_value,
                None if self.attr.as_str() == "entity_id" || self.attr.as_str() == "node_id" => {
                    AlgorithmParamValue::Int(node as i64)
                }
                None => self.default.clone(),
            };
            map.insert(node, value);
        }
        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Attach node attributes using a node map variable.
pub struct AttachNodeAttrStep {
    source: String,
    attr: AttrName,
    effective_at: Option<crate::TimestampMicros>,
}

impl AttachNodeAttrStep {
    pub fn new(source: impl Into<String>, attr: AttrName) -> Self {
        Self {
            source: source.into(),
            attr,
            effective_at: None,
        }
    }

    /// Set an explicit effective time; requires persistent atomic execution.
    pub fn with_effective_at(mut self, at: Option<crate::TimestampMicros>) -> Self {
        self.effective_at = at;
        self
    }
}

impl Step for AttachNodeAttrStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        if self.effective_at.is_some() {
            AtomicStepEffect::StagedProperties
        } else {
            AtomicStepEffect::PendingProperties
        }
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.attach_node_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Persist a node map as a graph attribute".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if let Ok(column) = scope.variables().node_column(&self.source) {
            let mut values = Vec::with_capacity(column.nodes().len());
            for (node, value) in column.iter() {
                if ctx.is_cancelled() {
                    return Err(anyhow!("attach_node_attr cancelled"));
                }
                if let Some(attr_value) = value.as_attr_value() {
                    values.push((node, attr_value));
                } else {
                    return Err(anyhow!(
                        "variable '{}' contains unsupported attribute type",
                        self.source
                    ));
                }
            }

            if let Some(at) = self.effective_at {
                return scope.attach_attrs_at(
                    ctx,
                    true,
                    HashMap::from([(self.attr.clone(), values)]),
                    at,
                );
            }
            return scope
                .attach_node_column(ctx, self.attr.clone(), values)
                .map_err(|err| anyhow!("failed to set node attributes: {err}"));
        }

        let map = scope.variables().node_map(&self.source)?;
        let mut entries: Vec<(NodeId, &AlgorithmParamValue)> =
            map.iter().map(|(node, value)| (*node, value)).collect();
        entries.sort_unstable_by_key(|(node, _)| *node);
        let mut values = Vec::with_capacity(entries.len());

        for (node, value) in entries {
            if ctx.is_cancelled() {
                return Err(anyhow!("attach_node_attr cancelled"));
            }
            if let Some(attr_value) = value.as_attr_value() {
                values.push((node, attr_value));
            } else {
                return Err(anyhow!(
                    "variable '{}' contains unsupported attribute type",
                    self.source
                ));
            }
        }

        let mut attrs = HashMap::new();
        attrs.insert(self.attr.clone(), values);
        if let Some(at) = self.effective_at {
            return scope.attach_attrs_at(ctx, true, attrs, at);
        }
        scope
            .attach_node_attrs(ctx, attrs)
            .map_err(|err| anyhow!("failed to set node attributes: {err}"))?;
        Ok(())
    }
}

/// Load an existing edge attribute into a variable.
pub struct LoadEdgeAttrStep {
    target: String,
    attr: AttrName,
    default: AlgorithmParamValue,
}

impl LoadEdgeAttrStep {
    pub fn new(target: impl Into<String>, attr: AttrName, default: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            attr,
            default,
        }
    }
}

impl Step for LoadEdgeAttrStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.load_edge_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Load an edge attribute into a variable".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().edge_set().len());
        for &edge in scope.edge_ids() {
            let value = scope
                .edge_attribute(ctx, edge, &self.attr)?
                .and_then(AlgorithmParamValue::from_attr_value)
                .unwrap_or_else(|| self.default.clone());
            map.insert(edge, value);
        }
        scope.variables_mut().set_edge_map(self.target.clone(), map);
        Ok(())
    }
}

/// Attach edge attributes using an edge map variable.
pub struct AttachEdgeAttrStep {
    source: String,
    attr: AttrName,
    effective_at: Option<crate::TimestampMicros>,
}

impl AttachEdgeAttrStep {
    pub fn new(source: impl Into<String>, attr: AttrName) -> Self {
        Self {
            source: source.into(),
            attr,
            effective_at: None,
        }
    }

    /// Set an explicit effective time; requires persistent atomic execution.
    pub fn with_effective_at(mut self, at: Option<crate::TimestampMicros>) -> Self {
        self.effective_at = at;
        self
    }
}

impl Step for AttachEdgeAttrStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        if self.effective_at.is_some() {
            AtomicStepEffect::StagedProperties
        } else {
            AtomicStepEffect::PendingProperties
        }
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.attach_edge_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Persist an edge map as an attribute".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().edge_map(&self.source)?;
        let mut entries: Vec<(EdgeId, &AlgorithmParamValue)> =
            map.iter().map(|(edge, value)| (*edge, value)).collect();
        entries.sort_unstable_by_key(|(edge, _)| *edge);
        let mut attrs = HashMap::new();
        let mut values = Vec::with_capacity(entries.len());

        for (edge, value) in entries {
            if let Some(attr_value) = value.as_attr_value() {
                values.push((edge, attr_value));
            } else {
                return Err(anyhow!(
                    "variable '{}' contains unsupported attribute type",
                    self.source
                ));
            }
        }

        attrs.insert(self.attr.clone(), values);
        if let Some(at) = self.effective_at {
            return scope.attach_attrs_at(ctx, false, attrs, at);
        }
        scope
            .attach_edge_attrs(ctx, attrs)
            .map_err(|err| anyhow!("failed to set edge attributes: {err}"))?;
        Ok(())
    }
}

/// Scale edge weights by a constant factor.
pub struct EdgeWeightScaleStep {
    attr: AttrName,
    factor: f64,
    target: String,
}

impl EdgeWeightScaleStep {
    pub fn new(attr: AttrName, factor: f64, target: impl Into<String>) -> Self {
        Self {
            attr,
            factor,
            target: target.into(),
        }
    }
}

impl Step for EdgeWeightScaleStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.edge_weight_scale"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Scale edge weights by factor {}", self.factor),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().edge_set().len());

        for &edge in scope.edge_ids() {
            if ctx.is_cancelled() {
                return Err(anyhow!("edge_weight_scale cancelled"));
            }

            let weight = scope
                .edge_attribute(ctx, edge, &self.attr)?
                .and_then(AlgorithmParamValue::from_attr_value)
                .unwrap_or(AlgorithmParamValue::Float(1.0));

            let scaled = match weight {
                AlgorithmParamValue::Float(v) => AlgorithmParamValue::Float(v * self.factor),
                AlgorithmParamValue::Int(v) => AlgorithmParamValue::Float(v as f64 * self.factor),
                other => other, // Pass through non-numeric values
            };

            map.insert(edge, scaled);
        }

        scope.variables_mut().set_edge_map(self.target.clone(), map);
        Ok(())
    }
}

struct AttachTableStep {
    source: String,
    columns: Vec<(AttrName, AttrName)>,
    effective_at: Option<crate::TimestampMicros>,
}

impl Step for AttachTableStep {
    fn id(&self) -> &'static str {
        "table.attach"
    }
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::StagedProperties
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("table attachment cancelled"));
        }
        let table = scope.variables().table(&self.source)?;
        let pending = ctx
            .property_batch
            .as_mut()
            .ok_or_else(|| anyhow!("table.attach requires a persistent atomic builder"))?;
        pending.attach_table(scope.subgraph(), table, &self.columns, self.effective_at)?;
        Ok(())
    }
}

pub(super) fn register_table_attachment(registry: &StepRegistry) -> Result<()> {
    registry.register(
        "table.attach",
        StepMetadata {
            id: "table.attach".into(),
            description: "Attach explicitly selected row-bound columns from this atomic run".into(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            for (key, _) in spec.params.iter() {
                if !matches!(key.as_str(), "source" | "columns" | "effective_at") {
                    return Err(anyhow!("table.attach does not support parameter '{key}'"));
                }
            }
            let source = spec.params.expect_text("source")?.to_string();
            if source.trim().is_empty() {
                return Err(anyhow!("table.attach requires a nonempty source"));
            }
            let columns = match spec.params.get("columns") {
                Some(AlgorithmParamValue::TextList(names)) => names
                    .iter()
                    .map(|name| (name.clone(), name.clone()))
                    .collect(),
                Some(AlgorithmParamValue::IntList(names)) if names.is_empty() => Vec::new(),
                Some(AlgorithmParamValue::Json(serde_json::Value::Array(names))) => names
                    .iter()
                    .map(|name| {
                        let name = name
                            .as_str()
                            .ok_or_else(|| anyhow!("table.attach columns must be strings"))?
                            .to_owned();
                        Ok((name.clone(), name))
                    })
                    .collect::<Result<Vec<_>>>()?,
                Some(AlgorithmParamValue::Json(serde_json::Value::Object(names))) => names
                    .iter()
                    .map(|(name, target)| {
                        Ok((
                            name.clone(),
                            target
                                .as_str()
                                .ok_or_else(|| {
                                    anyhow!("table.attach column targets must be strings")
                                })?
                                .to_owned(),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?,
                _ => {
                    return Err(anyhow!(
                    "table.attach requires explicit columns: names or a source-to-property mapping"
                ))
                }
            };
            let effective_at = match spec.params.get("effective_at") {
                None => None,
                Some(AlgorithmParamValue::Int(at)) => Some(*at),
                _ => {
                    return Err(anyhow!(
                        "effective_at must be signed integer Unix microseconds"
                    ))
                }
            };
            Ok(Box::new(AttachTableStep {
                source,
                columns,
                effective_at,
            }))
        },
    )?;
    Ok(())
}
