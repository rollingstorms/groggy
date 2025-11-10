//! Attribute loading and attaching step primitives.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::traits::SubgraphOperations;
use crate::types::{AttrName, EdgeId, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().node_set().len());
        for &node in scope.node_ids() {
            let value = match scope
                .subgraph()
                .get_node_attribute(node, &self.attr)?
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
}

impl AttachNodeAttrStep {
    pub fn new(source: impl Into<String>, attr: AttrName) -> Self {
        Self {
            source: source.into(),
            attr,
        }
    }
}

impl Step for AttachNodeAttrStep {
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

            if !values.is_empty() {
                scope
                    .subgraph()
                    .set_node_attr_column(self.attr.clone(), values)
                    .map_err(|err| anyhow!("failed to set node attributes: {err}"))?;
            }
            return Ok(());
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

        if values.is_empty() {
            return Ok(());
        }

        let mut attrs = HashMap::new();
        attrs.insert(self.attr.clone(), values);
        scope
            .subgraph()
            .set_node_attrs(attrs)
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().edge_set().len());
        for &edge in scope.edge_ids() {
            let value = scope
                .subgraph()
                .get_edge_attribute(edge, &self.attr)?
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
}

impl AttachEdgeAttrStep {
    pub fn new(source: impl Into<String>, attr: AttrName) -> Self {
        Self {
            source: source.into(),
            attr,
        }
    }
}

impl Step for AttachEdgeAttrStep {
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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

        if values.is_empty() {
            return Ok(());
        }

        attrs.insert(self.attr.clone(), values);
        scope
            .subgraph()
            .set_edge_attrs(attrs)
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
                .subgraph()
                .get_edge_attribute(edge, &self.attr)?
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
