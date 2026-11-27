//! Attribute loading and attaching step primitives.

use super::read_scope::{sealed, ReadStep, ReadStepScope};
use std::collections::HashMap;

use anyhow::{anyhow, Result};

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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
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

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}
impl sealed::Sealed for EdgeWeightScaleStep {}
impl ReadStep for EdgeWeightScaleStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("edge_weight_scale cancelled"));
        }
        if !self.factor.is_finite() {
            return Err(anyhow!("edge_weight_scale factor must be finite"));
        }
        scope.validate_selected_read()?;
        let mut map = HashMap::with_capacity(scope.edge_count());
        for edge in scope.edge_ids() {
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
                other => other,
            };
            if matches!(&scaled, AlgorithmParamValue::Float(value) if !value.is_finite()) {
                return Err(anyhow!(
                    "edge_weight_scale requires finite numeric input and result"
                ));
            }
            map.insert(edge, scaled);
        }
        if ctx.is_cancelled() {
            return Err(anyhow!("edge_weight_scale cancelled"));
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
    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
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

impl sealed::Sealed for AttachNodeAttrStep {}
impl ReadStep for AttachNodeAttrStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
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

impl sealed::Sealed for AttachEdgeAttrStep {}
impl ReadStep for AttachEdgeAttrStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
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

impl sealed::Sealed for LoadNodeAttrStep {}
impl ReadStep for LoadNodeAttrStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.node_count());
        for node in scope.node_ids() {
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

impl sealed::Sealed for LoadEdgeAttrStep {}
impl ReadStep for LoadEdgeAttrStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.edge_count());
        for edge in scope.edge_ids() {
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

impl sealed::Sealed for AttachTableStep {}
impl ReadStep for AttachTableStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("table attachment cancelled"));
        }
        let table = scope.variables().table(&self.source)?;
        scope.attach_table(ctx, table, &self.columns, self.effective_at)?;
        Ok(())
    }
}

#[cfg(test)]
mod scale_safety_tests {
    use super::super::core::StepVariables;
    use super::*;
    use crate::{subgraphs::Subgraph, AttrValue, Graph};
    use std::sync::{atomic::AtomicBool, Arc};
    fn selected(value: Option<AttrValue>) -> Subgraph {
        let mut graph = Graph::new_directed_manual();
        graph.add_nodes(2).unwrap();
        let edge = graph.add_edge(0, 1).unwrap();
        if let Some(v) = value {
            graph.set_edge_attr(edge, "weight".into(), v).unwrap();
        }
        Subgraph::from_nodes(
            graph.into_shared(),
            [0, 1].into_iter().collect(),
            "test".into(),
        )
        .unwrap()
    }
    #[test]
    fn invalid_scale_controls_inputs_and_cancellation_preserve_previous_output() {
        for (factor, value) in [
            (f64::NAN, None),
            (f64::INFINITY, None),
            (2.0, Some(AttrValue::Float64(f64::INFINITY))),
            (f64::MAX, Some(AttrValue::Int(2))),
        ] {
            let input = selected(value);
            let mut vars = StepVariables::default();
            let old = HashMap::from([(0, AlgorithmParamValue::Int(42))]);
            vars.set_edge_map("out", old.clone());
            let step = EdgeWeightScaleStep::new("weight".into(), factor, "out");
            assert!(step
                .apply(&mut Context::new(), &mut StepScope::new(&input, &mut vars))
                .is_err());
            assert_eq!(vars.edge_map("out").unwrap(), &old);
        }
        let input = selected(None);
        let mut vars = StepVariables::default();
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(EdgeWeightScaleStep::new("weight".into(), 1.0, "out")
            .apply(&mut ctx, &mut StepScope::new(&input, &mut vars))
            .is_err());
        assert!(vars.edge_map("out").is_err());
    }
    #[test]
    fn nonnumeric_values_and_legacy_unrepresentable_fallback_are_preserved() {
        for (input, expected) in [
            (Some(AttrValue::Bool(true)), AlgorithmParamValue::Bool(true)),
            (Some(AttrValue::Null), AlgorithmParamValue::None),
            (
                Some(AttrValue::Float64Vec(vec![1.0, 2.0])),
                AlgorithmParamValue::FloatList(vec![1.0, 2.0]),
            ),
            (
                Some(AttrValue::Date(
                    crate::types::CalendarDate::new(2020, 1, 2).unwrap(),
                )),
                AlgorithmParamValue::Float(2.0),
            ),
            (None, AlgorithmParamValue::Float(2.0)),
        ] {
            let input = selected(input);
            let mut vars = StepVariables::default();
            EdgeWeightScaleStep::new("weight".into(), 2.0, "out")
                .apply(&mut Context::new(), &mut StepScope::new(&input, &mut vars))
                .unwrap();
            assert_eq!(vars.edge_map("out").unwrap().get(&0), Some(&expected));
        }
    }
}
