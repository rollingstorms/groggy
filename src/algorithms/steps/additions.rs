//! Stage creation through the existing canonical mixed-batch acceptance path.
use super::core::{
    AtomicStepEffect, Step, StepMetadata, StepRegistry, StepScope, StepSpec, StepValue,
};
use super::read_scope::{sealed, ReadStep, ReadStepScope};
use crate::{
    algorithms::{AlgorithmParamValue, Context, CostHint},
    AttrName, AttrValue, BatchNodeRef,
};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

struct AddStep {
    nodes: bool,
    count: usize,
    edges: Vec<(BatchNodeRef, BatchNodeRef)>,
    attributes: HashMap<AttrName, AttrValue>,
    output: String,
    effective_at: Option<crate::TimestampMicros>,
}
impl Step for AddStep {
    fn id(&self) -> &'static str {
        if self.nodes {
            "graph.add_nodes"
        } else {
            "graph.add_edges"
        }
    }
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::Topology
    }
    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}
pub(super) fn endpoint(value: &serde_json::Value) -> Result<BatchNodeRef> {
    if let Some(id) = value.as_u64().and_then(|id| usize::try_from(id).ok()) {
        return Ok(BatchNodeRef::Existing(id));
    }
    if let Some(object) = value.as_object().filter(|o| o.len() == 1) {
        if let Some(position) = object
            .get("added")
            .and_then(|v| v.as_u64())
            .and_then(|v| usize::try_from(v).ok())
        {
            return Ok(BatchNodeRef::Added(position));
        }
    }
    bail!("edge endpoints must be nonnegative IDs or {{\"added\": position}} references")
}
fn instantiate(spec: &StepSpec, nodes: bool) -> Result<Box<dyn Step>> {
    let payload = if nodes { "count" } else { "edges" };
    for (key, _) in spec.params.iter() {
        if key != payload && !matches!(key.as_str(), "attributes" | "output" | "effective_at") {
            bail!("{} does not support parameter '{key}'", spec.id);
        }
    }
    let output = spec.params.expect_text("output")?.to_owned();
    if output.trim().is_empty() {
        bail!("{} requires a nonempty output", spec.id);
    }
    let effective_at = match spec.params.get("effective_at") {
        None => None,
        Some(AlgorithmParamValue::Int(at)) => Some(*at),
        _ => bail!("effective_at must be signed integer Unix microseconds"),
    };
    let attributes = match spec.params.get("attributes") {
        None => HashMap::new(),
        Some(AlgorithmParamValue::Json(serde_json::Value::Object(attrs))) => attrs
            .iter()
            .map(|(name, value)| {
                let value = crate::algorithms::builder::json_to_param(value.clone())?
                    .as_attr_value()
                    .ok_or_else(|| anyhow!("unsupported attribute value"))?;
                Ok((name.clone(), value))
            })
            .collect::<Result<_>>()?,
        _ => bail!("attributes must be an object of shared property values"),
    };
    let mut step = AddStep {
        nodes,
        count: 0,
        edges: Vec::new(),
        attributes,
        output,
        effective_at,
    };
    if nodes {
        step.count = match spec.params.get("count") {
            Some(AlgorithmParamValue::Int(count)) => usize::try_from(*count)
                .map_err(|_| anyhow!("count must be a nonnegative integer fitting usize"))?,
            _ => bail!("count must be a nonnegative integer"),
        };
    } else {
        step.edges = match spec.params.get("edges") {
            Some(AlgorithmParamValue::Json(serde_json::Value::Array(rows))) => rows
                .iter()
                .map(|row| {
                    let row = row
                        .as_array()
                        .filter(|row| row.len() == 2)
                        .ok_or_else(|| anyhow!("edges must contain [source, target] pairs"))?;
                    Ok((endpoint(&row[0])?, endpoint(&row[1])?))
                })
                .collect::<Result<_>>()?,
            Some(AlgorithmParamValue::IntList(rows)) if rows.is_empty() => Vec::new(),
            _ => bail!("edges must be an array of endpoint pairs"),
        };
    }
    Ok(Box::new(step))
}
pub(super) fn register(registry: &StepRegistry) -> Result<()> {
    for (id, nodes) in [("graph.add_nodes", true), ("graph.add_edges", false)] {
        registry.register(
            id,
            StepMetadata {
                id: id.into(),
                description: "Stage new entities without consuming IDs before acceptance".into(),
                cost_hint: CostHint::Linear,
            },
            move |spec| instantiate(spec, nodes),
        )?;
    }
    Ok(())
}

impl sealed::Sealed for AddStep {}
impl ReadStep for AddStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            bail!("topology addition cancelled");
        }
        if !ctx.persist_results() {
            bail!("{} requires persist=True", self.id());
        }
        let ids = scope.add(
            ctx,
            self.nodes,
            self.count,
            &self.edges,
            &self.attributes,
            self.effective_at,
        )?;
        let values = ids
            .into_iter()
            .map(|id| (id, AlgorithmParamValue::None))
            .collect();
        if ctx.is_cancelled() {
            bail!("topology addition cancelled");
        }
        scope.variables_mut().set_value(
            &self.output,
            if self.nodes {
                StepValue::NodeMap(values)
            } else {
                StepValue::EdgeMap(values)
            },
        );
        Ok(())
    }
}
