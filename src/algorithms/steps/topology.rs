//! Terminal topology effects: prepare removals, publish through one checked batch.
use super::core::{Step, StepMetadata, StepRegistry, StepScope, StepSpec, StepValue};
use crate::algorithms::{AlgorithmParamValue, Context, CostHint};
use anyhow::{anyhow, bail, Result};

struct RemoveStep {
    nodes: bool,
    input: RemovalInput,
    effective_at: Option<crate::TimestampMicros>,
}

enum RemovalInput {
    Ids(Vec<usize>),
    /// Every key in a typed node/edge map; values are deliberately not predicates.
    Keys(String),
}

impl Step for RemoveStep {
    fn id(&self) -> &'static str {
        if self.nodes {
            "graph.remove_nodes"
        } else {
            "graph.remove_edges"
        }
    }

    fn is_terminal_topology_effect(&self) -> bool {
        true
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if !ctx.persist_results() || ctx.property_batch.is_none() {
            bail!("{} requires a persistent local atomic builder", self.id());
        }
        if ctx.is_cancelled() {
            bail!("topology removal cancelled");
        }
        let ids = match &self.input {
            RemovalInput::Ids(ids) => ids.clone(),
            RemovalInput::Keys(source) => {
                let mut ids = match (self.nodes, scope.variables().value(source)) {
                    (true, Some(StepValue::NodeMap(map))) => {
                        map.keys().copied().collect::<Vec<_>>()
                    }
                    (true, Some(StepValue::NodeColumn(column))) => column.nodes().to_vec(),
                    (false, Some(StepValue::EdgeMap(map))) => map.keys().copied().collect(),
                    (_, None) => bail!("variable '{source}' not found"),
                    _ => bail!(
                        "{} requires a matching node/edge map or node column",
                        self.id()
                    ),
                };
                ids.sort_unstable();
                ids
            }
        };
        // Validate every target against the original execution selection before
        // touching staging. Final canonical validation also checks duplicate
        // removals, property conflicts, cascades, and temporal endpoint validity.
        let binding = scope.subgraph().graph();
        let graph = binding.borrow();
        for &id in &ids {
            if ctx.is_cancelled() {
                bail!("topology removal cancelled");
            }
            let selected = if self.nodes {
                scope.subgraph().nodes().contains(&id) && graph.has_node(id)
            } else {
                scope.subgraph().edges().contains(&id) && graph.has_edge(id)
            };
            if !selected {
                bail!(
                    "{} target {id} is outside the active execution selection",
                    self.id()
                );
            }
        }
        drop(graph);
        ctx.property_batch
            .as_mut()
            .ok_or_else(|| anyhow!("missing atomic effect scope"))?
            .stage_removals(self.nodes, ids, self.effective_at);
        Ok(())
    }
}

fn instantiate(spec: &StepSpec, nodes: bool) -> Result<Box<dyn Step>> {
    for (key, _) in spec.params.iter() {
        if !matches!(key.as_str(), "ids" | "source" | "effective_at") {
            bail!("{} does not support parameter '{key}'", spec.id);
        }
    }
    let input = match (spec.params.get("ids"), spec.params.get("source")) {
        (Some(_), None) => RemovalInput::Ids(super::typed_values::ids(spec, "ids")?),
        (None, Some(AlgorithmParamValue::Text(source))) if !source.trim().is_empty() => {
            RemovalInput::Keys(source.clone())
        }
        _ => bail!(
            "{} requires exactly one of ids or a nonempty source name",
            spec.id
        ),
    };
    let effective_at = match spec.params.get("effective_at") {
        None => None,
        Some(AlgorithmParamValue::Int(at)) => Some(*at),
        _ => bail!("effective_at must be a signed integer Unix timestamp in microseconds"),
    };
    Ok(Box::new(RemoveStep {
        nodes,
        input,
        effective_at,
    }))
}

pub(super) fn register(registry: &StepRegistry) -> Result<()> {
    for (id, nodes) in [("graph.remove_nodes", true), ("graph.remove_edges", false)] {
        registry.register(
            id,
            StepMetadata {
                id: id.into(),
                description: "Stage terminal removals in one atomic graph publication".into(),
                cost_hint: CostHint::Linear,
            },
            move |spec| instantiate(spec, nodes),
        )?;
    }
    Ok(())
}
