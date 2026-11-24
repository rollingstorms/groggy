//! Initialization step primitives for creating node and edge variables.

use super::read_scope::{sealed, ReadStep, ReadStepScope};
use std::collections::HashMap;

use anyhow::Result;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{AtomicStepEffect, Step, StepMetadata, StepScope};

/// Initialize a node variable with a constant value for all nodes.
pub struct InitNodesStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitNodesStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitNodesStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.init_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize a node variable with a constant value".to_string(),
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

/// Initialize a node variable with sequential index values (0, 1, 2, ...).
/// Uses ordered_nodes() for deterministic ordering.
pub struct InitNodesWithIndexStep {
    target: String,
}

impl InitNodesWithIndexStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for InitNodesWithIndexStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.init_nodes_with_index"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize a node variable with sequential indices".to_string(),
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

/// Initialize a scalar variable with a constant value.
pub struct InitScalarStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitScalarStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitScalarStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.init_scalar"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize a scalar variable with a constant value".to_string(),
            cost_hint: CostHint::Constant,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

/// Initialize an edge variable with a constant value for all edges.
pub struct InitEdgesStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitEdgesStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitEdgesStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.init_edges"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize an edge variable with a constant".to_string(),
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

/// Get the number of nodes in the current subgraph as a scalar variable.
pub struct GraphNodeCountStep {
    target: String,
}

impl GraphNodeCountStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for GraphNodeCountStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.graph_node_count"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Get the number of nodes in the graph".to_string(),
            cost_hint: CostHint::Constant,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

/// Get the number of edges in the current subgraph as a scalar variable.
pub struct GraphEdgeCountStep {
    target: String,
}

impl GraphEdgeCountStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for GraphEdgeCountStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.graph_edge_count"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Get the number of edges in the graph".to_string(),
            cost_hint: CostHint::Constant,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for InitNodesStep {}
impl ReadStep for InitNodesStep {
    fn apply_read(&self, _ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        let nodes = scope.ordered_nodes();
        let mut map = HashMap::with_capacity(nodes.len());
        for &node in nodes.iter() {
            map.insert(node, self.value.clone());
        }
        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

impl sealed::Sealed for InitNodesWithIndexStep {}
impl ReadStep for InitNodesWithIndexStep {
    fn apply_read(&self, _ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        let nodes = scope.ordered_nodes();
        let mut map = HashMap::with_capacity(nodes.len());

        for (idx, &node) in nodes.iter().enumerate() {
            map.insert(node, AlgorithmParamValue::Int(idx as i64));
        }

        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

impl sealed::Sealed for InitScalarStep {}
impl ReadStep for InitScalarStep {
    fn apply_read(&self, _ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        scope
            .variables_mut()
            .set_scalar(self.target.clone(), self.value.clone());
        Ok(())
    }
}

impl sealed::Sealed for InitEdgesStep {}
impl ReadStep for InitEdgesStep {
    fn apply_read(&self, _ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        let mut map = HashMap::new();
        for edge in scope.edge_ids() {
            map.insert(edge, self.value.clone());
        }
        scope.variables_mut().set_edge_map(self.target.clone(), map);
        Ok(())
    }
}

impl sealed::Sealed for GraphNodeCountStep {}
impl ReadStep for GraphNodeCountStep {
    fn apply_read(&self, _ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        let count = scope.node_count();
        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Int(count as i64));
        Ok(())
    }
}

impl sealed::Sealed for GraphEdgeCountStep {}
impl ReadStep for GraphEdgeCountStep {
    fn apply_read(&self, _ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        let count = scope.edge_count();
        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Int(count as i64));
        Ok(())
    }
}
