//! Initialization step primitives for creating node and edge variables.

use std::collections::HashMap;

use anyhow::Result;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::new();
        for &node in scope.subgraph().nodes() {
            map.insert(node, self.value.clone());
        }
        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::new();
        for &edge in scope.edge_ids() {
            map.insert(edge, self.value.clone());
        }
        scope.variables_mut().set_edge_map(self.target.clone(), map);
        Ok(())
    }
}
