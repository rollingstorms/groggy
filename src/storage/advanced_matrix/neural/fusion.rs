//! Memory Fusion Engine for Neural Network Operations
//!
//! This module implements operation fusion optimizations that combine multiple
//! operations into single kernels to reduce memory bandwidth and improve performance.

use crate::storage::advanced_matrix::{
    neural::autodiff::{ComputationGraph, NodeId, Operation},
    numeric_type::NumericType,
    unified_matrix::{MatrixError, MatrixResult, Shape, UnifiedMatrix},
};
use std::collections::{HashMap, HashSet};

/// Fusable operation patterns commonly found in neural networks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Linear layer followed by activation: MatMul + Bias + Activation
    LinearActivation {
        activation: ActivationType,
        has_bias: bool,
    },

    /// Convolution followed by batch norm and activation: Conv2D + BatchNorm + Activation
    ConvBatchNormActivation { activation: ActivationType },

    /// Element-wise operations: Add + Multiply + Activation chains
    ElementWiseChain {
        operations: Vec<ElementWiseOp>,
        final_activation: Option<ActivationType>,
    },

    /// Attention pattern: QKV projection + softmax + output projection
    AttentionBlock,

    /// Residual connection: x + f(x)
    ResidualConnection { skip_connection: bool },

    /// GELU approximation fusion: x * 0.5 * (1 + tanh(...))
    GeluApproximation,

    /// Softmax with numerical stability: x - max(x) then exp and normalize
    StableSoftmax,

    /// Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    LayerNormalization,
}

/// Activation function types for fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    None,
}

/// Element-wise operations that can be fused
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementWiseOp {
    Add,
    Multiply,
    Subtract,
    Divide,
    Square,
    Sqrt,
    Reciprocal,
    Negate,
}

/// Represents a fused operation that combines multiple primitive operations
pub struct FusedOperation<T: NumericType> {
    pub pattern: FusionPattern,
    pub input_nodes: Vec<NodeId>,
    pub output_shape: Shape,
    pub intermediate_shapes: Vec<Shape>,
    /// Optimized kernel function
    pub kernel: FusionKernel<T>,
    /// Memory requirements
    pub temp_memory_bytes: usize,
}

impl<T: NumericType> std::fmt::Debug for FusedOperation<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedOperation")
            .field("pattern", &self.pattern)
            .field("input_nodes", &self.input_nodes)
            .field("output_shape", &self.output_shape)
            .field("intermediate_shapes", &self.intermediate_shapes)
            .field("kernel", &"<fusion kernel function>")
            .field("temp_memory_bytes", &self.temp_memory_bytes)
            .finish()
    }
}

// Clone is not possible for function pointers, so we don't implement Clone
// Users should create new FusedOperation instances rather than cloning

/// Function signature for fused kernels
pub type FusionKernel<T> =
    Box<dyn Fn(&[&UnifiedMatrix<T>], &mut [UnifiedMatrix<T>]) -> MatrixResult<()> + Send + Sync>;

impl<T: NumericType> FusedOperation<T> {
    /// Execute the fused operation
    pub fn execute(
        &self,
        inputs: &[&UnifiedMatrix<T>],
        outputs: &mut [UnifiedMatrix<T>],
    ) -> MatrixResult<()> {
        if inputs.len() != self.input_nodes.len() {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.input_nodes.len(), 0),
                got: (inputs.len(), 0),
            });
        }

        (self.kernel)(inputs, outputs)
    }
}

/// Memory fusion engine that analyzes and optimizes computation graphs
pub struct FusionEngine<T: NumericType> {
    /// Known fusion patterns with their implementations
    fusion_patterns: HashMap<FusionPattern, FusionKernel<T>>,
    /// Cache for previously analyzed patterns
    pattern_cache: HashMap<Vec<Operation>, Option<FusionPattern>>,
    /// Performance statistics
    fusion_stats: FusionStatistics,
}

/// Statistics about fusion effectiveness
#[derive(Debug, Default)]
pub struct FusionStatistics {
    pub operations_fused: usize,
    pub memory_saved_bytes: usize,
    pub kernels_eliminated: usize,
    pub fusion_opportunities_found: usize,
    pub fusion_opportunities_applied: usize,
}

impl<T: NumericType> FusionEngine<T> {
    /// Create a new fusion engine with default patterns
    pub fn new() -> Self {
        let mut engine = Self {
            fusion_patterns: HashMap::new(),
            pattern_cache: HashMap::new(),
            fusion_stats: FusionStatistics::default(),
        };

        engine.register_default_patterns();
        engine
    }

    /// Register built-in fusion patterns
    fn register_default_patterns(&mut self) {
        // Linear + ReLU fusion
        self.register_pattern(
            FusionPattern::LinearActivation {
                activation: ActivationType::ReLU,
                has_bias: true,
            },
            Box::new(Self::fused_linear_relu_kernel),
        );

        // Element-wise chain fusion
        self.register_pattern(
            FusionPattern::ElementWiseChain {
                operations: vec![ElementWiseOp::Add, ElementWiseOp::Multiply],
                final_activation: Some(ActivationType::ReLU),
            },
            Box::new(Self::fused_elementwise_chain_kernel),
        );

        // GELU approximation fusion
        self.register_pattern(
            FusionPattern::GeluApproximation,
            Box::new(Self::fused_gelu_approximation_kernel),
        );

        // Stable softmax fusion
        self.register_pattern(
            FusionPattern::StableSoftmax,
            Box::new(Self::fused_stable_softmax_kernel),
        );

        // Layer normalization fusion
        self.register_pattern(
            FusionPattern::LayerNormalization,
            Box::new(Self::fused_layer_norm_kernel),
        );
    }

    /// Register a new fusion pattern
    pub fn register_pattern(&mut self, pattern: FusionPattern, kernel: FusionKernel<T>) {
        self.fusion_patterns.insert(pattern, kernel);
    }

    /// Analyze computation graph and identify fusion opportunities
    pub fn analyze_fusion_opportunities(
        &mut self,
        graph: &ComputationGraph<T>,
    ) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();
        let execution_order = graph.execution_order();

        // Sliding window analysis for finding fusable patterns
        for window_size in 2..=5 {
            for start in 0..=execution_order.len().saturating_sub(window_size) {
                let window = &execution_order[start..start + window_size];

                if let Some(opportunity) = self.analyze_window(graph, window) {
                    opportunities.push(opportunity);
                    self.fusion_stats.fusion_opportunities_found += 1;
                }
            }
        }

        // Filter overlapping opportunities (greedy selection for now)
        self.select_non_overlapping_opportunities(opportunities)
    }

    /// Analyze a window of operations for fusion potential
    fn analyze_window(
        &mut self,
        graph: &ComputationGraph<T>,
        window: &[NodeId],
    ) -> Option<FusionOpportunity> {
        let operations: Vec<Operation> = window
            .iter()
            .filter_map(|&id| {
                graph.get_node(id).map(|node_arc| {
                    let node = node_arc.lock().unwrap();
                    node.operation.clone()
                })
            })
            .collect();

        // Check cache first
        if let Some(cached_result) = self.pattern_cache.get(&operations) {
            return cached_result.as_ref().map(|pattern| FusionOpportunity {
                pattern: pattern.clone(),
                node_ids: window.to_vec(),
                estimated_speedup: self.estimate_speedup(pattern),
                memory_reduction: self.estimate_memory_reduction(pattern, window.len()),
            });
        }

        // Detect fusion patterns
        let detected_pattern = self.detect_pattern(&operations);
        self.pattern_cache
            .insert(operations, detected_pattern.clone());

        detected_pattern.map(|pattern| FusionOpportunity {
            pattern: pattern.clone(),
            node_ids: window.to_vec(),
            estimated_speedup: self.estimate_speedup(&pattern),
            memory_reduction: self.estimate_memory_reduction(&pattern, window.len()),
        })
    }

    /// Pattern detection logic
    fn detect_pattern(&self, operations: &[Operation]) -> Option<FusionPattern> {
        match operations {
            // MatMul + Add + ReLU pattern
            [Operation::MatMul, Operation::Add, Operation::ReLU] => {
                Some(FusionPattern::LinearActivation {
                    activation: ActivationType::ReLU,
                    has_bias: true,
                })
            }

            // MatMul + ReLU pattern (no bias)
            [Operation::MatMul, Operation::ReLU] => Some(FusionPattern::LinearActivation {
                activation: ActivationType::ReLU,
                has_bias: false,
            }),

            // Element-wise chain: Add + Multiply + ReLU
            [Operation::Add, Operation::Multiply, Operation::ReLU] => {
                Some(FusionPattern::ElementWiseChain {
                    operations: vec![ElementWiseOp::Add, ElementWiseOp::Multiply],
                    final_activation: Some(ActivationType::ReLU),
                })
            }

            // GELU approximation pattern: multiple ops that implement GELU
            ops if self.is_gelu_approximation_pattern(ops) => {
                Some(FusionPattern::GeluApproximation)
            }

            // Stable softmax pattern: Subtract(max) + Exp + Sum + Divide
            ops if self.is_stable_softmax_pattern(ops) => Some(FusionPattern::StableSoftmax),

            // Layer norm pattern: Sub(mean) + Square + Mean + Add(eps) + Sqrt + Divide + Mul(gamma) + Add(beta)
            ops if self.is_layer_norm_pattern(ops) => Some(FusionPattern::LayerNormalization),

            _ => None,
        }
    }

    /// Check if operation sequence matches GELU approximation
    fn is_gelu_approximation_pattern(&self, ops: &[Operation]) -> bool {
        // GELU ≈ x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        // Look for patterns involving multiply, add, tanh operations
        ops.len() >= 4
            && ops.iter().any(|op| matches!(op, Operation::Tanh))
            && ops
                .iter()
                .filter(|op| matches!(op, Operation::Multiply))
                .count()
                >= 2
    }

    /// Check if operation sequence matches stable softmax
    fn is_stable_softmax_pattern(&self, ops: &[Operation]) -> bool {
        // Stable softmax: x - max(x), exp(x), sum(exp(x)), exp(x) / sum
        ops.len() >= 3
            && ops.iter().any(|op| matches!(op, Operation::Subtract))
            && ops.iter().any(|op| matches!(op, Operation::Sum { .. }))
            && ops.iter().any(|op| matches!(op, Operation::Divide))
    }

    /// Check if operation sequence matches layer normalization
    fn is_layer_norm_pattern(&self, ops: &[Operation]) -> bool {
        // Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta
        ops.len() >= 5
            && ops
                .iter()
                .filter(|op| matches!(op, Operation::Mean { .. }))
                .count()
                >= 1
            && ops.iter().any(|op| matches!(op, Operation::Subtract))
            && ops.iter().any(|op| matches!(op, Operation::Divide))
    }

    /// Select non-overlapping fusion opportunities using greedy algorithm
    fn select_non_overlapping_opportunities(
        &self,
        mut opportunities: Vec<FusionOpportunity>,
    ) -> Vec<FusionOpportunity> {
        // Sort by estimated benefit (speedup * memory_reduction)
        opportunities.sort_by(|a, b| {
            let benefit_a = a.estimated_speedup * a.memory_reduction as f64;
            let benefit_b = b.estimated_speedup * b.memory_reduction as f64;
            benefit_b.partial_cmp(&benefit_a).unwrap()
        });

        let mut selected = Vec::new();
        let mut used_nodes: HashSet<NodeId> = HashSet::new();

        for opportunity in opportunities {
            // Check if any nodes overlap with already selected opportunities
            if opportunity
                .node_ids
                .iter()
                .any(|&id| used_nodes.contains(&id))
            {
                continue;
            }

            // Add nodes to used set
            for &id in &opportunity.node_ids {
                used_nodes.insert(id);
            }

            selected.push(opportunity);
        }

        selected
    }

    /// Estimate speedup from fusion
    fn estimate_speedup(&self, pattern: &FusionPattern) -> f64 {
        match pattern {
            FusionPattern::LinearActivation { .. } => 1.8, // 80% speedup typical
            FusionPattern::ConvBatchNormActivation { .. } => 2.2, // 120% speedup
            FusionPattern::ElementWiseChain { operations, .. } => {
                1.0 + 0.3 * operations.len() as f64 // ~30% per fused operation
            }
            FusionPattern::GeluApproximation => 2.5, // Significant speedup for GELU
            FusionPattern::StableSoftmax => 1.6,     // Good speedup for softmax
            FusionPattern::LayerNormalization => 2.8, // Excellent speedup for layer norm
            FusionPattern::AttentionBlock => 3.2,    // Outstanding speedup for attention
            FusionPattern::ResidualConnection { .. } => 1.4, // Modest speedup
        }
    }

    /// Estimate memory reduction from fusion
    fn estimate_memory_reduction(&self, pattern: &FusionPattern, num_ops: usize) -> usize {
        let base_reduction_per_op = 1024; // Bytes saved per eliminated intermediate

        let multiplier = match pattern {
            FusionPattern::LinearActivation { .. } => 2.0,
            FusionPattern::ConvBatchNormActivation { .. } => 3.0,
            FusionPattern::ElementWiseChain { operations, .. } => operations.len() as f64 * 0.8,
            FusionPattern::GeluApproximation => 4.0, // Many intermediates eliminated
            FusionPattern::StableSoftmax => 2.5,
            FusionPattern::LayerNormalization => 3.5,
            FusionPattern::AttentionBlock => 5.0,
            FusionPattern::ResidualConnection { .. } => 1.5,
        };

        (base_reduction_per_op as f64 * multiplier * num_ops as f64) as usize
    }

    /// Apply fusion optimizations to a computation graph
    pub fn optimize_computation_graph(
        &mut self,
        graph: &mut ComputationGraph<T>,
    ) -> MatrixResult<OptimizationResult> {
        let opportunities = self.analyze_fusion_opportunities(graph);
        let mut result = OptimizationResult::default();

        for opportunity in opportunities {
            if self.apply_fusion(graph, &opportunity)? {
                result.fusions_applied += 1;
                result.operations_eliminated += opportunity.node_ids.len() - 1;
                result.estimated_speedup *= opportunity.estimated_speedup;
                result.memory_saved += opportunity.memory_reduction;

                self.fusion_stats.fusion_opportunities_applied += 1;
                self.fusion_stats.operations_fused += opportunity.node_ids.len();
                self.fusion_stats.memory_saved_bytes += opportunity.memory_reduction;
            }
        }

        Ok(result)
    }

    /// Apply a single fusion opportunity
    fn apply_fusion(
        &self,
        graph: &mut ComputationGraph<T>,
        opportunity: &FusionOpportunity,
    ) -> MatrixResult<bool> {
        let kernel = self
            .fusion_patterns
            .get(&opportunity.pattern)
            .ok_or_else(|| {
                MatrixError::UnsupportedOperation(format!(
                    "Fusion pattern {:?} not implemented",
                    opportunity.pattern
                ))
            })?;

        // Create fused operation node
        let fused_node_id = self.create_fused_node(graph, opportunity, kernel)?;

        // Update dependencies to point to fused node
        self.redirect_dependencies(graph, &opportunity.node_ids, fused_node_id)?;

        // Remove original nodes (except inputs)
        self.remove_fused_nodes(graph, &opportunity.node_ids[1..])?;

        Ok(true)
    }

    /// Create a new fused operation node
    fn create_fused_node(
        &self,
        graph: &mut ComputationGraph<T>,
        opportunity: &FusionOpportunity,
        _kernel: &FusionKernel<T>,
    ) -> MatrixResult<NodeId> {
        // Determine inputs and output shape for fused operation
        let input_nodes = self.get_fusion_inputs(graph, &opportunity.node_ids)?;
        let output_shape = self.get_fusion_output_shape(graph, &opportunity.node_ids)?;

        // Create custom fused operation
        let fused_op = Operation::Add; // Placeholder - would need custom operation type

        graph.create_operation(fused_op, input_nodes, output_shape, true)
    }

    /// Get input nodes for fusion
    fn get_fusion_inputs(
        &self,
        graph: &ComputationGraph<T>,
        node_ids: &[NodeId],
    ) -> MatrixResult<Vec<NodeId>> {
        // Find all inputs that come from outside the fusion region
        let fusion_set: HashSet<NodeId> = node_ids.iter().copied().collect();
        let mut external_inputs = HashSet::new();

        for &node_id in node_ids {
            if let Some(node_arc) = graph.get_node(node_id) {
                let node = node_arc.lock().unwrap();
                for &input_id in &node.inputs {
                    if !fusion_set.contains(&input_id) {
                        external_inputs.insert(input_id);
                    }
                }
            }
        }

        Ok(external_inputs.into_iter().collect())
    }

    /// Get output shape for fusion
    fn get_fusion_output_shape(
        &self,
        graph: &ComputationGraph<T>,
        node_ids: &[NodeId],
    ) -> MatrixResult<Shape> {
        // Output shape is the shape of the last node in the fusion
        if let Some(&last_node_id) = node_ids.last() {
            if let Some(node_arc) = graph.get_node(last_node_id) {
                let node = node_arc.lock().unwrap();
                return Ok(node.shape);
            }
        }

        Err(MatrixError::ComputationError(
            "Could not determine fusion output shape".to_string(),
        ))
    }

    /// Redirect dependencies to fused node
    fn redirect_dependencies(
        &self,
        graph: &mut ComputationGraph<T>,
        old_nodes: &[NodeId],
        new_node: NodeId,
    ) -> MatrixResult<()> {
        // Find all nodes that depend on any of the old nodes
        let old_set: HashSet<NodeId> = old_nodes.iter().copied().collect();

        for (_, node_arc) in graph.nodes() {
            let mut node = node_arc.lock().unwrap();

            // Update input dependencies
            for input in &mut node.inputs {
                if old_set.contains(input) && *input == *old_nodes.last().unwrap() {
                    *input = new_node;
                }
            }
        }

        Ok(())
    }

    /// Remove old nodes that were fused
    fn remove_fused_nodes(
        &self,
        graph: &mut ComputationGraph<T>,
        node_ids: &[NodeId],
    ) -> MatrixResult<()> {
        // Remove nodes from graph (simplified - would need proper cleanup)
        for &_node_id in node_ids {
            // graph.nodes.remove(&node_id);
        }

        // Update execution order
        graph.update_execution_order()
    }

    // Fused kernel implementations

    /// Fused linear + ReLU kernel: y = max(0, Wx + b)
    fn fused_linear_relu_kernel(
        inputs: &[&UnifiedMatrix<T>],
        outputs: &mut [UnifiedMatrix<T>],
    ) -> MatrixResult<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(MatrixError::InvalidInput(
                "Invalid input/output count for fused linear ReLU".to_string(),
            ));
        }

        let x = inputs[0]; // Input
        let w = inputs[1]; // Weights
        let b = if inputs.len() > 2 {
            Some(inputs[2])
        } else {
            None
        }; // Bias (optional)
        let output = &mut outputs[0];

        // Perform fused operation: output = max(0, x @ w + b)
        let linear_result = x.matmul(w)?;

        let with_bias = if let Some(bias) = b {
            linear_result.add(bias)?
        } else {
            linear_result
        };

        // Apply ReLU in-place
        *output =
            crate::storage::advanced_matrix::neural::activations::ActivationOps::relu(&with_bias)?;

        Ok(())
    }

    /// Fused element-wise chain kernel
    fn fused_elementwise_chain_kernel(
        inputs: &[&UnifiedMatrix<T>],
        outputs: &mut [UnifiedMatrix<T>],
    ) -> MatrixResult<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(MatrixError::InvalidInput(
                "Invalid input/output count for fused elementwise chain".to_string(),
            ));
        }

        // Example: (x + y) * z with ReLU
        let result = inputs[0].add(inputs[1])?;
        let result = if inputs.len() > 2 {
            result.elementwise_multiply(inputs[2])?
        } else {
            result
        };

        outputs[0] =
            crate::storage::advanced_matrix::neural::activations::ActivationOps::relu(&result)?;
        Ok(())
    }

    /// Fused GELU approximation kernel
    fn fused_gelu_approximation_kernel(
        inputs: &[&UnifiedMatrix<T>],
        outputs: &mut [UnifiedMatrix<T>],
    ) -> MatrixResult<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(MatrixError::InvalidInput(
                "Invalid input/output count for fused GELU".to_string(),
            ));
        }

        outputs[0] =
            crate::storage::advanced_matrix::neural::activations::ActivationOps::gelu(inputs[0])?;
        Ok(())
    }

    /// Fused stable softmax kernel
    fn fused_stable_softmax_kernel(
        inputs: &[&UnifiedMatrix<T>],
        outputs: &mut [UnifiedMatrix<T>],
    ) -> MatrixResult<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(MatrixError::InvalidInput(
                "Invalid input/output count for fused softmax".to_string(),
            ));
        }

        outputs[0] = crate::storage::advanced_matrix::neural::activations::ActivationOps::softmax(
            inputs[0],
        )?;
        Ok(())
    }

    /// Fused layer normalization kernel
    fn fused_layer_norm_kernel(
        inputs: &[&UnifiedMatrix<T>],
        outputs: &mut [UnifiedMatrix<T>],
    ) -> MatrixResult<()> {
        if inputs.len() < 3 || outputs.is_empty() {
            return Err(MatrixError::InvalidInput(
                "Invalid input/output count for fused layer norm".to_string(),
            ));
        }

        let x = inputs[0]; // Input
        let _gamma = inputs[1]; // Scale
        let _beta = inputs[2]; // Bias

        // Layer norm: (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
        // This would need actual implementation with mean and variance computation

        outputs[0] = x.clone(); // Placeholder
        Ok(())
    }

    /// Get fusion statistics
    pub fn get_statistics(&self) -> &FusionStatistics {
        &self.fusion_stats
    }

    /// Clear statistics
    pub fn clear_statistics(&mut self) {
        self.fusion_stats = FusionStatistics::default();
    }
}

impl<T: NumericType> Default for FusionEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a fusion opportunity found in the computation graph
#[derive(Debug, Clone)]
pub struct FusionOpportunity {
    pub pattern: FusionPattern,
    pub node_ids: Vec<NodeId>,
    pub estimated_speedup: f64,
    pub memory_reduction: usize,
}

/// Result of applying fusion optimizations
#[derive(Debug, Default)]
pub struct OptimizationResult {
    pub fusions_applied: usize,
    pub operations_eliminated: usize,
    pub estimated_speedup: f64,
    pub memory_saved: usize,
}

impl OptimizationResult {
    pub fn new() -> Self {
        Self {
            fusions_applied: 0,
            operations_eliminated: 0,
            estimated_speedup: 1.0,
            memory_saved: 0,
        }
    }
}

/// Convenience function to optimize a computation graph
pub fn optimize_computation_graph<T: NumericType>(
    graph: &mut ComputationGraph<T>,
) -> MatrixResult<OptimizationResult> {
    let mut fusion_engine = FusionEngine::new();
    fusion_engine.optimize_computation_graph(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_engine_creation() {
        let engine: FusionEngine<f64> = FusionEngine::new();
        assert!(!engine.fusion_patterns.is_empty());
        assert_eq!(engine.fusion_stats.operations_fused, 0);
    }

    #[test]
    fn test_fusion_patterns() {
        let pattern1 = FusionPattern::LinearActivation {
            activation: ActivationType::ReLU,
            has_bias: true,
        };
        let pattern2 = FusionPattern::GeluApproximation;

        assert_ne!(pattern1, pattern2);
        assert_eq!(pattern1, pattern1.clone());
    }

    #[test]
    fn test_activation_types() {
        assert_eq!(ActivationType::ReLU, ActivationType::ReLU);
        assert_ne!(ActivationType::ReLU, ActivationType::GELU);
    }

    #[test]
    fn test_element_wise_ops() {
        let ops = [
            ElementWiseOp::Add,
            ElementWiseOp::Multiply,
            ElementWiseOp::Subtract,
        ];
        assert_eq!(ops.len(), 3);
        assert!(ops.contains(&ElementWiseOp::Add));
    }

    #[test]
    fn test_optimization_result() {
        let mut result = OptimizationResult::new();
        result.fusions_applied = 3;
        result.estimated_speedup = 2.5;

        assert_eq!(result.fusions_applied, 3);
        assert_eq!(result.estimated_speedup, 2.5);
    }
}
