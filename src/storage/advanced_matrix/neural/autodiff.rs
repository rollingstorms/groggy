//! Automatic Differentiation Framework
//! 
//! This module implements a basic automatic differentiation system for computing
//! gradients efficiently in neural network training and optimization.

use crate::storage::advanced_matrix::{
    numeric_type::NumericType,
    unified_matrix::{UnifiedMatrix, MatrixResult, MatrixError, Shape},
    backend::{ComputeBackendExt, OperationType},
};
use std::sync::{Arc, Weak, Mutex};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Unique identifier for computation graph nodes
pub type NodeId = usize;

/// Operations supported in the computation graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operation {
    // Basic arithmetic
    Add,
    Subtract, 
    Multiply,
    Divide,
    
    // Matrix operations
    MatMul,
    Transpose,
    Sum { axis: Option<usize> },
    Mean { axis: Option<usize> },
    
    // Neural network operations
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Softmax { axis: usize },
    
    // Convolution operations
    Conv2D { 
        kernel_size: (usize, usize),
        stride: (usize, usize), 
        padding: (usize, usize),
    },
    
    // Loss functions
    MSELoss,
    CrossEntropyLoss,
    
    // Leaf node (no operation)
    Leaf,
}

/// Node in the computation graph
#[derive(Debug)]
pub struct ComputationNode<T: NumericType> {
    pub id: NodeId,
    pub operation: Operation,
    pub inputs: Vec<NodeId>,
    pub shape: Shape,
    pub requires_grad: bool,
    pub gradient: Option<UnifiedMatrix<T>>,
    /// Reference to parent nodes (for gradient propagation)
    pub parents: Vec<Weak<Mutex<ComputationNode<T>>>>,
    /// Cached forward computation result
    pub value: Option<UnifiedMatrix<T>>,
}

impl<T: NumericType> ComputationNode<T> {
    pub fn new(id: NodeId, operation: Operation, shape: Shape, requires_grad: bool) -> Self {
        Self {
            id,
            operation,
            inputs: Vec::new(),
            shape,
            requires_grad,
            gradient: None,
            parents: Vec::new(),
            value: None,
        }
    }
    
    /// Add input node dependency
    pub fn add_input(&mut self, input_id: NodeId) {
        self.inputs.push(input_id);
    }
    
    /// Set gradient for this node
    pub fn set_gradient(&mut self, gradient: UnifiedMatrix<T>) -> MatrixResult<()> {
        if gradient.shape() != self.shape {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.shape.rows, self.shape.cols),
                got: (gradient.shape().rows, gradient.shape().cols),
            });
        }
        self.gradient = Some(gradient);
        Ok(())
    }
    
    /// Accumulate gradient (for nodes with multiple outputs)
    pub fn accumulate_gradient(&mut self, gradient: UnifiedMatrix<T>) -> MatrixResult<()> {
        match &mut self.gradient {
            None => self.set_gradient(gradient)?,
            Some(existing) => {
                let sum = existing.add(&gradient)?;
                *existing = sum;
            }
        }
        Ok(())
    }
}

/// Computation graph for automatic differentiation
#[derive(Debug, Clone)]
pub struct ComputationGraph<T: NumericType> {
    nodes: HashMap<NodeId, Arc<Mutex<ComputationNode<T>>>>,
    next_id: NodeId,
    /// Topologically sorted execution order
    execution_order: Vec<NodeId>,
}

impl<T: NumericType> ComputationGraph<T> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            execution_order: Vec::new(),
        }
    }
    
    /// Get execution order for fusion optimization
    pub fn execution_order(&self) -> &[NodeId] {
        &self.execution_order
    }
    
    /// Get nodes for graph analysis
    pub fn nodes(&self) -> &HashMap<NodeId, Arc<Mutex<ComputationNode<T>>>> {
        &self.nodes
    }
    
    // Removed duplicate method - see private implementation below
    
    /// Create a new leaf node (input or parameter)
    pub fn create_leaf(&mut self, value: UnifiedMatrix<T>, requires_grad: bool) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        
        let shape = value.shape();
        let mut node = ComputationNode::new(id, Operation::Leaf, shape, requires_grad);
        node.value = Some(value);
        
        self.nodes.insert(id, Arc::new(Mutex::new(node)));
        id
    }
    
    /// Create a new operation node
    pub fn create_operation(
        &mut self,
        operation: Operation,
        inputs: Vec<NodeId>,
        output_shape: Shape,
        requires_grad: bool,
    ) -> MatrixResult<NodeId> {
        let id = self.next_id;
        self.next_id += 1;
        
        let mut node = ComputationNode::new(id, operation, output_shape, requires_grad);
        
        // Add input dependencies and set parent references
        for &input_id in &inputs {
            node.add_input(input_id);
            if let Some(input_node) = self.nodes.get(&input_id) {
                let weak_ref = Arc::downgrade(&Arc::clone(input_node));
                node.parents.push(weak_ref);
            }
        }
        
        self.nodes.insert(id, Arc::new(Mutex::new(node)));
        
        // Update execution order
        self.update_execution_order()?;
        
        Ok(id)
    }
    
    /// Get node by ID
    pub fn get_node(&self, id: NodeId) -> Option<Arc<Mutex<ComputationNode<T>>>> {
        self.nodes.get(&id).map(Arc::clone)
    }
    
    /// Topological sort to determine execution order
    pub fn update_execution_order(&mut self) -> MatrixResult<()> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj_list: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        
        // Calculate in-degrees and build adjacency list
        for (&node_id, node_arc) in &self.nodes {
            in_degree.entry(node_id).or_insert(0);
            
            let node = node_arc.lock().unwrap();
            for &input_id in &node.inputs {
                adj_list.entry(input_id).or_insert_with(Vec::new).push(node_id);
                *in_degree.entry(node_id).or_insert(0) += 1;
            }
        }
        
        // Kahn's algorithm for topological sorting
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        let mut result = Vec::new();
        
        // Start with nodes that have no dependencies (in-degree 0)
        for (&node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node_id);
            }
        }
        
        while let Some(current) = queue.pop_front() {
            result.push(current);
            
            // Update in-degrees of dependent nodes
            if let Some(dependents) = adj_list.get(&current) {
                for &dependent in dependents {
                    if let Some(degree) = in_degree.get_mut(&dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }
        
        if result.len() != self.nodes.len() {
            return Err(MatrixError::ComputationError("Circular dependency detected in computation graph".to_string()));
        }
        
        self.execution_order = result;
        Ok(())
    }
    
    /// Forward pass through the computation graph
    pub fn forward(&self, output_node: NodeId) -> MatrixResult<UnifiedMatrix<T>> {
        for &node_id in &self.execution_order {
            if node_id > output_node {
                break; // Only compute up to the requested output
            }
            
            if let Some(node_arc) = self.nodes.get(&node_id) {
                let mut node = node_arc.lock().unwrap();
                
                // Skip if value is already computed
                if node.value.is_some() {
                    continue;
                }
                
                // Compute forward pass for this node
                let result = self.compute_forward_operation(&*node)?;
                node.value = Some(result);
            }
        }
        
        // Return the computed value for the output node
        if let Some(node_arc) = self.nodes.get(&output_node) {
            let node = node_arc.lock().unwrap();
            if let Some(ref value) = node.value {
                Ok(value.clone())
            } else {
                Err(MatrixError::ComputationError("Failed to compute forward pass".to_string()))
            }
        } else {
            Err(MatrixError::ComputationError("Output node not found".to_string()))
        }
    }
    
    /// Compute forward operation for a single node
    fn compute_forward_operation(&self, node: &ComputationNode<T>) -> MatrixResult<UnifiedMatrix<T>> {
        match &node.operation {
            Operation::Leaf => {
                Err(MatrixError::ComputationError("Leaf nodes should already have values".to_string()))
            }
            
            Operation::Add => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 2 {
                    return Err(MatrixError::ComputationError("Add operation requires exactly 2 inputs".to_string()));
                }
                inputs[0].add(&inputs[1])
            }
            
            Operation::Multiply => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 2 {
                    return Err(MatrixError::ComputationError("Multiply operation requires exactly 2 inputs".to_string()));
                }
                inputs[0].elementwise_multiply(&inputs[1])
            }
            
            Operation::MatMul => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 2 {
                    return Err(MatrixError::ComputationError("MatMul operation requires exactly 2 inputs".to_string()));
                }
                inputs[0].matmul(&inputs[1])
            }
            
            Operation::ReLU => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError("ReLU operation requires exactly 1 input".to_string()));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::relu(&inputs[0])
            }
            
            Operation::Sigmoid => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError("Sigmoid operation requires exactly 1 input".to_string()));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::sigmoid(&inputs[0])
            }
            
            Operation::Tanh => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError("Tanh operation requires exactly 1 input".to_string()));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::tanh(&inputs[0])
            }
            
            Operation::GELU => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError("GELU operation requires exactly 1 input".to_string()));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::gelu(&inputs[0])
            }
            
            Operation::Sum { axis } => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError("Sum operation requires exactly 1 input".to_string()));
                }
                // Implement sum reduction along specified axis
                self.reduce_sum(&inputs[0], *axis)
            }
            
            Operation::Transpose => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError("Transpose operation requires exactly 1 input".to_string()));
                }
                inputs[0].transpose()
            }
            
            _ => Err(MatrixError::UnsupportedOperation(format!("Operation {:?} not implemented", node.operation))),
        }
    }
    
    /// Get input values for a list of node IDs
    fn get_input_values(&self, input_ids: &[NodeId]) -> MatrixResult<Vec<UnifiedMatrix<T>>> {
        let mut inputs = Vec::new();
        for &id in input_ids {
            if let Some(node_arc) = self.nodes.get(&id) {
                let node = node_arc.lock().unwrap();
                if let Some(ref value) = node.value {
                    inputs.push(value.clone());
                } else {
                    return Err(MatrixError::ComputationError(format!("Input node {} has no computed value", id)));
                }
            } else {
                return Err(MatrixError::ComputationError(format!("Input node {} not found", id)));
            }
        }
        Ok(inputs)
    }
    
    /// Reduce sum along specified axis
    fn reduce_sum(&self, input: &UnifiedMatrix<T>, axis: Option<usize>) -> MatrixResult<UnifiedMatrix<T>> {
        match axis {
            None => {
                // Sum all elements
                let backend = input.backend_selector.select_backend(
                    OperationType::Sum,
                    input.len(),
                    T::DTYPE,
                    input.backend_hint.clone(),
                );
                // This would need actual implementation
                UnifiedMatrix::ones(1, 1) // Placeholder
            }
            Some(0) => {
                // Sum along rows (reduce rows, keep columns)
                UnifiedMatrix::ones(1, input.shape().cols)
            }
            Some(1) => {
                // Sum along columns (keep rows, reduce columns) 
                UnifiedMatrix::ones(input.shape().rows, 1)
            }
            Some(_) => Err(MatrixError::UnsupportedOperation("Only axis 0 and 1 supported for 2D matrices".to_string())),
        }
    }
    
    /// Backward pass (gradient computation)
    pub fn backward(&mut self, output_node: NodeId) -> MatrixResult<()> {
        // Initialize gradient for output node
        if let Some(node_arc) = self.nodes.get(&output_node) {
            let mut node = node_arc.lock().unwrap();
            if node.gradient.is_none() {
                // Initialize with gradient of 1 (assuming scalar output)
                let grad = UnifiedMatrix::ones(node.shape.rows, node.shape.cols)?;
                node.set_gradient(grad)?;
            }
        }
        
        // Propagate gradients in reverse topological order
        for &node_id in self.execution_order.iter().rev() {
            if node_id > output_node {
                continue;
            }
            
            if let Some(node_arc) = self.nodes.get(&node_id).map(Arc::clone) {
                self.compute_backward_operation(&node_arc)?;
            }
        }
        
        Ok(())
    }
    
    /// Compute backward pass for a single node
    fn compute_backward_operation(&self, node_arc: &Arc<Mutex<ComputationNode<T>>>) -> MatrixResult<()> {
        let node = node_arc.lock().unwrap();
        
        if !node.requires_grad || node.gradient.is_none() {
            return Ok(());
        }
        
        let grad_output = node.gradient.as_ref().unwrap();
        
        match &node.operation {
            Operation::Leaf => Ok(()), // Leaf nodes don't propagate gradients
            
            Operation::Add => {
                // d(a + b)/da = 1, d(a + b)/db = 1
                // Gradient flows through unchanged to both inputs
                for &input_id in &node.inputs {
                    if let Some(input_arc) = self.nodes.get(&input_id) {
                        let mut input_node = input_arc.lock().unwrap();
                        if input_node.requires_grad {
                            input_node.accumulate_gradient(grad_output.clone())?;
                        }
                    }
                }
                Ok(())
            }
            
            Operation::Multiply => {
                // d(a * b)/da = b, d(a * b)/db = a
                if node.inputs.len() == 2 {
                    let input_values = self.get_input_values(&node.inputs)?;
                    
                    // Gradient for first input: grad_output * second_input
                    if let Some(input_arc) = self.nodes.get(&node.inputs[0]) {
                        let mut input_node = input_arc.lock().unwrap();
                        if input_node.requires_grad {
                            let grad = grad_output.elementwise_multiply(&input_values[1])?;
                            input_node.accumulate_gradient(grad)?;
                        }
                    }
                    
                    // Gradient for second input: grad_output * first_input
                    if let Some(input_arc) = self.nodes.get(&node.inputs[1]) {
                        let mut input_node = input_arc.lock().unwrap();
                        if input_node.requires_grad {
                            let grad = grad_output.elementwise_multiply(&input_values[0])?;
                            input_node.accumulate_gradient(grad)?;
                        }
                    }
                }
                Ok(())
            }
            
            Operation::MatMul => {
                // d(A @ B)/dA = grad_output @ B^T
                // d(A @ B)/dB = A^T @ grad_output
                if node.inputs.len() == 2 {
                    let input_values = self.get_input_values(&node.inputs)?;
                    
                    // Gradient for first input (A)
                    if let Some(input_arc) = self.nodes.get(&node.inputs[0]) {
                        let mut input_node = input_arc.lock().unwrap();
                        if input_node.requires_grad {
                            let b_transposed = input_values[1].transpose()?;
                            let grad = grad_output.matmul(&b_transposed)?;
                            input_node.accumulate_gradient(grad)?;
                        }
                    }
                    
                    // Gradient for second input (B)
                    if let Some(input_arc) = self.nodes.get(&node.inputs[1]) {
                        let mut input_node = input_arc.lock().unwrap();
                        if input_node.requires_grad {
                            let a_transposed = input_values[0].transpose()?;
                            let grad = a_transposed.matmul(grad_output)?;
                            input_node.accumulate_gradient(grad)?;
                        }
                    }
                }
                Ok(())
            }
            
            Operation::ReLU => {
                // ReLU derivative: 1 if x > 0, 0 otherwise
                if let Some(&input_id) = node.inputs.first() {
                    if let Some(input_arc) = self.nodes.get(&input_id) {
                        let mut input_node = input_arc.lock().unwrap();
                        if input_node.requires_grad {
                            let input_value = input_node.value.as_ref().unwrap();
                            let relu_derivative = crate::storage::advanced_matrix::neural::activations::ActivationOps::relu_derivative(input_value)?;
                            let grad = grad_output.elementwise_multiply(&relu_derivative)?;
                            input_node.accumulate_gradient(grad)?;
                        }
                    }
                }
                Ok(())
            }
            
            _ => Err(MatrixError::UnsupportedOperation(format!("Backward pass for {:?} not implemented", node.operation))),
        }
    }
}

impl<T: NumericType> Default for ComputationGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level tensor with automatic differentiation
pub struct AutoDiffTensor<T: NumericType> {
    pub data: UnifiedMatrix<T>,
    pub node_id: NodeId,
    pub graph: Arc<Mutex<ComputationGraph<T>>>,
    pub requires_grad: bool,
}

impl<T: NumericType> AutoDiffTensor<T> {
    /// Create a new tensor with automatic differentiation
    pub fn new(data: UnifiedMatrix<T>, requires_grad: bool) -> Self {
        let mut graph = ComputationGraph::new();
        let node_id = graph.create_leaf(data.clone(), requires_grad);
        
        Self {
            data,
            node_id,
            graph: Arc::new(Mutex::new(graph)),
            requires_grad,
        }
    }
    
    /// Create a tensor from raw data
    pub fn from_data(data: Vec<T>, shape: (usize, usize), requires_grad: bool) -> MatrixResult<Self> {
        let matrix = UnifiedMatrix::from_data(data, shape.0, shape.1)?;
        Ok(Self::new(matrix, requires_grad))
    }
    
    /// Addition operation
    pub fn add(&self, other: &Self) -> MatrixResult<Self> {
        let result_data = self.data.add(&other.data)?;
        let result_shape = result_data.shape();
        
        let mut graph = self.graph.lock().unwrap();
        let result_requires_grad = self.requires_grad || other.requires_grad;
        let result_node_id = graph.create_operation(
            Operation::Add,
            vec![self.node_id, other.node_id],
            result_shape,
            result_requires_grad,
        )?;
        
        // Store the computed result
        if let Some(node_arc) = graph.get_node(result_node_id) {
            let mut node = node_arc.lock().unwrap();
            node.value = Some(result_data.clone());
        }
        
        Ok(Self {
            data: result_data,
            node_id: result_node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: result_requires_grad,
        })
    }
    
    /// Matrix multiplication operation
    pub fn matmul(&self, other: &Self) -> MatrixResult<Self> {
        let result_data = self.data.matmul(&other.data)?;
        let result_shape = result_data.shape();
        
        let mut graph = self.graph.lock().unwrap();
        let result_requires_grad = self.requires_grad || other.requires_grad;
        let result_node_id = graph.create_operation(
            Operation::MatMul,
            vec![self.node_id, other.node_id],
            result_shape,
            result_requires_grad,
        )?;
        
        // Store the computed result
        if let Some(node_arc) = graph.get_node(result_node_id) {
            let mut node = node_arc.lock().unwrap();
            node.value = Some(result_data.clone());
        }
        
        Ok(Self {
            data: result_data,
            node_id: result_node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: result_requires_grad,
        })
    }
    
    /// Apply ReLU activation
    pub fn relu(&self) -> MatrixResult<Self> {
        let result_data = crate::storage::advanced_matrix::neural::activations::ActivationOps::relu(&self.data)?;
        let result_shape = result_data.shape();
        
        let mut graph = self.graph.lock().unwrap();
        let result_node_id = graph.create_operation(
            Operation::ReLU,
            vec![self.node_id],
            result_shape,
            self.requires_grad,
        )?;
        
        // Store the computed result
        if let Some(node_arc) = graph.get_node(result_node_id) {
            let mut node = node_arc.lock().unwrap();
            node.value = Some(result_data.clone());
        }
        
        Ok(Self {
            data: result_data,
            node_id: result_node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: self.requires_grad,
        })
    }
    
    /// Compute gradients via backward pass
    pub fn backward(&self) -> MatrixResult<()> {
        let mut graph = self.graph.lock().unwrap();
        graph.backward(self.node_id)
    }
    
    /// Get the gradient for this tensor
    pub fn grad(&self) -> Option<UnifiedMatrix<T>> {
        let graph = self.graph.lock().unwrap();
        if let Some(node_arc) = graph.get_node(self.node_id) {
            let node = node_arc.lock().unwrap();
            node.gradient.clone()
        } else {
            None
        }
    }
}

/// Gradient tape for recording operations (alternative implementation approach)
pub struct GradientTape<T: NumericType> {
    operations: Vec<(Operation, Vec<NodeId>, NodeId)>,
    tensors: HashMap<NodeId, UnifiedMatrix<T>>,
    gradients: HashMap<NodeId, UnifiedMatrix<T>>,
    next_id: NodeId,
}

impl<T: NumericType> GradientTape<T> {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            tensors: HashMap::new(),
            gradients: HashMap::new(),
            next_id: 0,
        }
    }
    
    /// Record a tensor
    pub fn watch(&mut self, tensor: UnifiedMatrix<T>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.tensors.insert(id, tensor);
        id
    }
    
    /// Record an operation
    pub fn record_operation(&mut self, op: Operation, inputs: Vec<NodeId>, output: UnifiedMatrix<T>) -> NodeId {
        let output_id = self.next_id;
        self.next_id += 1;
        
        self.operations.push((op, inputs, output_id));
        self.tensors.insert(output_id, output);
        
        output_id
    }
}

impl<T: NumericType> Default for GradientTape<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenient function for backward pass
pub fn backward_pass<T: NumericType>(
    graph: &mut ComputationGraph<T>,
    output_node: NodeId,
) -> MatrixResult<()> {
    graph.backward(output_node)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_computation_graph_creation() {
        let graph: ComputationGraph<f64> = ComputationGraph::new();
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.next_id, 0);
    }
    
    #[test]
    fn test_operation_enum() {
        let add_op = Operation::Add;
        let relu_op = Operation::ReLU;
        let conv_op = Operation::Conv2D { 
            kernel_size: (3, 3), 
            stride: (1, 1), 
            padding: (1, 1) 
        };
        
        assert!(matches!(add_op, Operation::Add));
        assert!(matches!(relu_op, Operation::ReLU));
        assert!(matches!(conv_op, Operation::Conv2D { .. }));
    }
    
    #[test]
    fn test_node_creation() {
        let node: ComputationNode<f64> = ComputationNode::new(
            0, 
            Operation::ReLU, 
            Shape::new(10, 10), 
            true
        );
        
        assert_eq!(node.id, 0);
        assert!(node.requires_grad);
        assert!(matches!(node.operation, Operation::ReLU));
    }
}