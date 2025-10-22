//! Automatic Differentiation Framework
//!
//! This module implements a basic automatic differentiation system for computing
//! gradients efficiently in neural network training and optimization.

use crate::storage::advanced_matrix::{
    numeric_type::NumericType,
    unified_matrix::{MatrixError, MatrixResult, Shape, UnifiedMatrix},
};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, Weak};

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
    Power {
        exponent: u32,
    },
    Sum {
        axis: Option<usize>,
    },
    Mean {
        axis: Option<usize>,
    },

    // Neural network operations
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Softmax {
        axis: usize,
    },

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
                adj_list
                    .entry(input_id)
                    .or_insert_with(Vec::new)
                    .push(node_id);
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
            return Err(MatrixError::ComputationError(
                "Circular dependency detected in computation graph".to_string(),
            ));
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
                Err(MatrixError::ComputationError(
                    "Failed to compute forward pass".to_string(),
                ))
            }
        } else {
            Err(MatrixError::ComputationError(
                "Output node not found".to_string(),
            ))
        }
    }

    /// Compute forward operation for a single node
    fn compute_forward_operation(
        &self,
        node: &ComputationNode<T>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        match &node.operation {
            Operation::Leaf => Err(MatrixError::ComputationError(
                "Leaf node requested for forward compute without cached value".to_string(),
            )),

            Operation::Add => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 2 {
                    return Err(MatrixError::ComputationError(
                        "Add operation requires exactly 2 inputs".to_string(),
                    ));
                }
                inputs[0].add(&inputs[1])
            }

            Operation::Multiply => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 2 {
                    return Err(MatrixError::ComputationError(
                        "Multiply operation requires exactly 2 inputs".to_string(),
                    ));
                }
                inputs[0].elementwise_multiply(&inputs[1])
            }

            Operation::MatMul => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 2 {
                    return Err(MatrixError::ComputationError(
                        "MatMul operation requires exactly 2 inputs".to_string(),
                    ));
                }
                inputs[0].matmul(&inputs[1])
            }

            Operation::ReLU => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError(
                        "ReLU operation requires exactly 1 input".to_string(),
                    ));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::relu(
                    &inputs[0],
                )
            }

            Operation::Sigmoid => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError(
                        "Sigmoid operation requires exactly 1 input".to_string(),
                    ));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::sigmoid(
                    &inputs[0],
                )
            }

            Operation::Tanh => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError(
                        "Tanh operation requires exactly 1 input".to_string(),
                    ));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::tanh(
                    &inputs[0],
                )
            }

            Operation::GELU => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError(
                        "GELU operation requires exactly 1 input".to_string(),
                    ));
                }
                crate::storage::advanced_matrix::neural::activations::ActivationOps::gelu(
                    &inputs[0],
                )
            }

            Operation::Sum { axis } => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError(
                        "Sum operation requires exactly 1 input".to_string(),
                    ));
                }
                // Implement sum reduction along specified axis
                self.reduce_sum(&inputs[0], *axis)
            }

            Operation::Transpose => {
                let inputs = self.get_input_values(&node.inputs)?;
                if inputs.len() != 1 {
                    return Err(MatrixError::ComputationError(
                        "Transpose operation requires exactly 1 input".to_string(),
                    ));
                }
                inputs[0].transpose()
            }

            _ => Err(MatrixError::UnsupportedOperation(format!(
                "Operation {:?} not implemented",
                node.operation
            ))),
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
                    return Err(MatrixError::ComputationError(format!(
                        "Input node {} has no computed value",
                        id
                    )));
                }
            } else {
                return Err(MatrixError::ComputationError(format!(
                    "Input node {} not found",
                    id
                )));
            }
        }
        Ok(inputs)
    }

    /// Reduce sum along specified axis
    fn reduce_sum(
        &self,
        input: &UnifiedMatrix<T>,
        axis: Option<usize>,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        let (rows, cols) = (input.shape().rows, input.shape().cols);
        let data = input.to_vec()?; // row-major

        match axis {
            None => {
                // Sum all elements -> (1,1)
                let mut acc_f = 0.0f64;
                for v in &data {
                    acc_f += v.to_f64(); // <-- removed .unwrap_or(0.0)
                }
                let acc_t = T::from_f64(acc_f).ok_or_else(|| {
                    MatrixError::ComputationError("reduce_sum: from_f64 failed".into())
                })?;
                UnifiedMatrix::from_data(vec![acc_t], 1, 1)
            }
            Some(0) => {
                // Sum down rows -> (1, cols)
                let mut out_f = vec![0.0f64; cols];
                for r in 0..rows {
                    for c in 0..cols {
                        out_f[c] += data[r * cols + c].to_f64(); // <-- removed .unwrap_or(0.0)
                    }
                }
                let mut out_t = Vec::with_capacity(cols);
                for f in out_f {
                    out_t.push(T::from_f64(f).ok_or_else(|| {
                        MatrixError::ComputationError("reduce_sum axis=0: from_f64 failed".into())
                    })?);
                }
                UnifiedMatrix::from_data(out_t, 1, cols)
            }
            Some(1) => {
                // Sum across cols -> (rows, 1)
                let mut out_f = vec![0.0f64; rows];
                for r in 0..rows {
                    let mut acc_f = 0.0f64;
                    for c in 0..cols {
                        acc_f += data[r * cols + c].to_f64(); // <-- removed .unwrap_or(0.0)
                    }
                    out_f[r] = acc_f;
                }
                let mut out_t = Vec::with_capacity(rows);
                for f in out_f {
                    out_t.push(T::from_f64(f).ok_or_else(|| {
                        MatrixError::ComputationError("reduce_sum axis=1: from_f64 failed".into())
                    })?);
                }
                UnifiedMatrix::from_data(out_t, rows, 1)
            }
            Some(_) => Err(MatrixError::UnsupportedOperation(
                "Only axis 0 and 1 supported for 2D matrices".to_string(),
            )),
        }
    }

    /// Backward pass (gradient computation)
    pub fn backward(&mut self, output_node: NodeId) -> MatrixResult<()> {
        // 1) Clear all previously accumulated grads (avoid silent carry-over across calls)
        for node_arc in self.nodes.values() {
            let mut n = node_arc.lock().unwrap();
            n.gradient = None;
        }

        // 2) Seed the output grad with 1s (scalar or same shape)
        if let Some(node_arc) = self.nodes.get(&output_node) {
            let mut node = node_arc.lock().unwrap();
            if node.gradient.is_none() {
                let grad = UnifiedMatrix::ones(node.shape.rows, node.shape.cols)?;
                node.set_gradient(grad)?;
            }
        } else {
            return Err(MatrixError::ComputationError(
                "Output node not found".to_string(),
            ));
        }

        // 3) Propagate in reverse topological order up to output_node
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

    fn elementwise_pow_u32(
        &self,
        base: &UnifiedMatrix<T>,
        exp: u32,
    ) -> MatrixResult<UnifiedMatrix<T>> {
        if exp == 0 {
            return UnifiedMatrix::ones(base.shape().rows, base.shape().cols);
        }
        let mut out = base.clone();
        for _ in 1..exp {
            out = out.elementwise_multiply(base)?;
        }
        Ok(out)
    }

    /// Compute backward pass for a single node
    fn compute_backward_operation(
        &self,
        node_arc: &Arc<Mutex<ComputationNode<T>>>,
    ) -> MatrixResult<()> {
        let node = node_arc.lock().unwrap();

        if !node.requires_grad || node.gradient.is_none() {
            return Ok(());
        }

        let grad_output = node.gradient.as_ref().unwrap();

        match &node.operation {
            Operation::Leaf => Ok(()), // Leaf nodes don't propagate gradients

            Operation::Add => {
                if node.inputs.len() == 2 {
                    let ids = [node.inputs[0], node.inputs[1]];
                    let gs = vec![grad_output.clone(), grad_output.clone()];
                    drop(node);
                    return self.accumulate_per_input(&ids, gs);
                }
                Ok(())
            }

            Operation::Subtract => {
                if node.inputs.len() == 2 {
                    let neg = grad_output.scalar_multiply(T::from_f64(-1.0).unwrap())?;
                    let ids = [node.inputs[0], node.inputs[1]];
                    let gs = vec![grad_output.clone(), neg];
                    drop(node);
                    return self.accumulate_per_input(&ids, gs);
                }
                Ok(())
            }

            Operation::Sum { axis } => {
                // Upstream gradient shape matches sum output.
                // We need to broadcast it back to the input shape with ones.
                let grad_out = grad_output.clone();

                // Input node (single input)
                if let Some(&inp_id) = node.inputs.first() {
                    if let Some(inp_arc) = self.nodes.get(&inp_id) {
                        let mut inp = inp_arc.lock().unwrap();
                        if inp.requires_grad {
                            let (in_r, in_c) = (inp.shape.rows, inp.shape.cols);
                            let broadcasted = match axis {
                                None => {
                                    // (1,1) -> (in_r, in_c)
                                    let g = grad_out.to_vec()?[0];
                                    UnifiedMatrix::from_data(vec![g; in_r * in_c], in_r, in_c)?
                                }
                                Some(0) => {
                                    // (1, in_c) -> tile along rows
                                    let row = grad_out.to_vec()?;
                                    let mut out = Vec::with_capacity(in_r * in_c);
                                    for _r in 0..in_r {
                                        out.extend_from_slice(&row);
                                    }
                                    UnifiedMatrix::from_data(out, in_r, in_c)?
                                }
                                Some(1) => {
                                    // (in_r, 1) -> tile along cols
                                    let col = grad_out.to_vec()?;
                                    let mut out = Vec::with_capacity(in_r * in_c);
                                    for r in 0..in_r {
                                        for _c in 0..in_c {
                                            out.push(col[r]);
                                        }
                                    }
                                    UnifiedMatrix::from_data(out, in_r, in_c)?
                                }
                                Some(_) => {
                                    return Err(MatrixError::UnsupportedOperation(
                                        "Only axis 0 and 1 supported for 2D matrices".to_string(),
                                    ))
                                }
                            };
                            inp.accumulate_gradient(broadcasted)?;
                        }
                    }
                }
                Ok(())
            }

            Operation::Multiply => {
                if node.inputs.len() == 2 {
                    let ids = [node.inputs[0], node.inputs[1]];
                    let vals = self.get_input_values(&ids)?;

                    // g0 -> grad wrt first input: g * b
                    // g1 -> grad wrt second input: g * a
                    let g0 = grad_output.elementwise_multiply(&vals[1])?;
                    let g1 = grad_output.elementwise_multiply(&vals[0])?;

                    // Release the lock on the current node before mutating others
                    drop(node);

                    // This handles both distinct and duplicate ids (e.g., x*x) correctly.
                    return self.accumulate_per_input(&ids, vec![g0, g1]);
                }
                Ok(())
            }

            Operation::Power { exponent } => {
                if node.inputs.len() == 1 {
                    if let Some(input_arc) = self.nodes.get(&node.inputs[0]) {
                        let mut input_node = input_arc.lock().unwrap();
                        if input_node.requires_grad {
                            let a = input_node.value.as_ref().ok_or_else(|| {
                                MatrixError::ComputationError(
                                    "Missing forward value for Power input".to_string(),
                                )
                            })?;
                            match *exponent {
                                0 => {
                                    // d(1)/da = 0
                                    let z = UnifiedMatrix::zeros(
                                        grad_output.shape().rows,
                                        grad_output.shape().cols,
                                    )?;
                                    input_node.accumulate_gradient(z)?;
                                }
                                1 => {
                                    // d(a)/da = 1
                                    input_node.accumulate_gradient(grad_output.clone())?;
                                }
                                n => {
                                    // d(a^n)/da = n * a^(n-1) * grad_out
                                    let a_pow = self.elementwise_pow_u32(a, n - 1)?;
                                    let n_scalar = T::from_f64(n as f64).unwrap();
                                    let da = a_pow.scale(n_scalar)?; // n * a^(n-1)
                                    let g = grad_output.elementwise_multiply(&da)?; // chain rule
                                    input_node.accumulate_gradient(g)?;
                                }
                            }
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

            _ => Err(MatrixError::UnsupportedOperation(format!(
                "Backward pass for {:?} not implemented",
                node.operation
            ))),
        }
    }

    // Put this near other private helpers (impl<T: NumericType> ComputationGraph<T>)
    fn accumulate_per_input(
        &self,
        ids: &[NodeId],
        grads: Vec<UnifiedMatrix<T>>,
    ) -> MatrixResult<()> {
        use std::collections::HashMap;
        let mut merged: HashMap<NodeId, UnifiedMatrix<T>> = HashMap::new();

        for (i, &id) in ids.iter().enumerate() {
            if let Some(entry) = merged.get_mut(&id) {
                *entry = entry.add(&grads[i])?;
            } else {
                merged.insert(id, grads[i].clone());
            }
        }

        for (id, g) in merged {
            if let Some(n_arc) = self.nodes.get(&id) {
                let mut n = n_arc.lock().unwrap();
                if n.requires_grad {
                    n.accumulate_gradient(g)?;
                }
            }
        }
        Ok(())
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
    pub fn from_data(
        data: Vec<T>,
        shape: (usize, usize),
        requires_grad: bool,
    ) -> MatrixResult<Self> {
        let matrix = UnifiedMatrix::from_data(data, shape.0, shape.1)?;
        Ok(Self::new(matrix, requires_grad))
    }

    /// Create a leaf in an existing graph (same semantics as `new`, but bound to `graph`)
    pub fn new_in_graph(
        graph: &Arc<Mutex<ComputationGraph<T>>>,
        data: UnifiedMatrix<T>,
        requires_grad: bool,
    ) -> Self {
        let mut g = graph.lock().unwrap();
        let id = g.create_leaf(data.clone(), requires_grad);
        drop(g);
        Self {
            data,
            node_id: id,
            graph: Arc::clone(graph),
            requires_grad,
        }
    }

    /// Clone this tensor into `target` graph, preserving requires_grad (explicit on purpose).
    pub fn clone_to_graph(&self, target: &Arc<Mutex<ComputationGraph<T>>>) -> MatrixResult<Self> {
        let mut g = target.lock().unwrap();
        let id = g.create_leaf(self.data.clone(), self.requires_grad);
        drop(g);
        Ok(Self {
            data: self.data.clone(),
            node_id: id,
            graph: Arc::clone(target),
            requires_grad: self.requires_grad,
        })
    }

    /// Convenience: make a sibling leaf (same graph as `self`)
    pub fn like_new(&self, data: UnifiedMatrix<T>, requires_grad: bool) -> Self {
        Self::new_in_graph(&self.graph, data, requires_grad)
    }

    /// Create a tensor from raw data in an existing graph
    pub fn from_data_in_graph(
        graph: &Arc<Mutex<ComputationGraph<T>>>,
        data: Vec<T>,
        shape: (usize, usize),
        requires_grad: bool,
    ) -> MatrixResult<Self> {
        let matrix = UnifiedMatrix::from_data(data, shape.0, shape.1)?;
        Ok(Self::new_in_graph(graph, matrix, requires_grad))
    }

    /// Convenience: create a sibling tensor from raw data (same graph as `self`)
    pub fn like_from_data(
        &self,
        data: Vec<T>,
        shape: (usize, usize),
        requires_grad: bool,
    ) -> MatrixResult<Self> {
        let matrix = UnifiedMatrix::from_data(data, shape.0, shape.1)?;
        Ok(self.like_new(matrix, requires_grad))
    }

    /// (Optional) expose the graph so engines can bind to it
    pub fn graph_arc(&self) -> Arc<Mutex<ComputationGraph<T>>> {
        Arc::clone(&self.graph)
    }

    #[allow(dead_code)]
    fn ensure_same_graph(&self, other: &Self) -> MatrixResult<()> {
        if !Arc::ptr_eq(&self.graph, &other.graph) {
            return Err(MatrixError::ComputationError(
                "Binary operation on tensors from different graphs is not supported (merge graphs or create tensors in the same session)".to_string()
            ));
        }
        Ok(())
    }

    fn to_graph_const_only(&self, target: &Arc<Mutex<ComputationGraph<T>>>) -> MatrixResult<Self> {
        if self.requires_grad {
            return Err(MatrixError::ComputationError(
                "Cross-graph op with a grad-requiring tensor is not supported. \
                 Create tensors in the same session/graph."
                    .to_string(),
            ));
        }
        let mut g = target.lock().unwrap();
        let new_id = g.create_leaf(self.data.clone(), false);
        drop(g);
        Ok(Self {
            data: self.data.clone(),
            node_id: new_id,
            graph: Arc::clone(target),
            requires_grad: false,
        })
    }

    fn ensure_same_graph_or_import_const(&self, other: &Self) -> MatrixResult<Self> {
        if Arc::ptr_eq(&self.graph, &other.graph) {
            // Same graph: if other is a constant and would alias our node, force distinct leaf
            if !other.requires_grad && other.node_id == self.node_id {
                return other.to_graph_const_only(&self.graph);
            }
            // Reuse as-is
            return Ok(Self {
                data: other.data.clone(),
                node_id: other.node_id,
                graph: Arc::clone(&self.graph),
                requires_grad: other.requires_grad,
            });
        }

        // Different graphs: import constants; error if other needs grad
        if other.requires_grad {
            return Err(MatrixError::ComputationError(
                "Binary op on tensors from different graphs where the other requires grad is not supported"
                    .into(),
            ));
        }
        other.to_graph_const_only(&self.graph)
    }

    pub fn add(&self, other: &Self) -> MatrixResult<Self> {
        let other_in_self = self.ensure_same_graph_or_import_const(other)?;
        let result_data = self.data.add(&other_in_self.data)?;
        let result_shape = result_data.shape();

        let mut graph = self.graph.lock().unwrap();
        let result_requires_grad = self.requires_grad || other_in_self.requires_grad;
        let result_node_id = graph.create_operation(
            Operation::Add,
            vec![self.node_id, other_in_self.node_id],
            result_shape,
            result_requires_grad,
        )?;
        if let Some(node_arc) = graph.get_node(result_node_id) {
            node_arc.lock().unwrap().value = Some(result_data.clone());
        }
        Ok(Self {
            data: result_data,
            node_id: result_node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: result_requires_grad,
        })
    }

    pub fn subtract(&self, other: &Self) -> MatrixResult<Self> {
        let other_in_self = self.ensure_same_graph_or_import_const(other)?;
        let result_data = self.data.subtract(&other_in_self.data)?;
        let result_shape = result_data.shape();

        let mut graph = self.graph.lock().unwrap();
        let result_requires_grad = self.requires_grad || other_in_self.requires_grad;
        let result_node_id = graph.create_operation(
            Operation::Subtract,
            vec![self.node_id, other_in_self.node_id],
            result_shape,
            result_requires_grad,
        )?;
        if let Some(node_arc) = graph.get_node(result_node_id) {
            node_arc.lock().unwrap().value = Some(result_data.clone());
        }
        Ok(Self {
            data: result_data,
            node_id: result_node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: result_requires_grad,
        })
    }

    pub fn multiply(&self, other: &Self) -> MatrixResult<Self> {
        let other_in_self = self.ensure_same_graph_or_import_const(other)?;

        println!(
            "[mul:pre-op] self_id={}, other_in_self_id={}, same_graph={}",
            self.node_id,
            other_in_self.node_id,
            Arc::ptr_eq(&self.graph, &other_in_self.graph)
        );
        // debug_assert_ne!(
        //     self.node_id, other_in_self.node_id,
        //     "Duplicate input ids in multiply; import failed"
        // );

        let result_data = self.data.elementwise_multiply(&other_in_self.data)?;
        let result_shape = result_data.shape();

        let mut graph = self.graph.lock().unwrap();
        let result_requires_grad = self.requires_grad || other_in_self.requires_grad;
        let result_node_id = graph.create_operation(
            Operation::Multiply,
            vec![self.node_id, other_in_self.node_id],
            result_shape,
            result_requires_grad,
        )?;

        println!(
            "[mul:op] inputs=({}, {}), out={}",
            self.node_id, other_in_self.node_id, result_node_id
        );
        if let Some(node_arc) = graph.get_node(result_node_id) {
            node_arc.lock().unwrap().value = Some(result_data.clone());
        }
        Ok(Self {
            data: result_data,
            node_id: result_node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: result_requires_grad,
        })
    }

    pub fn matmul(&self, other: &Self) -> MatrixResult<Self> {
        let other_in_self = self.ensure_same_graph_or_import_const(other)?;
        let result_data = self.data.matmul(&other_in_self.data)?;
        let result_shape = result_data.shape();

        let mut graph = self.graph.lock().unwrap();
        let result_requires_grad = self.requires_grad || other_in_self.requires_grad;
        let result_node_id = graph.create_operation(
            Operation::MatMul,
            vec![self.node_id, other_in_self.node_id],
            result_shape,
            result_requires_grad,
        )?;
        if let Some(node_arc) = graph.get_node(result_node_id) {
            node_arc.lock().unwrap().value = Some(result_data.clone());
        }
        Ok(Self {
            data: result_data,
            node_id: result_node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: result_requires_grad,
        })
    }

    /// Sum reduction operation
    pub fn sum(&self, axis: Option<usize>) -> MatrixResult<Self> {
        let (rows, cols) = (self.data.shape().rows, self.data.shape().cols);
        let result_shape = match axis {
            None => Shape::new(1, 1),
            Some(0) => Shape::new(1, cols),
            Some(1) => Shape::new(rows, 1),
            Some(_) => {
                return Err(MatrixError::UnsupportedOperation(
                    "Only axis 0, 1, or None supported for sum".to_string(),
                ))
            }
        };

        // Use the graph op so forward/backward stay consistent with the node graph
        let mut graph = self.graph.lock().unwrap();
        let node_id = graph.create_operation(
            Operation::Sum { axis },
            vec![self.node_id],
            result_shape,
            self.requires_grad,
        )?;

        // Compute forward value here via the same reduction logic
        let result_data = graph.reduce_sum(&self.data, axis)?;
        if let Some(node_arc) = graph.get_node(node_id) {
            let mut node = node_arc.lock().unwrap();
            node.value = Some(result_data.clone());
        }

        Ok(Self {
            data: result_data,
            node_id,
            graph: Arc::clone(&self.graph),
            requires_grad: self.requires_grad,
        })
    }

    /// Apply ReLU activation
    pub fn relu(&self) -> MatrixResult<Self> {
        let result_data =
            crate::storage::advanced_matrix::neural::activations::ActivationOps::relu(&self.data)?;
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
    #[allow(dead_code)]
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
    pub fn record_operation(
        &mut self,
        op: Operation,
        inputs: Vec<NodeId>,
        output: UnifiedMatrix<T>,
    ) -> NodeId {
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
            padding: (1, 1),
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
            crate::storage::advanced_matrix::unified_matrix::Shape::new(10, 10),
            true,
        );

        assert_eq!(node.id, 0);
        assert!(node.requires_grad);
        assert!(matches!(node.operation, Operation::ReLU));
    }

    #[test]
    fn test_autodiff_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = AutoDiffTensor::from_data(data.clone(), (2, 2), true).unwrap();

        assert!(tensor.requires_grad);
        assert_eq!(tensor.data.rows(), 2);
        assert_eq!(tensor.data.cols(), 2);

        let extracted_data = tensor.data.to_vec().unwrap();
        assert_eq!(extracted_data, data);
    }

    #[test]
    fn test_simple_addition_forward() {
        let a_data = vec![1.0, 2.0];
        let b_data = vec![3.0, 4.0];

        let a = AutoDiffTensor::from_data(a_data, (1, 2), true).unwrap();
        let b = a.like_from_data(b_data, (1, 2), true).unwrap();

        let c = a.add(&b).unwrap();
        let result_data = c.data.to_vec().unwrap();

        assert_eq!(result_data, vec![4.0, 6.0]);
        assert!(c.requires_grad);
    }

    #[test]
    fn test_simple_subtraction_forward() {
        let a_data = vec![5.0, 8.0];
        let b_data = vec![2.0, 3.0];

        let a = AutoDiffTensor::from_data(a_data, (1, 2), true).unwrap();
        let b = a.like_from_data(b_data, (1, 2), true).unwrap();

        let c = a.subtract(&b).unwrap();
        let result_data = c.data.to_vec().unwrap();

        assert_eq!(result_data, vec![3.0, 5.0]);
        assert!(c.requires_grad);
    }

    #[test]
    fn test_simple_multiplication_forward() {
        let a_data = vec![2.0, 3.0];
        let b_data = vec![4.0, 5.0];

        let a = AutoDiffTensor::from_data(a_data, (1, 2), true).unwrap();
        let b = a.like_from_data(b_data, (1, 2), true).unwrap();

        let c = a.multiply(&b).unwrap();
        let result_data = c.data.to_vec().unwrap();

        assert_eq!(result_data, vec![8.0, 15.0]);
        assert!(c.requires_grad);
    }

    #[test]
    fn test_chain_operations() {
        let a_data = vec![1.0, 2.0];
        let b_data = vec![3.0, 4.0];
        let c_data = vec![0.5, 0.5];

        let a = AutoDiffTensor::from_data(a_data, (1, 2), true).unwrap();
        let b = a.like_from_data(b_data, (1, 2), true).unwrap();
        let c = a.like_from_data(c_data, (1, 2), true).unwrap();

        // Compute (a + b) * c
        let sum = a.add(&b).unwrap();
        let result = sum.multiply(&c).unwrap();

        let result_data = result.data.to_vec().unwrap();
        assert_eq!(result_data, vec![2.0, 3.0]); // (1+3)*0.5=2, (2+4)*0.5=3
    }

    #[test]
    fn test_gradient_computation_simple() {
        // Test backward pass: z = x + y, dz/dx = 1, dz/dy = 1
        let x_data = vec![2.0];
        let y_data = vec![3.0];

        let x = AutoDiffTensor::from_data(x_data, (1, 1), true).unwrap();
        let y = x.like_from_data(y_data, (1, 1), true).unwrap();

        let z = x.add(&y).unwrap();
        assert_eq!(z.data.to_vec().unwrap(), vec![5.0]);

        // Compute gradients
        z.backward().unwrap();

        // Check gradients
        let x_grad = x.grad().unwrap();
        let y_grad = y.grad().unwrap();

        assert_eq!(x_grad.to_vec().unwrap(), vec![1.0]);
        assert_eq!(y_grad.to_vec().unwrap(), vec![1.0]);
    }

    #[test]
    fn test_gradient_computation_subtract() {
        // Test backward pass: z = x - y, dz/dx = 1, dz/dy = -1
        let x_data = vec![5.0];
        let y_data = vec![2.0];

        let x = AutoDiffTensor::from_data(x_data, (1, 1), true).unwrap();
        let y = x.like_from_data(y_data, (1, 1), true).unwrap();

        let z = x.subtract(&y).unwrap();
        assert_eq!(z.data.to_vec().unwrap(), vec![3.0]);

        // Compute gradients
        z.backward().unwrap();

        // Check gradients
        let x_grad = x.grad().unwrap();
        let y_grad = y.grad().unwrap();

        assert_eq!(x_grad.to_vec().unwrap(), vec![1.0]);
        assert_eq!(y_grad.to_vec().unwrap(), vec![-1.0]);
    }

    #[test]
    fn test_gradient_computation_multiply() {
        // Test backward pass: z = x * y, dz/dx = y, dz/dy = x
        let x_data = vec![3.0];
        let y_data = vec![4.0];

        let x = AutoDiffTensor::from_data(x_data.clone(), (1, 1), true).unwrap();
        let y = x.like_from_data(y_data.clone(), (1, 1), true).unwrap();

        let z = x.multiply(&y).unwrap();
        assert_eq!(z.data.to_vec().unwrap(), vec![12.0]);

        // Compute gradients
        z.backward().unwrap();

        // Check gradients: dz/dx = y = 4, dz/dy = x = 3
        let x_grad = x.grad().unwrap();
        let y_grad = y.grad().unwrap();

        assert_eq!(x_grad.to_vec().unwrap(), vec![4.0]);
        assert_eq!(y_grad.to_vec().unwrap(), vec![3.0]);
    }

    #[test]
    fn test_chain_rule() {
        // Test chain rule: z = (x + y) * (x - y), dz/dx = (x-y) + (x+y) = 2x
        let x_data = vec![5.0];
        let y_data = vec![2.0];

        let x = AutoDiffTensor::from_data(x_data.clone(), (1, 1), true).unwrap();
        let y = x.like_from_data(y_data.clone(), (1, 1), true).unwrap();

        let sum = x.add(&y).unwrap(); // x + y = 7
        let diff = x.subtract(&y).unwrap(); // x - y = 3
        let z = sum.multiply(&diff).unwrap(); // (x+y)*(x-y) = 21

        assert_eq!(z.data.to_vec().unwrap(), vec![21.0]);

        // Compute gradients
        z.backward().unwrap();

        // Check gradients: dz/dx = 2x = 10
        let x_grad = x.grad().unwrap();
        assert_eq!(x_grad.to_vec().unwrap(), vec![10.0]);
    }

    #[test]
    fn test_matmul_forward() {
        // Test matrix multiplication: C = A @ B
        let a_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b_data = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix

        let a = AutoDiffTensor::from_data(a_data, (2, 2), true).unwrap();
        let b = a.like_from_data(b_data, (2, 2), true).unwrap();

        let c = a.matmul(&b).unwrap();
        let result = c.data.to_vec().unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_relu_forward() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = AutoDiffTensor::from_data(data, (1, 5), true).unwrap();

        let output = tensor.relu().unwrap();
        let result = output.data.to_vec().unwrap();

        // ReLU: max(0, x)
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sum_reduction() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let tensor = AutoDiffTensor::from_data(data, (2, 3), true).unwrap();

        // Test sum along axis 0 (columns)
        let sum_axis0 = tensor.sum(Some(0)).unwrap();
        assert_eq!(sum_axis0.data.shape().rows, 1);
        assert_eq!(sum_axis0.data.shape().cols, 3);

        // Test sum along axis 1 (rows)
        let sum_axis1 = tensor.sum(Some(1)).unwrap();
        assert_eq!(sum_axis1.data.shape().rows, 2);
        assert_eq!(sum_axis1.data.shape().cols, 1);

        // Test sum all elements
        let sum_all = tensor.sum(None).unwrap();
        assert_eq!(sum_all.data.shape().rows, 1);
        assert_eq!(sum_all.data.shape().cols, 1);
    }

    #[test]
    fn test_numerical_gradient_check() {
        // Numerical gradient checking: compare autodiff gradients with finite differences
        let x_val = 3.0;
        let y_val = 2.0;
        let h = 1e-5; // Small step for finite differences

        // Function: f(x, y) = x^2 + x*y + y^2 (approximated as x*x + x*y + y*y)
        let x = AutoDiffTensor::from_data(vec![x_val], (1, 1), true).unwrap();
        let y = x.like_from_data(vec![y_val], (1, 1), true).unwrap();

        let x2 = x.multiply(&x).unwrap();
        let xy = x.multiply(&y).unwrap();
        let y2 = y.multiply(&y).unwrap();
        let temp = x2.add(&xy).unwrap();
        let f = temp.add(&y2).unwrap();

        f.backward().unwrap();

        let autodiff_grad_x = x.grad().unwrap().to_vec().unwrap()[0];
        let autodiff_grad_y = y.grad().unwrap().to_vec().unwrap()[0];

        // Numerical gradients
        // df/dx ≈ (f(x+h, y) - f(x-h, y)) / (2h)
        let f_plus_x = (x_val + h) * (x_val + h) + (x_val + h) * y_val + y_val * y_val;
        let f_minus_x = (x_val - h) * (x_val - h) + (x_val - h) * y_val + y_val * y_val;
        let numerical_grad_x = (f_plus_x - f_minus_x) / (2.0 * h);

        let f_plus_y = x_val * x_val + x_val * (y_val + h) + (y_val + h) * (y_val + h);
        let f_minus_y = x_val * x_val + x_val * (y_val - h) + (y_val - h) * (y_val - h);
        let numerical_grad_y = (f_plus_y - f_minus_y) / (2.0 * h);

        // Check that autodiff and numerical gradients are close
        assert!(
            (autodiff_grad_x - numerical_grad_x).abs() < 1e-3,
            "Gradient mismatch for x: autodiff={}, numerical={}",
            autodiff_grad_x,
            numerical_grad_x
        );
        assert!(
            (autodiff_grad_y - numerical_grad_y).abs() < 1e-3,
            "Gradient mismatch for y: autodiff={}, numerical={}",
            autodiff_grad_y,
            numerical_grad_y
        );
    }

    #[test]
    fn test_graph_topology_order() {
        let x = AutoDiffTensor::from_data(vec![1.0], (1, 1), true).unwrap();
        let y = x.like_from_data(vec![2.0], (1, 1), true).unwrap();

        let z1 = x.add(&y).unwrap();
        let z2 = x.multiply(&y).unwrap();
        let final_result = z1.add(&z2).unwrap();

        // This should not panic - tests that graph topology is correct
        final_result.backward().unwrap();

        // All tensors should have gradients
        assert!(x.grad().is_some());
        assert!(y.grad().is_some());
    }

    #[test]
    fn grad_scalar_times_scalar_const() {
        let x = AutoDiffTensor::from_data(vec![1.5], (1, 1), true).unwrap();
        let two = x.like_from_data(vec![2.0], (1, 1), false).unwrap();
        let u = x.multiply(&two).unwrap(); // u = 2x
        u.backward().unwrap();
        let g = x.grad().unwrap().get(0, 0).unwrap().to_f64();
        assert!((g - 2.0).abs() < 1e-12);
    }

    #[test]
    fn grad_square_via_duplicate_input() {
        let x = AutoDiffTensor::from_data(vec![3.0], (1, 1), true).unwrap();
        let y = x.multiply(&x).unwrap(); // y = x^2
        y.backward().unwrap();
        let g = x.grad().unwrap().get(0, 0).unwrap().to_f64();
        assert!((g - 6.0).abs() < 1e-12); // 2x at x=3
    }

    #[test]
    fn dbg_mul_scalar_grad() {
        use super::AutoDiffTensor;
        let x = AutoDiffTensor::from_data(vec![1.5], (1, 1), true).unwrap();
        let two = AutoDiffTensor::from_data(vec![2.0], (1, 1), false).unwrap();
        let u = x.multiply(&two).unwrap();
        u.backward().unwrap();
        let g = x.grad().unwrap().get(0, 0).unwrap();
        eprintln!("[DBG Mul] du/dx = {}", g);
        assert!((g.to_f64() - 2.0).abs() < 1e-12);
    }
}
