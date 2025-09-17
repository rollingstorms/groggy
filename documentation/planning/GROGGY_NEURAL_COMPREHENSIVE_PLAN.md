# Groggy Neural Network Module - Comprehensive Implementation Plan

## 🎯 **Executive Summary**

This document outlines the complete implementation plan for `groggy.neural` - a high-performance neural network framework built on groggy's graph-native architecture. The module will provide PyTorch-style APIs while leveraging groggy's unique strengths: graph topology analysis, columnar storage, and automatic differentiation.

## 🚨 **Critical Foundation Issues to Address First**

### **Matrix Operation Return Type Inconsistencies**

**PROBLEM**: Current matrix operations return inconsistent types that break neural network gradient flow:

```rust
// ❌ BROKEN - Returns Vec<f64> instead of GraphMatrix
pub fn eigenvalue_decomposition(&self) -> GraphResult<(Vec<f64>, GraphMatrix<T>)>

// ❌ BROKEN - Non-matrix returns prevent gradient tracking
pub fn sum_axis(&self, axis: Axis) -> GraphResult<Vec<T>>
pub fn mean(&self) -> GraphResult<T>  // Returns scalar, not 1x1 matrix
```

**SOLUTION**: All matrix operations must return GraphMatrix types for gradient compatibility:

```rust
// ✅ FIXED - Always return matrices
pub fn eigenvalue_decomposition(&self) -> GraphResult<(GraphMatrix<T>, GraphMatrix<T>)>
pub fn sum_axis(&self, axis: Axis) -> GraphResult<GraphMatrix<T>>  // Column/row vector
pub fn mean(&self) -> GraphResult<GraphMatrix<T>>  // 1x1 matrix
```

**IMPACT**: Without this fix, neural networks cannot compute gradients through:
- Eigenvalue/eigenvector computations (PCA, attention mechanisms)
- Reduction operations (mean pooling, global average pooling)
- Statistical operations (batch normalization)

### **Gradient Compatibility Audit**

**All 118 matrix operations must support gradients. Current status:**

| Category | Operations | Gradient Support | Issues |
|----------|------------|------------------|--------|
| **Basic Math** | `+`, `-`, `*`, `/`, `**` | ✅ Complete | None |
| **Linear Algebra** | `matmul`, `transpose`, `inverse` | ✅ Complete | None |  
| **Decompositions** | `lu`, `qr`, `cholesky`, `svd` | ✅ Complete | None |
| **Reductions** | `sum`, `mean`, `min`, `max` | ❌ **BROKEN** | Return scalars/vectors |
| **Eigenvalues** | `eigenvalue_decomposition` | ❌ **BROKEN** | Returns Vec<f64> |
| **Reshaping** | `reshape`, `concatenate`, `stack` | ⚠️ **UNTESTED** | Need gradient tests |
| **Neural Ops** | `relu`, `sigmoid`, `tanh` | ✅ Complete | None |

## 🏗️ **Architecture Overview**

### **Design Philosophy**

1. **Graph-Enhanced Hybrid**: Traditional neural APIs with graph topology superpowers
2. **Zero-Copy Integration**: Neural tensors are GraphMatrix wrappers, not copies
3. **Gradient-First**: Every operation preserves gradient information
4. **Performance-Native**: Rust core with Python ergonomics
5. **Groggy-Distinctive**: Leverage unique graph analysis capabilities

### **Three-Tier Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON API LAYER                        │
│  groggy.neural.{layers, optim, loss, functional, models}   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      FFI LAYER                             │
│     python-groggy/src/ffi/neural/*.rs                     │
│     Pure bindings, zero business logic                     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     RUST CORE LAYER                        │
│      src/neural/{tensor, layers, optim, loss, graph}       │
│     All algorithms, computation graphs, performance        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 **Complete File Structure**

### **Rust Core (`src/neural/`)**

```
src/neural/
├── mod.rs                     # Public exports and module structure
├── tensor.rs                  # Tensor<T> wrapper around GraphMatrix
├── parameter.rs               # Parameter<T> with optimizer state
├── module.rs                  # Module trait and base implementations
├── graph_module.rs            # Graph-based neural networks (unique feature)
├── layers/
│   ├── mod.rs                 # Layer exports
│   ├── linear.rs              # Dense/fully-connected layers
│   ├── convolution.rs         # Conv1d, Conv2d, Conv3d
│   ├── normalization.rs       # BatchNorm, LayerNorm, GroupNorm
│   ├── activation.rs          # ReLU, GELU, Swish, Mish (differentiable)
│   ├── attention.rs           # MultiHeadAttention, SelfAttention
│   ├── transformer.rs         # TransformerEncoderLayer, etc.
│   ├── message_passing.rs     # GraphConv, GCN, GraphSAGE (groggy specialty)
│   ├── pooling.rs             # MaxPool, AvgPool, GlobalPool
│   ├── dropout.rs             # Dropout, DropPath
│   ├── embedding.rs           # Embedding, PositionalEmbedding
│   └── recurrent.rs           # LSTM, GRU, RNN
├── optim/
│   ├── mod.rs                 # Optimizer exports
│   ├── optimizer.rs           # Optimizer trait
│   ├── sgd.rs                 # SGD with momentum
│   ├── adam.rs                # Adam, AdamW
│   ├── rmsprop.rs             # RMSprop
│   ├── adagrad.rs             # Adagrad
│   └── scheduler.rs           # Learning rate scheduling
├── loss/
│   ├── mod.rs                 # Loss function exports
│   ├── mse.rs                 # Mean Squared Error
│   ├── cross_entropy.rs       # CrossEntropyLoss, NLLLoss
│   ├── bce.rs                 # Binary Cross Entropy
│   ├── huber.rs               # Huber Loss
│   └── custom.rs              # Custom loss function framework
├── functional/
│   ├── mod.rs                 # Functional operations
│   ├── activations.rs         # Functional activation functions
│   ├── convolution.rs         # Functional conv operations
│   ├── pooling.rs             # Functional pooling operations
│   └── linear.rs              # Functional linear operations
├── models/
│   ├── mod.rs                 # Pre-built model exports
│   ├── sequential.rs          # Sequential model container
│   ├── graph_model.rs         # Graph-based model (unique)
│   └── module_list.rs         # ModuleList, ModuleDict
├── training/
│   ├── mod.rs                 # Training utilities
│   ├── loop.rs                # Training loop abstractions
│   ├── metrics.rs             # Accuracy, F1, etc.
│   ├── callbacks.rs           # EarlyStopping, ModelCheckpoint
│   └── data.rs                # DataLoader integration
├── serialization/
│   ├── mod.rs                 # Model saving/loading
│   ├── state_dict.rs          # Parameter serialization
│   └── checkpoint.rs          # Training checkpoint management
└── utils/
    ├── mod.rs                 # Utility exports
    ├── initialization.rs      # Xavier, Kaiming, etc.
    ├── regularization.rs      # L1, L2 regularization
    └── gradient_utils.rs      # Gradient clipping, analysis
```

### **FFI Bindings (`python-groggy/src/ffi/neural/`)**

```
python-groggy/src/ffi/neural/
├── mod.rs                     # Neural FFI exports
├── tensor.rs                  # PyTensor bindings
├── parameter.rs               # PyParameter bindings
├── module.rs                  # PyModule bindings
├── layers/
│   ├── mod.rs                 # Layer FFI exports
│   ├── linear.rs              # PyLinear
│   ├── convolution.rs         # PyConv2d, etc.
│   ├── attention.rs           # PyMultiHeadAttention
│   ├── transformer.rs         # PyTransformerEncoderLayer
│   └── message_passing.rs     # PyGraphConv (groggy specialty)
├── optim/
│   ├── mod.rs                 # Optimizer FFI exports
│   ├── sgd.rs                 # PySGD
│   └── adam.rs                # PyAdam
├── loss/
│   ├── mod.rs                 # Loss FFI exports
│   ├── mse.rs                 # PyMSELoss
│   └── cross_entropy.rs       # PyCrossEntropyLoss
└── functional/
    ├── mod.rs                 # Functional FFI exports
    └── activations.rs         # Functional activation bindings
```

### **Python API (`python-groggy/python/groggy/neural/`)**

```
python-groggy/python/groggy/neural/
├── __init__.py                # Main neural exports
├── tensor.py                  # Tensor class with numpy-like API
├── parameter.py               # Parameter class
├── module.py                  # Module base class
├── layers/
│   ├── __init__.py            # Layer exports
│   ├── linear.py              # Linear layer
│   ├── convolution.py         # Conv2d, etc.
│   ├── normalization.py       # BatchNorm2d, etc.
│   ├── attention.py           # MultiHeadAttention
│   ├── transformer.py         # TransformerEncoderLayer
│   └── message_passing.py     # Graph neural network layers
├── optim/
│   ├── __init__.py            # Optimizer exports
│   ├── sgd.py                 # SGD optimizer
│   ├── adam.py                # Adam optimizer
│   └── scheduler.py           # Learning rate schedulers
├── loss/
│   ├── __init__.py            # Loss function exports
│   ├── mse.py                 # MSELoss
│   ├── cross_entropy.py       # CrossEntropyLoss
│   └── custom.py              # Custom loss framework
├── functional/
│   ├── __init__.py            # Functional exports
│   ├── activations.py         # relu, gelu, etc.
│   ├── convolution.py         # conv2d, etc.
│   └── pooling.py             # max_pool2d, etc.
├── models/
│   ├── __init__.py            # Model exports
│   ├── sequential.py          # Sequential model
│   └── graph_model.py         # Graph-based models (unique)
├── training/
│   ├── __init__.py            # Training exports
│   ├── trainer.py             # High-level training interface
│   └── callbacks.py           # Training callbacks
└── utils/
    ├── __init__.py            # Utility exports
    ├── data.py                # DataLoader utilities
    └── visualization.py       # Neural network visualization
```

## 🧮 **Core Data Structures**

### **Tensor<T> - Neural Matrix Wrapper**

```rust
/// Neural tensor built on GraphMatrix with autodiff support
pub struct Tensor<T: NumericType> {
    /// Underlying matrix data with computation graph
    matrix: GraphMatrix<T>,
    
    /// Gradient function for backward pass
    grad_fn: Option<Arc<dyn GradientFunction<T>>>,
    
    /// Device placement (CPU/GPU) - future extensibility  
    device: Device,
    
    /// Data type information
    dtype: DType,
    
    /// Training vs evaluation mode flag
    training: bool,
}

impl<T: NumericType> Tensor<T> {
    // Creation
    pub fn new(data: GraphMatrix<T>) -> Self
    pub fn zeros(shape: (usize, usize)) -> Self
    pub fn ones(shape: (usize, usize)) -> Self
    pub fn randn(shape: (usize, usize)) -> Self
    
    // Properties
    pub fn shape(&self) -> (usize, usize)
    pub fn requires_grad(&self) -> bool
    pub fn grad(&self) -> Option<&Tensor<T>>
    
    // Operations (all return Tensor for gradient tracking)
    pub fn add(&self, other: &Tensor<T>) -> GraphResult<Tensor<T>>
    pub fn matmul(&self, other: &Tensor<T>) -> GraphResult<Tensor<T>>
    pub fn relu(&self) -> GraphResult<Tensor<T>>
    
    // Autodiff
    pub fn backward(&mut self) -> GraphResult<()>
    pub fn zero_grad(&mut self)
    
    // Conversion
    pub fn detach(&self) -> Tensor<T>  // Remove from computation graph
    pub fn to_matrix(&self) -> &GraphMatrix<T>
}
```

### **Parameter<T> - Trainable Tensor**

```rust
/// Trainable parameter with optimizer state
pub struct Parameter<T: NumericType> {
    /// Parameter data (always requires_grad=true)
    tensor: Tensor<T>,
    
    /// Optimizer-specific state (momentum, running averages, etc.)
    optimizer_state: HashMap<String, f64>,
    
    /// Parameter name for debugging/serialization
    name: Option<String>,
}

impl<T: NumericType> Parameter<T> {
    pub fn new(tensor: Tensor<T>) -> Self
    pub fn from_matrix(matrix: GraphMatrix<T>) -> Self
    
    // Access
    pub fn data(&self) -> &Tensor<T>
    pub fn data_mut(&mut self) -> &mut Tensor<T>
    pub fn grad(&self) -> Option<&Tensor<T>>
    
    // Optimizer state management
    pub fn get_state(&self, key: &str) -> Option<f64>
    pub fn set_state(&mut self, key: &str, value: f64)
    pub fn clear_state(&mut self)
    
    // Updates
    pub fn apply_gradient(&mut self, learning_rate: f64) -> GraphResult<()>
    pub fn zero_grad(&mut self)
}
```

### **Module Trait - Layer Abstraction**

```rust
/// Base trait for all neural network layers
pub trait Module<T: NumericType>: Send + Sync {
    /// Forward pass computation
    fn forward(&self, input: &Tensor<T>) -> GraphResult<Tensor<T>>;
    
    /// Get all trainable parameters
    fn parameters(&self) -> Vec<&Parameter<T>>;
    
    /// Get mutable parameter references for optimization
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>>;
    
    /// Set training mode
    fn train(&mut self);
    
    /// Set evaluation mode
    fn eval(&mut self);
    
    /// Check if in training mode
    fn training(&self) -> bool;
    
    /// Get module name for debugging
    fn name(&self) -> &str { "Module" }
    
    /// Reset parameters to initial state
    fn reset_parameters(&mut self) -> GraphResult<()> { Ok(()) }
}
```

## 🎯 **Essential Neural Layers**

### **Linear Layer (Dense/Fully-Connected)**

```rust
/// Linear transformation: y = xW^T + b
pub struct Linear<T: NumericType> {
    /// Weight matrix (out_features × in_features)
    weight: Parameter<T>,
    
    /// Bias vector (out_features)
    bias: Option<Parameter<T>>,
    
    /// Input dimension
    in_features: usize,
    
    /// Output dimension  
    out_features: usize,
    
    /// Training mode flag
    training: bool,
}

impl<T: NumericType> Linear<T> {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> GraphResult<Self>
    
    /// Xavier/Glorot initialization
    fn init_weights(&mut self) -> GraphResult<()>
}

impl<T: NumericType> Module<T> for Linear<T> {
    fn forward(&self, input: &Tensor<T>) -> GraphResult<Tensor<T>> {
        // y = xW^T + b
        let output = input.matmul(&self.weight.data().transpose()?)?;
        if let Some(ref bias) = self.bias {
            output.add(bias.data())
        } else {
            Ok(output)
        }
    }
    
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
```

### **Attention Mechanism**

```rust
/// Multi-head self-attention mechanism
pub struct MultiHeadAttention<T: NumericType> {
    /// Number of attention heads
    num_heads: usize,
    
    /// Dimension of each head
    head_dim: usize,
    
    /// Total model dimension
    model_dim: usize,
    
    /// Query projection
    w_q: Linear<T>,
    
    /// Key projection
    w_k: Linear<T>,
    
    /// Value projection  
    w_v: Linear<T>,
    
    /// Output projection
    w_o: Linear<T>,
    
    /// Dropout for attention weights
    dropout: f64,
    
    /// Training mode
    training: bool,
}

impl<T: NumericType> MultiHeadAttention<T> {
    pub fn new(model_dim: usize, num_heads: usize, dropout: f64) -> GraphResult<Self>
    
    /// Scaled dot-product attention
    fn attention(&self, q: &Tensor<T>, k: &Tensor<T>, v: &Tensor<T>) -> GraphResult<Tensor<T>>
}

impl<T: NumericType> Module<T> for MultiHeadAttention<T> {
    fn forward(&self, input: &Tensor<T>) -> GraphResult<Tensor<T>> {
        // Multi-head attention computation
        // Q, K, V = input @ W_q, input @ W_k, input @ W_v
        // Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        unimplemented!("Complex attention mechanism")
    }
}
```

### **Message Passing (Graph Neural Networks)**

```rust
/// Graph convolutional layer - groggy's specialty
pub struct GraphConv<T: NumericType> {
    /// Linear transformation for node features
    linear: Linear<T>,
    
    /// Aggregation function (mean, max, sum)
    aggregation: AggregationType,
    
    /// Whether to include self-connections
    self_loops: bool,
    
    /// Normalization method
    normalization: Option<NormalizationType>,
}

#[derive(Debug, Clone)]
pub enum AggregationType {
    Mean,
    Sum, 
    Max,
    Attention, // Attention-based aggregation
}

impl<T: NumericType> GraphConv<T> {
    pub fn new(in_features: usize, out_features: usize, aggregation: AggregationType) -> GraphResult<Self>
}

impl<T: NumericType> Module<T> for GraphConv<T> {
    fn forward(&self, input: &Tensor<T>) -> GraphResult<Tensor<T>> {
        // Message passing on graph structure
        // 1. Linear transformation of node features
        // 2. Message aggregation from neighbors  
        // 3. Combination with self features
        unimplemented!("Graph convolution requires adjacency matrix")
    }
}
```

## ⚙️ **Optimizer Framework**

### **Optimizer Trait**

```rust
/// Base trait for all optimizers
pub trait Optimizer<T: NumericType>: Send + Sync {
    /// Perform one optimization step
    fn step(&mut self, parameters: &mut [Parameter<T>]) -> GraphResult<()>;
    
    /// Zero all parameter gradients
    fn zero_grad(&mut self, parameters: &mut [Parameter<T>]);
    
    /// Get learning rate
    fn learning_rate(&self) -> f64;
    
    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f64);
    
    /// Get optimizer state for serialization
    fn state_dict(&self) -> HashMap<String, f64>;
    
    /// Load optimizer state
    fn load_state_dict(&mut self, state: HashMap<String, f64>) -> GraphResult<()>;
}
```

### **SGD with Momentum**

```rust
/// Stochastic Gradient Descent with momentum
pub struct SGD<T: NumericType> {
    /// Learning rate
    learning_rate: f64,
    
    /// Momentum coefficient
    momentum: f64,
    
    /// Weight decay (L2 regularization)
    weight_decay: f64,
    
    /// Dampening factor for momentum
    dampening: f64,
    
    /// Whether to use Nesterov momentum
    nesterov: bool,
    
    /// Momentum buffers for each parameter
    momentum_buffers: HashMap<usize, Tensor<T>>,
}

impl<T: NumericType> SGD<T> {
    pub fn new(learning_rate: f64) -> Self
    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self
    pub fn with_weight_decay(learning_rate: f64, weight_decay: f64) -> Self
}

impl<T: NumericType> Optimizer<T> for SGD<T> {
    fn step(&mut self, parameters: &mut [Parameter<T>]) -> GraphResult<()> {
        for (param_id, param) in parameters.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                let mut update = grad.clone();
                
                // Apply weight decay
                if self.weight_decay != 0.0 {
                    update = update.add(&param.data().scale(self.weight_decay)?)?;
                }
                
                // Apply momentum
                if self.momentum != 0.0 {
                    let momentum_buffer = self.momentum_buffers
                        .entry(param_id)
                        .or_insert_with(|| Tensor::zeros(grad.shape()));
                    
                    *momentum_buffer = momentum_buffer
                        .scale(self.momentum)?
                        .add(&update.scale(1.0 - self.dampening)?)?;
                    
                    if self.nesterov {
                        update = update.add(&momentum_buffer.scale(self.momentum)?)?;
                    } else {
                        update = momentum_buffer.clone();
                    }
                }
                
                // Apply update: param = param - lr * update
                let new_data = param.data().sub(&update.scale(self.learning_rate)?)?;
                *param.data_mut() = new_data;
            }
        }
        Ok(())
    }
}
```

### **Adam Optimizer**

```rust
/// Adam optimizer with bias correction
pub struct Adam<T: NumericType> {
    /// Learning rate
    learning_rate: f64,
    
    /// Beta1 coefficient (momentum)
    beta1: f64,
    
    /// Beta2 coefficient (RMSprop)
    beta2: f64,
    
    /// Epsilon for numerical stability
    epsilon: f64,
    
    /// Weight decay
    weight_decay: f64,
    
    /// Current step number
    step_count: usize,
    
    /// First moment estimates
    momentum_buffers: HashMap<usize, Tensor<T>>,
    
    /// Second moment estimates
    velocity_buffers: HashMap<usize, Tensor<T>>,
}

impl<T: NumericType> Adam<T> {
    pub fn new(learning_rate: f64) -> Self
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self
}

impl<T: NumericType> Optimizer<T> for Adam<T> {
    fn step(&mut self, parameters: &mut [Parameter<T>]) -> GraphResult<()> {
        self.step_count += 1;
        
        for (param_id, param) in parameters.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // Get or initialize momentum and velocity buffers
                let m_t = self.momentum_buffers
                    .entry(param_id)
                    .or_insert_with(|| Tensor::zeros(grad.shape()));
                    
                let v_t = self.velocity_buffers
                    .entry(param_id)
                    .or_insert_with(|| Tensor::zeros(grad.shape()));
                
                // Update biased first moment estimate: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                *m_t = m_t.scale(self.beta1)?.add(&grad.scale(1.0 - self.beta1)?)?;
                
                // Update biased second moment estimate: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                let grad_squared = grad.elementwise_multiply(grad)?;
                *v_t = v_t.scale(self.beta2)?.add(&grad_squared.scale(1.0 - self.beta2)?)?;
                
                // Bias correction
                let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);
                
                let corrected_m = m_t.scale(1.0 / bias_correction1)?;
                let corrected_v = v_t.scale(1.0 / bias_correction2)?;
                
                // Update parameters: θ = θ - α * m̂ / (√v̂ + ε)
                let denominator = corrected_v.sqrt()?.add_scalar(self.epsilon)?;
                let update = corrected_m.elementwise_divide(&denominator)?;
                
                let new_data = param.data().sub(&update.scale(self.learning_rate)?)?;
                *param.data_mut() = new_data;
            }
        }
        Ok(())
    }
}
```

## 🎯 **Loss Functions**

### **Mean Squared Error**

```rust
/// Mean Squared Error loss function
pub struct MSELoss<T: NumericType> {
    /// Reduction method (mean, sum, none)
    reduction: Reduction,
}

#[derive(Debug, Clone)]
pub enum Reduction {
    Mean,   // Average over all elements
    Sum,    // Sum over all elements  
    None,   // No reduction (return full tensor)
}

impl<T: NumericType> MSELoss<T> {
    pub fn new() -> Self { Self { reduction: Reduction::Mean } }
    pub fn with_reduction(reduction: Reduction) -> Self { Self { reduction } }
    
    pub fn forward(&self, input: &Tensor<T>, target: &Tensor<T>) -> GraphResult<Tensor<T>> {
        // MSE = (input - target)²
        let diff = input.sub(target)?;
        let squared = diff.elementwise_multiply(&diff)?;
        
        match self.reduction {
            Reduction::Mean => squared.mean(),
            Reduction::Sum => squared.sum(),
            Reduction::None => Ok(squared),
        }
    }
}
```

### **Cross Entropy Loss**

```rust
/// Cross entropy loss for classification
pub struct CrossEntropyLoss<T: NumericType> {
    /// Reduction method
    reduction: Reduction,
    
    /// Label smoothing factor
    label_smoothing: f64,
    
    /// Ignore index for sparse targets
    ignore_index: Option<i64>,
}

impl<T: NumericType> CrossEntropyLoss<T> {
    pub fn new() -> Self
    pub fn with_label_smoothing(label_smoothing: f64) -> Self
    
    pub fn forward(&self, input: &Tensor<T>, target: &Tensor<T>) -> GraphResult<Tensor<T>> {
        // CrossEntropy = -Σ(target * log_softmax(input))
        let log_probs = input.log_softmax(1)?; // Softmax along class dimension
        let loss = target.elementwise_multiply(&log_probs)?.sum_axis(1)?.neg()?;
        
        match self.reduction {
            Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
            Reduction::None => Ok(loss),
        }
    }
}
```

## 🔄 **Training Loop Infrastructure**

### **High-Level Trainer**

```rust
/// High-level training interface
pub struct Trainer<T: NumericType> {
    /// Model to train
    model: Box<dyn Module<T>>,
    
    /// Loss function
    loss_fn: Box<dyn LossFunction<T>>,
    
    /// Optimizer
    optimizer: Box<dyn Optimizer<T>>,
    
    /// Training metrics
    metrics: Vec<Box<dyn Metric<T>>>,
    
    /// Training callbacks
    callbacks: Vec<Box<dyn Callback<T>>>,
    
    /// Device for computation
    device: Device,
}

impl<T: NumericType> Trainer<T> {
    pub fn new(
        model: Box<dyn Module<T>>,
        loss_fn: Box<dyn LossFunction<T>>,
        optimizer: Box<dyn Optimizer<T>>,
    ) -> Self
    
    /// Train for one epoch
    pub fn train_epoch(&mut self, dataloader: &DataLoader<T>) -> GraphResult<TrainingMetrics>
    
    /// Evaluate on validation set
    pub fn evaluate(&mut self, dataloader: &DataLoader<T>) -> GraphResult<EvaluationMetrics>
    
    /// Full training loop
    pub fn fit(&mut self, 
               train_loader: &DataLoader<T>, 
               val_loader: Option<&DataLoader<T>>,
               epochs: usize) -> GraphResult<TrainingHistory>
}
```

## 🌟 **Unique Groggy Features**

### **Graph-Based Neural Networks**

```rust
/// Graph-based neural network model - unique to groggy
pub struct GraphNeuralNetwork<T: NumericType> {
    /// Graph structure for the neural network topology
    network_graph: Graph,
    
    /// Node-to-layer mapping
    layer_map: HashMap<NodeId, Box<dyn Module<T>>>,
    
    /// Execution order (topological sort)
    execution_order: Vec<NodeId>,
    
    /// Input/output node mappings
    input_nodes: Vec<NodeId>,
    output_nodes: Vec<NodeId>,
}

impl<T: NumericType> GraphNeuralNetwork<T> {
    pub fn new() -> Self
    
    /// Add a layer as a node in the graph
    pub fn add_layer<M: Module<T> + 'static>(&mut self, name: &str, layer: M) -> NodeId
    
    /// Connect two layers with an edge
    pub fn connect(&mut self, from: NodeId, to: NodeId) -> GraphResult<()>
    
    /// Analyze network topology
    pub fn analyze_topology(&self) -> NetworkTopologyAnalysis
    
    /// Visualize network architecture  
    pub fn visualize(&self) -> NetworkVisualization
    
    /// Optimize network structure
    pub fn optimize_topology(&mut self) -> GraphResult<OptimizationReport>
}

impl<T: NumericType> Module<T> for GraphNeuralNetwork<T> {
    fn forward(&self, input: &Tensor<T>) -> GraphResult<Tensor<T>> {
        // Execute layers in topological order
        let mut activations = HashMap::new();
        
        // Set inputs
        for (i, &input_node) in self.input_nodes.iter().enumerate() {
            activations.insert(input_node, input.slice_batch(i)?);
        }
        
        // Forward pass through graph
        for &node_id in &self.execution_order {
            if let Some(layer) = self.layer_map.get(&node_id) {
                let inputs = self.collect_node_inputs(node_id, &activations)?;
                let output = layer.forward(&inputs)?;
                activations.insert(node_id, output);
            }
        }
        
        // Collect outputs
        self.collect_outputs(&activations)
    }
}
```

### **Graph Analysis for Neural Networks**

```python
# Unique groggy capabilities
model = groggy.neural.GraphNeuralNetwork()
model.add_layer("encoder", groggy.neural.Linear(784, 256))
model.add_layer("attention", groggy.neural.MultiHeadAttention(256, 8))
model.add_layer("decoder", groggy.neural.Linear(256, 10))

# Analyze network topology using graph algorithms
analysis = model.analyze_topology()
print(f"Network diameter: {analysis.diameter}")
print(f"Critical path: {analysis.critical_path}")
print(f"Parallelizable layers: {analysis.parallel_groups}")

# Optimize network structure
optimization_report = model.optimize_topology()
print(f"Memory reduction: {optimization_report.memory_savings}")
print(f"Compute reduction: {optimization_report.compute_savings}")
```

## 📋 **Implementation Phases**

### **Phase 1: Foundation (Week 1)**
**Goal**: Basic neural network training capability

**Priority Order:**
1. **🚨 CRITICAL: Fix matrix operation return types** (Day 1)
   - Fix `eigenvalue_decomposition` to return `(GraphMatrix<T>, GraphMatrix<T>)`
   - Fix reduction operations to return matrices not scalars
   - Add gradient compatibility tests

2. **Core Data Structures** (Day 2-3)
   - `Tensor<T>` wrapper around `GraphMatrix`
   - `Parameter<T>` with optimizer state
   - `Module` trait definition

3. **Essential Layers** (Day 4-5)
   - `Linear` layer with bias support
   - `ReLU`, `Sigmoid`, `Tanh` activations (reuse existing)
   - Basic layer composition

4. **Basic Optimizer** (Day 6-7)
   - `SGD` without momentum
   - Parameter update mechanics
   - Gradient zeroing

**Success Criteria:**
```python
# This should work at end of Phase 1
import groggy.neural as nn

model = nn.Linear(784, 10)
optimizer = nn.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training step
output = model(input_tensor)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### **Phase 2: Essential Components (Week 2)**
**Goal**: Complete basic neural network ecosystem

1. **Advanced Optimizers** (Day 1-2)
   - `SGD` with momentum
   - `Adam` optimizer
   - Learning rate scheduling

2. **Loss Functions** (Day 3-4)
   - `MSELoss` implementation
   - `CrossEntropyLoss` implementation  
   - Custom loss function framework

3. **Additional Layers** (Day 5-6)
   - `Conv2d` convolutional layer
   - `BatchNorm2d` normalization
   - `Dropout` regularization

4. **Model Containers** (Day 7)
   - `Sequential` model
   - `ModuleList`, `ModuleDict`
   - Parameter management utilities

### **Phase 3: Advanced Features (Week 3-4)**
**Goal**: Transformer and graph neural network support

1. **Attention Mechanisms** (Week 3)
   - `MultiHeadAttention`
   - `TransformerEncoderLayer`
   - Positional embedding

2. **Graph Neural Networks** (Week 4)
   - `GraphConv` message passing
   - Graph-based model architecture
   - Integration with groggy's graph analysis

3. **Training Infrastructure**
   - High-level `Trainer` class
   - Training callbacks
   - Model serialization

## 🎯 **Success Metrics**

### **Compatibility Benchmarks**
- [ ] Can train a simple MLP on MNIST
- [ ] Can train a CNN on CIFAR-10  
- [ ] Can train a Transformer on text classification
- [ ] Can train a GNN on graph classification
- [ ] Can serialize and load trained models
- [ ] Can visualize network architectures

### **Performance Benchmarks**
- [ ] Training speed within 2x of PyTorch on CPU
- [ ] Memory usage within 1.5x of PyTorch
- [ ] Gradient computation correctness verified
- [ ] Support for batch sizes up to 1024

### **API Completeness**
- [ ] All PyTorch-equivalent layers implemented
- [ ] All major optimizers available
- [ ] All common loss functions available
- [ ] Training loop utilities provided

## 🔧 **Technical Debt and Considerations**

### **Immediate Fixes Required**
1. **Matrix operation return types** - blocks all neural development
2. **Gradient compatibility audit** - verify all 118 operations work with autodiff
3. **Memory management** - ensure no leaks in computation graphs
4. **Error handling** - consistent error types across neural module

### **Design Decisions to Finalize**
1. **Batch dimension convention** - first or last axis?
2. **Device abstraction** - CPU-only initially, but plan for GPU
3. **Serialization format** - binary vs JSON vs custom
4. **Thread safety** - shared models across threads?

### **Future Extensibility**
1. **GPU support** - CUDA/ROCm integration path
2. **Distributed training** - multi-node training capability  
3. **Mixed precision** - FP16/BF16 support
4. **Dynamic graphs** - support for control flow in models

## 📊 **Conclusion**

This comprehensive plan establishes groggy.neural as a high-performance, graph-native neural network framework. The hybrid approach provides familiar PyTorch-style APIs while leveraging groggy's unique strengths in graph analysis and columnar storage.

**Key Differentiators:**
- **Graph-native architecture analysis**
- **Zero-copy tensor operations** 
- **Rust performance with Python ergonomics**
- **Built-in graph neural network support**
- **Integrated with groggy's graph ecosystem**

**Implementation Priority:**
1. **Fix matrix operation inconsistencies** (CRITICAL)
2. **Implement core training loop** (Week 1)
3. **Add essential layers and optimizers** (Week 2)
4. **Build advanced features** (Week 3-4)

This foundation will enable groggy to compete directly with PyTorch while offering unique graph-based capabilities that no other framework provides.