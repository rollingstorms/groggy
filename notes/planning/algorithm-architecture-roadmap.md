# Algorithm Architecture & Roadmap

## Vision & Goals

This document outlines the architecture for running algorithms in Groggy after the major release. The goal is to create a flexible, performant system where algorithms can be customized before or during runtime while keeping all heavy computation in Rust.

The core principle is **composability without compilation**: users assemble pre-compiled algorithm components in Python like connecting puzzle pieces or pipes, but the actual execution happens entirely in the Rust core. This enables rapid experimentation and customization while maintaining the performance guarantees that Groggy provides.

### Design Philosophy

The architecture follows a schema pattern where algorithms are defined as:

```
algorithm(subgraph) -> subgraph
```

This enables both direct application (`sg.apply(algorithm)`) and pipeline composition (`pipeline(step1, step2, step3)`). Python code defines the **structure** and **parameters** of algorithm pipelines, while Rust provides the **runtime** and **primitives**.

Users in Python are composing precompiled algorithm building blocks—they're connecting different pipes together, but those pipes run entirely in the Rust core without requiring recompilation.

---

## Architecture Outline

### 1. Rust Algorithm Core

The foundation lives in `src/algorithms/` with a trait-based design that keeps all heavy lifting and state management in Rust.

#### Algorithm Trait

Define a core trait that all algorithms implement:

```rust
// src/algorithms/mod.rs
use crate::graph::Subgraph;
use anyhow::Result;

pub trait Algorithm: Send + Sync {
    /// Unique identifier for this algorithm
    fn id(&self) -> &'static str;
    
    /// Optional metadata (version, cost hints, signature)
    fn metadata(&self) -> AlgorithmMetadata {
        AlgorithmMetadata::default()
    }
    
    /// Execute the algorithm on a subgraph
    fn execute(&self, ctx: &mut Context, sg: Subgraph) -> Result<Subgraph>;
}

pub struct AlgorithmMetadata {
    pub version: &'static str,
    pub description: &'static str,
    pub cost_hint: CostHint,
    pub supports_cancellation: bool,
}

pub struct Context {
    // Telemetry, metrics, cancellation tokens
    timers: HashMap<String, Instant>,
    iteration_counter: usize,
    cancel_token: Option<Arc<AtomicBool>>,
}

impl Context {
    pub fn with_scoped_timer<F, R>(&mut self, name: &str, f: F) -> R
    where F: FnOnce() -> R {
        let start = Instant::now();
        let result = f();
        self.record_duration(name, start.elapsed());
        result
    }
    
    pub fn emit_iteration(&mut self, iter: usize, updates: usize) {
        // Log or collect iteration statistics
    }
    
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.as_ref().map_or(false, |t| t.load(Ordering::Relaxed))
    }
}
```

#### Pipeline Infrastructure

Introduce a pipeline type that composes algorithms:

```rust
// src/algorithms/pipeline.rs
use super::{Algorithm, Context};
use crate::graph::Subgraph;

pub struct Pipeline {
    steps: Vec<Box<dyn Algorithm>>,
    metadata: PipelineMetadata,
}

impl Pipeline {
    pub fn run(&self, ctx: &mut Context, mut sg: Subgraph) -> Result<Subgraph> {
        for (idx, algo) in self.steps.iter().enumerate() {
            ctx.begin_step(idx, algo.id());
            sg = algo.execute(ctx, sg)?;
            
            if ctx.is_cancelled() {
                return Err(anyhow!("Pipeline cancelled at step {}", idx));
            }
        }
        Ok(sg)
    }
}

pub struct PipelineBuilder {
    steps: Vec<AlgorithmSpec>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }
    
    pub fn with_algorithm<F>(mut self, id: &str, configure: F) -> Self
    where F: FnOnce(&mut AlgorithmParams) {
        let mut params = AlgorithmParams::new();
        configure(&mut params);
        self.steps.push(AlgorithmSpec { id: id.to_string(), params });
        self
    }
    
    pub fn build(self, registry: &Registry) -> Result<Pipeline> {
        let mut compiled_steps = Vec::new();
        
        for spec in self.steps {
            let factory = registry.get(&spec.id)
                .ok_or_else(|| anyhow!("Unknown algorithm: {}", spec.id))?;
            let instance = factory.create(&spec.params)?;
            compiled_steps.push(instance);
        }
        
        Ok(Pipeline {
            steps: compiled_steps,
            metadata: PipelineMetadata::default(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmSpec {
    pub id: String,
    pub params: AlgorithmParams,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlgorithmParams {
    values: HashMap<String, ParamValue>,
}

impl AlgorithmParams {
    pub fn set<T: Into<ParamValue>>(&mut self, key: &str, value: T) {
        self.values.insert(key.to_string(), value.into());
    }
    
    pub fn get<T: TryFrom<ParamValue>>(&self, key: &str) -> Option<T> {
        self.values.get(key).and_then(|v| T::try_from(v.clone()).ok())
    }
}
```

#### Algorithm Registry

A central registry maps algorithm IDs to factory functions:

```rust
// src/algorithms/registry.rs
use super::{Algorithm, AlgorithmParams};
use std::collections::HashMap;
use anyhow::Result;

type FactoryFn = Box<dyn Fn(&AlgorithmParams) -> Result<Box<dyn Algorithm>> + Send + Sync>;

pub struct Registry {
    factories: HashMap<String, FactoryFn>,
}

impl Registry {
    pub fn new() -> Self {
        Self { factories: HashMap::new() }
    }
    
    pub fn register_factory<F>(&mut self, id: &str, factory: F)
    where
        F: Fn(&AlgorithmParams) -> Result<Box<dyn Algorithm>> + Send + Sync + 'static,
    {
        self.factories.insert(id.to_string(), Box::new(factory));
    }
    
    pub fn get(&self, id: &str) -> Option<&FactoryFn> {
        self.factories.get(id)
    }
    
    pub fn list_algorithms(&self) -> Vec<&str> {
        self.factories.keys().map(|s| s.as_str()).collect()
    }
}

// Global registry initialization
lazy_static! {
    pub static ref GLOBAL_REGISTRY: RwLock<Registry> = {
        let mut registry = Registry::new();
        
        // Register all algorithm categories
        crate::algorithms::community::register(&mut registry);
        crate::algorithms::centrality::register(&mut registry);
        crate::algorithms::pathfinding::register(&mut registry);
        
        RwLock::new(registry)
    };
}
```

#### Category-Based Organization

Algorithms are organized under `src/algorithms/{category}/`:

```
src/algorithms/
├── mod.rs              # Algorithm trait, Context, exports
├── pipeline.rs         # Pipeline, PipelineBuilder
├── registry.rs         # Registry, factory system
├── steps/              # Fine-grained primitives for DSL
│   ├── mod.rs
│   ├── init_nodes.rs
│   ├── map_nodes.rs
│   ├── attach_attr.rs
│   └── reductions.rs
├── community/          # Community detection algorithms
│   ├── mod.rs
│   ├── lpa.rs
│   ├── louvain.rs
│   └── modularity.rs
├── centrality/         # Centrality algorithms
│   ├── mod.rs
│   ├── betweenness.rs
│   ├── closeness.rs
│   └── pagerank.rs
└── pathfinding/        # Path algorithms
    ├── mod.rs
    ├── dijkstra.rs
    └── bfs.rs
```

---

### 2. Example: Label Propagation Algorithm

#### Rust Core Implementation

```rust
// src/algorithms/community/lpa.rs
use crate::algorithms::{Algorithm, Context, AlgorithmMetadata};
use crate::graph::Subgraph;
use anyhow::Result;

pub struct LabelPropagation {
    max_iter: usize,
    tolerance: f32,
    seed: Option<u64>,
}

impl LabelPropagation {
    pub fn new(max_iter: usize, tolerance: f32, seed: Option<u64>) -> Self {
        Self { max_iter, tolerance, seed }
    }
}

impl Algorithm for LabelPropagation {
    fn id(&self) -> &'static str {
        "community.lpa"
    }
    
    fn metadata(&self) -> AlgorithmMetadata {
        AlgorithmMetadata {
            version: "1.0.0",
            description: "Label Propagation Algorithm for community detection",
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
        }
    }
    
    fn execute(&self, ctx: &mut Context, mut sg: Subgraph) -> Result<Subgraph> {
        ctx.with_scoped_timer("community.lpa", || {
            // Initialize labels: each node gets its own ID as initial label
            let mut labels: HashMap<NodeId, u64> = sg
                .vertices()
                .map(|v| (v, v.as_u64()))
                .collect();
            
            let frontier: Vec<NodeId> = sg.vertices().collect();
            
            for iter in 0..self.max_iter {
                if ctx.is_cancelled() {
                    return Err(anyhow!("LPA cancelled at iteration {}", iter));
                }
                
                let mut updates = 0usize;
                let mut new_labels = labels.clone();
                
                // For each node, adopt the most common label among neighbors
                for &node in &frontier {
                    let neighbor_labels: Vec<u64> = sg
                        .neighbors(node)
                        .filter_map(|n| labels.get(&n).copied())
                        .collect();
                    
                    if !neighbor_labels.is_empty() {
                        let mode_label = mode(&neighbor_labels);
                        
                        if labels[&node] != mode_label {
                            new_labels.insert(node, mode_label);
                            updates += 1;
                        }
                    }
                }
                
                labels = new_labels;
                ctx.emit_iteration(iter, updates);
                
                // Check convergence
                let change_ratio = updates as f32 / frontier.len() as f32;
                if change_ratio <= self.tolerance {
                    break;
                }
            }
            
            // Attach labels to subgraph as node attribute
            for (node, label) in labels {
                sg.set_node_attr(node, "label", label)?;
            }
            
            Ok(sg)
        })
    }
}

fn mode(values: &[u64]) -> u64 {
    let mut counts = HashMap::new();
    for &val in values {
        *counts.entry(val).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|(_, count)| *count).map(|(val, _)| val).unwrap()
}
```

#### Registration

```rust
// src/algorithms/community/mod.rs
use crate::algorithms::{AlgorithmParams, Registry};
use super::lpa::LabelPropagation;

mod lpa;
mod louvain;

pub use lpa::LabelPropagation;

pub fn register(registry: &mut Registry) {
    registry.register_factory("community.lpa", |params| {
        let max_iter = params.get("max_iter").unwrap_or(20);
        let tolerance = params.get("tolerance").unwrap_or(0.001);
        let seed = params.get("seed");
        
        Ok(Box::new(LabelPropagation::new(max_iter, tolerance, seed)))
    });
    
    // Register other community algorithms...
}
```

#### Direct Rust Usage

```rust
use crate::algorithms::{PipelineBuilder, Context, GLOBAL_REGISTRY};

let registry = GLOBAL_REGISTRY.read().unwrap();

let mut pipeline = PipelineBuilder::new()
    .with_algorithm("community.lpa", |params| {
        params.set("max_iter", 30);
        params.set("tolerance", 0.0005);
    })
    .build(&registry)?;

let mut ctx = Context::new();
let result = pipeline.run(&mut ctx, input_subgraph)?;
```

---

### 3. FFI Layer

The FFI layer in `python-groggy/src/ffi/` provides thin shims that marshal data and expose Rust functionality to Python.

#### Pipeline FFI

```rust
// python-groggy/src/ffi/pipeline.rs
use pyo3::prelude::*;
use pyo3::types::PyDict;
use groggy::algorithms::{PipelineBuilder, Context, GLOBAL_REGISTRY};
use groggy::graph::Subgraph;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpec {
    pub steps: Vec<AlgorithmStepSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmStepSpec {
    pub id: String,
    pub params: HashMap<String, serde_json::Value>,
}

#[pyfunction]
pub fn build_pipeline(py: Python<'_>, spec_dict: &PyDict) -> PyResult<u64> {
    // Deserialize Python dict to PipelineSpec
    let spec: PipelineSpec = pythonize::depythonize(spec_dict)?;
    
    // Build pipeline (release GIL for potentially expensive validation)
    let pipeline = py.allow_threads(|| {
        let registry = GLOBAL_REGISTRY.read().unwrap();
        let mut builder = PipelineBuilder::new();
        
        for step in spec.steps {
            builder = builder.with_algorithm(&step.id, |params| {
                for (key, value) in step.params {
                    // Convert JSON value to AlgorithmParam
                    params.set(&key, value);
                }
            });
        }
        
        builder.build(&registry)
    })?;
    
    // Store pipeline in global cache, return handle
    let handle = PIPELINE_CACHE.write().unwrap().insert(pipeline);
    Ok(handle)
}

#[pyfunction]
pub fn run_pipeline(py: Python<'_>, handle: u64, subgraph: &PySubgraph) -> PyResult<PySubgraph> {
    let pipeline = PIPELINE_CACHE.read().unwrap()
        .get(handle)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid pipeline handle"))?
        .clone();
    
    let sg = subgraph.inner.clone();
    
    // Execute with GIL released
    let result = py.allow_threads(|| {
        let mut ctx = Context::new();
        pipeline.run(&mut ctx, sg)
    })?;
    
    Ok(PySubgraph { inner: Arc::new(result) })
}

#[pyfunction]
pub fn drop_pipeline(handle: u64) -> PyResult<()> {
    PIPELINE_CACHE.write().unwrap().remove(handle);
    Ok(())
}

lazy_static! {
    static ref PIPELINE_CACHE: RwLock<HandleCache<Pipeline>> = RwLock::new(HandleCache::new());
}
```

#### Algorithm Discovery FFI

```rust
// python-groggy/src/ffi/algorithms.rs
use pyo3::prelude::*;
use pyo3::types::PyList;
use groggy::algorithms::GLOBAL_REGISTRY;

#[pyfunction]
pub fn list_algorithms(py: Python<'_>) -> PyResult<PyObject> {
    let registry = GLOBAL_REGISTRY.read().unwrap();
    let algos = registry.list_algorithms();
    
    let py_list = PyList::new(py, algos);
    Ok(py_list.into())
}

#[pyfunction]
pub fn get_algorithm_metadata(algorithm_id: &str) -> PyResult<HashMap<String, String>> {
    let registry = GLOBAL_REGISTRY.read().unwrap();
    let factory = registry.get(algorithm_id)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Unknown algorithm"))?;
    
    // Create a dummy instance to extract metadata
    let instance = factory(&Default::default())?;
    let metadata = instance.metadata();
    
    Ok(HashMap::from([
        ("version".to_string(), metadata.version.to_string()),
        ("description".to_string(), metadata.description.to_string()),
    ]))
}
```

---

### 4. Python Layer

The Python layer provides user-facing APIs that wrap the FFI and enable intuitive algorithm composition.

#### Core Pipeline API

```python
# python-groggy/python/groggy/pipeline.py
from typing import List, Dict, Any, Optional, Callable
from ._groggy import ffi

class Pipeline:
    """A compiled algorithm pipeline."""
    
    def __init__(self, steps: List['AlgorithmHandle']):
        self._steps = steps
        self._handle: Optional[int] = None
    
    def _build(self) -> int:
        """Build the pipeline and return a handle."""
        spec = {
            "steps": [step.to_spec() for step in self._steps]
        }
        return ffi.build_pipeline(spec)
    
    def run(self, subgraph: 'Subgraph') -> 'Subgraph':
        """Execute the pipeline on a subgraph."""
        if self._handle is None:
            self._handle = self._build()
        
        return ffi.run_pipeline(self._handle, subgraph)
    
    def __del__(self):
        if self._handle is not None:
            ffi.drop_pipeline(self._handle)

def pipeline(*steps: 'AlgorithmHandle') -> Pipeline:
    """Create a pipeline from a sequence of algorithm steps."""
    return Pipeline(list(steps))
```

#### Algorithm Handle System

```python
# python-groggy/python/groggy/algorithms/base.py
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class AlgorithmHandle(ABC):
    """Base class for algorithm references."""
    
    @abstractmethod
    def to_spec(self) -> Dict[str, Any]:
        """Convert to a serializable spec for FFI."""
        pass

class RustAlgorithmHandle(AlgorithmHandle):
    """Reference to a Rust-implemented algorithm."""
    
    def __init__(self, algorithm_id: str, params: Optional[Dict[str, Any]] = None):
        self._id = algorithm_id
        self._params = params or {}
    
    def configure(self, **kwargs) -> 'RustAlgorithmHandle':
        """Return a new handle with updated parameters."""
        new_params = {**self._params, **kwargs}
        return RustAlgorithmHandle(self._id, new_params)
    
    def to_spec(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "params": self._params,
        }
    
    def __call__(self, **kwargs) -> 'RustAlgorithmHandle':
        """Alias for configure()."""
        return self.configure(**kwargs)

def algorithm(
    algorithm_id: str,
    defaults: Optional[Dict[str, Any]] = None,
    doc: Optional[str] = None,
) -> RustAlgorithmHandle:
    """Create a reference to a Rust algorithm."""
    handle = RustAlgorithmHandle(algorithm_id, defaults)
    if doc:
        handle.__doc__ = doc
    return handle
```

#### Pre-Registered Algorithms

```python
# python-groggy/python/groggy/algorithms/community.py
from .base import algorithm

lpa = algorithm(
    "community.lpa",
    defaults={"max_iter": 20, "tolerance": 0.001},
    doc="""Label Propagation Algorithm for community detection.
    
    Parameters:
        max_iter (int): Maximum number of iterations. Default: 20
        tolerance (float): Convergence threshold. Default: 0.001
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        Subgraph with 'label' attribute on nodes indicating communities.
    """,
)

louvain = algorithm(
    "community.louvain",
    defaults={"resolution": 1.0},
    doc="""Louvain method for community detection.""",
)
```

#### Subgraph Integration

```python
# python-groggy/python/groggy/subgraph.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .algorithms.base import AlgorithmHandle
    from .pipeline import Pipeline

class Subgraph:
    # ... existing Subgraph implementation ...
    
    def apply(self, algorithm: 'AlgorithmHandle | Pipeline') -> 'Subgraph':
        """Apply an algorithm or pipeline to this subgraph.
        
        Args:
            algorithm: Either an AlgorithmHandle or a Pipeline
        
        Returns:
            A new Subgraph with the algorithm applied
        
        Examples:
            >>> from groggy.algorithms.community import lpa
            >>> result = sg.apply(lpa)
            >>> result = sg.apply(lpa.configure(max_iter=50))
        """
        from .pipeline import Pipeline, pipeline as make_pipeline
        
        if isinstance(algorithm, Pipeline):
            return algorithm.run(self)
        else:
            # Wrap single algorithm in a pipeline
            return make_pipeline(algorithm).run(self)
```

#### Simple Usage Example

```python
from groggy import Subgraph
from groggy.algorithms.community import lpa

# Create a subgraph
sg = Subgraph.from_edges([(0, 1), (1, 2), (2, 0), (3, 4)])

# Apply algorithm with defaults
result = sg.apply(lpa)

# Apply with custom parameters
result = sg.apply(lpa.configure(max_iter=50, tolerance=0.0001))

# Or using call syntax
result = sg.apply(lpa(max_iter=50, tolerance=0.0001))
```

---

### 5. Python Builder DSL

For advanced users who want to compose custom algorithms from primitives without writing Rust.

#### Step Primitives

```rust
// src/algorithms/steps/mod.rs
pub trait Step: Send + Sync {
    fn id(&self) -> &'static str;
    fn execute(&self, ctx: &mut Context, sg: &mut Subgraph) -> Result<StepOutput>;
}

pub enum StepOutput {
    Subgraph(Subgraph),
    NodeData(HashMap<NodeId, Value>),
    Scalar(Value),
}
```

```rust
// src/algorithms/steps/init_nodes.rs
pub struct InitNodesStep {
    target_var: String,
    init_fn: InitFunction,
}

pub enum InitFunction {
    NodeId,              // node.id
    Constant(Value),     // constant value
    Attribute(String),   // node.attr_name
}

impl Step for InitNodesStep {
    fn id(&self) -> &'static str { "init_nodes" }
    
    fn execute(&self, ctx: &mut Context, sg: &mut Subgraph) -> Result<StepOutput> {
        let data: HashMap<NodeId, Value> = sg
            .vertices()
            .map(|node| {
                let value = match &self.init_fn {
                    InitFunction::NodeId => Value::U64(node.as_u64()),
                    InitFunction::Constant(v) => v.clone(),
                    InitFunction::Attribute(attr) => sg.get_node_attr(node, attr)?,
                };
                Ok((node, value))
            })
            .collect::<Result<_>>()?;
        
        Ok(StepOutput::NodeData(data))
    }
}
```

```rust
// src/algorithms/steps/map_nodes.rs
pub struct MapNodesStep {
    source_var: String,
    target_var: String,
    map_fn: MapFunction,
}

pub enum MapFunction {
    Mode,                          // mode(labels[neighbors(node)])
    Sum,                           // sum(values[neighbors(node)])
    Custom(RegisteredKernel),      // pre-registered function
}

impl Step for MapNodesStep {
    fn id(&self) -> &'static str { "map_nodes" }
    
    fn execute(&self, ctx: &mut Context, sg: &mut Subgraph) -> Result<StepOutput> {
        let source_data = ctx.get_var(&self.source_var)?;
        
        let mapped: HashMap<NodeId, Value> = sg
            .vertices()
            .map(|node| {
                let neighbors: Vec<Value> = sg
                    .neighbors(node)
                    .filter_map(|n| source_data.get(&n).cloned())
                    .collect();
                
                let result = match &self.map_fn {
                    MapFunction::Mode => mode(&neighbors)?,
                    MapFunction::Sum => sum(&neighbors)?,
                    MapFunction::Custom(kernel) => kernel.apply(&neighbors)?,
                };
                
                Ok((node, result))
            })
            .collect::<Result<_>>()?;
        
        Ok(StepOutput::NodeData(mapped))
    }
}
```

#### Python Builder API

```python
# python-groggy/python/groggy/builder.py
from typing import Any, Callable, Dict, Optional, List
from .algorithms.base import AlgorithmHandle

class AlgorithmBuilder:
    """Build custom algorithms from primitives."""
    
    def __init__(self, name: str):
        self._name = name
        self._steps: List[Dict[str, Any]] = []
        self._variables: Dict[str, str] = {}
    
    def input(self, name: str) -> 'VarHandle':
        """Declare an input variable."""
        self._variables[name] = "input"
        return VarHandle(self, name)
    
    def var(self, name: str, init_step: Dict[str, Any]) -> 'VarHandle':
        """Create a variable from a step."""
        self._steps.append(init_step)
        self._variables[name] = "computed"
        return VarHandle(self, name)
    
    def init_nodes(self, sg: 'VarHandle', fn: str) -> Dict[str, Any]:
        """Initialize node data.
        
        Args:
            sg: Subgraph variable
            fn: Initialization function ("node.id", constant, or "node.attr")
        """
        return {
            "op": "init_nodes",
            "fn": fn,
            "subgraph": sg.name,
        }
    
    def map_nodes(
        self,
        sg: 'VarHandle',
        fn: str,
        inputs: Dict[str, 'VarHandle'],
    ) -> 'VarHandle':
        """Map over nodes with a function.
        
        Args:
            sg: Subgraph variable
            fn: Map function (e.g., "mode(labels[neighbors(node)])")
            inputs: Input variables for the function
        """
        step = {
            "op": "map_nodes",
            "fn": fn,
            "subgraph": sg.name,
            "inputs": {k: v.name for k, v in inputs.items()},
        }
        self._steps.append(step)
        
        var_name = f"_mapped_{len(self._steps)}"
        self._variables[var_name] = "computed"
        return VarHandle(self, var_name)
    
    def attach_node_attr(
        self,
        sg: 'VarHandle',
        attr_name: str,
        values: 'VarHandle',
    ) -> 'VarHandle':
        """Attach node attributes to subgraph."""
        self._steps.append({
            "op": "attach_node_attr",
            "subgraph": sg.name,
            "attr": attr_name,
            "values": values.name,
        })
        return sg
    
    def compile(self) -> AlgorithmHandle:
        """Compile to a pipeline spec."""
        from .algorithms.base import BuilderAlgorithmHandle
        
        spec = {
            "name": self._name,
            "variables": self._variables,
            "steps": self._steps,
        }
        
        return BuilderAlgorithmHandle(spec)

class VarHandle:
    """Handle to a variable in the builder."""
    
    def __init__(self, builder: AlgorithmBuilder, name: str):
        self.builder = builder
        self.name = name

class BuilderAlgorithmHandle(AlgorithmHandle):
    """Handle for builder-created algorithms."""
    
    def __init__(self, spec: Dict[str, Any]):
        self._spec = spec
    
    def to_spec(self) -> Dict[str, Any]:
        return {
            "id": "builder.custom",
            "params": {"spec": self._spec},
        }
```

#### Full Builder Example: LPA

```python
# Example: Implement Label Propagation using the builder
from groggy.builder import AlgorithmBuilder

def lpa_custom(max_iter: int = 10):
    """Label Propagation Algorithm built with primitives."""
    builder = AlgorithmBuilder("label_propagation_custom")
    
    # Input subgraph
    sg = builder.input("subgraph")
    
    # Initialize labels: each node gets its ID
    labels = builder.var("labels", builder.init_nodes(sg, fn="node.id"))
    
    # Iteration loop
    for _ in range(max_iter):
        labels = builder.map_nodes(
            sg,
            fn="mode(labels[neighbors(node)])",
            inputs={"labels": labels},
        )
    
    # Attach final labels as node attribute
    builder.attach_node_attr(sg, "label", labels)
    
    return builder.compile()

# Usage
from groggy import Subgraph

sg = Subgraph.from_edges([(0, 1), (1, 2), (2, 0), (3, 4)])
lpa_algo = lpa_custom(max_iter=5)
result = sg.apply(lpa_algo)
```

#### Advanced Decorator-Based DSL

```python
# python-groggy/python/groggy/dsl.py
from typing import Callable
from .builder import AlgorithmBuilder, VarHandle
import inspect
import ast

def alg(name: str) -> Callable:
    """Decorator to convert a function into an algorithm spec.
    
    The decorated function uses special `step.*` calls that get
    compiled into a pipeline spec.
    """
    def decorator(fn: Callable) -> Callable:
        # Parse function AST to extract step calls
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        
        # Transform step.* calls into builder operations
        spec = _compile_function_to_spec(tree, name)
        
        # Return a factory function
        def algorithm_factory(**kwargs):
            # Instantiate spec with parameters
            from .algorithms.base import BuilderAlgorithmHandle
            final_spec = _apply_params(spec, kwargs)
            return BuilderAlgorithmHandle(final_spec)
        
        algorithm_factory.__name__ = fn.__name__
        algorithm_factory.__doc__ = fn.__doc__
        return algorithm_factory
    
    return decorator

# Namespace for step primitives in DSL
class step:
    @staticmethod
    def init_nodes(sg, fn):
        """Initialize node data (compiled at decoration time)."""
        return {"op": "init_nodes", "fn": fn, "sg": sg}
    
    @staticmethod
    def map_nodes(sg, fn, inputs):
        """Map nodes (compiled at decoration time)."""
        return {"op": "map_nodes", "fn": fn, "sg": sg, "inputs": inputs}
    
    @staticmethod
    def mode(values):
        """Mode aggregation function."""
        return {"fn": "mode", "inputs": values}
    
    @staticmethod
    def attach_nodes(sg, attr, values):
        """Attach node attributes."""
        return {"op": "attach_nodes", "sg": sg, "attr": attr, "values": values}

# Example usage with decorator
@alg("label_propagation")
def label_propagation(sg, max_iter: int = 10):
    """Label Propagation with decorator syntax."""
    
    labels = step.init_nodes(sg, lambda node: node.id)
    
    for _ in range(max_iter):
        labels = step.map_nodes(
            sg,
            fn=lambda node, labels: step.mode(labels[sg.neighbors(node)]),
            inputs={"labels": labels},
        )
    
    sg = step.attach_nodes(sg, "label", labels)
    return sg

# Usage
sg.apply(label_propagation(max_iter=5))
```

---

## Complete Roadmap

### Phase 1: Rust Core Foundation

**Goal:** Establish the trait-based algorithm system and pipeline infrastructure.

#### 1.1 Define Core Traits & Types

- [ ] Create `src/algorithms/mod.rs` with `Algorithm` trait
- [ ] Implement `Context` with telemetry, cancellation support
- [ ] Define `AlgorithmMetadata` and `CostHint` types
- [ ] Add `AlgorithmParams` with typed value support
- [ ] Write unit tests for trait implementations

#### 1.2 Build Pipeline System

- [ ] Implement `Pipeline` struct in `src/algorithms/pipeline.rs`
- [ ] Create `PipelineBuilder` with fluent API
- [ ] Add `PipelineSpec` serialization/deserialization
- [ ] Support pipeline validation and error handling
- [ ] Add integration tests for pipeline execution

#### 1.3 Create Algorithm Registry

- [ ] Implement `Registry` in `src/algorithms/registry.rs`
- [ ] Add factory registration and lookup
- [ ] Create global `GLOBAL_REGISTRY` with lazy initialization
- [ ] Support algorithm discovery and metadata queries
- [ ] Add tests for registration and retrieval

#### 1.4 Implement Step Primitives

- [ ] Design `Step` trait in `src/algorithms/steps/mod.rs`
- [ ] Implement `InitNodesStep` for node initialization
- [ ] Implement `MapNodesStep` for node transformations
- [ ] Implement `AttachNodeAttrStep` for attribute attachment
- [ ] Add reduction operations (sum, mode, min, max, etc.)
- [ ] Create step registry and validation
- [ ] Write comprehensive step tests

### Phase 2: Core Algorithm Implementations

**Goal:** Build out the initial algorithm catalog organized by category.

#### 2.1 Community Detection Algorithms

- [ ] Implement Label Propagation (`src/algorithms/community/lpa.rs`)
- [ ] Implement Louvain method (`src/algorithms/community/louvain.rs`)
- [ ] Add modularity calculation helpers
- [ ] Register all community algorithms
- [ ] Add benchmarks in `benches/community_algorithms.rs`
- [ ] Write integration tests

#### 2.2 Centrality Algorithms

- [ ] Implement PageRank (`src/algorithms/centrality/pagerank.rs`)
- [ ] Implement Betweenness centrality
- [ ] Implement Closeness centrality
- [ ] Register centrality algorithms
- [ ] Add benchmarks and tests

#### 2.3 Pathfinding Algorithms

- [ ] Implement Dijkstra's algorithm
- [ ] Implement BFS/DFS
- [ ] Implement A* pathfinding
- [ ] Register pathfinding algorithms
- [ ] Add benchmarks and tests

### Phase 3: FFI Bridge

**Goal:** Expose Rust algorithms and pipelines to Python with minimal overhead.

#### 3.1 Pipeline FFI

- [ ] Create `python-groggy/src/ffi/pipeline.rs`
- [ ] Implement `build_pipeline` FFI function
- [ ] Implement `run_pipeline` with GIL release
- [ ] Add `drop_pipeline` for cleanup
- [ ] Create pipeline handle cache system
- [ ] Add FFI error handling and conversion

#### 3.2 Algorithm Discovery FFI

- [ ] Implement `list_algorithms` FFI function
- [ ] Implement `get_algorithm_metadata` FFI function
- [ ] Add parameter validation helpers
- [ ] Expose algorithm categories to Python

#### 3.3 Subgraph Marshalling Extensions

- [ ] Extend subgraph FFI for attribute updates
- [ ] Add incremental label update support
- [ ] Optimize attribute serialization for algorithm results
- [ ] Add tests for FFI round-trips

### Phase 4: Python User API

**Goal:** Provide intuitive, Pythonic interfaces for algorithm usage.

#### 4.1 Core Pipeline API

- [ ] Create `python-groggy/python/groggy/pipeline.py`
- [ ] Implement `Pipeline` class wrapping FFI
- [ ] Add `pipeline()` factory function
- [ ] Integrate with `Subgraph.apply()`
- [ ] Write Python unit tests

#### 4.2 Algorithm Handle System

- [ ] Create `python-groggy/python/groggy/algorithms/base.py`
- [ ] Implement `AlgorithmHandle` base class
- [ ] Implement `RustAlgorithmHandle` for Rust algorithms
- [ ] Add `algorithm()` factory function
- [ ] Support parameter configuration and validation

#### 4.3 Pre-Registered Algorithms

- [ ] Create `python-groggy/python/groggy/algorithms/community.py`
- [ ] Expose `lpa`, `louvain` handles
- [ ] Create `python-groggy/python/groggy/algorithms/centrality.py`
- [ ] Expose `pagerank`, `betweenness`, etc.
- [ ] Create `python-groggy/python/groggy/algorithms/pathfinding.py`
- [ ] Add docstrings and type hints
- [ ] Write usage examples

#### 4.4 Discovery & Introspection

- [ ] Add `groggy.algorithms.list()` function
- [ ] Add `groggy.algorithms.info(algorithm_id)` function
- [ ] Support category filtering and search
- [ ] Create interactive documentation

### Phase 5: Python Builder DSL

**Goal:** Enable custom algorithm composition from Python without Rust compilation.

#### 5.1 Rust Step Interpreter

- [ ] Implement builder spec validation in Rust
- [ ] Create step executor that interprets specs
- [ ] Add variable/SSA slot management
- [ ] Support control flow (loops, conditionals)
- [ ] Add comprehensive validation and error messages

#### 5.2 Python Builder Primitives

- [ ] Create `python-groggy/python/groggy/builder.py`
- [ ] Implement `AlgorithmBuilder` class
- [ ] Add `VarHandle` for variable references
- [ ] Implement step methods (`init_nodes`, `map_nodes`, etc.)
- [ ] Add `compile()` to generate specs
- [ ] Write builder tests

#### 5.3 Function DSL

- [ ] Implement kernel registry for safe functions
- [ ] Support function string parsing/validation
- [ ] Add common kernels (mode, sum, min, max, etc.)
- [ ] Restrict to safe, pre-compiled operations

#### 5.4 Advanced Decorator DSL (Optional)

- [ ] Create `python-groggy/python/groggy/dsl.py`
- [ ] Implement `@alg` decorator
- [ ] Add AST transformation for step.* calls
- [ ] Support parameter binding and defaults
- [ ] Add decorator tests and examples

### Phase 6: Testing & Documentation

**Goal:** Ensure quality and usability through comprehensive testing and documentation.

#### 6.1 Rust Tests

- [ ] Unit tests for all algorithm implementations
- [ ] Integration tests for pipeline execution
- [ ] Tests for registry and factory system
- [ ] Benchmark suite in `benches/`
- [ ] Test cancellation and error paths

#### 6.2 Python Tests

- [ ] Unit tests for all Python APIs
- [ ] Integration tests with real graphs
- [ ] Tests for builder DSL
- [ ] Performance regression tests
- [ ] Cross-platform tests (Linux, macOS, Windows)

#### 6.3 Documentation

- [ ] Architecture doc (this document) in `docs/`
- [ ] API reference for Rust algorithms
- [ ] Python API documentation
- [ ] Tutorial: Using pre-built algorithms
- [ ] Tutorial: Building custom algorithms
- [ ] Migration guide from current API
- [ ] Performance guide and best practices

#### 6.4 Examples & Notebooks

- [ ] Example notebooks for each algorithm category
- [ ] Pipeline composition examples
- [ ] Builder DSL examples
- [ ] Real-world use case demonstrations
- [ ] Performance comparison notebooks

### Phase 7: Polish & Release

**Goal:** Prepare for production use in the major release.

#### 7.1 Performance Optimization

- [ ] Profile algorithm implementations
- [ ] Optimize hot paths identified in benchmarks
- [ ] Reduce FFI overhead
- [ ] Optimize pipeline compilation
- [ ] Add parallel execution for independent steps

#### 7.2 Error Handling & Validation

- [ ] Review all error messages for clarity
- [ ] Add helpful suggestions in errors
- [ ] Validate specs at build time
- [ ] Add runtime safety checks
- [ ] Test error propagation across FFI

#### 7.3 API Stability

- [ ] Review all public APIs
- [ ] Mark experimental features clearly
- [ ] Document stability guarantees
- [ ] Plan deprecation strategy
- [ ] Version algorithm specs

#### 7.4 Release Preparation

- [ ] Update CHANGELOG
- [ ] Write release notes
- [ ] Create migration guide
- [ ] Update README examples
- [ ] Prepare announcement materials

---

## Key Design Considerations

### Subgraph Interchange Format

Algorithms must maintain the O(1) amortized mutation budget when processing subgraphs. To achieve this:

- Use **Arc-backed columnar snapshots** for intermediate state so pipelines can reuse data without copying.
- Algorithms return new `Subgraph` instances rather than mutating in-place, preserving immutability at the Python level while allowing internal optimizations.
- Attribute updates use **copy-on-write** semantics to share unchanged data across pipeline steps.

### Cancellation & Async Support

Long-running algorithms must support graceful cancellation:

- `Context` carries a cancellation token (`Arc<AtomicBool>`) checked periodically during execution.
- FFI exposes `cancel_pipeline(handle)` that sets the token.
- Python wraps this in async-friendly APIs (e.g., integration with `asyncio.CancelledError`).

### Metadata & Discoverability

All algorithms expose rich metadata:

- **ID**: Unique identifier (e.g., `"community.lpa"`)
- **Version**: Semantic version for compatibility tracking
- **Description**: Human-readable documentation
- **Cost Hint**: Expected complexity (Linear, Quadratic, Cubic, etc.)
- **Parameters**: Schema for accepted parameters and defaults
- **Supports Cancellation**: Whether the algorithm can be interrupted

This metadata powers:

- Discovery via `groggy.algorithms.list()`
- Validation before pipeline execution
- Cost estimation for query planning
- Documentation generation

### Builder Spec Validation

Python-generated specs must be validated before execution:

- **Type checking**: Ensure variables have compatible types
- **Data-flow analysis**: Verify all variables are defined before use
- **Function resolution**: Check that all referenced functions exist in the kernel registry
- **Cost estimation**: Warn about expensive operations
- **Serialization stability**: Ensure specs are deterministic for diffing

Failed validation returns structured errors that point to the problematic step.

### Security & Safety

Since builders let users specify function calls, we must restrict them:

- **Whitelist approach**: Only pre-registered kernels are allowed (no arbitrary Python exec)
- **Sandboxed functions**: Even custom kernels run in a restricted environment
- **Resource limits**: Prevent infinite loops or memory exhaustion
- **Audit trail**: Log all spec executions for security review

---

## Open Questions & Future Extensions

### Stateful Algorithms

Some algorithms maintain state across invocations (e.g., incremental community detection). Consider:

- Adding a `StatefulAlgorithm` trait with `update()` method
- Supporting checkpoint/restore for long-running computations
- Designing a state storage API that works across FFI

### Distributed Execution

For very large graphs, algorithms may need to run distributed:

- Design a partition-aware `Algorithm` trait
- Support message passing between partitions
- Integrate with distributed graph storage backends

### GPU Acceleration

Certain algorithms benefit from GPU execution:

- Add GPU-aware step implementations
- Support hybrid CPU/GPU pipelines
- Provide graceful fallback when GPU unavailable

### Interactive Algorithms

Some use cases need real-time interaction:

- Support streaming updates to running algorithms
- Enable parameter tuning during execution
- Provide progress callbacks and visualization hooks

### Custom Python Extensions

While the builder DSL covers many cases, some users may want true Python-defined algorithms:

- Investigate safe Python callback mechanism
- Use `py_call` to invoke Python from Rust algorithms
- Ensure GIL is properly managed
- Profile performance impact

---

## Success Metrics

### Performance Targets

- **FFI overhead**: < 100ns per algorithm invocation
- **Pipeline compilation**: < 10ms for typical pipelines
- **Execution overhead**: < 5% compared to hand-written Rust
- **Memory overhead**: < 10% for intermediate pipeline state

### Developer Experience

- **Time to first algorithm**: < 5 minutes with pre-built algorithms
- **Time to custom algorithm**: < 30 minutes with builder DSL
- **Error message quality**: 90%+ of users understand errors without docs
- **Documentation coverage**: 100% of public APIs documented

### Compatibility

- **Rust API stability**: Semver guarantees for all public types
- **Python API stability**: PEP 387 compatibility policy
- **Spec versioning**: Forward compatibility for 2 minor versions
- **Platform support**: Linux, macOS, Windows on x86_64 and ARM64

---

## Conclusion

This architecture provides a powerful, flexible foundation for algorithm execution in Groggy. By keeping all heavy computation in Rust while exposing composable building blocks to Python, we achieve the best of both worlds: performance and usability.

The three-tier approach (Rust core → FFI shim → Python DSL) maintains clear separation of concerns while enabling rapid experimentation. Users can start with pre-built algorithms, compose them into pipelines, and eventually build custom algorithms—all without leaving Python or waiting for compilation.

The roadmap is ambitious but tractable, with each phase building on the previous. Following the established patterns in the repository (columnar operations, FFI safety, comprehensive testing) ensures consistency with the existing codebase.

The key to success is **maintaining discipline** around the architecture boundaries: business logic stays in Rust, FFI only marshals data, and Python focuses on composition and user experience. With this foundation, Groggy can support a rich ecosystem of graph algorithms while preserving its performance characteristics and ease of use.
