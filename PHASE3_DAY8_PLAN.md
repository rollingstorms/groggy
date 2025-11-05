# Phase 3, Day 8: Batch Compilation

**Date**: 2025-11-05  
**Status**: 🚧 In Progress

## Overview

Day 8 focuses on implementing batched execution - compiling the entire optimized IR into a single execution plan that can be dispatched to Rust with just one FFI call per algorithm run.

## Objectives

1. Design batch execution plan format
2. Implement batch plan generator from IR
3. Update FFI interface for batch execution
4. Add plan caching system
5. Validate correctness and measure performance

## Current State

### What We Have
- ✅ Optimized IR with 5 fusion passes
- ✅ Typed IR nodes (Core, Graph, Attr, Control)
- ✅ Dataflow analysis and dependencies
- ✅ 2.74x speedup from fusion alone

### What We Need
- [ ] BatchExecutionPlan data structure
- [ ] IR → BatchPlan compiler
- [ ] FFI `execute_batch_plan()` method
- [ ] Plan caching by algorithm hash
- [ ] Performance validation

## Design: Batch Execution Plan

### Concept

Instead of executing each IR operation individually (crossing FFI each time), we compile the entire IR into a single "execution plan" - a compact binary representation that Rust can interpret in one go.

**Before (current)**:
```python
# Python side - many FFI calls
ranks = init_nodes(1.0)           # FFI call 1
deg = get_degrees()                # FFI call 2
inv_deg = divide(1.0, deg)         # FFI call 3
for _ in range(100):               # Python loop
    contrib = mul(ranks, inv_deg)  # FFI call 4...
    # ... hundreds more FFI calls per iteration
```

**After (batched)**:
```python
# Python side - one FFI call
plan = compile_to_batch_plan(ir_graph)  # Compile once
result = execute_batch_plan(plan)        # Execute entire algorithm in Rust
```

### BatchExecutionPlan Format

```python
@dataclass
class BatchExecutionPlan:
    """
    Compiled execution plan for an algorithm.
    
    Represents the entire algorithm as a sequence of operations
    that can be executed in Rust without crossing back to Python.
    """
    name: str
    operations: List[BatchOp]
    variables: Dict[str, int]  # var name → slot index
    constants: Dict[int, Any]  # slot → value
    metadata: Dict[str, Any]
    
    def to_bytes(self) -> bytes:
        """Serialize to compact binary format for FFI."""
        
    @staticmethod
    def from_ir(ir_graph: IRGraph) -> 'BatchExecutionPlan':
        """Compile IR graph into batch execution plan."""
```

### BatchOp Format

```python
@dataclass
class BatchOp:
    """
    Single operation in batch execution plan.
    
    Analogous to a bytecode instruction:
    - opcode: what operation to perform
    - inputs: which slots to read from
    - output: which slot to write to
    - metadata: operation-specific data
    """
    opcode: str       # "add", "mul", "neighbor_agg", "loop_start", etc.
    inputs: List[int] # Input variable slot indices
    output: int       # Output variable slot index
    metadata: Dict    # Operation-specific parameters
```

### Example Compilation

**IR Graph**:
```
n1: constant(100) → n
damp1: constant(0.85) → damping
init1: div(1.0, n) → init_rank
loop1: loop(100)
  contrib1: div(ranks, degrees) → contrib
  ns1: neighbor_agg(contrib) → neighbor_sum
  r1: mul(neighbor_sum, damping) → ranks
```

**Batch Plan**:
```
Variables: {n: 0, damping: 1, init_rank: 2, ranks: 3, degrees: 4, contrib: 5, neighbor_sum: 6}

Operations:
  0: CONST_INT     [] → 0         {value: 100}
  1: CONST_FLOAT   [] → 1         {value: 0.85}
  2: DIV           [*, 0] → 2     {numerator: 1.0}
  3: INIT_NODES    [2] → 3        {}
  4: DEGREE        [] → 4         {}
  5: LOOP_START    [] → *         {count: 100}
  6:   DIV         [3, 4] → 5     {}
  7:   NEIGHBOR_AGG [5] → 6       {agg: "sum"}
  8:   MUL         [6, 1] → 3     {}  # Write back to ranks slot
  9: LOOP_END      [] → *         {}
```

## Implementation Tasks

### Task 1: BatchExecutionPlan Data Structure

**File**: `python-groggy/python/groggy/builder/execution/batch_plan.py`

```python
"""
Batch execution plan representation and compilation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import struct
import json

@dataclass
class BatchOp:
    """Single operation in batch execution plan."""
    opcode: str
    inputs: List[int]
    output: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize to binary format."""
        # Pack: opcode (2 bytes), num_inputs (1 byte), inputs (var bytes),
        # output (2 bytes), metadata (JSON)
        pass

@dataclass
class BatchExecutionPlan:
    """Compiled execution plan for an algorithm."""
    name: str
    operations: List[BatchOp]
    variables: Dict[str, int]
    constants: Dict[int, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize entire plan to binary."""
        pass
    
    def to_json(self) -> Dict:
        """Serialize to JSON for debugging."""
        pass
```

### Task 2: IR → BatchPlan Compiler

**File**: `python-groggy/python/groggy/builder/execution/compiler.py`

```python
"""
Compile optimized IR into batch execution plans.
"""
from groggy.builder.ir import IRGraph, IRNode
from .batch_plan import BatchExecutionPlan, BatchOp

class BatchCompiler:
    """Compiles IR graphs into batch execution plans."""
    
    def compile(self, ir_graph: IRGraph) -> BatchExecutionPlan:
        """
        Compile IR graph into batch execution plan.
        
        Steps:
        1. Topological sort of IR nodes (execution order)
        2. Assign variable slots
        3. Generate BatchOps from IR nodes
        4. Optimize operation sequence
        5. Pack into execution plan
        """
        # 1. Get execution order
        exec_order = self._topological_sort(ir_graph)
        
        # 2. Assign variable slots
        var_slots = self._assign_slots(exec_order)
        
        # 3. Generate operations
        ops = []
        for node in exec_order:
            batch_op = self._compile_node(node, var_slots)
            ops.append(batch_op)
        
        # 4. Extract constants
        constants = self._extract_constants(ir_graph, var_slots)
        
        # 5. Build plan
        plan = BatchExecutionPlan(
            name=ir_graph.name,
            operations=ops,
            variables=var_slots,
            constants=constants,
            metadata=ir_graph.metadata
        )
        
        return plan
    
    def _topological_sort(self, ir_graph: IRGraph) -> List[IRNode]:
        """Sort nodes in execution order."""
        pass
    
    def _assign_slots(self, nodes: List[IRNode]) -> Dict[str, int]:
        """Assign variable slot indices."""
        pass
    
    def _compile_node(self, node: IRNode, var_slots: Dict[str, int]) -> BatchOp:
        """Compile single IR node to BatchOp."""
        pass
    
    def _extract_constants(self, ir_graph: IRGraph, var_slots: Dict[str, int]) -> Dict[int, Any]:
        """Extract constant values."""
        pass
```

### Task 3: FFI Batch Execution

**File**: `python-groggy/src/ffi/execution.rs`

```rust
/// Execute a batch execution plan.
#[pyfunction]
pub fn execute_batch_plan(
    py: Python,
    graph: &PyGraph,
    plan_bytes: &[u8],
) -> PyResult<PyObject> {
    py.allow_threads(|| {
        // Deserialize plan
        let plan = BatchPlan::from_bytes(plan_bytes)?;
        
        // Allocate variable slots
        let mut slots = vec![Value::None; plan.num_slots];
        
        // Execute operations in sequence
        for op in &plan.operations {
            execute_op(graph, op, &mut slots)?;
        }
        
        // Return result from specified output slot
        Ok(slots[plan.output_slot].clone())
    })
}

fn execute_op(
    graph: &CSRGraph,
    op: &BatchOp,
    slots: &mut [Value],
) -> Result<()> {
    match op.opcode {
        OpCode::Add => {
            let a = &slots[op.inputs[0]];
            let b = &slots[op.inputs[1]];
            slots[op.output] = add(a, b);
        }
        OpCode::NeighborAgg => {
            let values = &slots[op.inputs[0]];
            slots[op.output] = neighbor_agg(graph, values, op.agg_type);
        }
        OpCode::LoopStart => {
            // Mark loop entry, save state for iteration
        }
        // ... handle all opcodes
    }
    Ok(())
}
```

### Task 4: Plan Caching

**File**: `python-groggy/python/groggy/builder/execution/cache.py`

```python
"""
Cache compiled batch execution plans.
"""
import hashlib
import pickle
from pathlib import Path
from typing import Optional
from .batch_plan import BatchExecutionPlan

class PlanCache:
    """Cache for compiled batch execution plans."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.groggy' / 'plan_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
    
    def get(self, ir_hash: str) -> Optional[BatchExecutionPlan]:
        """Get cached plan by IR hash."""
        # Check memory cache
        if ir_hash in self._memory_cache:
            return self._memory_cache[ir_hash]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{ir_hash}.plan"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                plan = pickle.load(f)
                self._memory_cache[ir_hash] = plan
                return plan
        
        return None
    
    def put(self, ir_hash: str, plan: BatchExecutionPlan):
        """Cache a compiled plan."""
        self._memory_cache[ir_hash] = plan
        
        cache_file = self.cache_dir / f"{ir_hash}.plan"
        with open(cache_file, 'wb') as f:
            pickle.dump(plan, f)
    
    def hash_ir(self, ir_graph: IRGraph) -> str:
        """Compute hash of IR graph for caching."""
        ir_json = ir_graph.to_json()
        ir_bytes = json.dumps(ir_json, sort_keys=True).encode()
        return hashlib.sha256(ir_bytes).hexdigest()
```

### Task 5: Integration with AlgorithmBuilder

**File**: Update `python-groggy/python/groggy/builder/algorithm_builder.py`

```python
class AlgorithmBuilder:
    def __init__(self, name: str, use_batched=True):
        # ...
        self.use_batched = use_batched
        self._plan_cache = PlanCache()
    
    def build(self) -> Dict:
        """Build algorithm specification."""
        if self.use_batched:
            # Optimize IR
            optimized_ir = optimize_ir(self.ir_graph)
            
            # Check cache
            ir_hash = self._plan_cache.hash_ir(optimized_ir)
            plan = self._plan_cache.get(ir_hash)
            
            if plan is None:
                # Compile to batch plan
                compiler = BatchCompiler()
                plan = compiler.compile(optimized_ir)
                
                # Cache it
                self._plan_cache.put(ir_hash, plan)
            
            return {
                'name': self.name,
                'type': 'batched',
                'plan': plan.to_bytes(),
                'metadata': self.metadata
            }
        else:
            # Legacy step-by-step execution
            return {
                'name': self.name,
                'type': 'steps',
                'steps': self.steps,
                'metadata': self.metadata
            }
```

## Testing Strategy

### Unit Tests

**File**: `tests/test_batch_compilation.py`

```python
def test_compile_simple_arithmetic():
    """Test compiling simple arithmetic to batch plan."""
    b = AlgorithmBuilder("test")
    x = b.init_nodes(1.0)
    y = x * 2.0 + 1.0
    b.attr.attach("output", y)
    
    compiler = BatchCompiler()
    plan = compiler.compile(b.ir_graph)
    
    assert len(plan.operations) > 0
    assert "output" in plan.variables

def test_compile_neighbor_agg():
    """Test compiling neighbor aggregation."""
    # ...

def test_compile_loop():
    """Test compiling loop construct."""
    # ...

def test_plan_serialization():
    """Test plan serialization to bytes."""
    # ...
```

### Integration Tests

**File**: `tests/test_batched_execution.py`

```python
def test_batched_pagerank():
    """Test full PageRank with batched execution."""
    G = create_test_graph()
    
    # Run with batched execution
    result_batched = G.pagerank(use_batched=True)
    
    # Run with legacy step-by-step
    result_legacy = G.pagerank(use_batched=False)
    
    # Should produce identical results
    assert_allclose(result_batched, result_legacy)

def test_plan_caching():
    """Test that plans are cached correctly."""
    G = create_test_graph()
    
    # First run - compiles
    t1 = time.time()
    result1 = G.pagerank()
    compile_time = time.time() - t1
    
    # Second run - uses cache
    t2 = time.time()
    result2 = G.pagerank()
    cached_time = time.time() - t2
    
    # Cached should be much faster
    assert cached_time < compile_time * 0.5
    assert_allclose(result1, result2)
```

## Performance Targets

### FFI Call Reduction

| Algorithm | Step-by-Step | Batched | Improvement |
|-----------|-------------|---------|-------------|
| PageRank (100 iter) | ~100,000 | 1 | 100,000x |
| Label Propagation | ~50,000 | 1 | 50,000x |
| Connected Components | ~1,000 | 1 | 1,000x |

### Execution Time

| Algorithm | Current (fused) | Target (batched) | Speedup |
|-----------|----------------|------------------|---------|
| PageRank | 310ms | <100ms | 3x+ |
| Label Propagation | 180ms | <60ms | 3x+ |

### Compilation Overhead

- First run (compile + execute): <1s compilation overhead
- Subsequent runs (cached): <1ms cache lookup

## Implementation Checklist

### Day 8 Tasks

- [ ] Create `builder/execution/` directory
- [ ] Implement `BatchExecutionPlan` and `BatchOp` classes
- [ ] Implement `BatchCompiler` with topological sort
- [ ] Implement opcode mapping for all IR node types
- [ ] Implement plan serialization (to_bytes, from_bytes)
- [ ] Implement `PlanCache` with memory and disk caching
- [ ] Add FFI `execute_batch_plan()` in Rust
- [ ] Implement batch operation executor in Rust
- [ ] Update `AlgorithmBuilder.build()` to use batching
- [ ] Write unit tests for compilation
- [ ] Write integration tests for execution
- [ ] Benchmark performance improvements
- [ ] Document batched execution system

## Expected Challenges

### Challenge 1: Topological Sort with Loops

Loops create cycles in the dependency graph. Need special handling:
- Treat loop body as subgraph
- Sort loop iterations separately
- Handle loop-carried dependencies

### Challenge 2: Variable Lifetime

Need to track when variables can be freed:
- Liveness analysis from Phase 1 Day 2
- Reuse slots when possible
- Handle loop variables carefully

### Challenge 3: FFI Data Transfer

Batch plans may be large:
- Use efficient binary serialization
- Consider compression for large plans
- Stream results if needed

### Challenge 4: Debugging

Batch execution is opaque:
- Add execution tracing mode
- Support step-through debugging
- Provide IR → BatchPlan visualization

## Success Criteria

- [ ] Batch compilation works for all IR node types
- [ ] FFI reduction: 100,000+ → 1 call per algorithm
- [ ] Performance: 3x+ speedup over fused execution
- [ ] Caching: <1ms overhead on cache hit
- [ ] Correctness: All tests pass, results match legacy
- [ ] Documentation: Clear user guide and examples

## Next Steps (Day 9-10)

- Day 9: Parallel execution of independent operations
- Day 10: Memory optimization and buffer reuse

---

**Status**: 🚧 Starting implementation  
**Phase**: 3 (Batched Execution)  
**Day**: 8 of 16
