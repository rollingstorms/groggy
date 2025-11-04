# Builder Implementation Quick Reference

**Purpose**: Fast lookup for implementing builder features  
**Audience**: Developer implementing the plan

---

## Core Operations Implementation

### Add to `builder.py`

```python
class CoreOps:
    """Core step primitives."""
    
    def __init__(self, builder: 'AlgorithmBuilder'):
        self.builder = builder
    
    def add(self, left: VarHandle, right: Union[VarHandle, float]) -> VarHandle:
        """Element-wise addition or scalar add."""
        var = self.builder._new_var("add")
        self.builder.steps.append({
            "type": "core.add",
            "left": left.name if isinstance(left, VarHandle) else left,
            "right": right.name if isinstance(right, VarHandle) else right,
            "output": var.name
        })
        return var
    
    def mul(self, left: VarHandle, scalar: float) -> VarHandle:
        """Scalar multiplication."""
        var = self.builder._new_var("mul")
        self.builder.steps.append({
            "type": "core.mul",
            "left": left.name,
            "right": scalar,
            "output": var.name
        })
        return var
    
    def normalize_sum(self, values: VarHandle) -> VarHandle:
        """Normalize so sum = 1.0."""
        var = self.builder._new_var("normalized")
        self.builder.steps.append({
            "type": "normalize_sum",
            "input": values.name,
            "output": var.name
        })
        return var
```

### Update `_encode_step()`

```python
def _encode_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
    step_type = step.get("type")
    
    # Existing cases...
    
    if step_type == "core.add":
        return {
            "id": "core.add",
            "params": {
                "left": step["left"],
                "right": step["right"],
                "target": step["output"]
            }
        }
    
    if step_type == "core.mul":
        return {
            "id": "core.mul",
            "params": {
                "left": step["left"],
                "right": step["right"],
                "target": step["output"]
            }
        }
    
    if step_type == "normalize_sum":
        return {
            "id": "core.normalize_values",
            "params": {
                "source": step["input"],
                "target": step["output"],
                "method": "sum",
                "epsilon": 1e-9
            }
        }
```

---

## Variable Management

### Add `var()` method

```python
class AlgorithmBuilder:
    def var(self, name: str, value: VarHandle) -> VarHandle:
        """
        Create or reassign a variable.
        
        Args:
            name: Variable name
            value: Value to assign
            
        Returns:
            VarHandle with the given name
        """
        # Track variable assignment
        handle = VarHandle(name, self)
        self.variables[name] = handle
        
        # Add alias step if needed (maps value to name)
        if value.name != name:
            self.steps.append({
                "type": "alias",
                "source": value.name,
                "target": name
            })
        
        return handle
```

### Update `_encode_step()` for alias

```python
if step_type == "alias":
    # No Rust step needed - just variable tracking
    # But we can add a comment or passthrough
    return {
        "id": "core.alias",  # Or skip this step entirely
        "params": {
            "source": step["source"],
            "target": step["target"]
        }
    }
```

**Alternative**: Don't generate steps for alias, just track in Python

---

## Map Operations

### Add `map_nodes()` method

```python
class AlgorithmBuilder:
    def map_nodes(self, 
                  fn: str,
                  inputs: Dict[str, VarHandle] = None) -> VarHandle:
        """
        Map expression over nodes with neighbor access.
        
        Args:
            fn: Expression string (e.g., "sum(ranks[neighbors(node)])")
            inputs: Variable context
            
        Returns:
            VarHandle for result
            
        Example:
            sums = builder.map_nodes(
                "sum(ranks[neighbors(node)])",
                inputs={"ranks": ranks}
            )
        """
        var = self._new_var("mapped")
        
        # Build inputs dict with variable names
        input_vars = {}
        if inputs:
            for key, val in inputs.items():
                input_vars[key] = val.name
        
        self.steps.append({
            "type": "map_nodes",
            "fn": fn,
            "inputs": input_vars,
            "output": var.name
        })
        
        return var
```

### Update `_encode_step()`

```python
if step_type == "map_nodes":
    # Parse expression to Rust format
    # For now, pass as JSON string
    return {
        "id": "core.map_nodes",
        "params": {
            "source": step["inputs"].get("ranks", ""),  # Main input
            "target": step["output"],
            "expr": self._parse_expression(step["fn"], step["inputs"])
        }
    }

def _parse_expression(self, fn: str, inputs: Dict) -> Dict:
    """
    Convert expression string to Rust Expr format.
    
    Simple version: just pass the function name and detect patterns
    """
    if "sum(" in fn and "neighbors(" in fn:
        # Extract variable name from pattern like "sum(ranks[neighbors(node)])"
        import re
        match = re.search(r'sum\((\w+)\[neighbors\(', fn)
        if match:
            var_name = match.group(1)
            return {
                "Sum": {
                    "NeighborMap": {
                        "var": var_name
                    }
                }
            }
    
    # Fallback: return as raw expression string
    return {"Raw": fn}
```

---

## Iteration Implementation

### Add `LoopContext` class

```python
class LoopContext:
    """Context manager for loop body."""
    
    def __init__(self, builder: 'AlgorithmBuilder', iterations: int):
        self.builder = builder
        self.iterations = iterations
        self.start_step = None
        self.loop_vars = {}  # Track variables at loop start
    
    def __enter__(self):
        # Mark start of loop body
        self.start_step = len(self.builder.steps)
        
        # Snapshot current variables
        self.loop_vars = dict(self.builder.variables)
        
        return self
    
    def __exit__(self, *args):
        # Unroll the loop
        self.builder._finalize_loop(
            self.start_step, 
            self.iterations,
            self.loop_vars
        )

class AlgorithmBuilder:
    def iterate(self, count: int) -> LoopContext:
        """
        Create a loop that repeats for `count` iterations.
        
        Example:
            with builder.iterate(20):
                x = builder.core.add(x, 1)
        """
        return LoopContext(self, count)
```

### Add loop unrolling logic

```python
class AlgorithmBuilder:
    def _finalize_loop(self, 
                      start_step: int, 
                      iterations: int,
                      loop_vars: Dict[str, VarHandle]):
        """
        Unroll loop by repeating steps.
        
        Args:
            start_step: Index where loop body starts
            iterations: Number of times to repeat
            loop_vars: Variables at loop start
        """
        # Extract loop body
        loop_body = self.steps[start_step:]
        
        # Remove loop body from main steps
        self.steps = self.steps[:start_step]
        
        # Track variable renames across iterations
        var_mapping = {v.name: v.name for v in loop_vars.values()}
        
        # Repeat body N times
        for iteration in range(iterations):
            for step in loop_body:
                # Clone step
                new_step = step.copy()
                
                # Rename variables for this iteration
                if "input" in new_step:
                    new_step["input"] = var_mapping.get(
                        new_step["input"], 
                        new_step["input"]
                    )
                
                if "left" in new_step:
                    new_step["left"] = var_mapping.get(
                        new_step["left"],
                        new_step["left"]
                    )
                
                if "right" in new_step and isinstance(new_step["right"], str):
                    new_step["right"] = var_mapping.get(
                        new_step["right"],
                        new_step["right"]
                    )
                
                # Generate unique output name for this iteration
                if "output" in new_step:
                    original_output = new_step["output"]
                    new_output = f"{original_output}_iter{iteration}"
                    new_step["output"] = new_output
                    
                    # Update mapping for next iteration
                    var_mapping[original_output] = new_output
                
                self.steps.append(new_step)
        
        # Add final aliases to restore original variable names
        for original, final in var_mapping.items():
            if original != final and original in loop_vars:
                self.steps.append({
                    "type": "alias",
                    "source": final,
                    "target": original
                })
                
                # Update variable handle
                self.variables[original].name = original
```

---

## Validation Integration

### Add FFI binding

```python
# In python-groggy/src/ffi/api/pipeline.rs

#[pyfunction]
pub fn validate_pipeline(py: Python, steps: Vec<PyStepSpec>) -> PyResult<PyValidationReport> {
    // Convert Python steps to Rust StepSpec
    let rust_steps: Vec<StepSpec> = steps
        .into_iter()
        .map(|s| s.to_rust())
        .collect();
    
    // Get schema registry
    let schema_registry = get_global_schema_registry();
    
    // Validate
    let validator = PipelineValidator::new(&schema_registry);
    let report = validator.validate(&rust_steps);
    
    // Convert to Python
    Ok(PyValidationReport::from(report))
}
```

### Add Python method

```python
class AlgorithmBuilder:
    def build(self, validate: bool = True) -> 'BuiltAlgorithm':
        """
        Build algorithm with optional validation.
        
        Args:
            validate: If True, validate pipeline before building
            
        Raises:
            ValidationError: If pipeline is invalid
        """
        if validate:
            report = self._validate()
            if not report.is_valid():
                raise ValidationError(report.format())
        
        return BuiltAlgorithm(self.name, self.steps)
    
    def _validate(self):
        """Validate pipeline using Rust validator."""
        from groggy import _groggy
        
        # Convert steps to Rust format
        rust_steps = [self._encode_step(s) for s in self.steps]
        
        # Call Rust validation
        return _groggy.validate_pipeline(rust_steps)
```

---

## Testing Template

### PageRank Test

```python
def test_builder_pagerank():
    """Build and execute PageRank using builder DSL."""
    from groggy import Graph, Subgraph
    from groggy.builder import AlgorithmBuilder
    
    # Create test graph
    graph = Graph()
    nodes = [graph.add_node() for _ in range(10)]
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i], nodes[i + 1])
    sg = Subgraph.from_graph(graph)
    
    # Build PageRank
    builder = AlgorithmBuilder("pagerank")
    ranks = builder.init_nodes(1.0)
    
    with builder.iterate(20):
        sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        damped = builder.core.mul(sums, 0.85)
        ranks = builder.var("ranks", 
            builder.core.add(damped, 0.15)
        )
        ranks = builder.core.normalize_sum(ranks)
    
    builder.attach_as("pagerank", ranks)
    
    # Execute
    algo = builder.build(validate=True)
    result = sg.apply(algo)
    
    # Verify
    from groggy.algorithms import pagerank
    expected = sg.apply(pagerank(iterations=20))
    
    for node in result.nodes():
        actual = result.get_node_attr(node, "pagerank")
        expect = expected.get_node_attr(node, "pagerank")
        assert abs(actual - expect) < 1e-6, \
            f"Node {node}: {actual} != {expect}"
```

---

## Common Patterns

### Scalar Operations

```python
# Multiply by scalar
scaled = builder.core.mul(values, 0.85)

# Add scalar
offset = builder.core.add(values, 0.15)

# Combine
result = builder.core.add(
    builder.core.mul(values, 0.85),
    0.15
)
```

### Neighbor Aggregation

```python
# Sum neighbor values
sums = builder.map_nodes(
    "sum(values[neighbors(node)])",
    inputs={"values": values}
)

# Mean neighbor values
means = builder.map_nodes(
    "mean(values[neighbors(node)])",
    inputs={"values": values}
)

# Most common neighbor value
modes = builder.map_nodes(
    "mode(labels[neighbors(node)])",
    inputs={"labels": labels}
)
```

### Variable Updates

```python
# Initial value
x = builder.init_nodes(0.0)

# Update in loop
with builder.iterate(10):
    x = builder.var("x", builder.core.add(x, 1))
    # x is now updated for next iteration
```

---

## Debugging Tips

### Print Generated Steps

```python
builder = AlgorithmBuilder("debug")
# ... build pipeline ...

# See what steps are generated
for i, step in enumerate(builder.steps):
    print(f"{i}: {step}")
```

### Validate Explicitly

```python
builder = AlgorithmBuilder("test")
# ... build pipeline ...

# Validate without building
report = builder._validate()
print(report.format())
```

### Check Variable Tracking

```python
builder = AlgorithmBuilder("test")
x = builder.init_nodes(1.0)
y = builder.core.add(x, 2.0)

# Check variable names
print(f"x: {x.name}")  # "nodes_0"
print(f"y: {y.name}")  # "add_1"
print(f"vars: {builder.variables}")
```

---

## Performance Notes

- Loop unrolling creates N copies of loop body - OK for N < 1000
- Expression parsing is one-time at build
- Validation is fast (< 5ms for typical pipelines)
- Execution speed = native (same Rust backend)

---

## Reference: Rust Step IDs

```
# Arithmetic
core.add, core.sub, core.mul, core.div

# Aggregations  
core.reduce_nodes (reducer: sum/mean/max/min)

# Normalization
core.normalize_values (method: sum/max/minmax)
core.normalize_node_values (legacy)

# Structural
core.node_degree
core.weighted_degree

# Attributes
core.init_nodes
core.load_node_attr
core.attach_node_attr

# Transformations
core.map_nodes
```

---

## Quick Checklist

Implementation order:

1. ✅ Add `CoreOps` class
2. ✅ Add `core.add()`, `core.mul()`
3. ✅ Add `core.normalize_sum()`
4. ✅ Update `_encode_step()` for new ops
5. ✅ Add `var()` method
6. ✅ Add `map_nodes()` method
7. ✅ Add `LoopContext` class
8. ✅ Add `iterate()` method
9. ✅ Implement loop unrolling
10. ✅ Write PageRank test
11. ✅ Write LPA test
12. ✅ Integrate validation
13. ✅ Add type hints

Ready to code! 🚀
