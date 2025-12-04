# VarHandle API

`VarHandle` is the core abstraction in the Groggy Algorithm Builder. It represents a variable in the algorithm's intermediate representation (IR) and provides operator overloading for natural mathematical syntax.

## Overview

Every operation in the builder creates and returns a `VarHandle`:

```python
@algorithm
def example(sG):
    x = sG.nodes(1.0)        # VarHandle
    y = x * 2.0              # VarHandle
    z = y + 1.0              # VarHandle
    result = z.normalize()    # VarHandle
    return result
```

## Constructor

**Direct construction not recommended.** VarHandles are created by the builder automatically.

```python
# DON'T DO THIS:
# var = VarHandle("my_var", builder)

# DO THIS:
var = sG.nodes(1.0)  # Builder creates VarHandle internally
```

## Attributes

### `name: str`

The internal variable name in the IR (e.g., `"node_0"`, `"add_3"`).

```python
x = sG.nodes(1.0)
print(x.name)  # "node_0"
```

### `builder: AlgorithmBuilder`

Reference to the parent builder (used internally for chaining operations).

## Arithmetic Operators

All arithmetic operators create new VarHandles representing the operation.

### Addition: `+`

```python
result = x + y       # Add two variables
result = x + 5.0     # Add scalar to variable
result = 5.0 + x     # Scalar + variable (commutative)
```

**IR Generated:**
```json
{
  "type": "core.add",
  "left": "x",
  "right": "y",
  "output": "add_0"
}
```

### Subtraction: `-`

```python
result = x - y       # Subtract variables
result = x - 5.0     # Subtract scalar
result = 5.0 - x     # Reverse subtraction
```

### Multiplication: `*`

```python
result = x * y       # Element-wise multiplication
result = x * 2.0     # Scale by scalar
```

**Common pattern:**
```python
# Weighted combination
weighted = alpha * x + (1 - alpha) * y
```

### Division: `/`

```python
result = x / y       # Element-wise division
result = x / 2.0     # Scale by reciprocal
```

**Division by zero handling:**
```python
# Safe division with epsilon
safe_div = x / (y + 1e-9)
```

### Negation: `-` (unary)

```python
negative = -x        # Negate all values
```

**Example:**
```python
# Reverse direction
reversed_flow = -outflow
```

## Comparison Operators

Comparison operators return VarHandles containing boolean masks (0.0 = false, 1.0 = true).

### Equality: `==`

```python
mask = x == 0.0      # Check if x equals zero
mask = x == y        # Element-wise equality
```

**Example:**
```python
is_sink = (degrees == 0.0)
```

### Inequality: `!=`

```python
mask = x != 0.0      # Non-zero check
```

### Greater Than: `>`

```python
mask = x > 0.5       # Values above threshold
mask = x > y         # Element-wise comparison
```

### Less Than: `<`

```python
mask = x < 0.5       # Values below threshold
```

### Greater or Equal: `>=`

```python
mask = x >= 0.0      # Non-negative values
```

### Less or Equal: `<=`

```python
mask = x <= 1.0      # Bounded values
```

## Fluent Methods

Fluent methods enable method chaining for readable code.

### `.where(if_true, if_false)`

Conditional selection based on boolean mask.

```python
result = mask.where(value_if_true, value_if_false)
```

**Example:**
```python
is_sink = (degrees == 0.0)
contrib = is_sink.where(0.0, ranks / degrees)
# If sink: contrib = 0.0
# Else:    contrib = ranks / degrees
```

**IR Generated:**
```json
{
  "type": "core.where",
  "condition": "is_sink",
  "if_true": "0.0",
  "if_false": "ranks_div_degrees",
  "output": "where_0"
}
```

### `.reduce(op: str)`

Aggregate all values to a single scalar.

**Supported operations:**
- `"sum"` - Sum all values
- `"mean"` - Average of all values
- `"min"` - Minimum value
- `"max"` - Maximum value

```python
total = values.reduce("sum")
average = values.reduce("mean")
min_val = values.reduce("min")
max_val = values.reduce("max")
```

**Example - PageRank sink handling:**
```python
sink_mass = is_sink.where(ranks, 0.0).reduce("sum")
```

**Returns:** VarHandle representing a scalar (broadcasted when used).

### `.normalize()`

Normalize values to sum to 1.0.

```python
normalized = values.normalize()
# Equivalent to: values / values.reduce("sum")
```

**Example:**
```python
@algorithm("pagerank")
def pagerank(sG, ...):
    # ... compute ranks ...
    return ranks.normalize()  # Final normalized PageRank scores
```

### `.degrees()`

Get the out-degree for each node.

```python
deg = node_values.degrees()
```

**Note:** Must be called on a VarHandle representing node values.

**Example:**
```python
ranks = sG.nodes(1.0 / sG.N)
deg = ranks.degrees()
inv_deg = 1.0 / (deg + 1e-9)
```

**IR Generated:**
```json
{
  "type": "graph.degree",
  "source": "ranks",
  "output": "degree_0"
}
```

## Matrix Notation

### Neighbor Aggregation: `sG @ values`

The `@` operator performs neighbor aggregation (sum by default).

```python
neighbor_sum = sG @ values
```

**Equivalent to:**
```python
neighbor_sum = sG.builder.graph_ops.neighbor_agg(values, agg="sum")
```

**Example - PageRank:**
```python
contrib = ranks / degrees
neighbor_sum = sG @ contrib
```

**How it works:**
```
For each node i:
  neighbor_sum[i] = sum(values[j] for j in neighbors(i))
```

**IR Generated:**
```json
{
  "type": "graph.neighbor_agg",
  "source": "values",
  "agg": "sum",
  "output": "neighbor_agg_0"
}
```

## Operator Precedence

Python's standard operator precedence applies:

```python
# Precedence (highest to lowest):
# 1. - (unary negation)
# 2. *, /
# 3. +, -
# 4. ==, !=, <, >, <=, >=

# Example:
result = x * 2.0 + y / 3.0  # Parsed as: (x * 2.0) + (y / 3.0)
```

**Use parentheses for clarity:**
```python
# Good
result = (x + y) / 2.0

# Unclear
result = x + y / 2.0  # Is it (x + y)/2 or x + (y/2)?
```

## Common Patterns

### Safe Division

```python
# Avoid division by zero
inv_deg = 1.0 / (degrees + 1e-9)
```

### Weighted Average

```python
weighted_avg = (alpha * x + (1 - alpha) * y)
```

### Clamping

```python
# Clamp to [0, 1]
clamped = values.where(values > 1.0, 1.0, values)
clamped = clamped.where(clamped < 0.0, 0.0, clamped)
```

### Conditional Computation

```python
is_active = (status == 1.0)
contribution = is_active.where(value, 0.0)
```

### Normalization Patterns

```python
# L1 normalization (sum to 1)
normalized = values / values.reduce("sum")

# Or use helper:
normalized = values.normalize()

# L2 normalization (unit length)
# (future: when pow() is available)
```

## Type Coercion

Scalars are automatically converted to VarHandles when needed:

```python
result = x + 5.0     # 5.0 → scalar VarHandle internally
result = x * 0.85    # 0.85 → scalar VarHandle internally
```

**Supported scalar types:**
- `float` (preferred)
- `int` (converted to float)

**Not supported:**
- `str`, `list`, `dict`, etc.

## Error Handling

### Invalid Operations

```python
# These will raise errors:
result = x + "string"      # TypeError: unsupported operand type
result = x @ y             # Only sG can use @
```

### Undefined Variables

The builder validates variable dependencies:

```python
@algorithm
def bad_algo(sG):
    result = undefined_var + 1.0  # NameError: undefined_var not defined
    return result
```

## Performance Considerations

### No Immediate Execution

Operators build IR, they don't execute immediately:

```python
x = sG.nodes(1.0)
y = x * 2.0          # No computation yet
z = y + 1.0          # Still no computation
algo = builder.build()  # Still just building IR
result = sg.apply(algo)  # NOW execution happens in Rust
```

### Chaining Efficiency

Long chains are fine - they're compiled, not interpreted:

```python
# This is efficient:
result = ((x * 2.0 + 1.0) / y).normalize()
# Single FFI call, fused execution in Rust
```

## Advanced Usage

### Manual Variable Naming

```python
# Normally variables are auto-named (add_0, mul_1, etc.)
# For debugging, you can create named variables:
ranks = sG.builder.var("ranks", initial_ranks)
# "ranks" will appear in IR, easier to trace
```

### Accessing IR

```python
x = sG.nodes(1.0)
y = x * 2.0
print(y.name)  # "mul_0"
print(y.builder)  # <AlgorithmBuilder object>
```

## Migration from Old API

### Before (explicit builder calls):

```python
result = builder.core.add(builder.core.mul(x, 2.0), 1.0)
mask = builder.core.compare(values, "gt", 0.5)
output = builder.core.where(mask, a, b)
```

### After (VarHandle operators):

```python
result = x * 2.0 + 1.0
mask = values > 0.5
output = mask.where(a, b)
```

**Reduction: 75-80% less code, 100% more readable.**

## Examples

### PageRank Iteration

```python
with sG.builder.iter.loop(max_iter):
    # Compute contribution from each node
    contrib = ranks / (degrees + 1e-9)
    
    # Aggregate from neighbors
    neighbor_sum = sG @ contrib
    
    # Update ranks
    ranks = sG.builder.var("ranks",
        damping * neighbor_sum + (1 - damping) / sG.N
    )
```

### Label Propagation Mode

```python
# Collect neighbor labels
neighbor_labels = sG.builder.graph_ops.collect_neighbor_values(labels)

# Find most common
most_common = sG.builder.core.mode(neighbor_labels)

# Update
labels = sG.builder.var("labels", most_common)
```

### Custom Centrality

```python
@algorithm
def custom_centrality(sG, alpha=0.5):
    degrees = sG.builder.graph_ops.degree()
    neighbors = sG.nodes(1.0)
    neighbor_count = sG @ neighbors
    
    # Blend degree and neighbor count
    centrality = alpha * degrees + (1 - alpha) * neighbor_count
    return centrality.normalize()
```

## See Also

- [CoreOps API](core.md) - Arithmetic operations
- [GraphOps API](graph.md) - Topology operations
- Operator overloading is covered in the main builder guide and examples.
- [Algorithm Examples](../tutorials/README.md)
