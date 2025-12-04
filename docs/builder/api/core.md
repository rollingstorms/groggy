# CoreOps API

`CoreOps` provides pure value-space operations: arithmetic, comparisons, reductions, and conditionals. These operations work on node or edge values without depending on graph topology.

## Overview

Access CoreOps through `sG.builder.core`:

```python
@algorithm
def example(sG):
    x = sG.nodes(1.0)
    y = sG.builder.core.add(x, 2.0)  # Or: y = x + 2.0
    return y
```

**Note:** Most CoreOps are accessible via VarHandle operators. Direct calls are rarely needed.

## Arithmetic Operations

### `add(left, right)`

Element-wise addition.

```python
result = sG.builder.core.add(x, y)
# Equivalent: result = x + y
```

**Parameters:**
- `left`: VarHandle or scalar
- `right`: VarHandle or scalar

**Returns:** VarHandle

**Example:**
```python
total = sG.builder.core.add(income, bonus)
# Or: total = income + bonus
```

### `sub(left, right)`

Element-wise subtraction.

```python
result = sG.builder.core.sub(x, y)
# Equivalent: result = x - y
```

### `mul(left, right)`

Element-wise multiplication.

```python
result = sG.builder.core.mul(x, 2.0)
# Equivalent: result = x * 2.0
```

**Example - Scaling:**
```python
scaled = sG.builder.core.mul(values, 0.85)
# Or: scaled = values * 0.85
```

### `div(left, right)`

Element-wise division.

```python
result = sG.builder.core.div(x, y)
# Equivalent: result = x / y
```

**Warning:** No automatic division-by-zero protection. Use `recip()` or add epsilon:

```python
# Bad (may divide by zero):
result = x / degrees

# Good:
result = x / (degrees + 1e-9)
```

### `recip(values, epsilon=1e-10)`

Reciprocal with epsilon for numerical stability.

```python
inv_deg = sG.builder.core.recip(degrees, epsilon=1e-9)
# Equivalent to: 1.0 / (degrees + 1e-9)
```

**Parameters:**
- `values`: VarHandle
- `epsilon`: Small value to prevent division by zero (default: 1e-10)

**Returns:** VarHandle

**Example - PageRank:**
```python
degrees = ranks.degrees()
inv_deg = sG.builder.core.recip(degrees, epsilon=1e-9)
contrib = ranks * inv_deg
```

## Comparison Operations

### `compare(left, op, right)`

Element-wise comparison returning boolean mask (0.0 or 1.0).

```python
mask = sG.builder.core.compare(values, "gt", 0.5)
# Equivalent: mask = values > 0.5
```

**Parameters:**
- `left`: VarHandle
- `op`: Comparison operator string
  - `"eq"` - Equal to (==)
  - `"ne"` - Not equal (!=)
  - `"lt"` - Less than (<)
  - `"le"` - Less or equal (<=)
  - `"gt"` - Greater than (>)
  - `"ge"` - Greater or equal (>=)
- `right`: VarHandle or scalar

**Returns:** VarHandle (boolean mask)

**Operator equivalents:**
```python
# These are equivalent:
mask = sG.builder.core.compare(x, "gt", 0.5)
mask = x > 0.5

mask = sG.builder.core.compare(x, "eq", y)
mask = x == y
```

**Example - Identify sinks:**
```python
degrees = ranks.degrees()
is_sink = sG.builder.core.compare(degrees, "eq", 0.0)
# Or: is_sink = degrees == 0.0
```

## Conditional Operations

### `where(condition, if_true, if_false)`

Conditional selection based on mask.

```python
result = sG.builder.core.where(mask, value_a, value_b)
# Equivalent: result = mask.where(value_a, value_b)
```

**Parameters:**
- `condition`: VarHandle (boolean mask)
- `if_true`: VarHandle or scalar (used where condition is 1.0)
- `if_false`: VarHandle or scalar (used where condition is 0.0)

**Returns:** VarHandle

**Semantics:**
```python
# For each element i:
# result[i] = if_true[i] if condition[i] == 1.0 else if_false[i]
```

**Example - Sink handling:**
```python
is_sink = degrees == 0.0
contrib = sG.builder.core.where(is_sink, 0.0, ranks / degrees)
# Or: contrib = is_sink.where(0.0, ranks / degrees)
```

**Example - Clamping:**
```python
# Clamp values to [0, 1]
too_high = values > 1.0
clamped = sG.builder.core.where(too_high, 1.0, values)
too_low = clamped < 0.0
clamped = sG.builder.core.where(too_low, 0.0, clamped)
```

## Reduction Operations

### `reduce_scalar(values, op="sum")`

Aggregate all values to a single scalar.

```python
total = sG.builder.core.reduce_scalar(values, op="sum")
# Equivalent: total = values.reduce("sum")
```

**Parameters:**
- `values`: VarHandle
- `op`: Aggregation operation
  - `"sum"` - Sum all values
  - `"mean"` - Average of all values
  - `"min"` - Minimum value
  - `"max"` - Maximum value

**Returns:** VarHandle (scalar, broadcasted when used)

**Examples:**
```python
# Sum
total = sG.builder.core.reduce_scalar(values, "sum")
# Or: total = values.reduce("sum")

# Average
avg = sG.builder.core.reduce_scalar(values, "mean")
# Or: avg = values.reduce("mean")

# Min/Max
min_val = sG.builder.core.reduce_scalar(values, "min")
max_val = sG.builder.core.reduce_scalar(values, "max")
```

**Example - PageRank sink mass:**
```python
is_sink = degrees == 0.0
sink_ranks = is_sink.where(ranks, 0.0)
sink_mass = sG.builder.core.reduce_scalar(sink_ranks, "sum")
```

### `normalize_sum(values)`

Normalize values to sum to 1.0.

```python
normalized = sG.builder.core.normalize_sum(values)
# Equivalent: normalized = values.normalize()
```

**Returns:** VarHandle

**Equivalent to:**
```python
total = values.reduce("sum")
normalized = values / total
```

**Example:**
```python
# Final PageRank normalization
return sG.builder.core.normalize_sum(ranks)
# Or: return ranks.normalize()
```

## Broadcasting Operations

### `broadcast_scalar(scalar, reference)`

Broadcast a scalar to match the shape of a reference variable.

```python
uniform = sG.builder.core.broadcast_scalar(inv_n_scalar, ranks)
```

**Parameters:**
- `scalar`: VarHandle representing a scalar value
- `reference`: VarHandle whose shape to match

**Returns:** VarHandle (broadcasted to reference shape)

**Example - Uniform distribution:**
```python
node_count = sG.builder.graph_node_count()
inv_n = sG.builder.core.recip(node_count)
uniform = sG.builder.core.broadcast_scalar(inv_n, ranks)
# Now uniform is a node-value map with 1/N everywhere
```

**Modern alternative:**
```python
# Instead of broadcast, use direct initialization:
uniform = sG.nodes(1.0 / sG.N)
```

## Math Operations

### `pow(base, exponent)`

Element-wise power operation.

```python
squared = sG.builder.core.pow(values, 2.0)
```

**Parameters:**
- `base`: VarHandle or scalar
- `exponent`: VarHandle or scalar

**Returns:** VarHandle

**Examples:**
```python
# Square values
squared = sG.builder.core.pow(values, 2.0)

# Square root (via pow)
sqrt_vals = sG.builder.core.pow(values, 0.5)

# Variable exponent
result = sG.builder.core.pow(base_values, exp_values)
```

### `abs(values)`

Element-wise absolute value.

```python
abs_vals = sG.builder.core.abs(values)
```

**Example:**
```python
# Distance metric
diff = x - y
distance = sG.builder.core.abs(diff)
```

### `sqrt(values)`

Element-wise square root.

```python
sqrt_vals = sG.builder.core.sqrt(values)
```

**Note:** Values should be non-negative.

**Example:**
```python
# Standard deviation (simplified)
variance = ((values - mean).pow(2)).reduce("mean")
std_dev = sG.builder.core.sqrt(variance)
```

### `exp(values)`

Element-wise exponential (e^x).

```python
exponentials = sG.builder.core.exp(values)
```

**Example - Softmax-style:**
```python
exp_vals = sG.builder.core.exp(logits)
total = exp_vals.reduce("sum")
probs = exp_vals / total
```

### `log(values, base=None)`

Element-wise logarithm.

```python
# Natural log
ln_vals = sG.builder.core.log(values)

# Log base 10
log10_vals = sG.builder.core.log(values, base=10.0)
```

**Parameters:**
- `values`: VarHandle (should be positive)
- `base`: Optional logarithm base (default: natural log)

**Example:**
```python
# Log-space computation
log_probs = sG.builder.core.log(probabilities)
```

### `min(left, right)`

Element-wise minimum.

```python
min_vals = sG.builder.core.min(values, 1.0)
```

**Parameters:**
- `left`: VarHandle or scalar
- `right`: VarHandle or scalar

**Example - Clamping:**
```python
# Cap values at 1.0
capped = sG.builder.core.min(values, 1.0)
```

### `max(left, right)`

Element-wise maximum.

```python
max_vals = sG.builder.core.max(values, 0.0)
```

**Example - Ensure non-negative:**
```python
positive = sG.builder.core.max(values, 0.0)
```

## Utility Operations

### `clip(values, min_value=None, max_value=None)`

Clamp values to [min_value, max_value].

```python
clipped = sG.builder.core.clip(values, min_value=0.0, max_value=1.0)
```

**Parameters:**
- `values`: VarHandle
- `min_value`: Optional minimum value
- `max_value`: Optional maximum value

At least one of `min_value` or `max_value` must be provided.

**Returns:** VarHandle

**Examples:**
```python
# Clamp to [0, 1]
normalized = sG.builder.core.clip(values, 0.0, 1.0)

# Clamp minimum only
safe_deg = sG.builder.core.clip(degrees, min_value=1.0)

# Clamp maximum only
bounded = sG.builder.core.clip(values, max_value=100.0)
```

### `mode(lists, tie_break="lowest")`

Find most common value in lists.

```python
most_common = sG.builder.core.mode(neighbor_labels)
```

**Parameters:**
- `lists`: VarHandle containing lists of values
- `tie_break`: Tie-breaking strategy
  - `"lowest"` - Choose lowest value (default)
  - `"highest"` - Choose highest value
  - `"keep"` - Keep first occurrence

**Returns:** VarHandle

**Used in Label Propagation:**
```python
neighbor_labels = sG.builder.graph_ops.collect_neighbor_values(labels)
most_common = sG.builder.core.mode(neighbor_labels, tie_break="lowest")
```

### `histogram(values, bins=10)`

Compute histogram of values.

```python
hist = sG.builder.core.histogram(values, bins=20)
```

**Parameters:**
- `values`: VarHandle
- `bins`: Number of histogram bins

**Returns:** VarHandle (histogram data structure)

**Example:**
```python
# Analyze degree distribution
degrees = sG.builder.graph_ops.degree()
degree_hist = sG.builder.core.histogram(degrees, bins=50)
```

### `update_in_place(target, values, mask=None)`

Update target variable in-place (for async algorithms).

```python
updated = sG.builder.core.update_in_place(target, new_values, mask)
```

**Parameters:**
- `target`: VarHandle to update
- `values`: New values to write
- `mask`: Optional boolean mask (update only where mask is 1.0)

**Returns:** VarHandle (same as target)

**Use case:** Asynchronous label propagation

## Deprecated Operations

These operations moved to other traits but remain for backward compatibility:

### `neighbor_agg(values, agg="sum", weights=None)` ⚠️ Deprecated

**Use instead:** `sG.builder.graph_ops.neighbor_agg()`

```python
# Deprecated:
neighbor_sum = sG.builder.core.neighbor_agg(values, "sum")

# Use:
neighbor_sum = sG.builder.graph_ops.neighbor_agg(values, "sum")
# Or: neighbor_sum = sG @ values
```

### `collect_neighbor_values(values, include_self=True)` ⚠️ Deprecated

**Use instead:** `sG.builder.graph_ops.collect_neighbor_values()`

### `neighbor_mode_update(target, ...)` ⚠️ Deprecated

**Use instead:** `sG.builder.graph_ops.neighbor_mode_update()`

## Common Patterns

### Safe Division

```python
# Bad (may divide by zero):
result = x / y

# Good:
result = x / (y + 1e-9)

# Best:
inv_y = sG.builder.core.recip(y, epsilon=1e-9)
result = x * inv_y
```

### Conditional Computation

```python
# Pattern: apply operation only when condition is true
is_active = status == 1.0
active_values = is_active.where(values, 0.0)
result = sG @ active_values
```

### Normalization

```python
# L1 normalization (sum to 1)
normalized = values / values.reduce("sum")
# Or: normalized = values.normalize()

# Min-max normalization
min_val = values.reduce("min")
max_val = values.reduce("max")
normalized = (values - min_val) / (max_val - min_val + 1e-9)
```

### Aggregation with Filtering

```python
# Sum only positive values
is_positive = values > 0.0
positive_sum = is_positive.where(values, 0.0).reduce("sum")
```

## Performance Notes

### Operator Fusion

Related operations may be fused by the optimizer:

```python
# These might be fused into a single kernel:
result = (x * 2.0 + 1.0) / y
```

### Reduction Efficiency

Reductions are efficient O(N) operations:

```python
# Don't worry about calling reduce multiple times:
mean = values.reduce("mean")
min_val = values.reduce("min")
max_val = values.reduce("max")
# Each is a fast single-pass operation
```

### Broadcast Overhead

Broadcasts are essentially free (lazy evaluation):

```python
# This is efficient:
uniform = 1.0 / sG.N
result = values + uniform  # Broadcast happens implicitly
```

## Examples

### Weighted Average

```python
@algorithm
def weighted_avg(sG, alpha=0.5):
    x = sG.builder.attr.load("feature_a", default=0.0)
    y = sG.builder.attr.load("feature_b", default=0.0)
    result = alpha * x + (1 - alpha) * y
    return result
```

### Standardization

```python
@algorithm
def standardize(sG, attr_name):
    values = sG.builder.attr.load(attr_name)
    mean = values.reduce("mean")
    
    # Variance
    diff = values - mean
    variance = (diff * diff).reduce("mean")
    std_dev = sG.builder.core.sqrt(variance)
    
    # Standardize
    standardized = diff / (std_dev + 1e-9)
    return standardized
```

### Thresholding

```python
@algorithm
def threshold(sG, attr_name, threshold=0.5):
    values = sG.builder.attr.load(attr_name)
    above_threshold = values > threshold
    result = above_threshold.where(1.0, 0.0)
    return result
```

## See Also

- [VarHandle API](varhandle.md) - Operator overloading
- [GraphOps API](graph.md) - Topology operations
- Migration notes available in repository docs
