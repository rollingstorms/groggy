# IterOps API

`IterOps` provides control flow operations: loops, convergence detection, and iteration strategies. These enable iterative graph algorithms like PageRank, label propagation, and belief propagation.

## Overview

Access IterOps through `sG.builder.iter`:

```python
@algorithm
def example(sG, max_iter=100):
    values = sG.nodes(1.0)
    
    with sG.builder.iter.loop(max_iter):
        # Iterative updates
        values = sG.builder.var("values", values * 2.0)
    
    return values
```

## Basic Loops

### `loop(count)`

Fixed iteration loop.

```python
with sG.builder.iter.loop(100):
    # Code here runs 100 times
    values = sG.builder.var("values", updated_values)
```

**Parameters:**
- `count`: Number of iterations (int)

**Returns:** Context manager

**Usage:**

```python
@algorithm
def pagerank(sG, max_iter=100, damping=0.85):
    ranks = sG.nodes(1.0 / sG.N)
    deg = ranks.degrees()
    
    with sG.builder.iter.loop(max_iter):
        contrib = ranks / (deg + 1e-9)
        neighbor_sum = sG @ contrib
        ranks = sG.builder.var("ranks",
            damping * neighbor_sum + (1 - damping) / sG.N
        )
    
    return ranks.normalize()
```

**Key points:**
- Updates must use `sG.builder.var()` to create loop-carried dependencies
- Variable name (e.g., `"ranks"`) identifies what gets updated
- Same variable name = update in place
- Different variable name = new variable

**Loop-Carried Dependencies:**

```python
with sG.builder.iter.loop(10):
    # ✅ Correct - updates 'values' variable
    values = sG.builder.var("values", values * 2.0)

# ❌ Incorrect - creates new variable each iteration (not updated)
with sG.builder.iter.loop(10):
    values = values * 2.0  # Doesn't propagate across iterations
```

### `loop_range(start, end, step=1)`

Loop with range (like Python's `range()`).

```python
with sG.builder.iter.loop_range(0, 100, step=1):
    # Iteration variable accessible as special variable
    pass
```

**Parameters:**
- `start`: Starting value (int)
- `end`: Ending value (exclusive, int)
- `step`: Step size (default: 1)

**Example:**

```python
@algorithm
def gradual_decay(sG):
    values = sG.nodes(100.0)
    
    with sG.builder.iter.loop_range(0, 100):
        # Decay by 1% each iteration
        values = sG.builder.var("values", values * 0.99)
    
    return values
```

**Note:** Iteration variable access is limited in current implementation.

## Convergence-Based Loops

### `until_converged(tolerance=1e-6, max_iter=1000, check_every=1)`

Loop until values converge.

```python
with sG.builder.iter.until_converged(tolerance=1e-6):
    values = sG.builder.var("values", updated_values)
```

**Parameters:**
- `tolerance`: Convergence threshold (float, default: 1e-6)
- `max_iter`: Maximum iterations (int, default: 1000)
- `check_every`: Check convergence every N iterations (default: 1)

**Returns:** Context manager

**Convergence criterion:**
```python
# Converged when:
max(abs(new_values - old_values)) < tolerance
```

**Example - PageRank with convergence:**

```python
@algorithm
def pagerank_converged(sG, tol=1e-6, damping=0.85):
    ranks = sG.nodes(1.0 / sG.N)
    deg = ranks.degrees()
    
    with sG.builder.iter.until_converged(tolerance=tol, max_iter=1000):
        contrib = ranks / (deg + 1e-9)
        neighbor_sum = sG @ contrib
        ranks = sG.builder.var("ranks",
            damping * neighbor_sum + (1 - damping) / sG.N
        )
    
    return ranks.normalize()
```

**Performance tip:**
```python
# Check convergence less frequently for speed
with sG.builder.iter.until_converged(tolerance=1e-6, check_every=10):
    # 10x fewer convergence checks
    values = sG.builder.var("values", updated)
```

### `while_condition(condition_var, max_iter=1000)`

Loop while condition is true.

```python
with sG.builder.iter.while_condition(has_changes, max_iter=100):
    # Loop while has_changes > 0
    pass
```

**Parameters:**
- `condition_var`: VarHandle (scalar, loop continues while > 0)
- `max_iter`: Maximum iterations (safety limit)

**Example - Flood fill:**

```python
@algorithm
def flood_fill(sG, source_mask):
    filled = source_mask
    changed = sG.builder.core.reduce_scalar(source_mask, "sum")
    
    with sG.builder.iter.while_condition(changed, max_iter=1000):
        # Propagate to neighbors
        neighbor_filled = sG @ filled
        newly_filled = neighbor_filled > 0.0
        
        # Update
        old_sum = filled.reduce("sum")
        filled = sG.builder.var("filled", newly_filled)
        new_sum = filled.reduce("sum")
        
        # Check if anything changed
        changed = sG.builder.var("changed", new_sum - old_sum)
    
    return filled
```

## Update Strategies

### `set_strategy(strategy)`

Set update strategy for the loop.

```python
sG.builder.iter.set_strategy("async")
with sG.builder.iter.loop(100):
    # Updates happen asynchronously
    pass
```

**Strategies:**

- `"sync"` - Synchronous updates (default)
  - All nodes updated simultaneously
  - Uses previous iteration's values
  - Deterministic

- `"async"` - Asynchronous updates
  - Nodes updated in order
  - Uses latest available values
  - Non-deterministic but often faster convergence

- `"random"` - Random order updates
  - Nodes updated in random order each iteration
  - Non-deterministic

**Example - Asynchronous Label Propagation:**

```python
@algorithm
def lpa_async(sG, max_iter=10):
    labels = sG.nodes(unique=True)
    
    # Set async strategy
    sG.builder.iter.set_strategy("async")
    
    with sG.builder.iter.loop(max_iter):
        labels = sG.builder.graph_ops.neighbor_mode_update(
            labels, include_self=True, ordered=True
        )
    
    return labels
```

**Comparison:**

| Strategy | Deterministic | Convergence Speed | Use Case |
|----------|--------------|-------------------|----------|
| `sync` | ✅ Yes | Moderate | PageRank, most algorithms |
| `async` | ❌ No | Fast | Label propagation, Gauss-Seidel |
| `random` | ❌ No | Variable | Stochastic algorithms |

### `with_schedule(schedule)`

Use custom update schedule.

```python
# Update every 2 iterations
schedule = sG.builder.iter.every_n_iterations(2)

with sG.builder.iter.loop(100):
    with sG.builder.iter.with_schedule(schedule):
        values = sG.builder.var("values", expensive_update)
```

**Schedules:**
- `every_n_iterations(n)` - Update every N iterations
- `exponential_backoff(base=2)` - Update less frequently over time
- `custom_schedule(iterations_list)` - Update at specific iterations

**Example - Progressive refinement:**

```python
@algorithm
def progressive_update(sG):
    fast_values = sG.nodes(1.0)
    slow_values = sG.nodes(1.0)
    
    with sG.builder.iter.loop(100):
        # Fast update every iteration
        fast_values = sG.builder.var("fast", fast_values * 0.99)
        
        # Slow update every 10 iterations
        with sG.builder.iter.with_schedule(
            sG.builder.iter.every_n_iterations(10)
        ):
            slow_values = sG.builder.var("slow", expensive_computation())
    
    return fast_values + slow_values
```

## Loop Control

### `break_if(condition)`

Break loop if condition is true.

```python
with sG.builder.iter.loop(1000):
    # ...
    converged = max_diff < tolerance
    sG.builder.iter.break_if(converged)
```

**Parameters:**
- `condition`: VarHandle (scalar boolean, break if > 0)

**Example:**

```python
@algorithm
def early_stopping(sG, threshold=1e-6):
    values = sG.nodes(1.0)
    
    with sG.builder.iter.loop(1000):
        old_values = values
        values = sG.builder.var("values", update_values(values))
        
        # Compute max difference
        diff = sG.builder.core.abs(values - old_values)
        max_diff = diff.reduce("max")
        
        # Break if converged
        has_converged = max_diff < threshold
        sG.builder.iter.break_if(has_converged)
    
    return values
```

### `continue_if(condition)`

Skip rest of iteration if condition is false.

```python
with sG.builder.iter.loop(100):
    should_update = check_condition()
    sG.builder.iter.continue_if(should_update)
    
    # This only runs if should_update is true
    values = sG.builder.var("values", expensive_update)
```

**Example - Conditional updates:**

```python
@algorithm
def conditional_propagation(sG, max_iter=100):
    values = sG.nodes(1.0)
    
    with sG.builder.iter.loop(max_iter):
        # Check if any values are active
        has_active = (values > 0.01).reduce("sum")
        
        # Skip iteration if nothing is active
        sG.builder.iter.continue_if(has_active)
        
        # Propagate (only if active nodes exist)
        values = sG.builder.var("values", sG @ values * 0.9)
    
    return values
```

## Nested Loops

Loops can be nested:

```python
with sG.builder.iter.loop(outer_iters):
    # Outer loop body
    
    with sG.builder.iter.loop(inner_iters):
        # Inner loop body
        values = sG.builder.var("values", updated)
```

**Example - Two-phase algorithm:**

```python
@algorithm
def two_phase(sG, outer=10, inner=50):
    values = sG.nodes(1.0)
    
    with sG.builder.iter.loop(outer):
        # Phase 1: Fast updates
        with sG.builder.iter.loop(inner):
            values = sG.builder.var("values", values * 0.99)
        
        # Phase 2: Refinement (runs once per outer iteration)
        values = sG.builder.var("values", values.normalize())
    
    return values
```

**Note:** Deep nesting can impact performance. Keep nesting depth reasonable.

## Iteration Metadata

### `iteration_count()`

Get current iteration number (within loop).

```python
with sG.builder.iter.loop(100):
    iter_num = sG.builder.iter.iteration_count()
    # Use iter_num in computation
```

**Returns:** VarHandle (scalar, current iteration 0-based)

**Example - Decay schedule:**

```python
@algorithm
def scheduled_decay(sG, max_iter=100):
    values = sG.nodes(100.0)
    
    with sG.builder.iter.loop(max_iter):
        iter_num = sG.builder.iter.iteration_count()
        
        # Decay rate decreases over time
        decay_rate = 0.99 - (iter_num / max_iter) * 0.01
        
        values = sG.builder.var("values", values * decay_rate)
    
    return values
```

### `total_iterations()`

Get total number of iterations (after loop completes).

```python
with sG.builder.iter.until_converged(tolerance=1e-6) as loop_info:
    values = sG.builder.var("values", updated)

total = loop_info.total_iterations()
```

**Note:** Only available after loop completes, not within loop.

## Common Patterns

### Fixed Iterations

Simple fixed-count loop:

```python
with sG.builder.iter.loop(100):
    values = sG.builder.var("values", update_function(values))
```

### Convergence Detection

Loop until change is small:

```python
with sG.builder.iter.until_converged(tolerance=1e-6, max_iter=1000):
    old = values
    values = sG.builder.var("values", update_function(values))
```

### Early Stopping

Manual convergence check:

```python
with sG.builder.iter.loop(1000):
    old = values
    values = sG.builder.var("values", update_function(values))
    
    diff = sG.builder.core.abs(values - old).reduce("max")
    converged = diff < tolerance
    sG.builder.iter.break_if(converged)
```

### Asynchronous Updates

Non-deterministic but faster:

```python
sG.builder.iter.set_strategy("async")
with sG.builder.iter.loop(100):
    values = sG.builder.var("values", update_function(values))
```

### Multi-Variable Updates

Update multiple variables:

```python
with sG.builder.iter.loop(100):
    new_x = update_x(x, y)
    new_y = update_y(x, y)
    
    x = sG.builder.var("x", new_x)
    y = sG.builder.var("y", new_y)
```

### Phased Updates

Different operations at different phases:

```python
with sG.builder.iter.loop(100):
    # Always do fast update
    values = sG.builder.var("values", fast_update(values))
    
    # Periodically do slow refinement
    iter_num = sG.builder.iter.iteration_count()
    is_refinement_iter = (iter_num % 10) == 0
    
    with sG.builder.iter.continue_if(is_refinement_iter):
        values = sG.builder.var("values", slow_refinement(values))
```

## Performance Considerations

### Convergence Checking Overhead

```python
# ❌ Expensive - checks every iteration
with sG.builder.iter.until_converged(tolerance=1e-6, check_every=1):
    values = sG.builder.var("values", update)

# ✅ Better - checks every 10 iterations
with sG.builder.iter.until_converged(tolerance=1e-6, check_every=10):
    values = sG.builder.var("values", update)
```

### Synchronous vs Asynchronous

```python
# Sync: Deterministic, moderate speed
sG.builder.iter.set_strategy("sync")

# Async: Non-deterministic, often 2-5x faster convergence
sG.builder.iter.set_strategy("async")
```

### Loop Body Complexity

Keep loop bodies focused:

```python
# ✅ Good - focused loop body
with sG.builder.iter.loop(100):
    values = sG.builder.var("values", core_update(values))

# ❌ Avoid - complex loop body
with sG.builder.iter.loop(100):
    # Many operations make loop harder to optimize
    a = complex_op_1()
    b = complex_op_2()
    c = complex_op_3()
    values = sG.builder.var("values", combine(a, b, c))
```

## Limitations

### No Dynamic Loop Counts

Loop count must be known at algorithm definition time:

```python
# ❌ Doesn't work - can't be dynamic
param = load_from_somewhere()
with sG.builder.iter.loop(param):
    pass

# ✅ Works - pass as parameter
@algorithm
def my_algo(sG, max_iter):
    with sG.builder.iter.loop(max_iter):
        pass
```

### No Arbitrary Break Conditions

Break conditions must be computable within the IR:

```python
# ❌ Can't use Python conditionals
if some_python_condition:
    break

# ✅ Use IR-based conditions
sG.builder.iter.break_if(ir_condition_var)
```

### Variable Scope

Variables updated in loops must use `sG.builder.var()`:

```python
# ❌ Doesn't create loop-carried dependency
with sG.builder.iter.loop(10):
    values = values * 2.0  # Creates new variable each time

# ✅ Correct
with sG.builder.iter.loop(10):
    values = sG.builder.var("values", values * 2.0)
```

## Examples

### PageRank

```python
@algorithm
def pagerank(sG, damping=0.85, max_iter=100, tol=1e-6):
    ranks = sG.nodes(1.0 / sG.N)
    deg = ranks.degrees()
    
    with sG.builder.iter.until_converged(tolerance=tol, max_iter=max_iter):
        contrib = ranks / (deg + 1e-9)
        neighbor_sum = sG @ contrib
        
        # Handle sinks
        is_sink = (deg == 0.0)
        sink_mass = is_sink.where(ranks, 0.0).reduce("sum")
        
        ranks = sG.builder.var("ranks",
            damping * neighbor_sum + 
            (1 - damping) / sG.N + 
            damping * sink_mass / sG.N
        )
    
    return ranks.normalize()
```

### Label Propagation (Async)

```python
@algorithm
def lpa_async(sG, max_iter=10):
    labels = sG.nodes(unique=True)
    
    sG.builder.iter.set_strategy("async")
    
    with sG.builder.iter.loop(max_iter):
        labels = sG.builder.graph_ops.neighbor_mode_update(
            labels, include_self=True, ordered=True
        )
    
    return labels
```

### Belief Propagation

```python
@algorithm
def belief_propagation(sG, max_iter=50, damping=0.5):
    beliefs = sG.nodes(1.0)
    messages = sG.nodes(0.0)
    
    with sG.builder.iter.loop(max_iter):
        # Update messages
        new_messages = sG @ beliefs
        messages = sG.builder.var("messages",
            damping * new_messages + (1 - damping) * messages
        )
        
        # Update beliefs
        beliefs = sG.builder.var("beliefs",
            beliefs * messages.normalize()
        )
    
    return beliefs.normalize()
```

### Iterative Refinement

```python
@algorithm
def iterative_refinement(sG, max_iter=100):
    coarse = sG.nodes(1.0)
    fine = sG.nodes(0.0)
    
    with sG.builder.iter.loop(max_iter):
        # Coarse update (every iteration)
        coarse = sG.builder.var("coarse", sG @ coarse * 0.9)
        
        # Fine update (every 10 iterations)
        iter_num = sG.builder.iter.iteration_count()
        is_fine_iter = (iter_num % 10) == 0
        
        with sG.builder.iter.continue_if(is_fine_iter):
            fine = sG.builder.var("fine", expensive_refinement(coarse))
    
    return coarse + fine
```

## See Also

- [CoreOps API](core.md) - Operations within loop bodies
- [GraphOps API](graph.md) - Neighbor operations in iterations
- [PageRank Tutorial](../tutorials/02_pagerank.md) - Iterative algorithm example
- [LPA Tutorial](../tutorials/03_lpa.md) - Asynchronous iteration example
