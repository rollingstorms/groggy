# Tomorrow's Quick Start Guide

**Date:** 2025-11-06  
**Focus:** Fix Loop Execution + Complete LPA + Validate Optimizations

---

## 🎯 Three Main Tasks

### Task 1: Fix Loop Execution (2-3 hours) ⚠️ CRITICAL
**Why:** Loops are 60-174x slower than native due to FFI crossings

**What to do:**
```python
# Current problem: This unrolls 100 times
with sG.builder.iter.loop(100):
    ranks = update(ranks)  # 100 FFI crossings!

# Need: Emit single IR node, execute in Rust
```

**Files to modify:**
1. `python-groggy/python/groggy/builder/traits/iter.py`
   - Update `LoopContext._finalize_loop()` 
   - Emit single `ControlIRNode` instead of unrolling

2. `src/builder/executor.rs`
   - Add loop execution support
   - Execute loop iterations natively

3. `python-groggy/src/ffi/builder.rs`
   - Add `Loop` variant to step types
   - Handle loop parameters (count, body)

**Test:**
```bash
python benchmark_builder_vs_native.py
# Should see PageRank drop from 0.514s to <0.020s (5k nodes)
```

---

### Task 2: Complete LPA (1-2 hours)
**Why:** LPA missing two operations, produces invalid results

**What to add:**

#### 2a. `collect_neighbor_values()` in GraphOps
```python
# Add to python-groggy/python/groggy/builder/traits/graph.py
def collect_neighbor_values(self, values, include_self=True):
    """Collect all neighbor values as a list per node."""
    var = self.builder._new_var("neighbor_values")
    self.builder.steps.append({
        "type": "graph.collect_neighbor_values",
        "source": values.name,
        "include_self": include_self,
        "output": var.name
    })
    return var
```

#### 2b. `mode()` in CoreOps
```python
# Add to python-groggy/python/groggy/builder/traits/core.py
def mode(self, values):
    """Return most common value."""
    var = self.builder._new_var("mode")
    self.builder.steps.append({
        "type": "core.mode",
        "source": values.name,
        "output": var.name
    })
    return var
```

#### 2c. Rust implementation
Add to `src/builder/executor.rs`:
- `execute_collect_neighbor_values()`
- `execute_mode()`

**Test:**
```python
# Update benchmark_builder_vs_native.py LPA
@algorithm("lpa")
def lpa(sG, max_iter=10):
    labels = sG.nodes(unique=True)
    b = sG.builder
    
    with sG.iterate(max_iter):
        neighbor_labels = b.graph.collect_neighbor_values(labels, include_self=True)
        labels = sG.var("labels", b.core.mode(neighbor_labels))
    
    return labels
```

---

### Task 3: Validate Optimizations (1-2 hours)
**Why:** Confirm all optimization passes work with native loops

**What to do:**

1. Run optimized PageRank:
```python
from groggy.builder.ir.optimizer import optimize_ir

algo = pagerank(damping=0.85, max_iter=100)
optimized = optimize_ir(algo.ir_graph, passes=['constant_fold', 'cse', 'fuse_arithmetic', 'fuse_neighbor', 'dce'])
# Apply optimized version
```

2. Measure impact:
```python
# Before optimization
unoptimized_time = benchmark(pagerank_unoptimized)

# After optimization  
optimized_time = benchmark(pagerank_optimized)

print(f"Speedup: {unoptimized_time / optimized_time:.2f}x")
```

3. Update performance tables in:
   - `IR_OPTIMIZATION_STATUS.md`
   - `BUILDER_PERFORMANCE_BASELINE.md`

---

## 📊 Success Criteria

### Loop Fix
- ✅ PageRank 5k nodes: <0.020s (currently 0.514s)
- ✅ PageRank 200k nodes: <0.300s (currently 19.5s)
- ✅ Results match native within 0.001 tolerance

### LPA
- ✅ Finds 3-13 communities (currently 13k+)
- ✅ Results match native implementation
- ✅ Performance within 2-3x of native

### Optimizations
- ✅ All passes apply without errors
- ✅ Fusion reduces operations by 10-20%
- ✅ Batching reduces FFI calls by 70%+
- ✅ Semantic preservation maintained

---

## 🚀 Quick Commands

### Run tests
```bash
pytest tests/test_ir_*.py -v      # All IR tests (should be 72/72)
pytest tests/test_simple_builder.py -v  # Builder integration
```

### Run benchmarks
```bash
python benchmark_builder_vs_native.py  # Main benchmark
```

### Build & install
```bash
maturin develop --release  # After Rust changes
```

### Check status
```bash
git status
git diff --stat
```

---

## 📁 Key Files

### Python (Loop Fix + LPA)
```
python-groggy/python/groggy/builder/
├── traits/
│   ├── iter.py         # ⚠️ Fix loop emission
│   ├── graph.py        # ➕ Add collect_neighbor_values
│   └── core.py         # ➕ Add mode
```

### Rust (Loop Execution + LPA Ops)
```
src/
├── builder/
│   └── executor.rs     # ⚠️ Add loop execution + new ops

python-groggy/src/ffi/
└── builder.rs          # ⚠️ Add Loop step type
```

### Tests & Benchmarks
```
tests/
└── test_simple_builder.py  # Test loop execution

benchmark_builder_vs_native.py  # Main validation
```

---

## 🐛 Debug Tips

### Loop not executing?
```python
# Check if loop node is being emitted
print(algo.ir_graph.pretty_print())
# Should see: ControlIRNode(type=loop, count=100)
# Not: 100 separate nodes

# Check execution
print(algo.steps)  # Should have Loop step
```

### LPA wrong results?
```python
# Check operations exist
print(dir(sG.builder.graph))  # Should have collect_neighbor_values
print(dir(sG.builder.core))   # Should have mode

# Check intermediate values
with sG.iterate(1):  # Just 1 iteration for debug
    neighbor_labels = b.graph.collect_neighbor_values(labels)
    print(f"Neighbor labels shape: {len(neighbor_labels)}")
```

### Optimizations not applying?
```python
# Check IR before optimization
print(f"Operations before: {len(algo.ir_graph.nodes)}")

# Run optimization
optimized = optimize_ir(algo.ir_graph)

# Check IR after
print(f"Operations after: {len(optimized.nodes)}")
print(f"Reduction: {(1 - len(optimized.nodes)/len(algo.ir_graph.nodes))*100:.1f}%")
```

---

## 📚 Reference Docs

If you get stuck, check:
- `IR_OPTIMIZATION_STATUS.md` - Current status & known issues
- `BUILDER_IR_OPTIMIZATION_PLAN.md` - Complete roadmap
- `OPTIMIZATION_PASSES.md` - How optimization works
- `LOOP_UNROLLING_FIX.md` - Loop bug details

---

## ⏱️ Time Budget (6-7 hours total)

| Task | Time | Priority |
|------|------|----------|
| Loop execution fix | 2-3h | ⚠️ Critical |
| LPA completion | 1-2h | Medium |
| Optimization validation | 1-2h | High |
| Documentation updates | 30m | Low |
| **Total** | **6-7h** | |

---

## 🎯 End Goal

By end of day:
1. ✅ Loops execute natively (60-174x faster)
2. ✅ LPA works correctly
3. ✅ All optimizations validated
4. ✅ Performance within 2-5x of native (before JIT)
5. ✅ Ready for Phase 4 (JIT compilation)

Let's make it happen! 🚀

---

**Pro tip:** Start with loop fix (biggest impact), then LPA (quick win), then optimization validation (confirms everything works together).
