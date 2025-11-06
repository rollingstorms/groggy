# Tomorrow's Quick Start Guide

**Task**: Fix variable tracking bug (30 min)  
**Goal**: Complete Phase 1, unlock benchmarking

---

## The 30-Minute Fix

### Edit File 1: Add rebuild method

**File**: `python-groggy/python/groggy/builder/ir/graph.py`

**Add this method to IRGraph class:**
```python
def rebuild_var_tracking(self):
    """Rebuild var_defs/var_uses after graph modifications."""
    self.var_defs.clear()
    self.var_uses.clear()
    
    for node in self.nodes:
        if node.output:
            self.var_defs[node.output] = node
        for inp in node.inputs:
            if isinstance(inp, str):
                self.var_uses[inp].append(node)
```

### Edit File 2: Call rebuild after fusion

**File**: `python-groggy/python/groggy/builder/ir/optimizer.py`

**Find `fuse_neighbor_operations()` and add the if block:**
```python
def fuse_neighbor_operations(self) -> bool:
    modified = False
    modified |= self._fuse_neighbor_mul_pattern()
    modified |= self._fuse_arithmetic_chains()
    
    # ADD THIS:
    if modified:
        self.ir.rebuild_var_tracking()
    
    return modified
```

### Test

```bash
cd /Users/michaelroth/Documents/Code/groggy
maturin develop --release
python test_fusion_perf.py
```

**Expected**: Runs without "variable not found" error

---

## Detailed Docs

- `NEXT_SESSION_START.md` - Full step-by-step
- `VARIABLE_REFERENCE_BUG.md` - Why this fix works
- `SESSION_FINAL_STATUS.md` - Complete status
- `LOOP_EXECUTION_FIX_PLAN.md` - Overall roadmap

---

## After the Fix

```bash
python test_fusion_perf.py
# Benchmark performance (expect 4-12x improvement)
# Document results
# Mark Phase 1 complete
```

**That's it!** 🚀
