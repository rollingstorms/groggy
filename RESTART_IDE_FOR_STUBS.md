# How to Fix "Unknown Type" in IDE After Stub Changes

## The Problem
After regenerating `.pyi` stub files, your IDE may still show "unknown type" because it cached the old stubs.

## The Solution: Restart Python Language Server

### VSCode
1. **Open Command Palette:** `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. **Type:** "Python: Restart Language Server"
3. **Hit Enter**
4. Wait 5-10 seconds for reload

### PyCharm
1. **Go to:** File → Invalidate Caches / Restart
2. **Select:** "Invalidate and Restart"
3. Wait for PyCharm to restart

### Jupyter / IPython
**Restart the kernel:**
```python
# In notebook, click "Restart Kernel" button
# Or run this code to reload:
import importlib
import groggy
importlib.reload(groggy)
```

### Alternative: Restart IDE Completely
If language server restart doesn't work:
1. Close your IDE completely
2. Reopen the project
3. Wait for indexing to complete

## Verify It Works

Create a test file and check type hints:

```python
# test_types.py
import groggy as gr

g = gr.Graph()
g.add_nodes(5)

# Hover over these variables in your IDE:
subgraph = g.view()          # Should show: Subgraph
nodes = g.nodes              # Should show: NodesAccessor  
all_nodes = g.nodes.all()    # Should show: Subgraph

# Tab completion should now work:
g.view().<TAB>               # Shows Subgraph methods
g.nodes.<TAB>                # Shows NodesAccessor methods
```

## What Should Work Now

✅ **Method chaining with types:**
- `g.view()` shows as `Subgraph` type
- `g.nodes` shows as `NodesAccessor` type
- `g.nodes.all()` shows as `Subgraph` type
- Chaining continues to work: `g.view().nodes.all()`

✅ **Tab autocomplete:**
- After `g.view().` you see Subgraph methods
- After `g.nodes.` you see NodesAccessor methods

✅ **Docstrings:**
- Shift+Tab in Jupyter shows method documentation
- Hover in VSCode/PyCharm shows parameter info

## Still Not Working?

1. **Check stub file exists:**
   ```python
   import groggy as gr
   from pathlib import Path
   stub = Path(gr.__file__).parent / "_groggy.pyi"
   print(f"Stub exists: {stub.exists()}")
   print(f"Location: {stub}")
   ```

2. **Check stub has forward references:**
   ```bash
   head -10 python-groggy/python/groggy/_groggy.pyi
   # Should see: from __future__ import annotations
   ```

3. **Regenerate stubs:**
   ```bash
   python scripts/generate_stubs.py
   ```

4. **Check Python version:**
   - Stubs require Python 3.7+
   - `from __future__ import annotations` requires Python 3.7+

## Common Issues

**Issue:** "Still shows `Any` type"
- **Fix:** Restart language server (see above)

**Issue:** "Can't find module groggy"
- **Fix:** Make sure you ran `maturin develop --release`

**Issue:** "Types work in VSCode but not PyCharm"
- **Fix:** Each IDE has its own cache - restart both separately

**Issue:** "Works in IDE but not in Jupyter"
- **Fix:** Restart Jupyter kernel (see above)

## Technical Details

The stub file (`_groggy.pyi`) contains:
- 9,227 lines of type signatures
- 56 classes with full method signatures
- 1,038 methods with docstrings
- 123 properties with `@property` decorator
- 15 inferred return types for chaining

The key fix was adding:
```python
from __future__ import annotations  # Enables forward references
```

This allows `Graph` (defined early in file) to reference `Subgraph` (defined later).

---

**Last Updated:** 2024-10-04  
**After any stub regeneration:** Restart your IDE's language server!
