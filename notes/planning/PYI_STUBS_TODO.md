# Python Type Stubs (.pyi) - Implementation TODO

## ✅ Phase 1: Basic Stub Generation (COMPLETED - Enhanced!)

- [x] Install stub generation tools
- [x] Create custom stub generator script (`scripts/generate_stubs.py`)
- [x] Generate comprehensive stubs for `_groggy` module
- [x] Test that stubs are complete (1038 methods across 56 classes!)
- [x] Create convenience script (`scripts/generate_stubs.sh`)
- [x] Commit stubs to repository
- [x] **NEW:** Add return type inference for method chaining
- [x] **NEW:** Properly handle PyO3 properties with `@property` decorator
- [x] **NEW:** Infer types for common chaining patterns (view, nodes, edges)

**Status:** Phase 1 complete WITH enhancements! Method chaining now works with autocomplete.

**Files Created:**
- `python-groggy/python/groggy/_groggy.pyi` (9227 lines, 56 classes, 1038 methods, 123 properties)
- `scripts/generate_stubs.py` (custom generator with type inference)
- `scripts/generate_stubs.sh` (convenience wrapper)

**Enhancements:**
- Return type inference: 11 key methods have proper return types
- Property detection: 123 properties marked with `@property`
- Method chaining works: `g.view() -> Subgraph`, `g.nodes -> NodesAccessor`

**How to Use:**
```python
# In Jupyter or IPython
import groggy as gr
g = gr.Graph()
g.<TAB>  # Shows all 64 methods with autocomplete!
g.add_node<SHIFT+TAB>  # Shows docstring!

# Method chaining with types!
g.view().<TAB>  # Shows Subgraph methods
g.nodes.<TAB>  # Shows NodesAccessor methods
g.nodes.all().<TAB>  # Shows Subgraph methods (because nodes.all() -> Subgraph)
```

## 🔄 Phase 2: Documentation Enhancement (TODO)

- [ ] Extract Rust doc comments from FFI source files
- [ ] Parse `/// doc comments` from `python-groggy/src/ffi/**/*.rs`
- [ ] Map Rust docs to Python method names
- [ ] Enhance stub generator to inject richer documentation
- [ ] Add type aliases (NodeId, EdgeId, etc.) for better type hints
- [ ] Improve parameter type hints beyond `*args, **kwargs`
- [ ] Add Examples sections to key methods
- [ ] **Handle `__getattr__` delegation patterns** (NEW)

**Estimated Time:** 8-10 hours (increased due to delegation handling)

**Priority:** Medium (current stubs work, but could be better)

### ⚠️ Known Issue: `__getattr__` Delegation

Some FFI classes use `__getattr__` for dynamic attribute access (not method delegation):

**Classes with `__getattr__`:**
- `NodesAccessor` / `EdgesAccessor` - dynamic attribute column access (`g.nodes.name`, `g.edges.weight`)
- `BaseTable` - dynamic column access
- `BaseArray` - potential dynamic access
- `Neighborhood` - delegation patterns

**Current Status:** 
- These classes show their direct methods correctly in stubs
- Dynamic attributes accessed via `__getattr__` won't autocomplete (expected)
- This is actually correct behavior - dynamic attributes can't be known at stub generation time

**Future Enhancement:**
- Could add `def __getattr__(self, name: str) -> Any: ...` to document the pattern
- Could add comments explaining dynamic access in docstrings

### Tasks:
1. Create Rust doc comment parser
   - Use regex or `syn` crate to parse Rust AST
   - Extract `/// ...` comments above `#[pymethods]`
   - Generate mapping: `{class_name: {method_name: doc_string}}`

2. Enhance stub generator
   - Load doc mapping
   - Merge with introspected signatures  
   - Generate richer stubs with full documentation

3. Add type aliases
   ```python
   # _groggy.pyi
   NodeId = int
   EdgeId = int
   AttrName = str
   StateId = str
   BranchName = str
   ```

4. Improve type hints (optional, harder)
   - Parse PyO3 `#[pyo3(signature = ...)]` attributes
   - Extract actual parameter names and types
   - Generate proper signatures instead of `*args, **kwargs`

## 🤖 Phase 3: Automation & CI (TODO)

- [ ] Add stub generation to `build.rs` (optional)
- [ ] Create CI check to validate stubs are up-to-date
- [ ] Add pre-commit hook (optional)
- [ ] Update CONTRIBUTING.md with stub regeneration instructions
- [ ] Add stub validation to test suite

**Estimated Time:** 3-4 hours

**Priority:** Low (can be done later)

### Tasks:
1. CI Validation Script
   ```bash
   # .github/workflows/check-stubs.yml
   - name: Check type stubs
     run: |
       maturin develop --release
       python scripts/generate_stubs.py
       git diff --exit-code python-groggy/python/groggy/_groggy.pyi
   ```

2. Update Documentation
   - Add to CONTRIBUTING.md
   - Explain when to regenerate stubs
   - Document the workflow

3. Optional: Build Integration
   ```rust
   // build.rs
   fn main() {
       println!("cargo:rerun-if-changed=python-groggy/src/ffi");
       // Optionally: trigger stub generation
   }
   ```

## 📋 Current Status Summary

**What Works Now:**
- ✅ Autocomplete in Jupyter/IPython
- ✅ Autocomplete in VSCode/PyCharm  
- ✅ Shift+Tab shows docstrings in Jupyter
- ✅ All 1038 methods are discoverable
- ✅ 56 classes fully stubbed
- ✅ 123 properties properly typed
- ✅ **Method chaining with type hints!**
- ✅ **Properties return correct types (NodesAccessor, EdgesAccessor, etc.)**

**What Could Be Better:**
- Most type hints still use `*args, **kwargs` for parameters
- Could have richer docs from Rust comments
- Not yet integrated into build/CI
- Some parameter names could be more specific (Phase 2 goal)

**Overall:** Phase 1 exceeds main goal! Discoverability AND method chaining both work.

## 🎯 Quick Reference

### Regenerate Stubs After Changes

```bash
# Option 1: Full rebuild + regenerate
./scripts/generate_stubs.sh

# Option 2: Skip build if already built
./scripts/generate_stubs.sh --skip-build

# Option 3: Manual
maturin develop --release
python scripts/generate_stubs.py
```

### Check Stubs Work

```python
# Test in Python REPL or Jupyter
import groggy
help(groggy.Graph)  # Should show class docs
help(groggy.Graph.add_node)  # Should show method docs

# Test autocomplete (in Jupyter/IPython)
g = groggy.Graph()
g.  # Hit TAB - should show methods
```

### Integration with IDEs

- **VSCode:** Install Python extension, stubs auto-detected
- **PyCharm:** Stubs auto-detected, no config needed
- **Jupyter:** Works automatically with Tab completion
- **mypy:** `mypy --strict your_script.py` will use stubs

## 🔗 Related Files

- `PYI_STUBS_PLANNING.md` - Original planning document
- `scripts/generate_stubs.py` - Stub generator script
- `scripts/generate_stubs.sh` - Convenience wrapper
- `python-groggy/python/groggy/_groggy.pyi` - Generated stubs
- `python-groggy/src/ffi/` - Rust FFI source (for Phase 2 doc extraction)

## 📝 Notes for Future

1. Consider adding `#[pyo3(text_signature = "...")]` to Rust code
   - Makes signatures show up in help() automatically
   - Reduces need for post-processing
   - Example: `#[pyo3(signature = (count))]`

2. Could generate separate `.pyi` per class
   - Would be easier to navigate
   - But current single file works fine

3. Watch for PyO3 updates
   - PyO3 might add better stub generation support
   - Check: https://github.com/PyO3/pyo3/issues

---

**Last Updated:** 2024-10-04  
**Phase 1 Completed:** 2024-10-04  
**Next Priority:** Add to datasets module plan, then Phase 2 (doc enhancement)
