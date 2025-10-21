# Python Type Stubs (.pyi) for Groggy - Planning Document

## Problem Statement

**Current Issue:** No autocomplete, no type hints, no inline documentation in Jupyter notebooks when using Groggy's Rust-backed Python API. This severely impacts developer experience and discoverability.

When users type `graph.` and hit Tab or Shift+Tab in Jupyter:
- ❌ No method suggestions appear
- ❌ No parameter hints
- ❌ No docstrings visible
- ❌ IDEs like VSCode/PyCharm can't provide IntelliSense

**Root Cause:** Groggy is a Rust/Python hybrid library using PyO3. The Python module `_groggy` is compiled Rust code (`.so`/`.pyd`), so Python tooling can't introspect it to discover methods, signatures, or docs.

**Scale:** ~900+ methods across ~65+ classes exported from Rust via PyO3

## Solution: Python Stub Files (.pyi)

Type stub files (`.pyi`) are Python's standard way to provide type information and documentation for compiled extensions. They act as a "shadow" interface that IDEs and tooling can read.

### What are .pyi files?
- Similar to C header files (`.h`)
- Contain type signatures but no implementation
- Python's type checkers (mypy, pyright) and IDEs use them
- Work with compiled extensions (Rust, C, C++)

### Example
```python
# _groggy.pyi
class Graph:
    """High-performance graph with Git-like version control."""
    
    def add_node(self) -> NodeId:
        """
        Add a new node to the graph.
        
        Returns:
            NodeId: ID of the newly created node
        """
        ...
    
    def add_nodes(self, count: int) -> List[NodeId]:
        """Add multiple nodes efficiently."""
        ...
```

## Key Questions

### 1. Manual vs Generated Stubs?

**Option A: Manual .pyi files** ❌
- Pros: Full control, can add rich docs
- Cons: **UNMAINTAINABLE** - 900+ methods, will drift out of sync

**Option B: Auto-generate from Rust** ✅ (Recommended)
- Pros: Always in sync, one source of truth, scalable
- Cons: Requires tooling/script

**Decision:** Must be generated. Manual maintenance for 900+ methods is not feasible.

### 2. What Tool to Use?

**Option A: pyo3-stubgen** (Existing Tool)
- Tool: https://github.com/PyO3/pyo3-stubgen
- Pros: 
  - Built specifically for PyO3
  - Reads compiled module at runtime
  - Already understands PyO3 patterns
- Cons:
  - May not capture Rust doc comments
  - Requires post-processing for quality
- Status: Actively maintained

**Option B: Custom Generator from Rust Source**
- Parse Rust FFI files directly
- Extract `#[pymethods]`, `#[pyclass]`, doc comments (`///`)
- Generate .pyi with full documentation
- Pros:
  - Can include Rust doc comments
  - Full control over output
  - Can validate during build
- Cons:
  - Need to build parser
  - More initial work
  - Maintenance burden

**Option C: Hybrid Approach** ✅ (Recommended)
- Use pyo3-stubgen as base generator
- Post-process to enhance with Rust doc comments
- Script can run as part of build/CI

**Decision Needed:** Start with pyo3-stubgen or build custom?

### 3. Where to Put .pyi Files?

**Python Package Structure:**
```
python-groggy/python/groggy/
├── __init__.py
├── _groggy.pyi          # ← Stub for compiled Rust module
├── graph.py
├── graph.pyi            # ← Optional: if graph.py needs stubs
├── generators.py
├── imports.py
└── ...
```

**Key Files Needed:**
- `_groggy.pyi` - Main stub for all Rust-exposed classes (Graph, SubgraphArray, etc.)
- `__init__.pyi` - Optional, for re-exports
- Individual `.pyi` for Python wrapper files (only if needed)

### 4. What Goes in the Stubs?

**Minimum (pyo3-stubgen output):**
```python
class Graph:
    def add_node(self) -> int: ...
    def add_nodes(self, count: int) -> List[int]: ...
```

**Enhanced (with doc comments from Rust):**
```python
class Graph:
    """
    High-performance graph library with memory optimization.
    
    Provides Git-like version control and advanced query capabilities.
    """
    
    def add_node(self) -> NodeId:
        """
        Add a new node to the graph.
        
        Returns:
            NodeId: ID of the newly created node
            
        Example:
            >>> g = Graph()
            >>> node_id = g.add_node()
        """
        ...
    
    def add_nodes(self, count: int) -> List[NodeId]:
        """
        Add multiple nodes efficiently in bulk.
        
        Args:
            count: Number of nodes to create
            
        Returns:
            List[NodeId]: List of newly created node IDs
        """
        ...
```

**Question:** Minimal first or enhanced from the start?

### 5. Integration with Build Process?

**Option A: Manual Generation**
- Developer runs script when methods change
- Commit `.pyi` files to git
- Pros: Simple, no build complexity
- Cons: Easy to forget, can drift

**Option B: Generate During Build**
- `maturin develop` auto-generates stubs
- Add to `build.rs` or post-build script
- Pros: Always up to date
- Cons: Slower builds

**Option C: CI Validation**
- CI checks if stubs are out of date
- Fails if `.pyi` doesn't match source
- Developer regenerates when needed
- Pros: Catches drift, no build overhead
- Cons: Need to implement check

**Recommended:** Option C (manual with CI validation)

## Proposed Architecture

### Phase 1: Basic Stub Generation (Quick Win)

1. **Install pyo3-stubgen:**
   ```bash
   pip install pyo3-stubgen
   ```

2. **Generate initial stubs:**
   ```bash
   # After building with maturin develop
   pyo3-stubgen groggy._groggy -o python-groggy/python/groggy/
   ```

3. **Test autocomplete in Jupyter:**
   ```python
   import groggy as gr
   g = gr.Graph()
   g.  # <-- Should show methods in autocomplete
   ```

4. **Commit stubs to repo**

**Estimated Time:** 1 hour

### Phase 2: Documentation Enhancement

1. **Extract Rust doc comments:**
   - Parse `python-groggy/src/ffi/**/*.rs`
   - Find `/// doc comments` above `#[pymethods]`
   - Map to method names

2. **Post-process .pyi files:**
   - Script that reads generated stubs
   - Injects Rust doc comments
   - Cleans up type hints (NodeId vs int, etc.)

3. **Custom type aliases:**
   ```python
   # _groggy.pyi
   NodeId = int
   EdgeId = int
   AttrName = str
   ```

**Estimated Time:** 4-6 hours

### Phase 3: Automation & CI

1. **Create generation script:**
   ```bash
   scripts/generate_stubs.sh
   ```
   - Builds extension
   - Runs pyo3-stubgen
   - Enhances with docs
   - Validates output

2. **Add CI check:**
   - GitHub Actions job
   - Runs stub generator
   - Checks for git diff
   - Fails if out of sync

3. **Documentation:**
   - Add to CONTRIBUTING.md
   - Explain when to regenerate

**Estimated Time:** 2-3 hours

## Implementation Script (Phase 1)

```bash
#!/bin/bash
# scripts/generate_stubs.sh

set -e

echo "🔨 Building Groggy extension..."
maturin develop --release

echo "📝 Generating type stubs..."
pyo3-stubgen groggy._groggy -o python-groggy/python/groggy/

echo "✅ Stubs generated at python-groggy/python/groggy/_groggy.pyi"
echo "📚 Test in Jupyter: import groggy; groggy.Graph() # then hit Tab"
```

## Documentation Extraction Strategy

### Rust Source Pattern:
```rust
#[pymethods]
impl PyGraph {
    /// Add a new node to the graph.
    /// 
    /// Returns the ID of the newly created node.
    #[pyo3(text_signature = "($self)")]
    fn add_node(&mut self) -> PyResult<usize> {
        // implementation
    }
}
```

### Parsing Approach:
1. Use regex or Rust parser (syn crate)
2. Extract doc comments before each method
3. Map to method names
4. Merge with pyo3-stubgen output

### Alternative: Add pyo3 text_signature
PyO3 supports `#[pyo3(text_signature = "...")]` which pyo3-stubgen can read:
```rust
#[pyo3(text_signature = "(count)")]
fn add_nodes(&mut self, count: usize) -> PyResult<Vec<usize>> {
```

This might be easier than post-processing!

## Success Metrics

- [ ] Autocomplete works in Jupyter (Tab completion)
- [ ] Docstrings appear (Shift+Tab in Jupyter)
- [ ] VSCode/PyCharm show IntelliSense
- [ ] mypy/pyright type checking works
- [ ] All 65+ classes have stubs
- [ ] Major methods have parameter hints
- [ ] CI validates stubs stay in sync

## Open Questions for Review

1. **Start with pyo3-stubgen or build custom parser?**
   - Recommendation: Start with pyo3-stubgen, see if it's good enough

2. **How much doc enhancement in Phase 1 vs Phase 2?**
   - Recommendation: Basic stubs first (1 hour), then enhance

3. **Should we add pyo3 text_signature annotations to Rust code?**
   - This might be easier than post-processing
   - Makes Rust source more verbose but improves Python UX

4. **Commit stubs to git or generate on install?**
   - Recommendation: Commit to git (better user experience)

5. **Type alias strategy?**
   - NodeId = int, EdgeId = int, etc.
   - Or use NewType pattern?

## Timeline Estimate

- **Phase 1 (Basic stubs):** 1-2 hours
  - Install pyo3-stubgen
  - Generate initial stubs
  - Test and commit
  
- **Phase 2 (Enhanced docs):** 4-6 hours
  - Doc extraction script
  - Post-processing
  - Manual cleanup
  
- **Phase 3 (Automation):** 2-3 hours
  - CI integration
  - Documentation
  
**Total:** 7-11 hours for full implementation

## Example Output Comparison

### Before (No Stubs):
```python
# Jupyter cell
import groggy as gr
g = gr.Graph()
g.<TAB>
# Shows: Nothing or generic __dict__, __class__, etc.
```

### After (With Stubs):
```python
# Jupyter cell
import groggy as gr
g = gr.Graph()
g.<TAB>
# Shows: add_node, add_nodes, add_edge, add_edges, node_count, ...
# Shift+Tab on g.add_node shows: "Add a new node to the graph. Returns: NodeId"
```

## Risks & Mitigation

**Risk 1:** Stubs drift from implementation
- Mitigation: CI validation checks

**Risk 2:** pyo3-stubgen doesn't capture all nuances
- Mitigation: Manual post-processing script

**Risk 3:** 900+ methods = huge .pyi file
- Mitigation: Split by class, one stub per major type

**Risk 4:** Type hints might be wrong (int vs NodeId)
- Mitigation: Manual review and aliases

## Recommendation

**Start with Phase 1 immediately:**
1. Install pyo3-stubgen
2. Generate basic stubs
3. Test in Jupyter
4. Commit and see improvement

This gives immediate value (autocomplete!) and we can enhance later.

**Then evaluate:** Is pyo3-stubgen good enough or do we need custom tooling?

---

## Next Steps

**Please review and decide:**
- [ ] Approve Phase 1 approach (pyo3-stubgen)
- [ ] Should we add text_signature to Rust code?
- [ ] Commit stubs to git? (Yes recommended)
- [ ] Priority: Basic stubs first or enhanced docs?
- [ ] Any concerns about 900+ methods in one file?

Once approved, I can implement Phase 1 in ~1 hour and get you autocomplete working!
