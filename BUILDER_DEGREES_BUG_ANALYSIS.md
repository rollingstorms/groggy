# Builder Debugging Session - December 2024

## Problem Summary

The builder-based PageRank and LPA algorithms were producing incorrect results with significant differences from native implementations:
- **PageRank**: max difference of 0.287 (expected < 1e-5)
- **LPA**: Community counts off by ~32 communities  
- **Pipeline warnings**: "redefines variable" warnings for loop iterations

## Discovery: Attribute vs Property Naming Conflict

### The "degree" Attribute Issue

### The Bug

When the builder pipeline attached node degrees using `builder.attach_as("degree", degrees)`, the values were stored correctly in the graph, but accessing them via `node.degree` returned incorrect values.

**Why?**
- The `PyNode` class in `python-groggy/src/ffi/entities/node.rs` has a built-in `degree()` property getter (line 147)
- This property getter uses `self.inner.degree()` from the `NodeOperations` trait
- Python property getters take precedence over attribute access via `__getattr__`
- When code accesses `node.degree`, it calls the property getter instead of retrieving the "degree" attribute

### Evidence

Debug output showed:
```
[DEBUG] NodeDegreeStep: final map = {0: Int(1), 1: Int(1), 2: Int(0)}  ✅ Correct
[DEBUG] AttachNodeAttrStep: node=0→1, node=1→1, node=2→0              ✅ Correct

But accessing via node.degree:
  Node 0: degree=1  ✅ Correct  
  Node 1: degree=2  ❌ Wrong (should be 1)
  Node 2: degree=1  ❌ Wrong (should be 0)

Accessing via raw attribute:
  Node 0: raw_attr=1  ✅ Correct
  Node 1: raw_attr=1  ✅ Correct  
  Node 2: raw_attr=0  ✅ Correct
```

The stored attributes were correct, but the property getter returned different values.

### Verification

Changing the attribute name from "degree" to "out_degree" made everything work:
```python
builder.attach_as("out_degree", degrees)  # Instead of "degree"
node.out_degree  # Access the custom attribute - ✅ Works!
node.degree      # Still returns the property getter value
```

## Solution

### Short-term Fix
All builder algorithms should avoid using "degree" as an attribute name. Use specific names like:
- `out_degree` for directed out-degree
- `in_degree` for directed in-degree  
- `total_degree` for undirected degree

### Files to Update

1. **benchmark_builder_vs_native.py**
   - Line 21: Change `degrees = builder.node_degrees(ranks)` usage
   - Update all degree attribute references throughout PageRank implementation

2. **tests/test_builder_pagerank.py**
   - Update `_pagerank_step` helper function
   - Change degree attribute references in all test cases

3. **tests/test_builder_core.py**
   - Update `test_builder_node_degrees_*` tests to use `out_degree` attribute name

### Long-term Considerations

**Property vs Attribute Precedence**
The current behavior where property getters shadow attributes is correct Python behavior and follows PyO3 conventions. However, it creates a footgun for users.

**Options:**
1. **Documentation** (recommended): Document reserved attribute names that conflict with properties
2. **Deprecation**: Consider deprecating the `degree` property in favor of explicit `out_degree`/`in_degree`/`total_degree` properties
3. **Validation**: Add builder validation that warns when attaching attributes that conflict with property names

## Status Update

### Completed
1. ✅ Identified and documented the degree attribute vs property conflict
2. ✅ Removed debug statements from Rust code
3. ✅ Verified NodeDegreeStep and AttachNodeAttrStep work correctly  
4. ✅ Created debug scripts demonstrating the issue

### Still To Investigate

The degree naming conflict was **not** the root cause of PageRank/LPA mismatches. The benchmark and test scripts don't actually use "degree" as an attribute name - they compute degrees inline.

Re-running tests after the fix still shows:
- PageRank diff: 0.0039 (still too high)
- Test assertion: `assert abs(pr_builder - pr_native) < 1e-6` fails

**Remaining Issues:**

1. **PageRank Algorithm Mismatch**: The builder PageRank logic doesn't match the native implementation
   - Possible causes: loop iteration issues, variable aliasing problems, or missing algorithm steps
   - Need to compare step-by-step: degree weighting, sink handling, teleport, normalization

2. **Loop Variable Redefinition Warnings**: 
   ```
   Pipeline validation: Step 27 (alias): redefines variable 'ranks'
   ```
   - Happening in `_finalize_loop` (builder.py:883)
   - Suggests alias resolution is broken for loop iterations

3. **LPA Community Count**: Off by ~32 communities
   - May be related to the loop aliasing issue
   - Or incorrect neighbor mode update semantics

## Next Steps

1. Debug PageRank step-by-step to find where builder diverges from native
   - Add intermediate value logging
   - Compare after each iteration
   - Check sink mass calculation

2. Fix loop variable aliasing in builder.py `_finalize_loop`
   - Ensure each iteration reads from previous iteration's output
   - Stop collapsing all aliases to the initial variable

3. Verify LPA after loop fix

4. Document reserved attribute names in builder docs

## Related Issues

- **LPA Warning**: `Pipeline validation: Step 3 (core.update_in_place): redefines variable 'nodes_0'`
  - This is a separate issue with loop alias handling in `_finalize_loop`
  - Needs investigation in `python-groggy/python/groggy/builder.py` around line 883

- **PageRank Sink Handling**: After fixing the degree naming issue, verify that sink node handling (nodes with out-degree=0) is correct in the builder implementation
