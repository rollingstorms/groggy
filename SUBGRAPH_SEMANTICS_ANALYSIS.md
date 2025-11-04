# Subgraph Semantics Analysis

## Problem Statement

The builder DSL currently uses `G` (GraphHandle) as the main input to algorithms, but in practice, algorithms **always** operate on subgraphs, not full graphs. This creates semantic confusion.

## Current Usage Pattern

```python
# User code
graph = Graph()
# ... build graph ...

# Create subgraph view (could be filtered subset or full graph)
sg = graph.view()  # Returns Subgraph
sg_filtered = graph.view().filter(...)  # Filtered subgraph

# Apply algorithm to SUBGRAPH
result = sg.apply(pagerank())
```

## Inside Algorithm Definition

```python
@algorithm("pagerank")
def pagerank(G, damping=0.85, max_iter=100):
    # G looks like "Graph" but is actually operating on a subgraph
    ranks = G.nodes(1.0 / G.N)  # N = number of nodes IN THE SUBGRAPH
    ...
```

## The Confusion

1. **Variable naming**: `G` conventionally means "Graph" but it's a subgraph view
2. **Operations**: All ops (node_count, degrees, neighbor_agg) work on subgraph
3. **User expectations**: Users might think they're modifying the full graph
4. **Semantic correctness**: The abstraction leaks

## Analysis: Is This Actually a Problem?

### Arguments FOR Refactoring

1. **Semantic Clarity**: Should be `SG` or `subgraph` to match reality
2. **User Mental Model**: Clearer that operations are scoped to a view
3. **Future Subsetting**: If we add `.subgraph(mask)`, it's clearer as sub-subgraph
4. **Documentation**: Less explanation needed if naming matches behavior

### Arguments AGAINST Refactoring

1. **Convention**: In graph theory papers, `G` is standard notation
2. **Abstraction**: Users don't need to know it's a subgraph internally
3. **Consistency**: NetworkX, igraph all use `G` even for subgraphs
4. **Simplicity**: Shorter variable name, less cognitive load

## Recommendation

**Option A: Semantic Rename (RECOMMENDED)**

Rename `GraphHandle` → `SubgraphHandle` and use `SG` in examples:

```python
@algorithm("pagerank")
def pagerank(SG, damping=0.85, max_iter=100):
    """
    Args:
        SG: SubgraphHandle representing the graph view to process
    """
    ranks = SG.nodes(1.0 / SG.N)  # Clear: N nodes in THIS subgraph
    ...
```

**Pros:**
- Semantically correct
- Minimal code changes (~5 files)
- Clear to users
- Aligns with Groggy's subgraph-first philosophy

**Cons:**
- Breaks from conventional `G` notation
- More verbose

**Option B: Keep G but Document**

Keep `GraphHandle` and `G`, but document clearly:

```python
@algorithm("pagerank")
def pagerank(G, damping=0.85, max_iter=100):
    """
    Args:
        G: GraphHandle representing a subgraph view. 
           All operations are scoped to this subgraph.
    """
    ranks = G.nodes(1.0 / G.N)  # N = nodes in subgraph
    ...
```

**Pros:**
- No code changes
- Familiar `G` notation
- Less verbose

**Cons:**
- Still semantically misleading
- Requires constant documentation

**Option C: Hybrid Approach (COMPROMISE)**

Keep `GraphHandle` class name (internal), but use `subgraph` parameter:

```python
@algorithm("pagerank")
def pagerank(subgraph, damping=0.85, max_iter=100):
    """
    Args:
        subgraph: Graph view to process
    """
    ranks = subgraph.nodes(1.0 / subgraph.N)
    ...
```

**Pros:**
- Clear parameter name
- No class rename needed
- Self-documenting

**Cons:**
- More verbose than `G` or `SG`

## Implementation Impact

### Files to Change (Option A):

1. `builder/varhandle.py` - Rename `GraphHandle` → `SubgraphHandle`
2. `builder/algorithm_builder.py` - Update `graph()` → `subgraph()` method
3. `builder/decorators.py` - Update parameter passing
4. `builder/examples.py` - Change `G` → `SG` in all examples
5. Documentation - Update all references

**Estimated effort**: 2-3 hours

### Breaking Changes

- **Option A**: Slight - `builder.graph()` → `builder.subgraph()`
- **Option B**: None
- **Option C**: None (parameter name is user choice)

## Real-World Analogy

**JAX/PyTorch**: Use `x`, `y` (generic) not `data` (semantic)
**Pandas**: Uses `df` even though it's a DataFrame
**NetworkX**: Uses `G` even for subgraphs
**GraphBLAS**: Uses `A` (matrix) regardless of view

→ **Industry convention leans toward generic short names**

## Decision Matrix

| Criterion | Option A (SG) | Option B (G+doc) | Option C (subgraph) |
|-----------|--------------|------------------|---------------------|
| Semantic correctness | ✅ High | ⚠️ Low | ✅ High |
| Code changes | 🔧 Moderate | ✅ None | ✅ Minimal |
| User familiarity | ⚠️ Unfamiliar | ✅ Standard | ✅ Clear |
| Documentation load | ✅ Low | ⚠️ High | ✅ Low |
| Verbosity | ✅ Short | ✅ Short | ⚠️ Long |

## Proposed Solution

**Recommendation: Option B (Document) with Option A (Rename) as future improvement**

**Short term (now)**:
1. Keep `GraphHandle` and `G` notation
2. Add clear docstrings explaining it's a subgraph view
3. Add a note in CONTRIBUTING.md about semantics

**Long term (Phase 4+)**:
1. Consider renaming to `SubgraphHandle` in major version
2. Update all examples to use `SG` consistently
3. Add deprecation path for `builder.graph()` → `builder.subgraph()`

**Rationale**:
- Aligns with industry conventions (NetworkX, GraphBLAS use `G`)
- Avoids breaking changes mid-refactor
- Can be addressed in comprehensive v1.0 release
- Documentation is low-cost fix for now

## Code Example: Current (Good Enough)

```python
@algorithm("pagerank")
def pagerank(G, damping=0.85, max_iter=100):
    """
    PageRank centrality algorithm.
    
    Args:
        G: GraphHandle representing the graph view to process.
           Note: All operations are scoped to this subgraph view.
        damping: Damping factor (0.85 typical)
        max_iter: Maximum iterations
        
    Returns:
        VarHandle with PageRank scores
    
    Note:
        The algorithm operates on the provided subgraph view.
        Use graph.view() to create a full graph view, or
        graph.view().filter(...) for a subset.
    """
    ranks = G.nodes(1.0 / G.N)  # G.N = node count in this view
    deg = ranks.degrees()        # Degrees within this view
    
    with G.builder.iter.loop(max_iter):
        neighbor_sum = G @ (ranks / (deg + 1e-9))
        ranks = G.builder.var("ranks",
            damping * neighbor_sum + (1 - damping) / G.N
        )
    
    return ranks.normalize()
```

## Conclusion

**Decision: DEFER major refactor, improve documentation**

The semantic issue is **valid** but not critical. The confusion is primarily conceptual, not technical. Since:

1. Industry uses `G` even for subgraphs (NetworkX, igraph)
2. The builder correctly operates on whatever view is provided
3. We're mid-refactor and shouldn't introduce breaking changes
4. Documentation can clarify the semantics

**Action Items**:
- ✅ Document in docstrings that `G` is a subgraph view
- ✅ Add semantic note to CONTRIBUTING.md
- ✅ Consider rename in future major version (v1.0+)
- ⏸️ No code changes needed now

**This is a "semantic clarity" issue, not a "correctness" bug.**

---

## Update: sG Convention Adopted (2025-11-04)

### Decision

After analysis, we've adopted **`sG`** as the standard parameter name for subgraph handles in algorithm definitions.

### Rationale

1. **Semantic Clarity**: `sG` clearly indicates "subgraph" not "graph"
2. **Concise**: Only 2 characters, as short as `G`
3. **Distinctive**: Different enough to be memorable
4. **Mathematical**: Follows notation style (lowercase prefix)
5. **Compromise**: Gets benefits of clarity without verbosity

### Implementation

All example algorithms now use `sG`:

```python
@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """
    Args:
        sG: Subgraph view to process (GraphHandle)
    """
    ranks = sG.nodes(1.0 / sG.N)
    deg = ranks.degrees()
    
    with sG.builder.iter.loop(max_iter):
        neighbor_sum = sG @ (ranks / (deg + 1e-9))
        ranks = sG.builder.var("ranks",
            damping * neighbor_sum + (1 - damping) / sG.N
        )
    
    return ranks.normalize()
```

### Files Updated

- `builder/examples.py` - All 7 example algorithms
- `builder/decorators.py` - Documentation and examples
- `builder/__init__.py` - Module docstring
- `BUILDER_SYNTAX_COMPARISON.md` - All code examples

### Convention

**Use `sG` (not `G`) in all new algorithm definitions to emphasize subgraph semantics.**

This makes it clear that:
- Operations are scoped to the provided view
- `sG.N` is the node count in the subgraph, not the full graph
- Neighbor aggregation works within the subgraph topology
- The algorithm operates on whatever view is provided

### Migration

Existing code using `G` continues to work - `sG` is a parameter name convention, not a breaking API change. Users can choose their preferred parameter name, but we recommend `sG` for consistency.
