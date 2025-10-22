# API Documentation Strategy - Hybrid Approach

## The Challenge

**Comprehensive API Analysis:**
- **53 unique objects** in Groggy ecosystem
- **921 total method calls** tested
- **501 methods** in 14 core API objects alone
- Complex delegation via `getattr` forwarding
- Meta API graph tracks all transformations

**Reality Check:** Hand-writing documentation for 501+ methods is impractical and unmaintainable.

## The Solution: Hybrid Documentation

### Core API Objects (14)

| Object | Methods | Success Rate |
|--------|---------|--------------|
| Graph | 50/65 | 76.9% |
| Subgraph | 32/60 | 53.3% |
| SubgraphArray | 10/14 | 71.4% |
| GraphTable | 13/22 | 59.1% |
| NodesTable | 21/33 | 63.6% |
| EdgesTable | 24/37 | 64.9% |
| BaseTable | 41/101 | 40.6% |
| NumArray | 14/17 | 82.4% |
| NodesArray | 9/13 | 69.2% |
| EdgesArray | 12/15 | 80.0% |
| GraphMatrix | 50/93 | 53.8% |
| NodesAccessor | 12/15 | 80.0% |
| EdgesAccessor | 14/16 | 87.5% |
| **TOTAL** | **501** | **-** |

### Hybrid Structure for Each API Page

#### Part 1: Hand-Crafted Theory (Top of Page)

**What we write manually:**

1. **Conceptual Overview** (~200 words)
   - What this object represents in Groggy
   - Role in the architecture
   - Key design decisions

2. **Architecture & Design** (~400 words)
   - How it fits in three-tier architecture
   - Rust Core implementation details
   - Why it exists (problem it solves)
   - Performance characteristics

3. **Object Transformations** (~300 words)
   - Visual diagram of transformation paths
   - What this can become
   - What can create this
   - Example delegation chains

4. **Common Patterns** (~400 words)
   - 3-5 real-world usage patterns
   - Combining multiple methods
   - Best practices
   - Anti-patterns to avoid

5. **Performance Notes** (~200 words)
   - Time/space complexity table
   - Optimization tips
   - Memory considerations

**Total hand-crafted per page: ~1,500 words**

#### Part 2: Auto-Generated Reference (Bottom of Page)

**Using mkdocstrings:**

```markdown
## Complete Method Reference

::: groggy.Graph
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
      members:
        - add_node
        - add_edge
        - connected_components
        # ... all 65 methods
```

**What mkdocstrings provides:**
- Method signatures from Python
- Parameter types and descriptions
- Return types
- Docstring content
- Automatically stays in sync with code

#### Part 3: Enhanced with Meta-Graph Data (Future)

**From comprehensive testing:**
- Return types discovered empirically
- Success rates for each method
- Known issues and workarounds
- Delegation chains mapped

## Implementation Strategy

### Phase 1: Template + Auto-gen (Current)

For each core object:

1. **Create hand-crafted sections** (theory, architecture, transformations, patterns)
2. **Add mkdocstrings block** for complete method reference
3. **Link to user guide** for tutorials

**Effort:** ~2-3 hours per object = ~40 hours total

### Phase 2: Enhance with Meta-Graph (Future)

1. **Extract method data** from comprehensive test graph
2. **Generate method tables** with return types and success rates
3. **Map delegation chains** automatically
4. **Create transformation diagrams** from meta-graph

**Effort:** ~1 week of tooling development

### Phase 3: Keep in Sync (Ongoing)

1. **mkdocstrings** auto-updates from docstrings
2. **Meta-graph** re-runs with each release
3. **Hand-crafted sections** updated for architectural changes

## Benefits of Hybrid Approach

### ✅ Completeness
- Every method documented via mkdocstrings
- No methods missed
- Always in sync with code

### ✅ Depth
- Hand-crafted theory sections provide context
- Architecture explanations
- Real-world patterns

### ✅ Maintainability
- Auto-gen reduces maintenance burden
- Docstrings are single source of truth
- Meta-graph tracks actual behavior

### ✅ Usability
- Theory for understanding "why"
- Auto-reference for looking up "how"
- Patterns for learning "when"

## Example: Graph API Page Structure

```markdown
# Graph API Reference

## Conceptual Overview
[Hand-crafted - 200 words explaining Graph's role]

## Architecture & Design
[Hand-crafted - 400 words on three-tier architecture]

### GraphSpace: Active State
[Explains topology tracking]

### GraphPool: Columnar Storage
[Explains attribute storage]

### HistoryForest: Version Control
[Explains Git-like features]

## Object Transformations
[Hand-crafted - 300 words + diagram]

Graph → Subgraph (via filtering)
Graph → GraphTable (via table())
Graph → BaseArray (via ["attr"])
Graph → GraphMatrix (via to_matrix())

## Common Patterns
[Hand-crafted - 400 words with 5 patterns]

### Pattern 1: Build → Query → Analyze
[Example code]

### Pattern 2: Delegation Chain
[Example code]

## Performance Notes
[Hand-crafted - 200 words + complexity table]

---

## Complete Method Reference

::: groggy.Graph
    [Auto-generated - all 65 methods]
```

## Files to Update

### Update Existing
- ✅ `docs/api/graph.md` - Add mkdocstrings block
- `docs/api/subgraph.md` - Create with hybrid structure
- `docs/api/accessors.md` - Create with hybrid structure
- ... (12 more core objects)

### New Templates
- `documentation/templates/api_page_template.md` - Reusable template
- `documentation/scripts/generate_api_skeleton.py` - Script to bootstrap new pages

### Configuration
- ✅ `mkdocs.yml` - Already configured with mkdocstrings
- `mkdocs.yml` - Add members lists for each object

## Next Steps

1. **Finish Graph API** - Add remaining hand-crafted sections
2. **Create template** - Standardize structure
3. **Generate skeletons** - Bootstrap all 14 core objects
4. **Fill in theory** - Write hand-crafted sections
5. **Validate** - Build docs and check rendering

## Success Metrics

- **Coverage**: All 501 methods documented
- **Depth**: Every object has theory/architecture sections
- **Usability**: Can find any method in <10 seconds
- **Maintainability**: Auto-gen reduces manual work by 70%

---

**Decision:** Proceed with hybrid approach, prioritizing complete coverage with auto-gen while maintaining depth through hand-crafted theory sections.
