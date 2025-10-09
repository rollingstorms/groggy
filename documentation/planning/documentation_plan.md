# Groggy Documentation Plan

**Status:** Draft - In Development
**Created:** October 9, 2025
**Purpose:** Comprehensive roadmap for creating usage-driven documentation covering all Groggy modules

---

## Overview

This document outlines the plan for creating comprehensive, test-driven documentation for Groggy. The documentation will be built from the ground up using existing test suites as the source of truth for demonstrated functionality.

### Core Philosophy

"Do what has been asked; nothing more, nothing less."

The documentation will:
- **Be usage-driven**: Every documented feature must have a working test demonstrating it
- **Follow the object graph**: Documentation organized by the natural flow between objects (Graph → SubgraphArray → Table → Array → Matrix)
- **Start from foundations**: Begin with core concepts and build upward
- **Be iterative**: Each module builds on the previous, creating a cohesive learning path

---

## The Origin Story: How Groggy Started

Groggy began with a mission: create a **high-performance Rust-based graph backend** combined with a **quick, intuitive Python frontend** capable of producing:
- Lightweight graph diagrams
- Heavy-duty machine learning algorithms
- Real-time graph analytics

### The Ultralight Example

The journey started with the "ultralight example" - an attempt to distill Groggy's essence into the smallest possible implementation. This led to key architectural decisions:

#### Core Architectural Insight: Separation of Structure and Attributes

**The Big Idea:** Nodes/edges and their attributes are stored completely separately.

```
Graph Information  ←→  Attribute Information
  (Structure)            (Signal/Data)
```

This separation enables:
- **Graph structure** (topology): nodes, edges, connections
- **Graph signal** (data): attributes, features, metadata

#### Columnar Storage Philosophy

To support both graph operations and machine learning workflows, Groggy needed rectangular data. The solution: **columnar storage architecture**.

Key objects in the ultralight example:
1. **AttributeValues**: Any data type storage
2. **Delta objects**: Track changes over time (dynamic graph support)
3. **GraphSpace**: State of the graph (which nodes/edges are alive)
4. **GraphPool**: Flyweight pool containing all attributes
5. **HistoryForest**: Git-like version control for graphs

**Critical Design Pattern:** Attributes are never stored inside nodes/edges. Nodes and edges only **point to** attributes. Everything is abstract and columnar.

---

## Core Conceptual Model

### 1. The Three-Tier Architecture

```
┌──────────────────────────────────────┐
│        Python API Layer              │  User-facing, chainable, intuitive
│  (Graph, Table, Array, Matrix)       │
├──────────────────────────────────────┤
│          FFI Bridge                  │  Pure translation, no logic
│         (PyO3 bindings)              │
├──────────────────────────────────────┤
│         Rust Core                    │  High-performance algorithms,
│  (Storage, State, Algorithms)        │  columnar operations
└──────────────────────────────────────┘
```

### 2. The Object Transformation Graph

Groggy itself can be thought of as a graph where:
- **Nodes** = Object types (Graph, Subgraph, Table, Array, Matrix)
- **Edges** = Methods that transform one type into another

Example transformation chains:
```
Graph → connected_components() → SubgraphArray
SubgraphArray → sample(5) → SubgraphArray
SubgraphArray → neighborhood(depth=2) → SubgraphArray
SubgraphArray → table() → GraphTable
GraphTable → agg() → AggregationResult
```

### 3. Immutable Views

Key concept: When working with the graph, we typically get **immutable views** that can be further manipulated:
- **Subgraphs** are views into the main graph
- **Tables** are snapshots of graph state
- **Arrays** are columnar views of attributes
- **Matrices** represent graph structure or embeddings

We send commands to the core engine, which renders the appropriate view.

### 4. Delegation & Chaining

The signature feature of Groggy's API:

```python
result = (
    g.connected_components()          # Returns SubgraphArray
     .sample(5)                       # Still SubgraphArray
     .neighborhood(depth=2)           # Expand neighborhoods
     .table()                         # Convert to GraphTable
     .agg({"weight": "mean"})         # Aggregate columns
)
```

Each object knows which methods it can delegate to and what types they return.

---

## Documentation Structure

### Top‑Level Table of Contents

To make the docs complete end‑to‑end, we’ll bracket the module chapters with **Front Matter** (theory/intro/overview/install) and **Back Matter** (utilities and appendices). This augments the module‑driven chapters below.

#### Front Matter (Theory & Getting Started)
0. **About Groggy** — what/why, audience, scope
0.1 **Install** — pip/maturin, Rust toolchain from source, optional extras
0.2 **Quickstart** — 5‑minute graph→table→matrix path
0.3 **Concepts & Architecture** —
   - Executive Overview & Philosophy
   - Origins: Ultralight Example (GraphSpace, GraphPool, HistoryForest)
   - Three‑Tier Architecture (Rust Core ↔ FFI ↔ Python)
   - Core Ideas: structure vs info, versioning, columnar
   - Connected Views: Graph↔Table↔Ray↔Matrix
   - Visualization preview & roadmap

### Organization by Module

Documentation will be organized following the test module structure, creating chapters for each major component:

#### Phase 1: Core Foundation (Chapters 1-3)
1. **Graph Core** (`test_graph_core.py`)
   - Graph creation and initialization
   - Node CRUD operations
   - Edge CRUD operations
   - Attribute management
   - Basic graph properties

2. **Subgraphs** (`test_subgraph_base.py`)
   - Creating subgraphs (slicing, filtering)
   - Subgraph operations
   - Relationship to parent graph
   - Converting back to Graph

3. **Accessors** (`test_accessors.py`)
   - NodesAccessor (`g.nodes`)
   - EdgesAccessor (`g.edges`)
   - Attribute access patterns
   - Bulk operations

#### Phase 2: Tabular Data (Chapters 4-6)
4. **Base Tables** (`test_table_base.py`)
   - Table fundamentals
   - Table operations (head, tail, shape)
   - Column operations
   - Table transformations

5. **Nodes & Edges Tables** (`test_nodes_edges_tables.py`)
   - NodesTable operations
   - EdgesTable operations
   - Attribute columns
   - Filtering and selection

6. **Graph Tables** (`test_graph_table.py`)
   - GraphTable as unified view
   - Bundle operations (save/load)
   - Merge operations
   - Graph reconstruction

#### Phase 3: Arrays & Vectors (Chapters 7-10)
7. **Base Arrays** (`test_base_array.py`)
   - Array fundamentals
   - Array operations
   - Type system
   - Array transformations

8. **Numeric Arrays** (`test_num_array.py`)
   - NumArray for numerical data
   - Statistical operations
   - Aggregations
   - Numpy interop

9. **Graph Arrays** (`test_graph_arrays.py`)
   - SubgraphArray
   - ComponentsArray
   - Array delegation patterns
   - Chaining examples

10. **Specialized Arrays** (`test_array_base.py`)
    - NodesArray
    - EdgesArray
    - MetaNodeArray
    - Custom array types

#### Phase 4: Matrix & Linear Algebra (Chapters 11-12)
11. **Matrix Base** (`test_matrix_base.py`)
    - Matrix fundamentals
    - Matrix operations
    - Sparse vs dense
    - Linear algebra operations

12. **Graph Matrices** (`test_graph_matrix.py`)
    - Adjacency matrices
    - Laplacian matrices
    - Spectral embeddings
    - Matrix-graph conversions

#### Phase 5: Advanced Features (Chapters 13-16)
13. **Algorithms** (`test_algorithms.py`)
    - Connected components
    - Shortest paths
    - Centrality measures
    - Community detection

14. **Neural Networks** (`test_neural.py`)
    - Neural graph modules
    - Automatic differentiation
    - Gradient computation
    - Training workflows

15. **Integration** (`test_integration.py`)
    - NetworkX compatibility
    - Pandas integration
    - Numpy/SciPy interop
    - Import/export workflows

16. **Performance** (`test_performance.py`)
    - Benchmarking patterns
    - Optimization techniques
    - Memory management
    - Scaling considerations

---

## Documentation Format

### Chapter Template

Each chapter will follow this structure:

```markdown
# Chapter X: [Module Name]

## Overview
Brief introduction to the module and its role in Groggy.

## Core Concepts
Key ideas needed to understand this module.

## Basic Usage

### Creating [Object]
Simplest examples of creating the object.

### Essential Operations
The 5-10 most common operations with examples.

### Common Patterns
Frequently used combinations and workflows.

## Detailed API Reference

### Method Categories
Organized by functionality, not alphabetically.

### Method Details
For each method:
- Purpose and use case
- Parameters (with types and defaults)
- Return type and structure
- Example usage
- Related methods

## Integration Examples
How this module works with others.

## Common Pitfalls
What to watch out for.

## Performance Notes
When relevant: complexity, memory usage, optimization tips.

## Related Documentation
Links to related chapters and concepts.
```

### Example Format

Every example must:
1. Be runnable as-is (no placeholders)
2. Have a comment explaining what it demonstrates
3. Show both input and expected output where helpful
4. Be extracted from or validated by tests

```python
# Create a graph and run connected components
import groggy as gr

g = gr.generators.karate_club()

# Run algorithm (writes 'component' attribute to nodes)
g.connected_components(inplace=True, label='component')

# Get number of components
num_components = len(g.nodes['component'].unique())
print(f"Found {num_components} components")
# Output: Found 1 components
```

---

## Documentation Toolchain

### Chosen Stack: MkDocs + Material

**Primary Documentation Site:**
- **MkDocs** with **Material for MkDocs** theme
- **mkdocstrings** for Python API auto-documentation
- **mdBook** or **rustdoc** for Rust core documentation (linked section)
- Hosted on **GitHub Pages** or **Vercel**

**Why This Stack:**
- Fast, Markdown-first workflow
- Beautiful, responsive theme out of the box
- Easy versioning and search
- Excellent Python integration via mkdocstrings
- Simple deployment pipeline
- Works perfectly with our test-driven documentation approach

**Directory Structure:**
```
groggy/
├── docs/                          # MkDocs documentation root
│   ├── index.md                   # Landing page
│   ├── about.md                   # Chapter 0: About Groggy
│   ├── install.md                 # Chapter 0.1: Install
│   ├── quickstart.md              # Chapter 0.2: Quickstart
│   ├── concepts/                  # Chapter 0.3: Concepts & Architecture
│   │   ├── overview.md
│   │   ├── origins.md             # Ultralight example
│   │   ├── architecture.md        # Three-tier architecture
│   │   └── connected-views.md
│   ├── guide/                     # Main chapters (1-16)
│   │   ├── graph-core.md          # Chapter 1
│   │   ├── subgraphs.md           # Chapter 2
│   │   ├── accessors.md           # Chapter 3
│   │   ├── tables-base.md         # Chapter 4
│   │   └── ...
│   ├── api/                       # Hand-crafted API reference (theory-driven)
│   │   ├── graph.md               # Graph API with architecture context
│   │   ├── subgraph.md            # Subgraph as view concept
│   │   ├── table.md               # GraphTable with columnar explanation
│   │   ├── array.md               # Array types and transformations
│   │   ├── matrix.md              # Matrix operations and graph projections
│   │   └── ...
│   ├── rust/                      # Rust core documentation (linked)
│   ├── appendices/                # Appendices A-I
│   │   ├── cli.md
│   │   ├── file-formats.md
│   │   ├── errors.md
│   │   ├── glossary.md
│   │   └── ...
│   └── examples/                  # Runnable example gallery
├── mkdocs.yml                     # MkDocs configuration
├── README.md                      # Project README (maintained separately)
└── documentation/                 # Planning docs (current location)
    └── planning/
        └── documentation_plan.md  # This file
```

**API Documentation Philosophy:**

The API reference is NOT just auto-generated signatures. Each API page follows this structure:

1. **Conceptual Overview** - How this object fits into Groggy's architecture
2. **Theory & Design** - Why it exists, what problem it solves, architectural context
3. **Object Transformations** - What this can become (delegation chains)
4. **Core Methods** - Hand-written documentation with:
   - Purpose in context of the architecture
   - Usage examples from tests
   - Return types and transformation patterns
   - Performance characteristics
   - Related methods and cross-references
5. **Common Patterns** - Real-world usage combining multiple methods
6. **Auto-generated Reference** - Complete signature list (supplementary)

**Example: API page for Graph**
```markdown
# Graph API Reference

## Conceptual Overview
The Graph is the foundational object in Groggy, representing the **active state**
of your graph topology combined with the **columnar attribute pool**. Under the
hood, it maintains a GraphSpace (which nodes/edges are alive) and a GraphPool
(where attributes are stored).

## Architecture & Design
In Groggy's three-tier architecture, the Python Graph object is a thin wrapper
over the Rust core's high-performance graph implementation...

## Object Transformations
Graph objects can transform into:
- **Subgraph** via `g.nodes[...]`, `g.edges[...]`, `g.subgraph(...)`
- **GraphTable** via `g.table()`
- **BaseArray** via `g["attribute_name"]`
- **GraphMatrix** via `g.to_matrix()`, `g.laplacian_matrix()`, etc.

## Core Methods

### `add_node(**attrs) -> NodeId`
Adds a single node to the graph with optional attributes.

**Architectural Context:** This operation appends to the columnar attribute pool
and marks the node as alive in the GraphSpace. Attributes are stored separately
from the node structure itself.

**Usage:**
```python
import groggy as gr
g = gr.Graph()
alice = g.add_node(name="Alice", age=30)
```

**Returns:** Integer node ID for reference in edge creation and queries.

**Related:** `add_nodes()`, `remove_node()`, `g.nodes.set_attrs()`

### ...
```

**README Strategy:**
- Start with existing README.md as foundation
- Keep README focused on quick start and overview
- Link to full documentation site for comprehensive guides
- README serves as entry point, docs site is the deep dive

---

## API Documentation Workflow

### For Each API Page:

**Step 1: Generate Initial Scaffold** (Auto-generated)
- Use mkdocstrings to generate method signatures
- Extract docstrings and type hints
- Create basic structure

**Step 2: Add Conceptual Overview** (Hand-written)
- Explain what this object represents in Groggy's architecture
- Reference the three-tier architecture (Rust Core → FFI → Python)
- Explain how it fits into the ultralight example concepts (GraphSpace, GraphPool, etc.)

**Step 3: Document Theory & Design** (Hand-written)
- Why does this object exist? What problem does it solve?
- How does it relate to the columnar storage architecture?
- What are the performance characteristics?
- When should users choose this vs alternatives?

**Step 4: Map Object Transformations** (Hand-written)
- What can this object become? (delegation chains)
- What can create this object? (reverse transformations)
- Visual diagram showing transformation graph
- Link to related objects

**Step 5: Document Core Methods** (Hand-written, test-driven)
- For each method:
  - **Architectural context**: How does this work under the hood?
  - **Usage example**: Extract from tests, show real usage
  - **Return type & transformations**: What does this become?
  - **Performance notes**: Complexity, memory implications
  - **Related methods**: Cross-reference similar/complementary methods

**Step 6: Show Common Patterns** (Hand-written, test-driven)
- Real-world usage combining multiple methods
- Delegation chain examples
- Integration patterns (e.g., with pandas, numpy)
- Anti-patterns and what to avoid

**Step 7: Include Auto-generated Reference** (Supplementary)
- Complete method list with signatures
- Auto-generated from docstrings
- Serves as quick reference for users who already understand concepts

### Quality Checklist for Each API Page:

- [ ] Conceptual overview explains the "why" not just the "what"
- [ ] Architecture section references three-tier design
- [ ] Transformation graph shows all delegation paths
- [ ] Each method has architectural context, not just usage
- [ ] Examples are extracted from actual tests
- [ ] Performance characteristics documented
- [ ] Cross-references to related objects/methods
- [ ] Common patterns show real-world usage
- [ ] Anti-patterns highlighted
- [ ] Auto-generated reference included as supplement

---

## Implementation Plan

### Stage 0: Front Matter & Toolchain Setup (Week 1)
- [x] Create this planning document
- [ ] Install and configure MkDocs + Material theme
- [ ] Set up mkdocstrings for Python API documentation
- [ ] Configure documentation build pipeline
- [ ] Create base documentation structure (directories, templates)
- [ ] Set up GitHub Pages/Vercel deployment
- [ ] Establish example validation pipeline
- [ ] Chapter 0: About Groggy (what/why, audience, scope)
- [ ] Chapter 0.1: Install (pip, maturin, from source)
- [ ] Chapter 0.2: Quickstart (5-minute graph→table→array→matrix path)
- [ ] Chapter 0.3: Concepts & Architecture
  - [ ] Executive overview & philosophy
  - [ ] Origins: Ultralight example (GraphSpace, GraphPool, HistoryForest)
  - [ ] Three-tier architecture (Rust Core ↔ FFI ↔ Python)
  - [ ] Core ideas: structure vs info, versioning, columnar
  - [ ] Connected views: Graph↔Table↔Array↔Matrix
  - [ ] Visualization preview & roadmap

### Stage 1: Core Foundation - Phase 1 (Weeks 2-3)
- [ ] Chapter 1: Graph Core
  - [ ] Graph creation and initialization
  - [ ] Node CRUD operations
  - [ ] Edge CRUD operations
  - [ ] Attribute management
  - [ ] Basic graph properties
- [ ] Chapter 2: Subgraphs
  - [ ] Creating subgraphs (slicing, filtering)
  - [ ] Subgraph operations
  - [ ] Relationship to parent graph
  - [ ] Converting back to Graph
- [ ] Chapter 3: Accessors
  - [ ] NodesAccessor and EdgesAccessor
  - [ ] Attribute access patterns
  - [ ] Bulk operations
- [ ] **API Reference (Phase 1):**
  - [ ] `api/graph.md` - Hand-craft with architecture context, theory, transformations
  - [ ] `api/subgraph.md` - Emphasize view concept, relationship to Graph
  - [ ] `api/accessors.md` - Document nodes/edges accessor patterns
  - [ ] Each page: Conceptual overview → Theory → Transformations → Methods → Patterns
- [ ] Validate all examples run successfully

### Stage 2: Tabular Layer - Phase 2 (Weeks 4-5)
- [ ] Chapter 4: Base Tables
  - [ ] Table fundamentals
  - [ ] Table operations (head, tail, shape)
  - [ ] Column operations
  - [ ] Table transformations
- [ ] Chapter 5: Nodes & Edges Tables
  - [ ] NodesTable and EdgesTable operations
  - [ ] Attribute columns
  - [ ] Filtering and selection
- [ ] Chapter 6: Graph Tables
  - [ ] GraphTable as unified view
  - [ ] Bundle operations (save/load)
  - [ ] Merge operations
  - [ ] Graph reconstruction
- [ ] **API Reference (Phase 2):**
  - [ ] `api/table.md` - Columnar storage explanation, why tables matter
  - [ ] `api/nodes-table.md` - NodesTable with architecture context
  - [ ] `api/edges-table.md` - EdgesTable relationships
  - [ ] `api/graph-table.md` - Unified view concept, bundle format theory
  - [ ] Each page: Conceptual overview → Theory → Transformations → Methods → Patterns
- [ ] Create comprehensive table usage guide

### Stage 3: Array Layer - Phase 3 (Weeks 6-7)
- [ ] Chapter 7: Base Arrays
  - [ ] Array fundamentals
  - [ ] Array operations
  - [ ] Type system
  - [ ] Array transformations
- [ ] Chapter 8: Numeric Arrays
  - [ ] NumArray for numerical data
  - [ ] Statistical operations
  - [ ] Aggregations
  - [ ] Numpy interop
- [ ] Chapter 9: Graph Arrays
  - [ ] SubgraphArray
  - [ ] ComponentsArray
  - [ ] Array delegation patterns
  - [ ] Chaining examples
- [ ] Chapter 10: Specialized Arrays
  - [ ] NodesArray and EdgesArray
  - [ ] MetaNodeArray
  - [ ] Custom array types
- [ ] **API Reference (Phase 3):**
  - [ ] `api/base-array.md` - Columnar attributes, array as view concept
  - [ ] `api/num-array.md` - Numeric operations, why separate from base
  - [ ] `api/subgraph-array.md` - Array of subgraphs, delegation chains theory
  - [ ] `api/nodes-array.md` - NodesArray transformations
  - [ ] `api/edges-array.md` - EdgesArray transformations
  - [ ] Each page: Conceptual overview → Theory → Transformations → Methods → Patterns
- [ ] Document delegation chains comprehensively

### Stage 4: Matrix & Advanced - Phase 4 (Weeks 8-9)
- [ ] Chapter 11: Matrix Base
  - [ ] Matrix fundamentals
  - [ ] Matrix operations
  - [ ] Sparse vs dense
  - [ ] Linear algebra operations
- [ ] Chapter 12: Graph Matrices
  - [ ] Adjacency matrices
  - [ ] Laplacian matrices
  - [ ] Spectral embeddings
  - [ ] Matrix-graph conversions
- [ ] Chapter 13: Algorithms
  - [ ] Connected components
  - [ ] Shortest paths
  - [ ] Centrality measures
  - [ ] Community detection
- [ ] Chapter 14: Neural Networks
  - [ ] Neural graph modules
  - [ ] Automatic differentiation
  - [ ] Gradient computation
  - [ ] Training workflows
- [ ] **API Reference (Phase 4):**
  - [ ] `api/graph-matrix.md` - Graph as matrix projection, spectral theory
  - [ ] `api/algorithms.md` - Algorithm design philosophy, when to use each
  - [ ] `api/neural.md` - Neural module architecture, autodiff theory
  - [ ] Each page: Conceptual overview → Theory → Transformations → Methods → Patterns

### Stage 5: Integration & Advanced Topics - Phase 5 (Week 10)
- [ ] Chapter 15: Integration
  - [ ] NetworkX compatibility
  - [ ] Pandas integration
  - [ ] Numpy/SciPy interop
  - [ ] Import/export workflows
- [ ] Chapter 16: Performance
  - [ ] Benchmarking patterns
  - [ ] Optimization techniques
  - [ ] Memory management
  - [ ] Scaling considerations
- [ ] **API Reference (Phase 5):**
  - [ ] `api/generators.md` - Built-in graph generators, use cases
  - [ ] `api/io.md` - Import/export philosophy, format choices
  - [ ] `api/networkx-compat.md` - Interop design, when to convert
  - [ ] Each page: Conceptual overview → Theory → Transformations → Methods → Patterns
- [ ] Cross-reference validation across all API pages
- [ ] Example index creation
- [ ] Verify all transformation paths documented

### Stage 6: Back Matter & Polish (Week 11)
- [ ] Appendix A: **CLI & Utilities** (command reference, env vars, config)
- [ ] Appendix B: **File Formats & IO** (Parquet/Arrow/CSV; bundles; on-disk layout)
- [ ] Appendix C: **Error Reference & Troubleshooting** (common errors, causes, remedies)
- [ ] Appendix D: **Glossary** (Graph, Subgraph, Table, Array, Matrix, View, etc.)
- [ ] Appendix E: **Design Decisions (ADRs)** (architectural choices, rationale)
- [ ] Appendix F: **Example Index** (cross-links by task; searchable snippets)
- [ ] Appendix G: **Performance Cookbook** (memory notes, batching tips, materialization costs)
- [ ] Appendix H: **Integration Guides** (pandas/Arrow/PyG/NetworkX in one place)
- [ ] Appendix I: **Versioning & Migration Notes** (breaking changes, upgrade paths)
- [ ] Final review and editing
- [ ] Consistency check across all chapters
- [ ] Link validation

--- 

## Back Matter: Appendices (Outline)

These appendices will live under `docs/` and be linked from the sidebar.

- **Appendix A — CLI & Utilities**: command reference, env vars, config conventions.
- **Appendix B — File Formats & IO**: on‑disk layout, schema fingerprints, checksums, reproducibility.
- **Appendix C — Error Reference**: common errors, causes, remedies, return/exception mapping.
- **Appendix D — Glossary**: Graph, Subgraph, Table, Ray, Matrix, View, Induced, Materialize, etc.
- **Appendix E — ADRs**: rationale for architectural choices; links to PRs.
- **Appendix F — Example Index**: index of runnable snippets by task.
- **Appendix G — Performance Cookbook**: memory notes, materialization costs, batching tips.
- **Appendix H — Integration Guides**: pandas/Arrow/PyG/NetworkX interop in one place.
- **Appendix I — Versioning & Migration**: breaking changes, upgrade steps, mapping tables.


## Test-Driven Documentation Workflow

### For Each Chapter:

1. **Identify test module** (`test_X.py`)
2. **Read all tests** to understand covered functionality
3. **Group tests by concept** (creation, modification, query, etc.)
4. **Extract examples** from successful tests
5. **Write narrative** connecting the examples
6. **Validate examples** run independently
7. **Add to documentation**

### Example Extraction Pattern:

From test:
```python
# Example: Build → Inspect → Query → Algorithm → Views → Viz
# Goal: demonstrate connected views and common ops in ~20 lines.
# Remember: everything is a graph.

import groggy as gr

# ───────────────────────────────────────────────
# 1. Build a tiny graph
# ───────────────────────────────────────────────
g = gr.Graph()

# add nodes
alice = g.add_node(name="Alice", age=29)
bob   = g.add_node(name="Bob",   club="Purple", active=True, age=55)
carol = g.add_node(name="Carol", club="Blue",   active=True, age=31)

# add edges
g.add_edge(alice, bob,   weight=5)
g.add_edge(alice, carol, weight=2)
g.add_edge(bob,   carol, weight=1)

# ───────────────────────────────────────────────
# 2. Inspect (Graph → Table)
# ───────────────────────────────────────────────
nodes_tbl = g.nodes.table()
edges_tbl = g.edges.table()

print("Nodes table:")
print(nodes_tbl.head())   # columns: [id, name, age, club, active, ...]
print("\nEdges table:")
print(edges_tbl.head())   # columns: [src, dst, weight, ...]

# ───────────────────────────────────────────────
# 3. Query / Slice → returns Subgraph
# ───────────────────────────────────────────────
purple_nodes  = g.nodes[g.nodes["club"] == "Purple"]
younger_nodes = g.nodes[g.nodes["age"]  <  30]

print(f"\nPurple nodes:  {len(purple_nodes)}")
print(f"Younger nodes: {len(younger_nodes)}")

# ───────────────────────────────────────────────
# 4. Algorithm → Connected Components
# ───────────────────────────────────────────────
g.connected_components(inplace=True, label="component")
num_components = len(g.nodes["component"].unique())
print(f"\nConnected components: {num_components}")

# ───────────────────────────────────────────────
# 5. Views → Array → Matrix
# ───────────────────────────────────────────────
ages = g.nodes["age"]
mean_age = ages.mean()
print(f"\nMean age: {mean_age:.1f}")

L = g.laplacian_matrix()
print(f"Laplacian shape: {L.shape}")

# ───────────────────────────────────────────────
# 6. Viz → color by computed attribute
# ───────────────────────────────────────────────
print("\nRendering visualization...")
g.viz.show(node_color="component")

# ───────────────────────────────────────────────
# Everything connects. Everything is a graph.
# ───────────────────────────────────────────────
```

### 3. Emphasize Transformations
Document what becomes what:
- `Graph → table() → GraphTable`
- `Graph → nodes[:10] → Subgraph`
- `Graph["attr"] → BaseArray`
- `Subgraph → table() → GraphTable`

### 4. Real Use Cases
Every example should answer: "When would I use this?"

```python
# Use case: Find all nodes in the largest component
components = g.connected_components()
largest = components.sorted_by_size().first()
nodes_in_largest = largest.node_ids()
```

### 5. Progressive Disclosure
- Basic usage first (80% of use cases)
- Advanced features later (20% of use cases)
- Expert tips in separate sections

---

## Success Criteria

### For Each Chapter:
- [ ] All examples run successfully
- [ ] All documented features have test coverage
- [ ] Cross-references are accurate
- [ ] Code-to-prose ratio is balanced (show, don't just tell)
- [ ] Covers at least 90% of public API

### For Overall Documentation:
- [ ] Complete learning path from basics to advanced
- [ ] Easy to find examples for common tasks
- [ ] Clear about what returns what
- [ ] Delegation chains well documented
- [ ] Integration examples comprehensive

---

## Living Document Notes

This plan will evolve as we work through the documentation. Key areas to watch:

1. **Missing test coverage** - Document gaps for future test development
2. **API changes** - Keep synchronized with code changes
3. **User feedback** - Incorporate questions and confusion points
4. **Performance discoveries** - Add benchmarks and optimization notes
5. **Integration patterns** - Document newly discovered workflows

---

## Appendix A: Object Transformation Map

Visual reference for delegation chains:

```
Graph
├─→ Subgraph (via nodes[...], edges[...], subgraph())
├─→ GraphTable (via table())
├─→ BaseArray (via g["attr_name"])
├─→ NodesArray (via nodes.array())
├─→ EdgesArray (via edges.array())
└─→ GraphMatrix (via to_matrix(), spectral(), etc.)

Subgraph
├─→ Graph (via to_graph())
├─→ GraphTable (via table())
├─→ SubgraphArray (via connected_components(), etc.)
├─→ BaseArray (via ["attr_name"])
└─→ GraphMatrix (via to_matrix())

SubgraphArray
├─→ GraphTable (via table())
├─→ Subgraph (via [index], first(), last())
└─→ SubgraphArray (via filter(), sample(), neighborhood())

GraphTable
├─→ Graph (via to_graph())
├─→ NodesTable (via nodes)
├─→ EdgesTable (via edges)
├─→ BaseArray (via ["column_name"])
└─→ AggregationResult (via agg())

BaseArray
├─→ NumArray (for numeric data)
├─→ Table (via to_table())
└─→ Python list (via to_list())

GraphMatrix
├─→ NumArray (via to_array())
├─→ Graph (via to_graph())
└─→ Numpy array (via to_numpy())
```

---

## Appendix B: Test Coverage Summary

Current test module organization:

| Module | File | Primary Focus |
|--------|------|---------------|
| Graph Core | `test_graph_core.py` | Graph CRUD, basic operations |
| Subgraphs | `test_subgraph_base.py` | Subgraph creation and operations |
| Accessors | `test_accessors.py` | Node/edge accessors, bulk operations |
| Table Base | `test_table_base.py` | Base table operations |
| Nodes/Edges Tables | `test_nodes_edges_tables.py` | Specialized table types |
| Graph Tables | `test_graph_table.py` | GraphTable and bundles |
| Base Arrays | `test_base_array.py` | Array fundamentals |
| Num Arrays | `test_num_array.py` | Numeric array operations |
| Graph Arrays | `test_graph_arrays.py` | SubgraphArray and delegation |
| Matrix Base | `test_matrix_base.py` | Matrix fundamentals |
| Graph Matrices | `test_graph_matrix.py` | Graph-specific matrices |
| Algorithms | `test_algorithms.py` | Graph algorithms |
| Neural | `test_neural.py` | Neural network support |
| Integration | `test_integration.py` | External library integration |
| Performance | `test_performance.py` | Benchmarks and optimization |

---

## Appendix C: Dense Usage Reference Integration

The existing "Dense Usage Reference" in CLAUDE.md provides an excellent skeleton for the documentation. Key sections to expand:

1. **Quickest Start** → Chapter 1 introduction
2. **Manual Construction** → Chapter 1, detailed examples
3. **Filters & Queries** → Chapter 3 (Accessors)
4. **Algorithms** → Chapter 13
5. **Delegation Chaining** → Featured in Chapters 9-10
6. **Arrays/Matrix Mixing** → Chapters 11-12
7. **Builders & Merge** → Chapter 6
8. **IO** → Chapter 6 (GraphTable bundles)

These examples should be expanded with:
- More detailed explanations
- Edge cases and variations
- Performance considerations
- When to use each pattern

---

## Notes for Claude Code

When working on documentation:

1. **Always check tests first** - Don't document features that aren't tested
2. **Run examples** - Every example must execute successfully
3. **Show types** - Make clear what type each operation returns
4. **Link generously** - Cross-reference related concepts
5. **Keep examples minimal** - Don't combine too many concepts in one example
6. **Explain the "why"** - Not just how to use it, but when and why
7. **Follow the transformation graph** - Help users understand the object flow

Remember: "Everything is a graph" - even the documentation structure itself!
