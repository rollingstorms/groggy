# Appendices

**Reference documentation and supplementary material**

---

This section contains reference material to supplement the main documentation:

## Available Appendices

### [Glossary](glossary.md)
Complete terminology reference with definitions, examples, and cross-references for all Groggy concepts.

**What's inside:**
- 50+ terms with clear definitions
- Code examples for key concepts
- Common abbreviations
- Quick reference tables

**Use this when:** You need to understand terminology or look up unfamiliar concepts.

---

### [Design Decisions (ADRs)](design-decisions.md)
Architectural Decision Records documenting the key design choices made in Groggy.

**What's inside:**
- 11 major architectural decisions
- Rationale and alternatives considered
- Consequences and trade-offs
- Design principles summary

**Use this when:** You want to understand why Groggy works the way it does.

---

### [Performance Cookbook](performance-cookbook.md)
Practical optimization patterns and recipes for high-performance graph operations.

**What's inside:**
- 11 optimization recipes
- Performance anti-patterns to avoid
- Profiling and benchmarking tools
- Complexity guidelines by graph size

**Use this when:** You need to optimize your graph operations or debug performance issues.

---

### [Temporal Extensions Guide](temporal-extensions-guide.md)
Comprehensive guide to temporal graph analytics and time-series features in Groggy.

**What's inside:**
- Core temporal concepts (snapshots, index, scope, deltas)
- Common temporal analysis patterns
- Best practices and performance optimization
- Example use cases and code
- Troubleshooting guide

**Use this when:** You need to analyze graph evolution, track changes over time, or perform temporal queries.

---

## Quick Links

### By Topic

**Learning Groggy:**
- Start with the [Glossary](glossary.md) to understand terminology
- Read [Design Decisions](design-decisions.md) to understand the architecture
- Refer to [Performance Cookbook](performance-cookbook.md) for optimization
- See [Temporal Extensions Guide](temporal-extensions-guide.md) for time-series analysis

**Understanding Architecture:**
- [Design Decisions](design-decisions.md) - Why these choices?
- [Glossary](glossary.md) - What do terms mean?
- [Concepts: Architecture](../concepts/architecture.md) - How does it work?

**Optimizing Performance:**
- [Performance Cookbook](performance-cookbook.md) - Optimization recipes
- [Performance Guide](../guide/performance.md) - Comprehensive tutorial
- [Design Decisions](design-decisions.md) - Performance trade-offs

**Temporal Analysis:**
- [Temporal Extensions Guide](temporal-extensions-guide.md) - Complete temporal features guide
- [User Guides](../guide/graph-core.md) - Basic graph operations
- [API Reference](../api/graph.md) - Temporal API methods

---

## Reference Material

All appendices are designed as quick reference guides. They complement the main documentation:

- **[Getting Started](../index.md)** - Begin here if you're new
- **[User Guides](../guide/graph-core.md)** - Learn by doing
- **[API Reference](../api/graph.md)** - Look up methods
- **[Concepts](../concepts/overview.md)** - Understand the design

---

## Using the Appendices

### Glossary
**Best for:** Looking up unfamiliar terms, understanding concepts

**Example use:**
> "What's the difference between a Subgraph and a SubgraphArray?"
> → Look it up in the [Glossary](glossary.md)

### Design Decisions
**Best for:** Understanding architectural choices, learning the "why"

**Example use:**
> "Why does Groggy use columnar storage?"
> → Read ADR-003 in [Design Decisions](design-decisions.md)

### Performance Cookbook
**Best for:** Optimizing code, debugging performance issues

**Example use:**
> "My graph filtering is slow. How do I speed it up?"
> → Check Recipe 1 in [Performance Cookbook](performance-cookbook.md)

### Temporal Extensions Guide
**Best for:** Working with temporal data, analyzing graph evolution

**Example use:**
> "How do I query the graph as it was last week?"
> → Check the Time-Travel Queries pattern in [Temporal Extensions Guide](temporal-extensions-guide.md)

---

## Contributing

Found an issue or have a suggestion?

- **Report issues:** [GitHub Issues](https://github.com/rollingstorms/groggy/issues)
- **Suggest improvements:** [GitHub Discussions](https://github.com/rollingstorms/groggy/discussions)
- **Request new content:** Let us know what appendices would be helpful!

---

*These appendices are living documents. They're updated as Groggy evolves.*
