# Advanced Algorithm Roadmap

## 🎯 Vision & Scope

This roadmap expands Groggy beyond the v0.5 foundations, layering in temporal analytics, advanced
algorithm suites, and richer builder/pipeline infrastructure. The goal is to establish Groggy as a
comprehensive graph analytics platform that balances performance (Rust core), composability (pipeline
architecture), and usability (Python DSL).

### Strategic Goals

**Temporal First-Class Citizenship** – Treat change history as queryable time-series data, enabling
drift analysis, burst detection, and historical pattern mining without custom code.

**Algorithm Breadth** – Cover the standard graph algorithm families (community, centrality, pathfinding,
decomposition, transform, statistical) so users reach for Groggy instead of stitching together
multiple libraries.

**Builder Maturity** – Expand the step primitive catalog so custom algorithms can be composed in Python
without dropping into Rust, while maintaining performance through columnar execution.

**Production Readiness** – Comprehensive testing, benchmarking, documentation, and error handling to
support mission-critical workloads.

### Non-Goals

**Complete Coverage** – We focus on *commonly used* algorithms and patterns, not exhaustive catalogs.
Specialized or esoteric algorithms belong in extensions or community packages.

**Distributed Execution** – Initial phases target single-machine performance. Distributed support
(partitioning, message passing) is a future extension.

**ML Integration** – While we provide feature engineering primitives, deep learning / GNN integration
is out of scope for this roadmap.

---

## 📊 Current Status

**Completed (v0.5.0)**
- ✅ Algorithm trait system and pipeline infrastructure
- ✅ Core algorithms: LPA, Louvain, PageRank, Betweenness, Closeness, Dijkstra, BFS, DFS, A*
- ✅ FFI bridge with thread-safe registry
- ✅ Python user API with discovery, handles, and pipelines
- ✅ Simplified builder DSL with step interpreter
- ✅ 304/304 Rust tests passing, 69/69 Python tests passing

**In Progress**
- 🚧 Temporal extensions (see `../temporal-extensions-plan.md` - separate planning document)
- 🚧 Visualization streaming architecture (see `../viz_module/`)

**Upcoming (This Roadmap)**
- ⏭️ Expanded builder step primitives (Phase 1)
- ⏭️ Additional algorithm categories (Phases 2-4)
- ⏭️ Builder / pipeline meta-infrastructure (Phase 5)
- ⏭️ Testing, benchmarking, and documentation (Phase 6)

---

## Note on Temporal Extensions

Temporal extensions are documented separately in `../temporal-extensions-plan.md` due to their complexity and scope. That plan covers:
- TemporalSnapshot and ExistenceIndex infrastructure
- TemporalIndex for efficient history queries
- AlgorithmContext temporal extensions
- Temporal algorithm steps (diff, window aggregation, filtering)
- Integration with existing systems
- Full implementation roadmap (8-9 weeks)

This roadmap focuses on expanding the non-temporal algorithm catalog and builder infrastructure.

---

## Phase Documents

Each phase has been split into its own document for easier navigation and focused planning:

### Core Algorithm Phases

1. **[Phase 1: Builder Core Extensions](PHASE_1_BUILDER_CORE.md)** (4-6 weeks)
   - 40+ step primitives across 10 categories
   - Infrastructure: schema registry, FFI runtime, validation
   - Foundation for all subsequent algorithm work

2. **[Phase 2: Community Algorithms](PHASE_2_COMMUNITY.md)** (6-8 weeks)
   - 10+ community detection algorithms (Leiden, Infomap, Girvan-Newman, etc.)
   - Shared infrastructure (quality metrics, dendrogram support)
   - Testing strategy and benchmarks

3. **[Phase 3: Centrality Algorithms](PHASE_3_CENTRALITY.md)** (4-6 weeks)
   - 10+ centrality measures (Degree, Eigenvector, Katz, Harmonic, etc.)
   - Normalization and convergence infrastructure
   - Performance benchmarks

4. **[Phase 4: Pathfinding Algorithms](PHASE_4_PATHFINDING.md)** (4-5 weeks)
   - Advanced pathfinding (Bellman-Ford, Floyd-Warshall, Yen's k-shortest, etc.)
   - All-pairs algorithms and optimizations
   - Path representation infrastructure

### New Algorithm Categories

5. **[New Algorithm Categories (Phases 4A-4D)](PHASE_4_NEW_CATEGORIES.md)** (16-21 weeks total)
   - **4A: Decomposition** - Spectral analysis, graph signal processing (6-8 weeks)
   - **4B: Transform** - Graph transformations and normalization (3-4 weeks)
   - **4C: Temporal** - Dynamic graph algorithms (4-5 weeks)
   - **4D: Statistical** - Graph measures and models (3-4 weeks)

### Infrastructure & Polish

6. **[Phase 5: Builder/Pipeline Meta Infrastructure](PHASE_5_META_INFRASTRUCTURE.md)** (3-4 weeks)
   - DSL expression language with macros
   - Pipeline manifest export/import (JSON/TOML)
   - CLI inspection tools
   - Parameter schema validation

7. **[Phase 6: Testing, Documentation, Polish](PHASE_6_POLISH.md)** (Ongoing + 2-3 weeks final)
   - Comprehensive testing strategy
   - Documentation coverage
   - Examples and tutorials
   - Release preparation

### Supporting Documents

- **[Success Metrics & Timeline](METRICS_AND_TIMELINE.md)**
  - Coverage, performance, quality, and usability metrics
  - Gantt chart with dependencies
  - Critical path analysis

- **[Risks & Mitigations](RISKS_AND_MITIGATIONS.md)**
  - Technical, resource, and scope risks
  - Mitigation strategies

- **[Implementation Style Guide](STYLE_GUIDE.md)**
  - Builder core conventions
  - Rust algorithm patterns
  - Python API guidelines
  - Code review checklist

---

## Quick Start

**For Algorithm Implementers**: Start with [Phase 1](PHASE_1_BUILDER_CORE.md) to understand the builder primitives, then move to your specific algorithm category (Phases 2-4).

**For Infrastructure Work**: See [Phase 5](PHASE_5_META_INFRASTRUCTURE.md) for meta-infrastructure and [Phase 6](PHASE_6_POLISH.md) for testing/polish work.

**For Project Planning**: Review [Metrics & Timeline](METRICS_AND_TIMELINE.md) for overall schedule and dependencies.

---

## Document Version

**Version**: 1.0  
**Last Updated**: 2024-01-15  
**Status**: Planning / RFC  
**Approval**: Pending review
