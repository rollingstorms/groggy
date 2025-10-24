## 🚨 Risks & Mitigations

### Technical Risks

**Risk**: Linear algebra dependencies (Phase 4A) introduce complexity  
**Mitigation**: Evaluate `nalgebra`, `faer`, and `ndarray` early. Choose based on performance and API ergonomics. Consider optional feature flags.

**Risk**: Builder DSL becomes too complex (Phase 1, 5)  
**Mitigation**: Keep primitives simple and composable. Provide high-level macros for common patterns. Gather user feedback early.

**Risk**: FFI overhead grows with more complex types (Phase 5)  
**Mitigation**: Benchmark marshalling costs. Use zero-copy where possible. Profile and optimize hot paths.

**Risk**: Test suite execution time becomes prohibitive (Phase 6)  
**Mitigation**: Parallelize tests. Use test categorization (unit, integration, slow). Run full suite in CI, subset locally.

### Resource Risks

**Risk**: Timeline slips due to underestimated complexity  
**Mitigation**: Buffer time in estimates (30%). Prioritize ruthlessly (High/Medium/Low). Ship incrementally.

**Risk**: Burnout from sustained effort  
**Mitigation**: Break work into digestible chunks. Celebrate milestones. Maintain quality over speed.

### Scope Risks

**Risk**: Feature creep (algorithms beyond roadmap)  
**Mitigation**: Strict prioritization. Defer "nice to have" items to future releases. Focus on breadth first, depth second.

**Risk**: External dependencies change (libraries, Python versions)  
**Mitigation**: Pin dependency versions. Monitor for breaking changes. Budget time for adaptation.

---

## 🎓 Learning & Experimentation

### Experimental Algorithm Families (Future Extensions)

Beyond this roadmap, these families could be explored:

**Streaming/Incremental Updates** – Leverage ChangeTracker for incremental community detection, rolling centrality, online anomaly detection.

**Structural Embeddings** – Node2Vec, DeepWalk, graph neural network foundations via sampling and aggregation primitives.

**Motif & Pattern Mining** – Subgraph enumeration, frequent pattern discovery, graphlet counting.

**Reachability & Flow** – Max-flow/min-cut, reachability indices, flow-based community detection.

**Graph Sketches & Sampling** – MinHash for similarity, reservoir sampling, approximate query answering.

**Explainability** – Trace paths, collect evidence, influence scoring for algorithm decisions.

### Research Directions

**Distributed Execution** – Partition graphs, message passing, integration with distributed storage.

**GPU Acceleration** – Offload linear algebra (decomposition) and graph traversal to GPU.

**Interactive Algorithms** – Real-time parameter tuning, progress visualization, streaming updates.

**ML Integration** – Graph feature extraction for GNNs, embedding pipelines, prediction interfaces.

---

