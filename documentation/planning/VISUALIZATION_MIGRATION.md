# Visualization Backend Migration Plan: Realtime as Default

## Overview

This document outlines the migration strategy for consolidating Groggy's visualization system to use the **Realtime Backend** as the primary/default visualization engine, deprecating the Streaming Backend. This migration enables advanced n-dimensional interaction controls, modular embedding support, and unified architecture.

> **Note (2024-xx-xx):** The historical Python `VizModule` bindings referenced here have been retired in favour of the lighter `VizAccessor` interface. The plan remains archived for context only.

## Current Architecture Issues

### Dual Backend Problem
- **Streaming Backend**: Simple, limited controls, basic layouts
- **Realtime Backend**: Advanced controls, n-dimensional embeddings, sophisticated physics
- **User Impact**: Honeycomb n-dimensional rotation controls exist but aren't accessible through default API
- **Developer Impact**: Code duplication, feature fragmentation, maintenance overhead

### Feature Gaps
- Honeycomb `HoneycombInteractionController` exists in realtime but not connected to `g.viz.show()`
- N-dimensional rotation matrices implemented but not exposed
- Advanced physics simulation available but unused
- Modular embedding system (UMAP, t-SNE, PCA) only in realtime backend

## Migration Strategy: Option 3 - Realtime as Default

### Phase 1: API Compatibility Layer
**Goal**: Ensure all existing streaming backend functionality works through realtime backend

#### 1.1 Method Mapping
```python
# Current API (preserved)
g.viz.show()                    # Now uses realtime backend
g.viz.show_honeycomb()          # Enhanced with n-dimensional controls
g.viz.honeycomb(params...)      # Full physics and interaction support
g.viz.stream()                  # Deprecated but functional fallback
```

#### 1.2 Configuration Migration
```rust
// src/viz/mod.rs - Updated default backend
impl VizModule {
    pub fn show(&mut self) -> GraphResult<RenderResult> {
        // OLD: self.render(VizBackend::Streaming, RenderOptions::default())
        self.render(VizBackend::Realtime, RenderOptions::default())
    }

    pub fn show_honeycomb(&mut self) -> GraphResult<RenderResult> {
        let options = RenderOptions {
            layout: LayoutMethod::Honeycomb,
            interaction_controls: InteractionLevel::Advanced,
            embedding_dimensions: 4, // Enable n-dimensional controls
        };
        self.render(VizBackend::Realtime, options)
    }
}
```

### Phase 2: Feature Consolidation
**Goal**: Migrate all streaming backend features to realtime backend

#### 2.1 Basic Layout Support
- [x] **Force-directed layouts**: Already in realtime backend
- [x] **Honeycomb layouts**: Advanced implementation exists
- [ ] **Grid layouts**: Need to port from streaming backend
- [ ] **Circular layouts**: Need to port from streaming backend

#### 2.2 Rendering Capabilities
- [x] **WebSocket streaming**: Enhanced version in realtime
- [x] **60 FPS updates**: Performance monitoring included
- [ ] **Static HTML export**: Port from streaming backend
- [ ] **PNG/SVG export**: Add to realtime backend

#### 2.3 Interaction Controls
- [x] **N-dimensional rotation**: `HoneycombInteractionController`
- [x] **Canvas dragging**: Advanced momentum physics
- [x] **Node dragging**: Real-time position updates
- [x] **Zoom controls**: Smooth scaling with limits
- [ ] **Selection tools**: Port from streaming backend

### Phase 3: Advanced Feature Activation
**Goal**: Enable realtime backend's advanced capabilities by default

#### 3.1 N-Dimensional Embeddings
```python
# Enable by default for honeycomb layouts
g.viz.show_honeycomb()  # Now includes:
# - 4D+ embedding spaces
# - Multi-modal rotation controls
# - Momentum-based physics
# - Real-time dimension projection
```

#### 3.2 Embedding Method Support
```python
# Modular embedding system now available
g.viz.set_embedding("umap", dimensions=8)     # UMAP with 8D space
g.viz.set_embedding("tsne", perplexity=30)    # t-SNE optimization
g.viz.set_embedding("pca", components=6)      # PCA dimensional reduction
```

#### 3.3 Real-time Analytics
```python
# Performance monitoring and adaptive quality
g.viz.show(analytics=True)  # Live FPS, memory, interaction metrics
g.viz.set_quality_mode("adaptive")  # Auto-adjust based on performance
```

## Implementation Steps

### Step 1: Backend Default Switch (Immediate)
```rust
// src/viz/mod.rs
impl VizModule {
    pub fn show(&mut self) -> GraphResult<RenderResult> {
        self.render(VizBackend::Realtime, RenderOptions::default())
    }
}
```

### Step 2: Streaming Feature Audit (Week 1)
- [ ] Identify all streaming-only features
- [ ] Create feature parity checklist
- [ ] Document API compatibility requirements

### Step 3: Feature Migration (Week 2-3)
- [ ] Port missing layout algorithms to realtime backend
- [ ] Add static export capabilities to realtime backend
- [ ] Implement selection tools in realtime controls

### Step 4: Testing & Validation (Week 4)
- [ ] Comprehensive test suite for all migrated features
- [ ] Performance benchmarking vs. streaming backend
- [ ] User acceptance testing with existing scripts

### Step 5: Documentation & Cleanup (Week 5)
- [ ] Update all documentation to reflect realtime-first approach
- [ ] Add deprecation warnings to streaming backend
- [ ] Create migration guide for users

## Feature Parity Matrix

| Feature | Streaming Backend | Realtime Backend | Migration Status |
|---------|------------------|------------------|------------------|
| **Core Visualization** | | | |
| Basic graph rendering | ✅ | ✅ | ✅ Complete |
| Force-directed layout | ✅ | ✅ | ✅ Complete |
| Honeycomb layout | ✅ | ✅ Enhanced | ✅ Complete |
| Grid layout | ✅ | ❌ | 🔄 In Progress |
| Circular layout | ✅ | ❌ | 🔄 In Progress |
| **Interaction Controls** | | | |
| Basic pan/zoom | ✅ | ✅ | ✅ Complete |
| Node dragging | ✅ | ✅ Enhanced | ✅ Complete |
| Canvas dragging | ✅ | ✅ Enhanced | ✅ Complete |
| N-dimensional rotation | ❌ | ✅ | ✅ New Feature |
| Selection tools | ✅ | ❌ | 🔄 In Progress |
| **Export & Rendering** | | | |
| WebSocket streaming | ✅ | ✅ Enhanced | ✅ Complete |
| Static HTML export | ✅ | ❌ | 🔄 In Progress |
| PNG export | ✅ | ❌ | 🔄 In Progress |
| SVG export | ✅ | ❌ | 🔄 In Progress |
| **Advanced Features** | | | |
| Real-time updates | ✅ | ✅ Enhanced | ✅ Complete |
| Performance monitoring | ❌ | ✅ | ✅ New Feature |
| Multi-dimensional embeddings | ❌ | ✅ | ✅ New Feature |
| Physics simulation | ❌ | ✅ | ✅ New Feature |
| Adaptive quality | ❌ | ✅ | ✅ New Feature |

## Risk Mitigation

### Compatibility Risks
- **Risk**: Existing user scripts break
- **Mitigation**: Maintain streaming backend as fallback during transition
- **Testing**: Comprehensive regression test suite

### Performance Risks
- **Risk**: Realtime backend too heavy for simple use cases
- **Mitigation**: Implement "lite" mode for basic visualizations
- **Monitoring**: Performance benchmarking throughout migration

### Feature Loss Risks
- **Risk**: Streaming-only features not migrated
- **Mitigation**: Complete feature audit and parity checklist
- **Validation**: User acceptance testing with diverse workloads

## Success Metrics

### Technical Metrics
- [ ] 100% API compatibility with existing scripts
- [ ] <10% performance regression for basic operations
- [ ] >50% performance improvement for complex interactions
- [ ] Zero critical bugs in migrated features

### User Experience Metrics
- [ ] N-dimensional honeycomb controls accessible via `g.viz.show_honeycomb()`
- [ ] Smooth 60 FPS interactions for graphs <1000 nodes
- [ ] Real-time analytics provide actionable performance insights
- [ ] Modular embedding system enables advanced research workflows

## Timeline

- **Week 1**: Complete feature audit and start grid/circular layout migration
- **Week 2**: Finish layout migration, add export capabilities
- **Week 3**: Implement selection tools, performance optimization
- **Week 4**: Comprehensive testing, bug fixes
- **Week 5**: Documentation, deprecation notices, release preparation

## Post-Migration Benefits

### For Users
- **Enhanced Honeycomb**: N-dimensional rotation controls work out of the box
- **Modular Embeddings**: Access to UMAP, t-SNE, PCA with configurable parameters
- **Better Performance**: Real-time analytics and adaptive quality
- **Advanced Physics**: Momentum-based interactions and smooth animations

### For Developers
- **Unified Codebase**: Single backend reduces maintenance overhead
- **Extensible Architecture**: Modular design enables rapid feature development
- **Performance Monitoring**: Built-in analytics for optimization guidance
- **Future-Proof**: Foundation for advanced visualization research

## Rollback Plan

If critical issues arise during migration:

1. **Immediate**: Revert default backend to streaming in `src/viz/mod.rs`
2. **Short-term**: Maintain dual-backend support with explicit selection
3. **Long-term**: Address issues in realtime backend, re-attempt migration

## Conclusion

Migrating to the Realtime Backend as default resolves the current architecture split while unlocking advanced n-dimensional interaction capabilities. The comprehensive feature parity matrix ensures no functionality is lost, while the phased approach minimizes risk to existing users.

The migration directly addresses the original issue: honeycomb n-dimensional rotation controls will be accessible through the standard `g.viz.show_honeycomb()` API, providing the modular, extensible visualization system the user requested.
