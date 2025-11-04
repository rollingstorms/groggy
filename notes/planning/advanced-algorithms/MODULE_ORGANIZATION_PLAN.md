# Phase 1 Module Organization Plan

## File Assignment Summary

### Existing Modules to Extend

**`structural.rs`** - Graph structure operations
- ✅ NodeDegreeStep
- ⏸️ WeightedDegreeStep
- ⏸️ KCoreMarkStep
- ⏸️ TriangleCountStep
- ⏸️ EdgeWeightSumStep

**`attributes.rs`** - Attribute operations
- ✅ LoadNodeAttrStep, LoadEdgeAttrStep
- ✅ AttachNodeAttrStep, AttachEdgeAttrStep
- ⏸️ EdgeWeightScaleStep (operates on edge attributes)

**`normalization.rs`** - Value normalization
- ✅ NormalizeNodeValuesStep (Sum/Max/MinMax)
- ⏸️ StandardizeStep (Z-score)
- ⏸️ ClipValuesStep

**`aggregations.rs`** - Statistical operations
- ✅ ReduceNodeValuesStep (Sum/Min/Max/Mean)
- ⏸️ StdDevStep
- ⏸️ MedianStep
- ⏸️ ModeStep
- ⏸️ QuantileStep
- ⏸️ EntropyStep
- ⏸️ HistogramStep

### New Modules to Create

**`filtering.rs`** - Ordering and filtering (Section 1.5)
- ⏸️ SortNodesByAttrStep
- ⏸️ FilterEdgesByAttrStep
- ⏸️ FilterNodesByAttrStep
- ⏸️ TopKStep
- ⏸️ Predicate enum

**`sampling.rs`** - Random sampling (Section 1.6)
- ⏸️ SampleNodesStep
- ⏸️ SampleEdgesStep
- ⏸️ ReservoirSampleStep
- ⏸️ SampleSpec enum

**`pathfinding.rs`** - Path utilities (Section 1.8)
- ⏸️ ShortestPathMapStep
- ⏸️ KShortestPathsStep
- ⏸️ RandomWalkStep

**`community.rs`** - Community detection helpers (Section 1.9)
- ⏸️ CommunitySeedStep
- ⏸️ ModularityGainStep
- ⏸️ LabelPropagateStep
- ⏸️ SeedStrategy enum

**`flow.rs`** - Network flow (Section 1.10)
- ⏸️ FlowUpdateStep
- ⏸️ ResidualCapacityStep

## Implementation Priority

### Phase 1.2 (Current Focus)
Extend `structural.rs` with 5 new steps

### Phase 1.3
Extend `normalization.rs` with 2 new steps

### Phase 1.5
Create `filtering.rs` with 4 steps + Predicate enum

### Phase 1.6
Create `sampling.rs` with 3 steps + SampleSpec enum

### Phase 1.7
Extend `aggregations.rs` with 6 new statistical steps

### Phase 1.8
Create `pathfinding.rs` with 3 steps

### Phase 1.9
Create `community.rs` with 3 steps + SeedStrategy enum

### Phase 1.10
Create `flow.rs` with 2 steps
