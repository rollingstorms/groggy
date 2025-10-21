# Graph Datasets Module - Architecture Planning Document

## Overview
Add dataset loading capabilities to Groggy for loading graph datasets from HuggingFace and other sources, with proper train/test/validation splits suitable for machine learning workflows.

## Motivation
- Enable data science workflows with standard graph datasets (MUTAG, PROTEINS, etc.)
- Support HuggingFace `graphs-datasets` collection: https://huggingface.co/graphs-datasets
- Provide efficient train/test/validation splits without data duplication
- Integrate with existing SubgraphArray for lightweight graph collections

## Key Design Questions

### 1. Rust Core vs Python-Only?

**Option A: Rust Core Module** ✓ (Recommended)
- Pros:
  - Dataset parsing/conversion logic in Rust = faster, more robust
  - Can leverage Rust HTTP clients (reqwest) for efficient downloads
  - Future-proof: can add caching, streaming, parallel loading
  - Consistent with project architecture (algorithms in Rust)
  - Could support binary formats efficiently (Arrow, Parquet for graphs)
- Cons:
  - More initial work
  - Need to handle HTTP/IO in Rust
  - FFI layer for dataset metadata

**Option B: Python-Only Wrapper**
- Pros:
  - Faster to implement
  - Easy integration with HuggingFace `datasets` library
  - Simpler dependencies (just Python packages)
- Cons:
  - Parsing overhead stays in Python
  - Less efficient for large datasets
  - Doesn't follow project's "algorithms in Rust" principle
  - Tighter coupling to HuggingFace's format

**Decision Needed:** Which approach aligns better with Groggy's architecture?

### 2. SubgraphArray vs Individual Graphs?

From user context: *"these pytorch data object are more like subgrapharrays, so we have train_set is a subgraph_array"*

**Proposed Design:**
```python
# Return SubgraphArray for each split
train_graphs = load_dataset("MUTAG", split="train")  # -> SubgraphArray
test_graphs = load_dataset("MUTAG", split="test")    # -> SubgraphArray

# SubgraphArray is lightweight, size(g1 + g2) ≈ size(g1.merge(g2))
# Perfect for ML: train_loader iterates over train_graphs
```

**Questions:**
- Does current SubgraphArray support iteration for ML loops?
- Should we add DataLoader-like features to SubgraphArray later?
- Graph-level labels: where to store them? (no graph-level attrs yet)

### 3. Dataset Format Support

**Priority 1: HuggingFace graphs-datasets**
- Format: Parquet/Arrow with schema:
  ```
  {
    'num_nodes': int,
    'edge_index': [[sources], [targets]],
    'node_feat': array or None,
    'edge_attr': array or None, 
    'y': label (graph classification)
  }
  ```
- Examples: MUTAG, PROTEINS, NCI1, ENZYMES, etc.
- Access: `load_dataset("graphs-datasets/MUTAG")`

**Priority 2: PyTorch Geometric format** (future)
- Many datasets available
- Similar structure to HF
- Could reuse conversion logic

**Priority 3: NetworkX datasets** (future)
- Built-in karate club, etc.
- Easy to add

## Proposed Architecture

### Rust Core (`src/datasets/` or `src/io/datasets/`)

```rust
// src/datasets/mod.rs
pub mod loaders;
pub mod converters;
pub mod cache;

// Core types
pub struct Dataset {
    pub name: String,
    pub splits: Vec<Split>,
    pub metadata: DatasetMetadata,
}

pub struct Split {
    pub name: String,  // "train", "test", "validation"
    pub graphs: Vec<Graph>,
}

// Converter from HuggingFace format
pub struct HuggingFaceConverter;
impl HuggingFaceConverter {
    pub fn convert_graph(hf_data: HFGraph) -> Result<Graph>;
    pub fn convert_dataset(hf_dataset: HFDataset) -> Result<Dataset>;
}
```

### Python FFI (`python-groggy/src/ffi/datasets.rs`)

```rust
#[pyfunction]
fn load_dataset(
    name: &str,
    source: &str,
    splits: Option<Vec<String>>,
    cache_dir: Option<String>,
) -> PyResult<PyObject> {
    // Call Rust core, marshal results
    // Return SubgraphArray for each split
}
```

### Python API (`python-groggy/python/groggy/datasets.py`)

```python
def load_dataset(
    name: str,
    source: str = "huggingface",
    splits: Optional[List[str]] = None,
) -> Union[SubgraphArray, Tuple[SubgraphArray, ...]]:
    """
    Load graph dataset with train/test splits.
    
    Examples:
        >>> train, test = load_dataset("MUTAG", splits=["train", "test"])
        >>> train  # SubgraphArray with training graphs
        >>> test   # SubgraphArray with test graphs
    """
```

## Implementation Strategy

### Phase 1: Python Prototype (Quick Validation)
- Pure Python implementation using HuggingFace `datasets` library
- Validate format conversion, test with MUTAG dataset
- Confirm SubgraphArray integration works
- **Goal:** Working demo in notebook before building Rust core

### Phase 2: Rust Core (Production)
- Move conversion logic to Rust
- Add HTTP client for direct downloads (optional, could still use HF)
- Implement caching strategy
- Optimize bulk graph creation

### Phase 3: Advanced Features
- Streaming for large datasets
- Custom dataset registration
- PyG format support
- Graph-level attributes in core

## Data Flow

```
HuggingFace Dataset (Parquet/Arrow)
         ↓
    [Parser/Converter] (Rust or Python?)
         ↓
    List of Graphs (Rust Graph objects)
         ↓
    SubgraphArray per split (train/test/valid)
         ↓
    Python API returns tuple of SubgraphArray
```

## Open Questions for Review

1. **Rust vs Python for initial implementation?**
   - Start with Python prototype, migrate to Rust later?
   - Or go straight to Rust core?

2. **Dependency strategy:**
   - Option A: Use HuggingFace `datasets` library (Python or Rust bindings?)
   - Option B: Implement our own HTTP + Parquet reader
   - Option C: Hybrid (HF for discovery, Rust for parsing)

3. **Where to store graph-level labels?**
   - Current workaround: store on first node with special key
   - Wait for graph-level attributes feature?
   - Add to SubgraphArray as separate array?

4. **SubgraphArray enhancements needed?**
   - Iteration protocol for ML loops?
   - Indexing: `train_graphs[i]` → single Graph?
   - Slicing: `train_graphs[0:10]` → SubgraphArray?

5. **Dataset discovery:**
   - Should we list available datasets from HF?
   - Auto-download with caching?
   - Require manual download first?

## Example Usage (Target API)

```python
import groggy as gr
from groggy.datasets import load_dataset

# Simple case: load with default splits
train_graphs, test_graphs = load_dataset(
    "MUTAG",
    source="huggingface"
)

print(f"Training: {len(train_graphs)} graphs")
print(f"Test: {len(test_graphs)} graphs")

# Iterate for ML training
for graph in train_graphs:
    features = graph.nodes.features  # Node features
    label = graph.label  # Graph label (where stored?)
    # ... train GNN model

# Advanced: custom splits
train, val, test = load_dataset(
    "PROTEINS",
    splits=["train", "validation", "test"]
)

# Single split
all_graphs = load_dataset("MUTAG", splits=["train"])
```

## Integration with Existing Code

- **SubgraphArray:** Already exists, use as return type
- **Graph creation:** Use bulk `add_nodes()`, `add_edges()` (already optimized)
- **Attributes:** Use `set_node_attr()`, `set_edge_attr()` (existing API)
- **imports.py:** Similar pattern, but for datasets not CSVs

## Success Criteria

- [ ] Load MUTAG dataset from HuggingFace
- [ ] Convert to Groggy graphs with node/edge features
- [ ] Return train/test as separate SubgraphArray objects
- [ ] Verify memory efficiency: no unnecessary duplication
- [ ] Document in notebook with visualization example
- [ ] Test with 2-3 different datasets (MUTAG, PROTEINS, ENZYMES)

## Timeline Estimate

- **Phase 1 (Python):** 2-3 hours
  - Pure Python implementation
  - Test with MUTAG
  - Demo notebook
  
- **Phase 2 (Rust Core):** 1-2 days
  - Rust dataset module
  - FFI bindings
  - Testing across datasets
  
- **Phase 3 (Advanced):** Future
  - As needed for performance/features

## Recommendation

**Start with Phase 1 (Python prototype)** because:
1. Validate the API design quickly
2. Test SubgraphArray integration
3. Identify any missing features (graph-level labels, etc.)
4. Get working demo for data science workflows
5. Then migrate performance-critical parts to Rust in Phase 2

The Python prototype will inform the Rust design and ensure we're solving the right problem.

---

## Next Steps

**Please review and edit this plan:**
- [ ] Choose Rust vs Python approach
- [ ] Clarify SubgraphArray requirements
- [ ] Decide on graph-level label storage
- [ ] Add any missing use cases or requirements
- [ ] Approve phased approach or suggest alternative

Once approved, implementation can begin with clear direction.
