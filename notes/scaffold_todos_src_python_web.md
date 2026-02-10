# Scaffold TODO/FIXME Listing (src, python-groggy, web)

## python-groggy/python/groggy/_groggy.cpython-39-darwin.so
- L166860: // TODO: apply nodes_changed / edges_* when UI requires it
- L169243: <!-- TODO(flat-embedding): add toggle + sliders for the flat embedding energy once implemented. -->
- L270519: XX$XXX$XXXXXXXXXXXX$XX$XXXXXXXX$$X$$$$XXXXXXXX$XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX$XXL$$$$D	
- L270527: 			$$9X$XXX$$$$$$$XXXXXXXXX$XXXXXXX$$XL$$$$D
- L270578: LXXXX$$$XXX$XXXXXXXXXLXXXXXXXXXXXX$$XXXXXX$$XXXX$$XXX$XXX$XX$$$X$XXXX$XXX$$$$XXXXX$XXX$$$$$X$$L$$$
- L270594: :$$$$$$$XXX$$
- L270639: `d$$$X$$$XXX$$$$XL$D$$x
- L270646: $$XXX$$$X$$XX$$XXXXXX$$XXXXXXXXXX$$XXXXXXXX$$$X$$$$X$D$$
- L270656: 3$X$X$$$$$$$$$$$$$XL$D$$'$$$X$$XXX$D$$x
- L270657: $$XX$XXXX		$$$$XXX$$X$X$XXXXXXXXXXX$$$XXXXXL$D$$
- L270667: $XXXXXX$$XXXXXX$$$X$XXX$$$$$XXXXXX$$$$$X$$$$$$X$XX$$$$$X$X$X$XXXXXXX$$$$$$XX$$$$$$XXXD$$$$$		

## python-groggy/python/groggy/builder/ir/nodes.py
- L501: - Execution blocks (TODO: may support in future)
- L502: - Conditionals (if/where - TODO)

## python-groggy/src/ffi/api/graph.rs
- L223: // TODO: We should validate that this ID doesn't already exist
- L500: // TODO: Refactor to use config/builder pattern
- L1440: // TODO: Core doesn't have aggregate_nodes_custom, implement if needed
- L1682: // TODO: This could be more efficient with a proper attribute iteration API

## python-groggy/src/ffi/experimental.rs
- L146: // TODO: Implement actual PageRank algorithm
- L164: // TODO: Implement community detection (Louvain, etc.)

## python-groggy/src/ffi/storage/accessors.rs
- L1186: // TODO: Copy attributes from original nodes
- L1509: // TODO: Implement proper conversion from NodesAccessor to SubgraphArray
- L2754: // TODO: Copy edge attributes
- L3135: // TODO: Implement proper conversion from EdgesAccessor to SubgraphArray

## python-groggy/src/ffi/storage/array.rs
- L53: /// TODO: Fix PyO3 type inference issue

## python-groggy/src/ffi/storage/components.rs
- L138: // TODO: Refactor to use config/builder pattern
- L407: None // TODO: Consider changing ArrayOps to return owned values for some types
- L467: // TODO: Refactor to use config/builder pattern

## python-groggy/src/ffi/storage/edges_array.rs
- L266: // TODO: Implement proper table conversion once table array structure is clarified

## python-groggy/src/ffi/storage/matrix.rs
- L204: // TODO: Implement graph integration in Phase 2
- L373: // TODO: Support advanced indexing like slices, arrays, etc.
- L2307: // TODO: Implement proper matrix-to-table conversion

## python-groggy/src/ffi/storage/nodes_array.rs
- L238: // TODO: Implement proper table conversion once table array structure is clarified

## python-groggy/src/ffi/storage/subgraph_array.rs
- L259: // TODO: Refactor to use config/builder pattern

## python-groggy/src/ffi/storage/table.rs
- L3313: // TODO: Implement attribute conversion (temporarily disabled to fix compilation)
- L4102: // TODO: Convert NodesTable to GraphDataSource for VizAccessor
- L5088: // TODO: Convert EdgesTable to GraphDataSource for VizAccessor
- L5690: py_dict.set_item("checksum_verified", false)?; // TODO: Implement full verification
- L5720: // TODO: Implement proper conversion from GraphTable to NodesAccessor
- L5729: // TODO: Implement proper conversion from GraphTable to EdgesAccessor
- L5738: // TODO: Implement proper conversion from GraphTable to SubgraphArray

## python-groggy/src/ffi/subgraphs/subgraph.rs
- L1391: // TODO: Implement hierarchical navigation in future version
- L1399: // TODO: Implement hierarchical navigation in future version
- L1471: // TODO: Refactor to use config/builder pattern

## python-groggy/src/ffi/types.rs
- L62: // TODO: Convert JSON back to Python objects (lists, dicts) using eval or json module

## python-groggy/src/lib.rs
- L497: // TODO: Add PyArrayArrayIterator once we implement proper PyBaseArray conversion

## src/algorithms/execution/batch_executor.rs
- L608: #[ignore] // TODO: Fix test - needs proper subgraph setup and Graph API
- L610: // TODO: Need proper Graph -> Subgraph conversion for test setup

## src/algorithms/execution/jit/compiler.rs
- L366: // TODO: These need StepScope access - will be handled in Phase 2

## src/algorithms/execution/jit/mod.rs
- L61: // TODO: Add StepScope-dependent operations (Load/StoreNodeProp, NeighborAggregate)

## src/api/graph.rs
- L1073: // TODO: Validate subgraph exists in pool
- L1153: // TODO: Add subgraph existence validation when pool supports it
- L1843: // TODO: When HistoryForest is implemented:
- L2057: // TODO: Implement complex query composition when needed
- L2068: /// TODO: Implement when GraphView is designed
- L2070: //     // TODO: GraphView::new(&self.pool, &self.query_engine)
- L2867: // TODO: Implement true sparse matrix support if needed
- L2929: // TODO: Implement transition matrix transformation

## src/errors.rs
- L537: // TODO: Add cases for other error types
- L574: // TODO: Add suggestions for other error types
- L748: // TODO: Add display implementations for all error variants
- L756: // TODO: For errors that wrap other errors (like IoError), return the underlying error
- L801: }; // TODO: Add macros for other common error types

## src/lib.rs
- L96: //! // Advanced querying (TODO: Update when NodeFilter API is finalized)
- L99: //! // Time travel - view the graph at any point in history (TODO: Implement)
- L103: //! // Merge branches with conflict resolution (TODO: Implement)
- L108: //! ADVANCED FEATURES (TODO: Update when query API is implemented):
- L281: // TODO: Re-enable when view_at_state is implemented
- L289: // TODO: Re-enable when current_branch field is available in statistics
- L302: // TODO: Re-enable when tag functionality is implemented
- L326: // TODO: Re-enable when configuration API is implemented

## src/query/traversal.rs
- L22: // use rayon::prelude::*; // TODO: Re-enable when parallel traversal is implemented
- L42: #[allow(dead_code)] // TODO: Implement configuration system
- L1167: #[allow(dead_code)] // TODO: Implement queue-based traversal

## src/state/change_tracker.rs
- L352: first_change_time: None, // TODO: Could track timestamps in strategy

## src/state/history.rs
- L313: description: None, // TODO: add descriptions to branches
- L314: created_at: 0,     // TODO: track creation time
- L315: created_by: "".to_string(), // TODO: track creator
- L317: is_current: false, // TODO: we need to track current branch in HistoryForest
- L831: let nodes_removed = Vec::new(); // TODO: when we implement node removal
- L833: let edges_removed = Vec::new(); // TODO: when we implement edge removal

## src/state/ref_manager.rs
- L596: // TODO: Uncomment when RefManager is implemented
- L610: // TODO: Test branch lifecycle
- L638: // TODO: Test branch switching
- L657: // TODO: Test tag lifecycle
- L681: // TODO: Test cleanup of invalid references
- L701: // TODO: Test error conditions

## src/state/space.rs
- L541: // TODO: Implement subgraph tracking in GraphSpace
- L786: // TODO: Graph provides change summary now - placeholder for now

## src/state/state.rs
- L713: // TODO: Complex merge algorithm
- L715: todo!("Implement merge_snapshots")
- L725: // TODO:
- L737: todo!("Implement validate_snapshot")

## src/storage/advanced_matrix/backend.rs
- L716: // TODO: Add GPU backends when available
- L936: // TODO: Implement actual benchmarking
- L971: #[ignore] // gemv method not currently implemented in NativeBackend

## src/storage/advanced_matrix/memory.rs
- L307: // TODO: Implement views for other backends
- L322: // TODO: Implement actual synchronization between backends
- L490: // TODO: Implement prefetching based on operation patterns

## src/storage/advanced_matrix/neural/convolution.rs
- L420: todo!("Direct convolution implementation")
- L430: todo!("FFT convolution implementation")
- L440: todo!("Winograd convolution implementation")

## src/storage/array/bool_array.rs
- L146: // TODO: Better error handling

## src/storage/matrix/matrix_core.rs
- L171: // TODO: Implement proper identity matrix in UnifiedMatrix
- L392: // TODO: Implement proper zero counting when UnifiedMatrix exposes element iteration
- L633: // TODO: Complete Conv2D integration - API mismatch between Conv2D (expects ConvTensor) and GraphMatrix (UnifiedMatrix)
- L715: // TODO: Implement type casting when UnifiedMatrix supports it
- L724: // TODO: Implement sparsity detection based on UnifiedMatrix data
- L1045: // TODO: Implement proper standard deviation calculation

## src/storage/pool.rs
- L68: #[allow(dead_code)] // TODO: Implement byte pool reuse
- L938: // TODO: Implement subgraph attribute storage
- L950: // TODO: Implement subgraph attribute storage
- L979: // TODO: Add memory usage, load factors, etc.

## src/storage/table/base.rs
- L3866: // TODO: Implement proper Parquet support
- L3873: // TODO: Implement proper Parquet support
- L4966: /// TODO: Implement vertical stacking/concatenation for tables
- L4969: // TODO: Allow stacking tables with different schemas by:
- L4978: "Stack requires tables with identical column schemas (TODO: support different schemas)".to_string()
- L5029: /// TODO: Implement horizontal concatenation for tables
- L5031: // TODO: Implement horizontal concatenation by:
- L5044: format!("{}_y", col_name) // TODO: Better conflict resolution
- L5068: // TODO: Proper column order management for concatenated tables

## src/storage/table/graph_table.rs
- L1366: // TODO: Add option to convert edges table or combined table
- L1539: checksums: HashMap::new(), // TODO: Implement checksums
- L1792: // TODO: Verify checksums, file integrity, etc.

## src/storage/table/integration_tests.rs
- L6: // TODO: These tests use outdated BaseArray::with_name() API that no longer exists.
- L18: #[ignore] // TODO: Update to use current BaseArray API
- L54: #[ignore] // TODO: Update to use current BaseArray API
- L103: #[ignore] // TODO: Update to use current BaseArray API
- L130: #[ignore] // TODO: Update to use current BaseArray API
- L173: #[ignore] // TODO: Update to use current BaseArray API

## src/subgraphs/composer.rs
- L656: EdgeStrategy::ContractAll => ExternalEdgeStrategy::Aggregate, // TODO: Implement contract properly

## src/subgraphs/hierarchical.rs
- L491: // TODO: Implement parent tracking in future iteration
- L511: // TODO: Implement hierarchy level calculation in future iteration
- L516: // For now, return self as root (TODO: implement proper hierarchy traversal)
- L524: // TODO: Implement sibling discovery in future iteration

## src/subgraphs/neighborhood.rs
- L66: // TODO: Extract nodes and edges from result properly

## src/subgraphs/subgraph.rs
- L98: /// TODO: Generate proper IDs through GraphPool storage
- L702: /// # let subgraph: Subgraph = todo!(); // Placeholder for actual subgraph
- L730: /// # let subgraph: Subgraph = todo!(); // Placeholder for actual subgraph
- L1368: // TODO: Create EntityNode wrappers for the nodes in this subgraph

## src/subgraphs/visualization.rs
- L96: is_directed: true, // TODO: Get from graph if available

## src/traits/graph_entity.rs
- L208: // TODO: Implement cycle detection for hierarchical structures

## src/traits/meta_operations.rs
- L87: // TODO: Implement re-aggregation logic
- L148: // TODO: Store and retrieve original edge IDs during collapse
- L161: // TODO: Implement meta-edge expansion

## src/traits/node_operations.rs
- L169: // TODO: Implement ComponentSubgraph when needed
- L282: // TODO: Implement PathSubgraph when needed

## src/traits/subgraph_operations.rs
- L959: // Create separate meta-edges for each original edge (TODO: implement properly)
- L1160: // TODO: Implement parent tracking for hierarchical subgraphs
- L1169: // TODO: Implement child tracking for hierarchical subgraphs

## src/utils/config.rs
- L187: // TODO: Initialize all other fields with balanced default values
- L192: // TODO: Initialize remaining fields with reasonable defaults
- L230: // TODO: Implement with all fields
- L247: // TODO: Implement with all fields
- L288: // TODO: Implement with all fields
- L413: // TODO: Set environment variables based on current config

## src/viz/embeddings/debug.rs
- L364: // TODO: Implement connectivity check
- L370: // TODO: Implement clustering coefficient calculation
- L405: // TODO: Implement actual memory usage measurement

## src/viz/embeddings/mod.rs
- L404: // TODO: Implement interleaving strategy
- L405: todo!("Interleave combination strategy not yet implemented")

## src/viz/embeddings/spectral.rs
- L65: let weight = 1.0; // TODO: get actual edge weight if needed
- L181: // TODO: Add connectivity check

## src/viz/mod.rs
- L19: // use ; // TODO: add missing import
- L527: // TODO: Properly extract graph structure from data_source

## src/viz/realtime/engine.rs
- L1485: // TODO: Implement quality-only recomputation
- L1494: // TODO: Implement real-time filtering
- L1503: // TODO: Implement node selection
- L1512: // TODO: Implement zoom animation
- L1517: // TODO: Implement view panning
- L1525: // TODO: Implement incremental node addition
- L1533: // TODO: Implement incremental edge addition
- L1538: // TODO: Implement incremental node removal
- L1543: // TODO: Implement view reset
- L1548: // TODO: Implement incremental update processing
- L1553: // TODO: Implement filter transition updates
- L1558: // TODO: Implement other dynamic aspect updates
- L1681: // TODO: Add node to graph and update positions
- L1684: // TODO: Remove node from graph and positions
- L1687: // TODO: Add edge to graph
- L1690: // TODO: Remove edge from graph
- L1696: // TODO: Update node attributes in graph
- L1702: // TODO: Update edge attributes in graph

## src/viz/realtime/server/realtime_server.rs
- L115: let temp_graph = crate::api::graph::Graph::new(); // TODO: Convert snapshot to proper graph

## src/viz/streaming/mod.rs
- L16: // pub use ; // TODO: add missing export

## web/app.js
- L560: // TODO: apply nodes_changed / edges_* when UI requires it

## web/index.html
- L90: <!-- TODO(flat-embedding): add toggle + sliders for the flat embedding energy once implemented. -->
