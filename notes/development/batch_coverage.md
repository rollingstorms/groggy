# Batch Coverage Inventory

This is a first-pass inventory of builder step types and their batch execution status.

## Batch-Supported Today

- Core arithmetic: `core.add`, `core.sub`, `core.mul`, `core.div`, `core.constant`
- Graph: `graph.neighbor_sum`, `graph.neighbor_mean`, `graph.neighbor_min`, `graph.neighbor_max`, `graph.neighbor_agg`
- Loads/stores: `init_nodes_with_index`, `attach_attr`, `alias`

## Batch-Optimized (Lowered/Fused)

- `fused_madd` (from `mul` + `add`)
- `fused_axpy` (from `mul` + `add` where one input is scalar)

## Needs BatchPlan Support (Target List)

- Core ops: `core.abs`, `core.clip`, `core.compare`, `core.where`, `core.recip`, `core.sqrt`, `core.exp`, `core.log`, `core.pow`, `core.min`, `core.max`
- Scalar ops: `core.reduce_scalar`, `core.broadcast_scalar`, `init_scalar`
- Graph ops: `core.collect_neighbor_values`, `core.mode_list`, `core.neighbor_mode_update`, `core.update_in_place`
- Other common steps: `normalize`, `normalize_sum`, `node_degree`, `graph_node_count`, `graph_edge_count`, `init_nodes`, `load_attr`, `load_edge_attr`, `map_nodes`

## Non-Batchable / Non-Computational

These steps are tied to visualization or table/array display and should remain outside loop batching.

- `reset_camera`, `set_layout`, `update_positions`, `focus_node`, `table`, `array`

## IR-Only / Internal

These appear in IR files but are not pipeline step types.

- `binary_op`, `const`, `core`, `constant`, `call`, `load_node_prop`, `store_node_prop`, `neighbor_aggregate`

## Notes

- This list is based on static scanning of step declarations in `python-groggy/python/groggy/`.
- The goal is to extend BatchPlan + BatchExecutor to cover the full Target List.
- Non-batchable items should stay correct via fallback execution.
