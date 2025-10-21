# Comprehensive Groggy Library Test Report
*Generated: 2025-10-09 22:32:11*

## Executive Summary

- **Overall Success Rate**: 66.6%
- **Objects Tested**: 27
- **Methods Tested**: 918
- **Release Readiness**: ⚠️ Needs Work

## Object Performance

| Object | Success Rate | Status |
|--------|--------------|--------|
| DisplayConfig | 100.0% | 🟢 |
| GroggyError | 100.0% | 🟢 |
| InvalidInputError | 100.0% | 🟢 |
| ComponentsArray | 88.9% | 🟡 |
| NumArray_from_builder | 88.2% | 🟡 |
| EdgesAccessor | 87.5% | 🟡 |
| NumArray | 82.4% | 🟡 |
| NodesAccessor | 80.0% | 🟡 |
| EdgesArray | 80.0% | 🟡 |
| SubgraphArray | 78.6% | 🟡 |
| EdgeId | 77.8% | 🟡 |
| NodeId | 77.8% | 🟡 |
| StateId | 77.8% | 🟡 |
| AttrName | 76.6% | 🟡 |
| BranchName | 76.6% | 🟡 |
| Graph | 76.6% | 🟡 |
| GraphMatrix_from_builder | 74.2% | 🟡 |
| BaseArray_from_builder | 71.9% | 🟡 |
| NodesArray | 69.2% | 🔴 |
| BaseTable_from_builder | 66.3% | 🔴 |
| EdgesTable | 64.9% | 🔴 |
| NodesTable | 63.6% | 🔴 |
| TableArray | 62.5% | 🔴 |
| GraphTable | 59.1% | 🔴 |
| GraphMatrix | 53.8% | 🔴 |
| Subgraph | 53.4% | 🔴 |
| BaseTable | 40.6% | 🔴 |

## Detailed Results

### Graph (76.6%)

**✅ Working methods (49)**: add_edge, add_node, add_nodes, aggregate, all_edge_attribute_names, all_node_attribute_names, bfs, branches, commit, commit_history, contains_edge, contains_node, create_branch, density, dfs, edge_attribute_keys, edge_count, edge_endpoints, edge_ids, edges, filter_edges, filter_nodes, get_edge_attr, get_node_attr, get_node_mapping, group_by, group_nodes_by_attribute, has_edge, has_edge_attribute, has_node, has_node_attribute, has_uncommitted_changes, is_connected, is_directed, is_empty, is_undirected, laplacian_matrix, neighborhood, node_attribute_keys, node_count, node_ids, nodes, remove_edge, remove_node, shortest_path, table, to_matrix, to_networkx, view

**❌ Failed methods (15)**: add_edges, add_graph, checkout_branch, get_edge_attrs, get_node_attrs, historical_view, neighbors, remove_edges, remove_nodes, resolve_string_id_to_node, set_edge_attr, set_edge_attrs, set_node_attr, set_node_attrs, transition_matrix

### NodesAccessor (80.0%)

**✅ Working methods (12)**: all, array, attribute_names, attributes, base, group_by, ids, matrix, meta, subgraphs, table, viz

**❌ Failed methods (3)**: filter, get_meta_node, set_attrs

### EdgesAccessor (87.5%)

**✅ Working methods (14)**: all, array, attribute_names, attributes, base, group_by, ids, matrix, meta, sources, table, targets, viz, weight_matrix

**❌ Failed methods (2)**: filter, set_attrs

### GraphTable (59.1%)

**✅ Working methods (13)**: auto_assign_edge_ids, edges, head, is_empty, ncols, nodes, nrows, shape, stats, tail, to_graph, validate, viz

**❌ Failed methods (9)**: from_federated_bundles, get_bundle_info, load_bundle, merge, save_bundle, to_edges, to_nodes, to_subgraphs, verify_bundle

### NodesTable (63.6%)

**✅ Working methods (21)**: base_table, drop_columns, group_by, head, interactive, interactive_embed, interactive_viz, into_base_table, is_empty, iter, ncols, node_ids, nrows, rich_display, select, shape, sort_by, sort_values, tail, to_pandas, viz

**❌ Failed methods (12)**: filter, filter_by_attr, from_csv, from_dict, from_json, from_parquet, slice, to_csv, to_json, to_parquet, unique_attr_values, with_attributes

### EdgesTable (64.9%)

**✅ Working methods (24)**: as_tuples, auto_assign_edge_ids, base_table, drop_columns, edge_ids, group_by, head, interactive, interactive_embed, interactive_viz, into_base_table, iter, ncols, nrows, rich_display, select, shape, sort_by, sort_values, sources, tail, targets, to_pandas, viz

**❌ Failed methods (13)**: filter, filter_by_attr, filter_by_sources, filter_by_targets, from_csv, from_dict, from_json, from_parquet, slice, to_csv, to_json, to_parquet, unique_attr_values

### NodesArray (69.2%)

**✅ Working methods (9)**: first, is_empty, iter, last, stats, table, to_list, total_node_count, union

**❌ Failed methods (4)**: contains, filter, filter_by_size, interactive

### EdgesArray (80.0%)

**✅ Working methods (12)**: filter_by_size, filter_by_weight, first, is_empty, iter, last, nodes, stats, table, to_list, total_edge_count, union

**❌ Failed methods (3)**: contains, filter, interactive

### NumArray (82.4%)

**✅ Working methods (14)**: count, dtype, first, is_empty, last, max, mean, min, nunique, std, sum, to_list, unique, var

**❌ Failed methods (3)**: contains, reshape, to_type

### SubgraphArray (78.6%)

**✅ Working methods (11)**: collapse, collect, edges_table, is_empty, merge, nodes_table, sample, summary, table, to_list, viz

**❌ Failed methods (3)**: extract_node_attribute, group_by, map

### TableArray (62.5%)

**✅ Working methods (5)**: collect, filter, is_empty, iter, to_list

**❌ Failed methods (3)**: agg, extract_column, map

### Subgraph (53.4%)

**✅ Working methods (31)**: adjacency_list, child_meta_nodes, collapse, connected_components, degree, density, edge_count, edge_ids, edges, edges_table, entity_type, has_meta_nodes, hierarchy_level, in_degree, is_connected, is_empty, meta_nodes, neighborhood, node_count, node_ids, nodes, out_degree, parent_meta_node, sample, summary, table, to_edges, to_graph, to_matrix, to_nodes, viz

**❌ Failed methods (27)**: bfs, calculate_similarity, clustering_coefficient, contains_edge, contains_node, dfs, edge_endpoints, filter_edges, filter_nodes, get_edge_attribute, get_node_attribute, group_by, has_edge, has_edge_between, has_node, has_path, induced_subgraph, intersect_with, merge_with, neighbors, set_edge_attrs, set_node_attrs, shortest_path_subgraph, subgraph_from_edges, subtract_from, to_networkx, transitivity

### ComponentsArray (88.9%)

**✅ Working methods (8)**: collapse, largest_component, neighborhood, sample, sizes, table, to_list, viz

**❌ Failed methods (1)**: filter

### GraphMatrix (53.8%)

**✅ Working methods (50)**: abs, apply, columns, data, dense, dense_html_repr, dtype, elu, exp, flatten, gelu, grad, identity, is_empty, is_numeric, is_sparse, is_square, is_symmetric, iter_columns, iter_rows, leaky_relu, log, map, max, mean, min, norm, norm_inf, norm_l1, preview, qr_decomposition, rank, relu, requires_grad, rich_display, shape, sigmoid, softmax, sqrt, sum, summary, svd, tanh, to_base_array, to_dict, to_list, to_numpy, to_pandas, to_table_for_streaming, transpose

**❌ Failed methods (43)**: backward, cholesky_decomposition, concatenate, determinant, dropout, eigenvalue_decomposition, elementwise_multiply, filter, from_base_array, from_data, from_flattened, from_graph_attributes, get, get_cell, get_column, get_column_by_name, get_row, inverse, lu_decomposition, max_axis, mean_axis, min_axis, multiply, ones, power, repeat, requires_grad_, reshape, scalar_multiply, set, solve, split, stack, std_axis, sum_axis, tile, to_degree_matrix, to_laplacian, to_normalized_laplacian, trace, var_axis, zero_grad, zeros

### BaseArray_from_builder (71.9%)

**✅ Working methods (46)**: append, append_element, contains, count, cummax, cummin, cumsum, describe, drop_duplicates_elements, drop_elements, dropna, dtype, extend, extend_elements, fillna, get, get_percentile, has_nulls, head, infer_numeric_type, is_empty, is_numeric, isna, iter, len, notna, null_count, numeric_compatibility_info, nunique, pct_change, percentile, percentiles, quantile, quantiles, remove, reverse, shift, sort, tail, to_list, to_table, to_table_with_name, to_table_with_prefix, to_table_with_suffix, unique, value_counts

**❌ Failed methods (18)**: apply, apply_to_each, corr, cov, expanding, filter, insert, map, max, mean, median, min, rolling, std, sum, to_num_array, to_type, var

### NumArray_from_builder (88.2%)

**✅ Working methods (15)**: contains, count, dtype, first, is_empty, last, max, mean, min, nunique, std, sum, to_list, unique, var

**❌ Failed methods (2)**: reshape, to_type

### BaseTable_from_builder (66.3%)

**✅ Working methods (67)**: add_prefix, add_suffix, agg, aggregate, append, apply, apply_to_columns, apply_to_rows, assign, check_outliers, column, column_info, column_names, columns, corr, cov, cummax, cummin, cumsum, describe, drop_columns, drop_duplicates, drop_rows, dropna, dropna_subset, extend, extend_rows, fillna, fillna_all, from_dict, get_column_numeric, get_column_raw, group_by, groupby, groupby_single, has_column, has_nulls, head, is_empty, isna, iter, median, melt, ncols, notna, nrows, null_counts, parse_join_on, pct_change, profile, rename, rich_display, select, shape, sort_by, sort_values, std, tail, to_csv, to_json, to_nodes_table, to_pandas, to_parquet, to_type, validate_schema, value_counts, var

**❌ Failed methods (34)**: append_row, corr_columns, cov_columns, expanding, expanding_all, filter, from_csv, from_json, from_parquet, get_percentile, group_by_agg, intersect, isin, join, nlargest, nsmallest, percentile, percentiles, pivot_table, quantile, quantiles, query, reorder_columns, rolling, rolling_all, sample, set_column, set_value, set_values_by_mask, set_values_by_range, shift, slice, to_edges_table, union

### GraphMatrix_from_builder (74.2%)

**✅ Working methods (69)**: abs, columns, data, dense, dense_html_repr, determinant, dropout, dtype, eigenvalue_decomposition, elu, exp, flatten, from_data, gelu, get_column, get_row, grad, identity, inverse, is_empty, is_numeric, is_sparse, is_square, is_symmetric, iter_columns, iter_rows, leaky_relu, log, lu_decomposition, max, max_axis, mean, mean_axis, min, min_axis, multiply, norm, norm_inf, norm_l1, power, preview, qr_decomposition, rank, relu, requires_grad, requires_grad_, rich_display, shape, sigmoid, softmax, sqrt, std_axis, sum, sum_axis, summary, svd, tanh, to_base_array, to_degree_matrix, to_dict, to_laplacian, to_list, to_normalized_laplacian, to_numpy, to_pandas, to_table_for_streaming, trace, transpose, var_axis

**❌ Failed methods (24)**: apply, backward, cholesky_decomposition, concatenate, elementwise_multiply, filter, from_base_array, from_flattened, from_graph_attributes, get, get_cell, get_column_by_name, map, ones, repeat, reshape, scalar_multiply, set, solve, split, stack, tile, zero_grad, zeros

### AttrName (76.6%)

**✅ Working methods (36)**: capitalize, casefold, count, encode, endswith, expandtabs, find, format, format_map, index, isalnum, isalpha, isascii, isdecimal, isdigit, isidentifier, islower, isnumeric, isprintable, isspace, istitle, isupper, lower, lstrip, maketrans, rfind, rindex, rsplit, rstrip, split, splitlines, startswith, strip, swapcase, title, upper

**❌ Failed methods (11)**: center, join, ljust, partition, removeprefix, removesuffix, replace, rjust, rpartition, translate, zfill

### BaseTable (40.6%)

**✅ Working methods (41)**: apply, apply_to_columns, apply_to_rows, column, column_info, column_names, columns, corr, cov, cummax, cummin, cumsum, describe, drop_columns, drop_duplicates, dropna, get_column_raw, group_by, groupby_single, has_column, has_nulls, head, is_empty, isna, iter, ncols, nlargest, notna, nrows, nsmallest, null_counts, pct_change, profile, rich_display, select, shape, sort_by, sort_values, tail, to_pandas, value_counts

**❌ Failed methods (60)**: add_prefix, add_suffix, agg, aggregate, append, append_row, assign, check_outliers, corr_columns, cov_columns, drop_rows, dropna_subset, expanding, expanding_all, extend, extend_rows, fillna, fillna_all, filter, from_csv, from_dict, from_json, from_parquet, get_column_numeric, get_percentile, group_by_agg, groupby, intersect, isin, join, median, melt, parse_join_on, percentile, percentiles, pivot_table, quantile, quantiles, query, rename, reorder_columns, rolling, rolling_all, sample, set_column, set_value, set_values_by_mask, set_values_by_range, shift, slice, std, to_csv, to_edges_table, to_json, to_nodes_table, to_parquet, to_type, union, validate_schema, var

### BranchName (76.6%)

**✅ Working methods (36)**: capitalize, casefold, count, encode, endswith, expandtabs, find, format, format_map, index, isalnum, isalpha, isascii, isdecimal, isdigit, isidentifier, islower, isnumeric, isprintable, isspace, istitle, isupper, lower, lstrip, maketrans, rfind, rindex, rsplit, rstrip, split, splitlines, startswith, strip, swapcase, title, upper

**❌ Failed methods (11)**: center, join, ljust, partition, removeprefix, removesuffix, replace, rjust, rpartition, translate, zfill

### DisplayConfig (100.0%)

**✅ Working methods (3)**: default, max_cols, max_rows

### EdgeId (77.8%)

**✅ Working methods (7)**: as_integer_ratio, bit_length, conjugate, denominator, imag, numerator, real

**❌ Failed methods (2)**: from_bytes, to_bytes

### GroggyError (100.0%)

**✅ Working methods (2)**: args, with_traceback

### InvalidInputError (100.0%)

**✅ Working methods (2)**: args, with_traceback

### NodeId (77.8%)

**✅ Working methods (7)**: as_integer_ratio, bit_length, conjugate, denominator, imag, numerator, real

**❌ Failed methods (2)**: from_bytes, to_bytes

### StateId (77.8%)

**✅ Working methods (7)**: as_integer_ratio, bit_length, conjugate, denominator, imag, numerator, real

**❌ Failed methods (2)**: from_bytes, to_bytes

