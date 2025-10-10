# BaseTable API Reference

**Type**: `groggy.BaseTable`

---

## Overview

Base class for tabular data operations shared by NodesTable and EdgesTable.

**Primary Use Cases:**
- Generic table operations
- Column-based data access
- Aggregations and transformations

**Related Objects:**
- `NodesTable`
- `EdgesTable`
- `BaseArray`

---

## Complete Method Reference

The following methods are available on `BaseTable` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `add_prefix()` | `?` | ✗ |
| `add_suffix()` | `?` | ✗ |
| `agg()` | `?` | ✗ |
| `aggregate()` | `?` | ✗ |
| `append()` | `?` | ✗ |
| `append_row()` | `?` | ✗ |
| `apply()` | `BaseTable` | ✓ |
| `apply_to_columns()` | `BaseTable` | ✓ |
| `apply_to_rows()` | `BaseTable` | ✓ |
| `assign()` | `?` | ✗ |
| `check_outliers()` | `?` | ✗ |
| `column()` | `BaseArray` | ✓ |
| `column_info()` | `dict` | ✓ |
| `column_names()` | `list` | ✓ |
| `columns()` | `list` | ✓ |
| `corr()` | `BaseTable` | ✓ |
| `corr_columns()` | `?` | ✗ |
| `cov()` | `BaseTable` | ✓ |
| `cov_columns()` | `?` | ✗ |
| `cummax()` | `BaseArray` | ✓ |
| `cummin()` | `BaseArray` | ✓ |
| `cumsum()` | `BaseArray` | ✓ |
| `describe()` | `BaseTable` | ✓ |
| `drop_columns()` | `BaseTable` | ✓ |
| `drop_duplicates()` | `BaseTable` | ✓ |
| `drop_rows()` | `?` | ✗ |
| `dropna()` | `BaseTable` | ✓ |
| `dropna_subset()` | `?` | ✗ |
| `expanding()` | `?` | ✗ |
| `expanding_all()` | `?` | ✗ |
| `extend()` | `?` | ✗ |
| `extend_rows()` | `?` | ✗ |
| `fillna()` | `?` | ✗ |
| `fillna_all()` | `?` | ✗ |
| `filter()` | `?` | ✗ |
| `from_csv()` | `?` | ✗ |
| `from_dict()` | `?` | ✗ |
| `from_json()` | `?` | ✗ |
| `from_parquet()` | `?` | ✗ |
| `get_column_numeric()` | `?` | ✗ |
| `get_column_raw()` | `BaseArray` | ✓ |
| `get_percentile()` | `?` | ✗ |
| `group_by()` | `TableArray` | ✓ |
| `group_by_agg()` | `?` | ✗ |
| `groupby()` | `?` | ✗ |
| `groupby_single()` | `TableArray` | ✓ |
| `has_column()` | `bool` | ✓ |
| `has_nulls()` | `bool` | ✓ |
| `head()` | `BaseTable` | ✓ |
| `intersect()` | `?` | ✗ |
| `is_empty()` | `bool` | ✓ |
| `isin()` | `?` | ✗ |
| `isna()` | `BaseTable` | ✓ |
| `iter()` | `BaseTableRowIterator` | ✓ |
| `join()` | `?` | ✗ |
| `median()` | `?` | ✗ |
| `melt()` | `?` | ✗ |
| `ncols()` | `int` | ✓ |
| `nlargest()` | `BaseTable` | ✓ |
| `notna()` | `BaseTable` | ✓ |
| `nrows()` | `int` | ✓ |
| `nsmallest()` | `BaseTable` | ✓ |
| `null_counts()` | `dict` | ✓ |
| `parse_join_on()` | `?` | ✗ |
| `pct_change()` | `BaseArray` | ✓ |
| `percentile()` | `?` | ✗ |
| `percentiles()` | `?` | ✗ |
| `pivot_table()` | `?` | ✗ |
| `profile()` | `BaseTable` | ✓ |
| `quantile()` | `?` | ✗ |
| `quantiles()` | `?` | ✗ |
| `query()` | `?` | ✗ |
| `rename()` | `?` | ✗ |
| `reorder_columns()` | `?` | ✗ |
| `rich_display()` | `str` | ✓ |
| `rolling()` | `?` | ✗ |
| `rolling_all()` | `?` | ✗ |
| `sample()` | `?` | ✗ |
| `select()` | `BaseTable` | ✓ |
| `set_column()` | `?` | ✗ |
| `set_value()` | `?` | ✗ |
| `set_values_by_mask()` | `?` | ✗ |
| `set_values_by_range()` | `?` | ✗ |
| `shape()` | `tuple` | ✓ |
| `shift()` | `?` | ✗ |
| `slice()` | `?` | ✗ |
| `sort_by()` | `BaseTable` | ✓ |
| `sort_values()` | `BaseTable` | ✓ |
| `std()` | `?` | ✗ |
| `tail()` | `BaseTable` | ✓ |
| `to_csv()` | `?` | ✗ |
| `to_edges_table()` | `?` | ✗ |
| `to_json()` | `?` | ✗ |
| `to_nodes_table()` | `?` | ✗ |
| `to_pandas()` | `DataFrame` | ✓ |
| `to_parquet()` | `?` | ✗ |
| `to_type()` | `?` | ✗ |
| `union()` | `?` | ✗ |
| `validate_schema()` | `?` | ✗ |
| `value_counts()` | `BaseTable` | ✓ |
| `var()` | `?` | ✗ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Method Categories

### Creation & Construction
_None in this category._


### Queries & Inspection
- **`add_prefix()`** → `?`
- **`add_suffix()`** → `?`
- **`aggregate()`** → `?`
- **`append()`** → `?`
- **`append_row()`** → `?`
- **`apply()`** → `BaseTable`
- **`apply_to_columns()`** → `BaseTable`
- **`apply_to_rows()`** → `BaseTable`
- **`assign()`** → `?`
- **`check_outliers()`** → `?`
- **`column()`** → `BaseArray`
- **`column_info()`** → `dict`
- **`column_names()`** → `list`
- **`columns()`** → `list`
- **`corr()`** → `BaseTable`
- **`corr_columns()`** → `?`
- **`cov()`** → `BaseTable`
- **`cov_columns()`** → `?`
- **`cummax()`** → `BaseArray`
- **`cummin()`** → `BaseArray`
- **`cumsum()`** → `BaseArray`
- **`describe()`** → `BaseTable`
- **`drop_columns()`** → `BaseTable`
- **`drop_duplicates()`** → `BaseTable`
- **`drop_rows()`** → `?`
- **`dropna()`** → `BaseTable`
- **`dropna_subset()`** → `?`
- **`expanding()`** → `?`
- **`expanding_all()`** → `?`
- **`extend()`** → `?`
- **`extend_rows()`** → `?`
- **`fillna()`** → `?`
- **`fillna_all()`** → `?`
- **`from_csv()`** → `?`
- **`from_dict()`** → `?`
- **`from_json()`** → `?`
- **`from_parquet()`** → `?`
- **`get_column_numeric()`** → `?`
- **`get_column_raw()`** → `BaseArray`
- **`get_percentile()`** → `?`
- **`group_by_agg()`** → `?`
- **`groupby()`** → `?`
- **`groupby_single()`** → `TableArray`
- **`has_column()`** → `bool`
- **`has_nulls()`** → `bool`
- **`head()`** → `BaseTable`
- **`intersect()`** → `?`
- **`is_empty()`** → `bool`
- **`isin()`** → `?`
- **`isna()`** → `BaseTable`
- **`iter()`** → `BaseTableRowIterator`
- **`join()`** → `?`
- **`median()`** → `?`
- **`melt()`** → `?`
- **`ncols()`** → `int`
- **`nlargest()`** → `BaseTable`
- **`notna()`** → `BaseTable`
- **`nrows()`** → `int`
- **`nsmallest()`** → `BaseTable`
- **`null_counts()`** → `dict`
- **`parse_join_on()`** → `?`
- **`pct_change()`** → `BaseArray`
- **`percentile()`** → `?`
- **`percentiles()`** → `?`
- **`pivot_table()`** → `?`
- **`profile()`** → `BaseTable`
- **`quantile()`** → `?`
- **`quantiles()`** → `?`
- **`query()`** → `?`
- **`rename()`** → `?`
- **`reorder_columns()`** → `?`
- **`rich_display()`** → `str`
- **`rolling()`** → `?`
- **`rolling_all()`** → `?`
- **`set_column()`** → `?`
- **`set_value()`** → `?`
- **`set_values_by_mask()`** → `?`
- **`set_values_by_range()`** → `?`
- **`shape()`** → `tuple`
- **`shift()`** → `?`
- **`slice()`** → `?`
- **`sort_by()`** → `BaseTable`
- **`sort_values()`** → `BaseTable`
- **`std()`** → `?`
- **`tail()`** → `BaseTable`
- **`to_edges_table()`** → `?`
- **`to_nodes_table()`** → `?`
- **`to_type()`** → `?`
- **`union()`** → `?`
- **`validate_schema()`** → `?`
- **`value_counts()`** → `BaseTable`
- **`var()`** → `?`


### Transformations
- **`agg()`** → `?`
- **`filter()`** → `?`
- **`group_by()`** → `TableArray`
- **`sample()`** → `?`
- **`select()`** → `BaseTable`


### Algorithms
_None in this category._


### State Management
_None in this category._


### I/O & Export
- **`to_csv()`** → `?`
- **`to_json()`** → `?`
- **`to_pandas()`** → `DataFrame`
- **`to_parquet()`** → `?`


---

## Object Transformations

`BaseTable` can transform into:

- **BaseTable → BaseArray**: `table["column"]`
- **BaseTable → DataFrame**: `table.to_pandas()`
- **BaseTable → AggregationResult**: `table.agg(...)`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/tables.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How BaseTable works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains
