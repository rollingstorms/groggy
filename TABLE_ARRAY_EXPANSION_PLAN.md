# TableArray Method Expansion Plan

## Overview

This document outlines the plan to expand the `TableArray` class with additional methods to provide comprehensive functionality for working with arrays of tables. TableArray is created from operations like `table.group_by()` and represents a collection of related tables.

## Current State

### Existing Methods (3 total)
- `agg(agg_specs: Dict[str, str]) -> BaseTable` - Aggregate across all tables with column-wise functions
- `iter_agg(agg_specs: Dict[str, str]) -> BaseTable` - Iterator-style aggregation (delegates to agg)
- `to_list() -> List[BaseTable]` - Convert to Python list of tables

## Proposed Methods by Category

### 🔥 High Priority - Per-Table Operations (Return TableArray)

These methods apply the same operation to each table in the array and return a new TableArray.

| Method | Signature | Return Type | Description | Example |
|--------|-----------|-------------|-------------|---------|
| `sample` | `sample(n: int) -> TableArray` | TableArray | Sample n rows from each table | `groups.sample(10)` |
| `head` | `head(n: int = 5) -> TableArray` | TableArray | Get first n rows from each table | `groups.head(5)` |
| `tail` | `tail(n: int = 5) -> TableArray` | TableArray | Get last n rows from each table | `groups.tail(3)` |
| `filter` | `filter(predicate: Callable) -> TableArray` | TableArray | Filter each table with same predicate | `groups.filter(lambda row: row['age'] > 25)` |
| `select` | `select(columns: List[str]) -> TableArray` | TableArray | Select columns from each table | `groups.select(['name', 'age'])` |
| `sort_by` | `sort_by(column: str, ascending: bool = True) -> TableArray` | TableArray | Sort each table by column | `groups.sort_by('score', False)` |
| `drop_columns` | `drop_columns(columns: List[str]) -> TableArray` | TableArray | Drop columns from each table | `groups.drop_columns(['temp_col'])` |
| `rename` | `rename(mapping: Dict[str, str]) -> TableArray` | TableArray | Rename columns in each table | `groups.rename({'old_name': 'new_name'})` |
| `apply` | `apply(func: Callable) -> TableArray` | TableArray | Apply function to each table (returns tables) | `groups.apply(lambda t: t.fillna(0))` |
| `apply_to_list` | `apply_to_list(func: Callable) -> List[Any]` | List | Apply function and return list of results | `groups.apply_to_list(lambda t: t.nrows)` |
| `apply_to_array` | `apply_to_array(func: Callable) -> BaseArray` | BaseArray | Apply function and return array of results | `groups.apply_to_array(lambda t: t.mean('age'))` |
| `apply_reduce` | `apply_reduce(func: Callable, reduce_func: Callable, init: Any) -> Any` | Any | Apply function then reduce to single value | `groups.apply_reduce(lambda t: t.nrows, lambda a,b: a+b, 0)` |

### 📊 Array Information Methods (Return Values)

These methods provide information about the TableArray itself.

| Method | Signature | Return Type | Description | Example |
|--------|-----------|-------------|-------------|---------|
| `len` | `len() -> int` | int | Number of tables in array | `len(groups)` |
| `__len__` | `__len__() -> int` | int | Python built-in len support | `len(groups)` |
| `count` | `count() -> int` | int | Total rows across all tables | `groups.count()` |
| `shape` | `shape() -> Tuple[int, int, int]` | Tuple | (num_tables, total_rows, num_cols) | `groups.shape()` |
| `describe` | `describe() -> BaseTable` | BaseTable | Statistics for each table | `groups.describe()` |

### 🎯 Array Aggregation Methods (Return BaseTable)

These methods combine data across all tables in the array.

| Method | Signature | Return Type | Description | Example |
|--------|-----------|-------------|-------------|---------|
| `concat` | `concat() -> BaseTable` | BaseTable | Concatenate all tables into one | `groups.concat()` |
| `union` | `union() -> BaseTable` | BaseTable | Union all tables (remove duplicates) | `groups.union()` |
| `merge_all` | `merge_all(on: str, how: str = 'inner') -> BaseTable` | BaseTable | Merge all tables on common column | `groups.merge_all('id')` |
| `value_counts` | `value_counts(column: str) -> BaseTable` | BaseTable | Value counts across all tables | `groups.value_counts('category')` |
| `unique` | `unique(column: str) -> BaseArray` | BaseArray | Unique values across all tables | `groups.unique('status')` |

### 📈 Statistical Aggregation (Return Values)

These methods compute statistics across all tables.

| Method | Signature | Return Type | Description | Example |
|--------|-----------|-------------|-------------|---------|
| `sum` | `sum(column: str) -> float` | float | Sum column across all tables | `groups.sum('revenue')` |
| `mean` | `mean(column: str) -> float` | float | Mean of column across all tables | `groups.mean('score')` |
| `min` | `min(column: str) -> AttrValue` | AttrValue | Min value across all tables | `groups.min('date')` |
| `max` | `max(column: str) -> AttrValue` | AttrValue | Max value across all tables | `groups.max('price')` |
| `std` | `std(column: str) -> float` | float | Standard deviation across all tables | `groups.std('variance')` |

### 🔄 Advanced Array Operations

| Method | Signature | Return Type | Description | Example |
|--------|-----------|-------------|-------------|---------|
| `map` | `map(func: Callable) -> TableArray` | TableArray | Apply function to each table | `groups.map(lambda t: t.head(10))` |
| `reduce` | `reduce(func: Callable, initial=None) -> Any` | Any | Reduce tables to single value | `groups.reduce(lambda a, b: a + b.nrows, 0)` |
| `zip` | `zip(other: TableArray) -> TableArray` | TableArray | Zip with another TableArray | `groups.zip(other_groups)` |
| `flatten` | `flatten() -> BaseTable` | BaseTable | Alias for concat() | `groups.flatten()` |

### 💾 I/O and Conversion Methods

| Method | Signature | Return Type | Description | Example |
|--------|-----------|-------------|-------------|---------|
| `to_csv` | `to_csv(prefix: str, path: str = '.') -> None` | None | Export each table to CSV | `groups.to_csv('group_')` |
| `to_pandas` | `to_pandas() -> List[pd.DataFrame]` | List[DataFrame] | Convert to pandas DataFrames | `groups.to_pandas()` |

## Implementation Priority

### Phase 1: Essential Operations (High Value, Low Complexity)
1. `len()` / `__len__()` - Basic array info
2. `head()` / `tail()` - Quick inspection
3. `sample()` - Sampling from groups
4. `select()` - Column selection
5. `concat()` - Recombine groups

### Phase 2: Core Data Operations
6. `filter()` - Data filtering
7. `sort_by()` - Sorting within groups
8. `count()` - Row counting
9. `describe()` - Quick statistics
10. `apply()` - Functional operations

### Phase 3: Advanced Features
11. `map()` - Functional programming
12. `value_counts()` - Cross-group analysis
13. `union()` - Deduplication
14. Statistical methods (`sum`, `mean`, `min`, `max`, `std`)

### Phase 4: Specialized Operations
15. `merge_all()` - Complex joining
16. `zip()` - Array combination
17. I/O methods (`to_csv`, `to_pandas`)
18. `reduce()` - Custom aggregation

## Implementation Notes

### Return Type Patterns
- **TableArray → TableArray**: Operations that transform each table independently
- **TableArray → BaseTable**: Operations that aggregate/combine all tables
- **TableArray → Scalar**: Operations that compute single values across all tables
- **TableArray → BaseArray**: Operations that extract arrays of values

### Error Handling
- Empty TableArray should handle gracefully (return empty results or raise clear errors)
- Column validation should check all tables in array
- Type mismatches should provide clear error messages

### Performance Considerations
- Operations should be applied efficiently across all tables
- Large TableArrays should not cause memory issues
- Consider lazy evaluation for chained operations

### Consistency with Existing API
- Method signatures should match BaseTable/BaseArray equivalents where possible
- Parameter defaults should be consistent
- Error messages should follow existing patterns

## Example Usage Patterns

```python
# Group and analyze
groups = table.group_by('department')

# Quick inspection
print(f"Number of departments: {len(groups)}")
top_employees = groups.head(5)  # Top 5 in each department

# Filter and sample
active_groups = groups.filter(lambda t: t['active'].all())
samples = active_groups.sample(10)

# Statistical analysis
dept_stats = groups.describe()
total_salary = groups.sum('salary')
avg_score = groups.mean('performance_score')

# Recombine data
all_top_performers = groups.filter(lambda t: t['score'] > 90).concat()

# Functional operations with flexible return types
normalized = groups.apply(lambda t: t.apply(lambda col: col / col.max()))  # TableArray of normalized tables
row_counts = groups.apply_to_list(lambda t: t.nrows)  # List of row counts [10, 15, 8]
mean_ages = groups.apply_to_array(lambda t: t.mean('age'))  # BaseArray of mean ages
total_rows = groups.apply_reduce(lambda t: t.nrows, lambda a,b: a+b, 0)  # Single total: 33
```

## Success Metrics

- **API Completeness**: TableArray supports 80% of common table operations
- **Performance**: Operations scale linearly with number of tables
- **Usability**: Intuitive method signatures matching pandas/polars patterns
- **Documentation**: All methods have clear examples and docstrings