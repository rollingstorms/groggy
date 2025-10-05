# Groggy Table Operations - Implementation Roadmap

## Executive Summary

Groggy currently has **~15% of expected table operations** implemented. This roadmap prioritizes the **85% missing functionality** by user impact and implementation complexity.

**Current Status:**
- ✅ Basic table structure (head, tail, shape, columns)
- ✅ Simple filtering and selection
- ✅ Basic I/O (CSV, Parquet, JSON)
- ✅ Foundation for aggregation and joins
- ❌ **Missing: Core statistics, missing value handling, string ops, reshaping**

---

## **PHASE 1: IMMEDIATE WINS** 🔥
*Target: 2-3 weeks, High user impact, Low-medium complexity*

### 1.1 Core Column Statistics (CRITICAL)
**User Pain:** Cannot compute basic stats like `table['sales'].sum()`

**Implementation needed:**
```rust
// In BaseArray<AttrValue>
impl BaseArray<AttrValue> {
    pub fn sum(&self) -> GraphResult<AttrValue>
    pub fn mean(&self) -> GraphResult<f64>
    pub fn min(&self) -> GraphResult<AttrValue>
    pub fn max(&self) -> GraphResult<AttrValue>
    pub fn count(&self) -> usize
    pub fn nunique(&self) -> usize
}
```

**Python API:**
```python
table['sales'].sum()      # → AttrValue::Int(15000)
table['price'].mean()     # → 45.67
table['name'].nunique()   # → 120
```

**Effort:** Medium (requires type-aware aggregation)

### 1.2 Table.describe() Method
**User Need:** Quick statistical overview

```python
table.describe()
# →        count    mean     std     min     max
#   sales   1000   150.5   45.2    10     500
#   price   1000    25.3    8.1     5      80
```

**Implementation:** Combine column stats into summary table
**Effort:** Low (builds on 1.1)

### 1.3 Missing Value Detection
**User Pain:** No way to handle missing/null values

```python
table.isna()           # → Boolean table
table['col'].notna()   # → Boolean array
table.dropna()         # → Table without null rows
```

**Effort:** Medium (requires null handling throughout type system)

### 1.4 Random Sampling
**User Need:** Quick data exploration

```python
table.sample(100)      # → 100 random rows
table.sample(frac=0.1) # → 10% of data
```

**Effort:** Low (straightforward random selection)

---

## **PHASE 2: DATA MANIPULATION ESSENTIALS** 🎯
*Target: 3-4 weeks, Essential for data workflows*

### 2.1 Enhanced GroupBy API
**Current:** Only `group_by_agg(groups, specs)`
**Needed:** Fluent pandas-style API

```python
# Current (clunky):
table.group_by_agg(['category'], {'sales': 'sum', 'count': 'count'})

# Needed (fluent):
table.groupby('category').sales.sum()
table.groupby('category').agg({'sales': 'sum', 'price': 'mean'})
table.groupby(['region', 'category']).sum()
```

**Implementation:**
- Create `GroupBy` intermediate object
- Chain statistics methods
- Delegate to existing `group_by_agg`

**Effort:** Medium-High (API design + delegation)

### 2.2 Column Renaming & Management
**User Pain:** Cannot rename columns

```python
table.rename({'old_name': 'new_name'})
table.rename(str.upper)  # Apply function to all column names
table.add_prefix('sales_')
table.reorder(['name', 'age', 'salary'])
```

**Effort:** Low-Medium (column metadata manipulation)

### 2.3 Row Operations
**User Need:** Add/remove rows

```python
table.append({'name': 'John', 'age': 30})  # Add single row
table.extend(other_table)                   # Add multiple rows
table.drop([0, 5, 10])                     # Remove specific rows
table.drop_duplicates()                     # Remove duplicate rows
```

**Effort:** Medium (requires row manipulation in storage layer)

### 2.4 String Operations Foundation
**User Pain:** Cannot process text data

```python
table['name'].str.upper()
table['email'].str.contains('@gmail.com')
table['phone'].str.replace('-', '')
table['description'].str.len()
```

**Implementation:** Add `.str` accessor returning `StringAccessor`
**Effort:** Medium (new accessor pattern + string operations)

---

## **PHASE 3: ADVANCED QUERIES & FILTERING** 📊
*Target: 2-3 weeks, Improves user experience*

### 3.1 Query Language
**User Need:** SQL-like filtering

```python
table.query("age > 25 and salary < 100000")
table.query("name.str.startswith('A') and active == True")
table.where(table['sales'] > table['sales'].mean())
```

**Implementation:**
- Extend existing predicate evaluation
- Add support for column references and string methods

**Effort:** High (expression parsing and evaluation)

### 3.2 Advanced Filtering
```python
table.between('age', 18, 65)
table.isin('category', ['A', 'B', 'C'])
table.nlargest(10, 'salary')
table.nsmallest(5, 'age')
```

**Effort:** Low-Medium (specific filter implementations)

### 3.3 Multi-Column Sorting
**Current:** Only single column sorting
**Needed:** Multi-column with mixed order

```python
table.sort_values(['region', 'sales'], ascending=[True, False])
table.sort_values('sales').head(10)  # Top 10 by sales
```

**Effort:** Medium (enhance existing sort implementation)

---

## **PHASE 4: DATA RESHAPING** 🔄
*Target: 4-5 weeks, Advanced analytics*

### 4.1 Pivot Operations
**User Need:** Reshape data for analysis

```python
table.pivot(index='region', columns='month', values='sales')
table.pivot_table(index='region', columns='category',
                  values='sales', aggfunc='sum')
```

**Implementation:** Complex reshaping with aggregation
**Effort:** High (new data structure transformations)

### 4.2 Melt Operations
```python
table.melt(id_vars=['name'], value_vars=['jan_sales', 'feb_sales'],
           var_name='month', value_name='sales')
```

**Effort:** Medium-High (inverse of pivot)

### 4.3 Advanced Joins
**Current:** Basic inner/left join
**Needed:** Complete join functionality

```python
table.join(other, on=['key1', 'key2'], how='outer', suffixes=['_x', '_y'])
table.cross_join(other)  # Cartesian product
```

**Effort:** Medium (extend existing join infrastructure)

---

## **PHASE 5: ANALYTICS & QUALITY** 🏗️
*Target: 3-4 weeks, Production readiness*

### 5.1 Advanced Statistics
```python
table['sales'].quantile([0.25, 0.5, 0.75])
table.corr()  # Correlation matrix
table['price'].rolling(window=7).mean()  # Rolling average
```

**Effort:** Medium-High (statistical algorithms)

### 5.2 Data Quality Tools
```python
table.profile()                    # Data profiling report
table.validate_schema(schema)      # Schema validation
table.check_outliers(method='iqr') # Outlier detection
```

**Effort:** Medium (analysis and reporting tools)

### 5.3 Performance Optimizations
- Lazy evaluation for chained operations
- Vectorized operations
- Memory-efficient algorithms
- Parallel processing for large datasets

**Effort:** High (performance engineering)

---

## **IMPLEMENTATION PRIORITIES BY USER IMPACT**

### 🔥 **Critical (Start Immediately)**
1. **Column statistics** (`sum`, `mean`, `count`) - Daily user need
2. **Missing value handling** (`isna`, `dropna`) - Data cleaning essential
3. **`table.describe()`** - Data exploration standard
4. **Random sampling** - Quick exploration

### 🎯 **High Priority (Next Sprint)**
1. **Enhanced GroupBy API** - Core analytics workflow
2. **Column renaming** - Basic data manipulation
3. **String operations** - Text processing capability
4. **Row insertion/deletion** - Data modification

### 📊 **Medium Priority (Following Month)**
1. **Query language** - Advanced filtering
2. **Multi-column sorting** - Better data organization
3. **Drop duplicates** - Data cleaning
4. **Advanced filtering** (`isin`, `between`)

### 🔄 **Lower Priority (Future Releases)**
1. **Pivot/melt operations** - Advanced reshaping
2. **Advanced joins** - Complex data merging
3. **Time series operations** - Specialized analytics
4. **Data quality tools** - Production features

---

## **ARCHITECTURAL CONSIDERATIONS**

### Core Implementation Strategy
1. **Rust Core First:** Implement operations in `src/storage/table/base.rs`
2. **FFI Layer:** Expose through `python-groggy/src/ffi/storage/table.rs`
3. **Python API:** Add convenience methods and accessors
4. **Type Safety:** Maintain strong typing throughout

### Performance Requirements
- **Column statistics:** O(n) single pass
- **GroupBy operations:** O(n log n) with sorting
- **Joins:** O(n + m) for hash joins
- **Memory usage:** Linear with data size

### API Design Principles
1. **Pandas compatibility** where possible
2. **Method chaining** for fluent workflows
3. **Type preservation** through operations
4. **Error handling** with clear messages

---

## **EFFORT ESTIMATION SUMMARY**

| Phase | Duration | Operations Count | User Impact | Complexity |
|-------|----------|------------------|-------------|------------|
| Phase 1 | 2-3 weeks | 15 operations | Critical | Low-Medium |
| Phase 2 | 3-4 weeks | 20 operations | High | Medium-High |
| Phase 3 | 2-3 weeks | 12 operations | Medium | Medium-High |
| Phase 4 | 4-5 weeks | 8 operations | Medium | High |
| Phase 5 | 3-4 weeks | 10 operations | Low-Medium | Medium-High |

**Total:** ~65 major operations over 14-19 weeks

---

## **SUCCESS METRICS**

### Functional Completeness
- **Target:** 80% of pandas table operations by end of roadmap
- **Milestone 1:** Core statistics (Phase 1) - 40% completeness
- **Milestone 2:** GroupBy + manipulation (Phase 2) - 60% completeness

### User Experience
- **API consistency:** All operations follow same patterns
- **Performance:** No operation >2x slower than pandas equivalent
- **Documentation:** Every operation has examples and docstrings

### Quality Gates
- **Test coverage:** >95% for all new operations
- **Memory safety:** No leaks in FFI layer
- **Error handling:** Graceful failures with helpful messages

This roadmap transforms Groggy from a basic table implementation to a complete data manipulation toolkit competitive with established libraries.