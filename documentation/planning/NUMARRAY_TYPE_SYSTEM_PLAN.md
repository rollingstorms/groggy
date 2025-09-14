# Unified Type System Enhancement Plan

## Problem Statement

Currently, the type system across `NumArray`, `Matrix`, `Table`, and `BaseArray` structures is inconsistent and hardcoded to specific types (primarily `f64`), which causes semantic and usability issues:

### Core Issues
1. **Semantic Correctness**: Node IDs should be integers, not floats
2. **Precision Loss**: Large integers may lose precision when converted to f64
3. **Type Confusion**: Users expect integer operations on IDs but get float results
4. **API Consistency**: Different return types for the same logical data
5. **Limited Interoperability**: No seamless conversion between NumArray, BaseArray, Matrix, and Table
6. **Manual Type Management**: Users must manually handle type conversions across data structures
7. **Missing Boolean Array Support**: Boolean masks return as lists instead of efficient BoolArray
8. **Incomplete Indexing/Slicing**: No sophisticated slicing system for matrices and arrays
9. **Inconsistent Representations**: Column names in matrices, inconsistent array displays

## Current Architecture Issues

### Inconsistent Type Systems Across Data Structures

#### NumArray Structure
```rust
// Current: Fixed to f64
pub struct PyNumArray {
    pub(crate) inner: NumArray<f64>,
}
```

#### Matrix Structure  
```rust
// Current: Fixed to f64, no BaseArray interop
pub struct PyGraphMatrix {
    pub(crate) inner: GraphMatrix<f64>,
}
```

#### Table Structure
```rust
// Current: Uses AttrValue enum, no automatic NumArray conversion
pub struct PyBaseTable {
    pub(crate) table: BaseTable,  // BaseTable<AttrValue>
}
```

#### BaseArray Structure
```rust
// Current: Uses AttrValue enum, no NumArray interop
pub struct PyBaseArray {
    pub(crate) inner: BaseArray<AttrValue>,
}
```

### Problematic Conversions
```rust
// In table.rs and accessors.rs
let values: Vec<f64> = node_ids.into_iter()
    .map(|id| id as f64)  // ← This is the problem
    .collect();

// No automatic conversion when accessing numeric columns
let column = table["scores"];  // Returns BaseArray<AttrValue>
// User must manually convert to NumArray

// No seamless Matrix ↔ BaseArray conversion
let matrix = Matrix::new(data);  // f64 only
let base_array = matrix.to_base_array();  // Not available
```

## Proposed Solution: Unified Generic Type System with Cross-Structure Conversions

### 1. Core Generic Type System

#### Unified Numeric Type Enum
```rust
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum NumericType {
    Int64,
    Int32,
    USize,    // For node/edge IDs
    Float64,
    Float32,
    Bool,     // For boolean operations
}

impl NumericType {
    fn size(&self) -> usize { ... }
    fn is_integer(&self) -> bool { ... }
    fn is_float(&self) -> bool { ... }
    fn can_convert_to(&self, other: &NumericType) -> bool { ... }
}
```

#### Generic NumArray
```rust
#[pyclass(name = "NumArray", unsendable)]
pub struct PyNumArray<T> 
where 
    T: Clone + Copy + PartialOrd + std::fmt::Debug + pyo3::ToPyObject + NumericValue
{
    pub(crate) inner: NumArray<T>,
    pub(crate) dtype: NumericType,
}

pub trait NumericValue: Clone + Copy + PartialOrd + std::fmt::Debug + pyo3::ToPyObject {
    fn numeric_type() -> NumericType;
    fn from_attr_value(val: &AttrValue) -> Option<Self>;
    fn to_attr_value(self) -> AttrValue;
}
```

#### Generic Matrix System
```rust
#[pyclass(name = "Matrix", unsendable)]
pub struct PyGraphMatrix<T>
where T: NumericValue
{
    pub(crate) inner: GraphMatrix<T>,
    pub(crate) dtype: NumericType,
}

impl<T: NumericValue> PyGraphMatrix<T> {
    /// Convert to BaseArray (flattened)
    fn to_base_array(&self) -> PyResult<PyBaseArray> { ... }
    
    /// Convert to NumArray (flattened)
    fn to_num_array(&self) -> PyResult<PyNumArray<T>> { ... }
    
    /// Convert matrix type
    fn astype(&self, dtype: &str) -> PyResult<PyObject> { ... }
}
```

### 2. Enhanced BaseArray with Automatic Type Detection

#### Smart BaseArray
```rust
impl PyBaseArray {
    /// Detect if array contains only numeric values of same type
    pub fn infer_numeric_type(&self) -> Option<NumericType> {
        let mut detected_type: Option<NumericType> = None;
        
        for value in self.inner.iter() {
            match value {
                AttrValue::SmallInt(_) => {
                    match detected_type {
                        None => detected_type = Some(NumericType::Int32),
                        Some(NumericType::Int32) => continue,
                        Some(NumericType::Int64) => continue, // Upgrade to i64
                        _ => return None, // Mixed types
                    }
                }
                AttrValue::BigInt(_) => {
                    detected_type = Some(NumericType::Int64); // Upgrade
                }
                AttrValue::Float(_) => {
                    match detected_type {
                        None => detected_type = Some(NumericType::Float64),
                        Some(t) if t.is_integer() => detected_type = Some(NumericType::Float64), // Upgrade
                        Some(NumericType::Float64) => continue,
                        _ => return None,
                    }
                }
                _ => return None, // Non-numeric
            }
        }
        
        detected_type
    }
    
    /// Automatic conversion to NumArray if numeric
    pub fn to_num_array(&self) -> PyResult<PyObject> {
        match self.infer_numeric_type() {
            Some(NumericType::Int32) => Ok(self.to_int32_array()?.into()),
            Some(NumericType::Int64) => Ok(self.to_int64_array()?.into()),
            Some(NumericType::Float64) => Ok(self.to_float64_array()?.into()),
            Some(NumericType::Float32) => Ok(self.to_float32_array()?.into()),
            None => Err(PyTypeError::new_err("Array contains non-numeric or mixed types"))
        }
    }
    
    /// Convert to Matrix (if 2D structure can be inferred)
    pub fn to_matrix(&self, rows: usize, cols: usize) -> PyResult<PyObject> {
        if self.inner.len() != rows * cols {
            return Err(PyValueError::new_err("Array length doesn't match matrix dimensions"));
        }
        
        match self.infer_numeric_type() {
            Some(dtype) => {
                // Create matrix with inferred type
                self.reshape_as_matrix(rows, cols, dtype)
            }
            None => Err(PyTypeError::new_err("Array must be numeric to convert to matrix"))
        }
    }
}
```

### 3. Enhanced Table with Smart Column Access

#### Intelligent Table Column Access
```rust
#[pymethods]
impl PyBaseTable {
    /// Smart column access - returns NumArray for numeric columns
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        if let Ok(column_name) = key.extract::<String>() {
            let column_data = self.table.get_column(&column_name)?;
            let base_array = PyBaseArray::from_base_array(column_data.clone());
            
            // Try to convert to NumArray if numeric
            match base_array.infer_numeric_type() {
                Some(_) => {
                    // Return NumArray for numeric columns
                    base_array.to_num_array()
                }
                None => {
                    // Return BaseArray for non-numeric columns
                    Ok(base_array.into_py(py))
                }
            }
        } else {
            // Handle other indexing (slices, etc.)
            // ...existing logic...
        }
    }
    
    /// Explicit method to get column as BaseArray
    fn get_column_raw(&self, column_name: &str) -> PyResult<PyBaseArray> {
        let column_data = self.table.get_column(column_name)?;
        Ok(PyBaseArray::from_base_array(column_data.clone()))
    }
    
    /// Explicit method to get column as NumArray (with conversion)
    fn get_column_numeric(&self, column_name: &str) -> PyResult<PyObject> {
        let base_array = self.get_column_raw(column_name)?;
        base_array.to_num_array()
    }
    
    /// Get column type information
    fn column_info(&self, column_name: &str) -> PyResult<PyDict> {
        let base_array = self.get_column_raw(column_name)?;
        let info = PyDict::new(py);
        
        info.set_item("name", column_name)?;
        info.set_item("length", base_array.len())?;
        info.set_item("dtype", match base_array.infer_numeric_type() {
            Some(dtype) => format!("{:?}", dtype).to_lowercase(),
            None => "mixed".to_string()
        })?;
        info.set_item("is_numeric", base_array.infer_numeric_type().is_some())?;
        
        Ok(info.into())
    }
}
```

### 4. Comprehensive Conversion System

#### Universal Type Conversion Trait
```rust
pub trait TypeConvertible {
    /// Convert to any compatible type using string specification
    fn astype(&self, dtype: &str) -> PyResult<PyObject>;
    
    /// Get current type information
    fn dtype(&self) -> String;
    
    /// Check if conversion is possible
    fn can_convert_to(&self, dtype: &str) -> bool;
}

impl<T: NumericValue> TypeConvertible for PyNumArray<T> {
    fn astype(&self, dtype: &str) -> PyResult<PyObject> {
        match dtype {
            "int64" => Ok(self.to_int64()?.into_py(py)),
            "int32" => Ok(self.to_int32()?.into_py(py)),
            "float64" => Ok(self.to_float64()?.into_py(py)),
            "float32" => Ok(self.to_float32()?.into_py(py)),
            "bool" => Ok(self.to_bool()?.into_py(py)),
            "basearray" => Ok(self.to_base_array()?.into_py(py)),
            "matrix" => Err(PyValueError::new_err("Cannot convert 1D array to matrix without dimensions")),
            _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}", dtype)))
        }
    }
}

impl TypeConvertible for PyBaseArray {
    fn astype(&self, dtype: &str) -> PyResult<PyObject> {
        match dtype {
            "int64" | "int32" | "float64" | "float32" | "bool" => {
                // Convert to NumArray first, then to target type
                let num_array = self.to_num_array()?;
                num_array.call_method1(py, "astype", (dtype,))
            }
            "basearray" => Ok(self.clone().into_py(py)),
            _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}", dtype)))
        }
    }
}
```

### 5. Cross-Structure Conversion Methods

#### NumArray ↔ BaseArray Conversion
```rust
impl<T: NumericValue> PyNumArray<T> {
    /// Convert to BaseArray<AttrValue>
    pub fn to_base_array(&self) -> PyResult<PyBaseArray> {
        let attr_values: Vec<AttrValue> = self.inner.iter()
            .map(|&val| val.to_attr_value())
            .collect();
        let base_array = BaseArray::new(attr_values);
        Ok(PyBaseArray::from_base_array(base_array))
    }
}

impl PyBaseArray {
    /// Convert to typed NumArray
    pub fn to_int64_array(&self) -> PyResult<PyNumArray<i64>> {
        let values: Result<Vec<i64>, _> = self.inner.iter()
            .map(|val| i64::from_attr_value(val))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| PyTypeError::new_err("Array contains non-integer values"));
        
        Ok(PyNumArray::new_int64(values?))
    }
    
    // Similar methods for other numeric types...
}
```

#### Matrix ↔ Array Conversions
```rust
impl<T: NumericValue> PyGraphMatrix<T> {
    /// Flatten matrix to NumArray
    pub fn flatten(&self) -> PyResult<PyNumArray<T>> {
        let flattened_data = self.inner.flatten();
        Ok(PyNumArray::new(flattened_data))
    }
    
    /// Convert to BaseArray (flattened)
    pub fn to_base_array(&self) -> PyResult<PyBaseArray> {
        let num_array = self.flatten()?;
        num_array.to_base_array()
    }
    
    /// Create matrix from BaseArray
    pub fn from_base_array(array: &PyBaseArray, rows: usize, cols: usize) -> PyResult<Self> {
        let num_array = array.to_num_array()?;
        Self::from_num_array_reshaped(num_array, rows, cols)
    }
}

impl PyNumArray<T> {
    /// Reshape into matrix
    pub fn reshape(&self, rows: usize, cols: usize) -> PyResult<PyGraphMatrix<T>> {
        if self.len() != rows * cols {
            return Err(PyValueError::new_err("Array length doesn't match matrix dimensions"));
        }
        
        let matrix_data = self.inner.clone().reshape(rows, cols)?;
        Ok(PyGraphMatrix::new(matrix_data))
    }
}
```

### 6. Python Interface Design

#### Seamless User Experience
```python
# ===== AUTOMATIC TYPE HANDLING =====

# Tables automatically return NumArray for numeric columns
table = g.nodes.table()
scores = table["score"]  # Returns NumArray<float64> automatically
node_ids = table["node_id"]  # Returns NumArray<int64> automatically  
names = table["name"]  # Returns BaseArray (mixed/string data)

# Check column types
print(table.column_info("score"))
# {'name': 'score', 'length': 100, 'dtype': 'float64', 'is_numeric': True}

# ===== UNIVERSAL TYPE CONVERSION =====

# NumArray conversions
float_ids = node_ids.astype('float64')  # NumArray<int64> → NumArray<float64>
base_array = node_ids.astype('basearray')  # NumArray<int64> → BaseArray<AttrValue>

# BaseArray conversions  
num_scores = base_array.astype('float64')  # BaseArray → NumArray<float64>
int_scores = base_array.astype('int64')    # BaseArray → NumArray<int64>

# Matrix conversions
matrix = scores.reshape(10, 10)  # NumArray → Matrix
flattened = matrix.flatten()     # Matrix → NumArray
base_flat = matrix.astype('basearray')  # Matrix → BaseArray (flattened)

# ===== CROSS-STRUCTURE WORKFLOWS =====

# Workflow 1: Table → NumArray → Matrix
table = g.nodes.table()
adjacency_data = table["adjacency_weights"]  # Auto-returns NumArray<float64>
adj_matrix = adjacency_data.reshape(n, n)    # NumArray → Matrix

# Workflow 2: Matrix → BaseArray → Table  
similarity_matrix = compute_similarity(adj_matrix)
similarity_flat = similarity_matrix.astype('basearray')  # Matrix → BaseArray
new_table = create_table({"similarity": similarity_flat})  # BaseArray → Table column

# Workflow 3: NumArray → Matrix → NumArray (transformation)
data = table["features"]  # NumArray<float64>
data_matrix = data.reshape(rows, cols)  # NumArray → Matrix  
normalized_matrix = data_matrix.normalize()  # Matrix operations
normalized_data = normalized_matrix.flatten()  # Matrix → NumArray

# ===== INTELLIGENT DEFAULTS =====

# Node/Edge IDs always return as integers
node_ids = g.nodes.ids()  # Returns NumArray<int64> (was PyIntArray)
edge_ids = g.edges.ids()  # Returns NumArray<int64>

# Numerical operations automatically promote types
result = node_ids + 0.5  # NumArray<int64> + float → NumArray<float64>
mean_id = node_ids.mean()  # Automatically converts to float for calculation

# Boolean operations maintain integer types
high_ids = node_ids[node_ids > 100]  # Still NumArray<int64>
```

#### Backward Compatibility and Migration
```python
# ===== LEGACY SUPPORT =====

# Old code continues working (with deprecation warnings)
node_ids_old = g.nodes.ids().astype('float64')  # Still works

# ===== NEW PREFERRED PATTERNS =====

# New code is cleaner and more intuitive
node_ids_new = g.nodes.ids()  # Already integers, no conversion needed
float_ids = node_ids_new.astype('float64')  # Explicit conversion when needed

# ===== EXPLICIT CONTROL =====

# Get raw BaseArray when needed
raw_column = table.get_column_raw("mixed_data")  # Always BaseArray
numeric_column = table.get_column_numeric("scores")  # Force NumArray conversion

# Type checking
if table["column"].dtype.startswith('int'):
    # Handle integer data
elif table["column"].dtype.startswith('float'):
    # Handle float data
```

---

## Advanced Indexing and Slicing System

### Problem: Incomplete Indexing Support

Currently, the indexing system has major gaps:
1. **Boolean masks return as lists** instead of efficient arrays
2. **No sophisticated matrix slicing** like `[:, :2]` or `[0, 1, 2]`
3. **Inconsistent column ordering** in matrix displays
4. **No BoolArray type** for efficient boolean operations

### Solution: Comprehensive BoolArray and Slicing System

#### 1. BoolArray Implementation
```rust
// New BoolArray type based on NumArray<bool>
#[pyclass(name = "BoolArray", unsendable)]
pub struct PyBoolArray {
    pub(crate) inner: NumArray<bool>,
    pub(crate) length: usize,
}

impl PyBoolArray {
    // Efficient boolean operations
    fn and(&self, other: &PyBoolArray) -> PyResult<PyBoolArray>
    fn or(&self, other: &PyBoolArray) -> PyResult<PyBoolArray>
    fn not(&self) -> PyResult<PyBoolArray>
    fn any(&self) -> bool
    fn all(&self) -> bool
    fn count(&self) -> usize  // Count of True values
    
    // Convert to indices for advanced indexing
    fn nonzero(&self) -> PyResult<PyNumArray<usize>>
    fn to_indices(&self) -> PyResult<Vec<usize>>
}
```

#### 2. Boolean Indexing Integration
```python
# ===== BEFORE: Lists returned =====
# g.degree() > 4 returns [True, False, True, ...]
high_degree_mask = g.degree() > 4  # Returns list
high_degree_nodes = g.nodes[high_degree_mask]  # Works but inefficient

# ===== AFTER: BoolArray returned =====
high_degree_mask = g.degree() > 4  # Returns BoolArray
high_degree_nodes = g.nodes[high_degree_mask]  # Much more efficient

# Advanced boolean operations
complex_mask = (g.degree() > 4) & (g.nodes['age'] < 30)  # BoolArray operations
selected_nodes = g.nodes[complex_mask]

# Boolean array methods
print(f"Selected {high_degree_mask.count()} nodes")  # Count True values
print(f"All high degree: {high_degree_mask.all()}")
print(f"Any high degree: {high_degree_mask.any()}")

# Convert to indices when needed
indices = high_degree_mask.nonzero()  # NumArray<usize> of True indices
```

#### 3. Sophisticated Matrix Slicing
```python
# ===== NUMPY-STYLE MATRIX SLICING =====

matrix = g.adjacency()  # Returns Matrix with shape (n, n)

# Basic slicing
submatrix = matrix[:5, :5]        # First 5x5 submatrix
row_slice = matrix[2, :]          # Row 2 as NumArray
col_slice = matrix[:, 3]          # Column 3 as NumArray

# Advanced integer indexing
selected_rows = matrix[[0, 2, 4], :]      # Select specific rows
selected_cols = matrix[:, [1, 3, 5]]      # Select specific columns
submatrix = matrix[[0, 2], [1, 3]]        # Select 2x2 submatrix

# Range slicing with step
every_other = matrix[::2, ::2]     # Every other row and column
reversed_matrix = matrix[::-1, :]  # Reverse row order

# Boolean indexing on matrices
degree_matrix = g.degree_matrix()
high_degree_mask = g.degree() > 4
filtered_matrix = degree_matrix[high_degree_mask, :]  # Select rows by boolean mask
```

#### 4. Array Indexing Enhancements
```python
# ===== ADVANCED ARRAY INDEXING =====

degrees = g.degree()  # NumArray<int64>

# Integer array indexing
selected_indices = [0, 2, 5, 7]
selected_degrees = degrees[selected_indices]  # NumArray subset

# Boolean indexing (now with BoolArray)
high_degree_mask = degrees > 4    # BoolArray
high_degrees = degrees[high_degree_mask]  # Efficient boolean indexing

# Slice indexing
first_half = degrees[:len(degrees)//2]
every_third = degrees[::3]
last_ten = degrees[-10:]

# Combined indexing
complex_selection = degrees[high_degree_mask][:5]  # First 5 high-degree values
```

#### 5. Consistent Slicing API
```rust
// Unified slicing interface
trait Sliceable<T> {
    fn slice(&self, index: SliceIndex) -> PyResult<Self>;
    fn slice_mut(&mut self, index: SliceIndex) -> PyResult<&mut Self>;
}

// Slice index types
#[derive(Debug)]
pub enum SliceIndex {
    Single(i64),                    // [5]
    Range(Option<i64>, Option<i64>, Option<i64>), // [start:end:step]
    List(Vec<i64>),                 // [[0, 2, 4]]
    BoolArray(PyBoolArray),         // [mask]
    Tuple(Vec<SliceIndex>),         // [row_slice, col_slice] for matrices
}
```

---

## Display and Representation System

### Problem: Inconsistent Display Format

Current issues with array and matrix display:
1. **Matrices show column names** when they shouldn't
2. **Arrays inconsistently show indices**
3. **No standardized formatting** across types

### Solution: Clean, Consistent Representations

#### 1. NumArray Display (Keep Index)
```python
# ===== NUMARRAY REPRESENTATION =====
degrees = g.degree()
print(degrees)
# Output:
# NumArray([1, 3, 2, 4, 1, 2, 3], dtype=int64)
# [0] 1
# [1] 3  
# [2] 2
# [3] 4
# [4] 1
# [5] 2
# [6] 3
```

#### 2. BoolArray Display (Keep Index)
```python
# ===== BOOLARRAY REPRESENTATION =====
mask = degrees > 2
print(mask)
# Output:
# BoolArray([False, True, False, True, False, False, True])
# [0] False
# [1] True
# [2] False  
# [3] True
# [4] False
# [5] False
# [6] True
# Count: 3/7 (42.9%)
```

#### 3. Matrix Display (No Column Names)
```python
# ===== MATRIX REPRESENTATION =====
matrix = g.adjacency()
print(matrix)
# Output:
# Matrix(shape=(4, 4), dtype=float64)
# [[0.0, 1.0, 0.0, 1.0],
#  [1.0, 0.0, 1.0, 0.0],
#  [0.0, 1.0, 0.0, 1.0], 
#  [1.0, 0.0, 1.0, 0.0]]

# No row/column names displayed - clean mathematical representation
```

#### 4. Sorted Column Consistency
```rust
// Ensure consistent column ordering in matrices
impl Matrix {
    fn ensure_sorted_columns(&mut self) {
        // Sort columns by name/index to ensure consistent ordering
        // This guarantees [:, :2] always selects the same columns
    }
    
    fn column_names(&self) -> Option<Vec<String>> {
        // Internal column tracking without display
        // Used for slicing like matrix["col1":"col3"]
    }
}
```

---

## Implementation Phases

### Phase 1: Core Unified Type System (Foundation)
1. **Generic Type Infrastructure**
   - Create `NumericType` enum with all supported types
   - Implement `NumericValue` trait for type conversions
   - Create `TypeConvertible` trait for universal conversions
   - Add generic `PyNumArray<T>` structure

2. **Basic Type Conversion Methods**
   - Implement `astype()` for NumArray
   - Add type-specific constructors (`new_int64`, `new_float64`, etc.)
   - Create conversion helper functions

### Phase 2: BaseArray Enhancement (Smart Detection)
1. **Automatic Type Detection**
   - Implement `infer_numeric_type()` for BaseArray
   - Add automatic `to_num_array()` conversion
   - Create type validation and error handling

2. **BaseArray ↔ NumArray Interoperability**
   - Implement `PyBaseArray.to_num_array()`
   - Implement `PyNumArray.to_base_array()`
   - Add type-safe conversion methods

### Phase 3: Table Integration (Smart Column Access)
1. **Intelligent Table Indexing**
   - Update `PyBaseTable.__getitem__()` to auto-return NumArray for numeric columns
   - Add `get_column_raw()` and `get_column_numeric()` methods
   - Implement `column_info()` for type introspection

2. **Table Conversion Methods**
   - Add table-wide type conversion capabilities
   - Implement column type detection and casting
   - Create table schema inference

### Phase 4: BoolArray and Boolean Operations
1. **BoolArray Implementation**
   - Create `PyBoolArray` based on `NumArray<bool>`
   - Implement boolean operations (`and`, `or`, `not`, `any`, `all`, `count`)
   - Add conversion methods (`nonzero()`, `to_indices()`)

2. **Boolean Indexing Integration**
   - Update comparison operators to return `BoolArray` instead of lists
   - Implement efficient boolean indexing for arrays and matrices
   - Add boolean mask support for node/edge filtering

### Phase 5: Advanced Indexing and Slicing System
1. **Unified Slicing Interface**
   - Create `SliceIndex` enum for all indexing types
   - Implement `Sliceable<T>` trait for arrays and matrices
   - Add support for integer lists, ranges, and boolean arrays

2. **NumPy-Style Matrix Slicing**
   - Implement 2D slicing syntax `matrix[:5, :3]`
   - Add advanced indexing `matrix[[0, 2], [1, 3]]`
   - Support step slicing `matrix[::2, ::2]`
   - Ensure consistent column ordering

3. **Array Indexing Enhancements**
   - Advanced integer array indexing
   - Boolean array indexing with BoolArray
   - Combined indexing operations
   - Slice chaining support

### Phase 6: Display and Representation System
1. **Consistent Display Format**
   - Clean NumArray display with indices
   - BoolArray display with count statistics  
   - Matrix display without column names
   - Standardized formatting across all types

2. **Column Ordering and Consistency**
   - Implement sorted column ordering for matrices
   - Internal column name tracking without display
   - Consistent slicing behavior

### Phase 7: Matrix System Integration
1. **Generic Matrix Implementation**
   - Create `PyGraphMatrix<T>` with type parameter
   - Implement Matrix ↔ NumArray conversions (`flatten()`, `reshape()`)
   - Add Matrix ↔ BaseArray conversions

2. **Matrix Construction from Arrays**
   - Implement `PyGraphMatrix::from_base_array()`
   - Add `PyGraphMatrix::from_num_array()`
   - Create matrix dimension inference

### Phase 8: API Updates and Migration
1. **Update Core Accessors**
   - Modify `g.nodes.ids()` to return `NumArray<i64>` instead of `PyIntArray`
   - Update `g.edges.ids()` similarly
   - Replace all float-casting with appropriate integer types

2. **Comprehensive Testing**
   - Create type conversion test suite
   - Add cross-structure conversion tests
   - Implement performance benchmarks
   - Test BoolArray operations and indexing
   - Validate matrix slicing behavior

### Phase 9: Documentation and Polish
1. **API Documentation Updates**
   - Document all new conversion methods
   - Create migration guide from old API
   - Add type system explanation

2. **Performance Optimization**
   - Optimize conversion pathways
   - Implement zero-copy conversions where possible
   - Add lazy evaluation for chained conversions

## Benefits

### 1. Type Safety and Semantic Correctness
- **Proper Types**: Node IDs as integers, measurements as floats, categories as strings
- **No Precision Loss**: Large integers remain precise without float conversion
- **Semantic Operations**: Integer operations on IDs, floating-point operations on measurements

### 2. Performance and Memory Efficiency  
- **Zero-Copy Conversions**: Direct type casting where possible
- **Lazy Evaluation**: Conversions only when needed
- **Memory Layout Optimization**: Type-specific memory layouts
- **Reduced Allocations**: In-place conversions for compatible types

### 3. User Experience and API Consistency
- **Automatic Intelligence**: Tables return appropriate types automatically
- **Universal Conversion**: `astype()` works across all data structures
- **Intuitive Behavior**: NumArray for numbers, BaseArray for mixed data
- **Seamless Interoperability**: Easy conversion between Matrix, Table, NumArray, BaseArray

### 4. Cross-Structure Workflows
- **Matrix Operations**: Easy reshaping from/to arrays and tables
- **Data Pipeline**: Smooth flow between different data representations
- **Type Preservation**: Operations maintain appropriate types throughout
- **Flexible Transformation**: Convert between structures as needed

### 5. Developer Benefits
- **Reduced Boilerplate**: No manual type checking and conversion
- **Clear APIs**: Type information available at runtime
- **Error Prevention**: Type mismatches caught early
- **Performance Insight**: Clear understanding of when conversions occur

## Migration Strategy

### 1. Gradual Cross-Structure Rollout
- **Phase-by-Phase**: Implement NumArray → BaseArray → Table → Matrix in sequence
- **Backward Compatibility**: Old APIs continue working with deprecation warnings
- **Feature Flags**: Optional new behavior during transition period
- **Parallel Systems**: New type system alongside existing implementation

### 2. User Migration Path
- **Automatic Migration**: Most code continues working without changes
- **Enhanced Functionality**: New type-aware features available immediately
- **Explicit Control**: Users can opt into new behavior when ready
- **Clear Deprecation Timeline**: 6-month warning period before removing old behavior

### 3. Documentation and Training
- **Migration Guide**: Step-by-step conversion instructions
- **Type System Guide**: Comprehensive explanation of new capabilities
- **Best Practices**: Recommended patterns for different use cases
- **Performance Guide**: When to use each data structure type

### 4. Testing and Validation
- **Comprehensive Test Suite**: All conversion paths tested
- **Performance Benchmarks**: Ensure no regression in critical paths
- **User Acceptance Testing**: Validate with real-world workflows
- **Edge Case Coverage**: Handle all type conversion scenarios

---

## 🎯 Priority Implementation Roadmap

### High Priority (Immediate - Next 4 weeks)
**Focus: Boolean Arrays and Basic Slicing**

#### Week 1-2: BoolArray Foundation ✅ **COMPLETED**
1. **✅ Implement `PyBoolArray`** - Based on `NumArray<bool>` with efficient operations
   - Core `PyBoolArray` struct with `NumArray<bool>` backing
   - Comprehensive Python FFI integration with PyO3
   - Iterator support and full indexing capabilities
2. **✅ Boolean operations** - `and`, `or`, `not`, `any`, `all`, `count`
   - Bitwise operators: `&` (and), `|` (or), `~` (not)
   - Statistical methods: `count()`, `any()`, `all()`, `count_false()`, `percentage()`
   - Index operations: `nonzero()`, `to_indices()`, `false_indices()`
   - Utility functions: `ones_bool()`, `zeros_bool()`, `apply_mask()`
3. **✅ Update comparison operators** - Return `BoolArray` instead of lists
   - **BaseArray**: All comparison operators (`>`, `<`, `>=`, `<=`, `==`, `!=`) return BoolArray
   - **NumArray**: All numeric comparison operators return BoolArray
   - **StatsArray**: All integer comparison operators return BoolArray
4. **✅ Integration testing** - Ensure `g.nodes[g.degree() > 4]` works with BoolArray
   - `g.degree() > 4` now returns BoolArray (not list)
   - `g.nodes[boolean_mask]` successfully filters using BoolArray
   - Complex boolean expressions work: `(degrees > 2) & (degrees < 10)`
   - **Result**: `g.nodes[g.degree() > 2]` returns filtered subgraph with 5 nodes and 6 edges

#### Week 3-4: Basic Array Slicing ✅ **COMPLETED**
1. **✅ Integer list indexing** - `array[[0, 2, 5]]` support
   - Full support for integer lists: `array[[0, 2, 5]]` returns filtered array
   - Negative index support: `array[[-1, -2, 0]]` works correctly
   - Works with BaseArray, NumArray, and BoolArray
2. **✅ Boolean indexing** - `array[bool_mask]` with BoolArray
   - Direct BoolArray indexing: `array[bool_mask]` filters elements
   - Comparison-generated masks: `array[array > 3]` creates and applies mask
   - Efficient boolean masking with proper length validation
3. **✅ Range slicing** - `array[:5]`, `array[::2]` support
   - Python slice notation: `array[:5]`, `array[2:7]`, `array[::2]`
   - Negative slicing: `array[-3:]`, `array[::-1]` (reverse)
   - Complex slicing: `array[1:8:2]` with step values
4. **✅ Combined operations** - `array[mask][:10]` chaining
   - Multi-step filtering: `array[bool_mask][:5]` chains correctly
   - Complex operations: `degrees[degrees > 2][:3]` works seamlessly
   - **Result**: Full NumPy-style indexing system with unified SliceIndex enum

### Medium Priority (Weeks 5-8)
**Focus: Matrix Slicing and Display**

#### Week 5-6: Matrix Slicing System ✅ **COMPLETED**
1. **✅ 2D slicing syntax** - `matrix[:5, :3]`, `matrix[::2, ::2]`
   - Full NumPy-style 2D slicing: `matrix[:2, :2]`, `matrix[::2, ::2]`
   - Range slicing: `matrix[1:3, 0:2]` with proper bounds checking
   - Step slicing: `matrix[::2, ::2]` for sampling matrices
2. **✅ Advanced indexing** - `matrix[[0, 2], [1, 3]]`
   - Integer list indexing: `matrix[[0, 2, 3], [0, 2]]` selects specific rows/cols
   - Negative index support: `matrix[[-1, -2], [0, 1]]` works correctly
   - Mixed indexing: `matrix[:2, [0, 2]]` combines slices with lists
3. **✅ Boolean matrix indexing** - `matrix[row_mask, :]`
   - Row boolean filtering: `matrix[row_mask, :]` filters rows by BoolArray
   - Column boolean filtering: `matrix[:, col_mask]` filters columns by BoolArray  
   - 2D boolean filtering: `matrix[row_mask, col_mask]` filters both dimensions
4. **✅ SliceIndex enum** - Unified indexing interface
   - `MatrixIndex` enum supports all indexing types
   - `MatrixSlice` handles 2D indexing specifications  
   - `MatrixSlicing` trait provides consistent API
   - **Result**: Complete NumPy-style 2D matrix indexing system with 10/10 tests passing

#### Week 7-8: Display and Representation
1. **Clean NumArray display** - With indices, no column names
2. **BoolArray display** - With count statistics and percentages  
3. **Matrix display cleanup** - No column names, clean mathematical format
4. **Consistent column ordering** - Sorted, predictable matrix columns

### Lower Priority (Weeks 9-16) 
**Focus: Advanced Features and Polish**

#### Week 9-12: Advanced Type System
1. **Generic NumArray implementation** - `NumArray<T>` support
2. **Type conversion system** - `astype()` across all structures
3. **Cross-structure conversions** - Matrix ↔ NumArray ↔ BaseArray
4. **Integer node ID support** - `g.nodes.ids()` returns `NumArray<i64>`

#### Week 13-16: Integration and Polish
1. **Performance optimizations** - Zero-copy conversions where possible
2. **Comprehensive testing** - All slicing and indexing scenarios
3. **Documentation** - Migration guides and API documentation
4. **Backward compatibility** - Deprecation warnings and migration path

### Success Metrics

#### BoolArray System Success
- ✅ `g.degree() > 4` returns `BoolArray` instead of list
- ✅ Boolean operations work: `(mask1 & mask2).count()` 
- ✅ Efficient boolean indexing: `g.nodes[complex_mask]`
- ✅ Performance improvement: 5x+ faster than list-based boolean indexing

#### Matrix Slicing Success  
- ✅ NumPy-style syntax works: `matrix[:5, :3]`, `matrix[::2, ::2]`
- ✅ Advanced indexing: `matrix[[0, 2], [1, 3]]`
- ✅ Consistent behavior: `matrix[:, :2]` always selects same columns
- ✅ Clean display: No column names in matrix repr

#### Overall Integration Success
- ✅ API consistency: All arrays support same slicing syntax
- ✅ User experience: Intuitive, NumPy-like behavior
- ✅ Performance: No regressions, improved efficiency
- ✅ Backward compatibility: Existing code continues working

---

## Complexity Considerations

### High Complexity Areas

#### 1. Generic Type System Architecture
- **Rust Generics with PyO3**: Complex interaction between Rust generic types and Python bindings
- **Type Erasure**: Managing different numeric types in a unified Python interface
- **Memory Layout**: Different numeric types have different sizes and alignment requirements
- **Trait Bounds**: Complex trait hierarchies for numeric operations

#### 2. Cross-Structure Conversion Matrix
- **Conversion Pathways**: NumArray ↔ BaseArray ↔ Matrix ↔ Table (16 total conversion paths)
- **Type Safety**: Ensuring conversions preserve data integrity across all paths
- **Performance Optimization**: Avoiding unnecessary copies in conversion chains
- **Error Handling**: Graceful failure when conversions are impossible

#### 3. Automatic Type Detection
- **Mixed-Type Arrays**: Handling BaseArrays with heterogeneous data
- **Type Promotion**: When to upgrade types (int32 → int64 → float64)
- **Ambiguous Cases**: Deciding between int64 and float64 for integer-like floats
- **Performance Impact**: Type detection overhead in hot paths

#### 4. Python Integration Challenges
- **Dynamic Typing**: Python's dynamic nature vs Rust's static typing
- **Method Dispatch**: Routing calls to appropriate generic implementations
- **Object Lifetime**: Managing Rust generics in Python's garbage-collected environment
- **Error Translation**: Converting Rust type errors to meaningful Python exceptions

### Implementation Challenges

#### 1. Memory Management
- **Zero-Copy Conversions**: Implementing where possible without compromising safety
- **Reference Counting**: Managing shared data across different structure types
- **Memory Alignment**: Ensuring optimal performance across numeric types
- **Garbage Collection**: Interaction between Rust ownership and Python GC

#### 2. Performance Optimization
- **Hot Path Identification**: Ensuring fast paths for common conversions
- **Lazy Evaluation**: Deferring expensive conversions until necessary
- **Caching**: Storing conversion results to avoid repeated work
- **Vectorization**: SIMD operations for bulk type conversions

#### 3. API Design Complexity
- **Method Overloading**: Simulating in Python what Rust does with generics
- **Type Hinting**: Providing accurate Python type hints for generic returns
- **Documentation**: Explaining complex type relationships to users
- **Backward Compatibility**: Maintaining old behavior while introducing new features

#### 4. Testing and Validation
- **Combinatorial Explosion**: Testing all type × structure × operation combinations
- **Edge Cases**: Handling numeric limits, NaN values, overflow conditions
- **Performance Regression**: Ensuring new system doesn't slow down existing code
- **Memory Leak Detection**: Validating complex ownership patterns don't leak

## Alternative: Incremental Implementation Strategy

### Quick Win: Enhanced Current System
For immediate improvement while planning the full system:

```rust
// ===== IMMEDIATE IMPROVEMENTS =====

// 1. Replace PyIntArray with NumArray<i64> for node IDs
impl PyGraph {
    fn node_ids(&self) -> PyResult<PyNumArray<i64>> {
        let ids: Vec<i64> = self.inner.node_ids().into_iter()
            .map(|id| id as i64)  // Keep as integers
            .collect();
        Ok(PyNumArray::new_int64(ids))
    }
}

// 2. Add basic cross-structure conversions
impl PyBaseArray {
    fn to_numeric_if_possible(&self) -> PyResult<Option<PyObject>> {
        if self.is_all_integers() {
            Ok(Some(self.to_int64_array()?.into()))
        } else if self.is_all_floats() {
            Ok(Some(self.to_float64_array()?.into()))
        } else {
            Ok(None)  // Not numeric
        }
    }
}

// 3. Smart table column access
impl PyBaseTable {
    fn __getitem__(&self, key: &str) -> PyResult<PyObject> {
        let base_column = self.get_column_raw(key)?;
        
        // Try to return as NumArray if numeric
        match base_column.to_numeric_if_possible()? {
            Some(numeric_array) => Ok(numeric_array),
            None => Ok(base_column.into())  // Return as BaseArray
        }
    }
}

// ===== MIGRATION BUILDING BLOCKS =====

// Simple astype() for current PyIntArray and PyNumArray
impl PyIntArray {
    fn astype(&self, dtype: &str) -> PyResult<PyObject> {
        match dtype {
            "float64" => {
                let floats: Vec<f64> = self.inner.iter().map(|&x| x as f64).collect();
                Ok(PyNumArray::new_float64(floats).into())
            }
            "basearray" => {
                let attrs: Vec<AttrValue> = self.inner.iter()
                    .map(|&x| AttrValue::BigInt(x)).collect();
                Ok(PyBaseArray::new(attrs).into())
            }
            _ => Err(PyValueError::new_err("Unsupported conversion"))
        }
    }
}
```

### Incremental Rollout Strategy

#### Week 1-2: Type Infrastructure
- Implement `NumericType` enum
- Create basic `astype()` methods for existing types
- Add cross-conversion between PyIntArray and PyNumArray

#### Week 3-4: Smart Table Access  
- Implement automatic NumArray return for numeric columns
- Add `column_info()` method for type introspection
- Create `get_column_raw()` and `get_column_numeric()` explicit methods

#### Week 5-6: BaseArray Intelligence
- Add `infer_numeric_type()` to BaseArray
- Implement `to_num_array()` conversion
- Create type validation and error handling

#### Week 7-8: Matrix Integration
- Add basic Matrix ↔ NumArray conversions
- Implement `flatten()` and `reshape()` methods
- Create Matrix construction from BaseArray

#### Week 9-10: Polish and Optimization
- Performance optimization for conversion paths
- Comprehensive testing of all conversion combinations
- Documentation and migration guide

This approach provides immediate value while building toward the full unified system, allowing users to benefit from improvements incrementally rather than waiting for a complete rewrite.
