GraphMatrix API
===============

GraphMatrix provides efficient matrix operations for homogeneous multi-column data with linear algebra support.

Constructor
-----------

.. function:: groggy.matrix(data, dtype=None)

   Create a GraphMatrix from 2D data.

   :param data: 2D array-like data (list of lists, NumPy array, etc.)
   :type data: list or numpy.ndarray
   :param str dtype: Optional data type specification
   :returns: New GraphMatrix instance
   :rtype: GraphMatrix

   **Example:**

   .. code-block:: python

      import groggy as gr
      
      # From 2D list
      data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      matrix = gr.matrix(data)
      
      # From NumPy array
      import numpy as np
      np_data = np.array([[1.0, 2.0], [3.0, 4.0]])
      matrix = gr.matrix(np_data)

Properties
----------

.. attribute:: GraphMatrix.shape

   Dimensions of the matrix as (rows, columns).

   :type: tuple

.. attribute:: GraphMatrix.dtype

   Data type of matrix elements.

   :type: str

.. attribute:: GraphMatrix.is_square

   Whether the matrix is square (rows == columns).

   :type: bool

.. attribute:: GraphMatrix.is_sparse

   Whether matrix is stored in sparse format.

   :type: bool

.. attribute:: GraphMatrix.rows

   Number of rows in the matrix.

   :type: int

.. attribute:: GraphMatrix.cols

   Number of columns in the matrix.

   :type: int

Element Access
--------------

.. method:: GraphMatrix.__getitem__(indices)

   Access matrix elements, rows, columns, or submatrices.

   :param indices: Index specification
   :type indices: int, slice, tuple, or list
   :returns: Single element, GraphArray, or GraphMatrix
   :rtype: Any, GraphArray, or GraphMatrix

   **Examples:**

   .. code-block:: python

      # Single element
      element = matrix[1, 2]
      
      # Row access (returns GraphArray)
      row = matrix[0]
      row = matrix[0, :]
      
      # Column access (returns GraphArray)
      col = matrix[:, 1]
      
      # Submatrix
      submatrix = matrix[0:2, 1:3]
      
      # Multiple rows/columns
      rows = matrix[[0, 2]]
      cols = matrix[:, [0, 2]]

.. method:: GraphMatrix.__setitem__(indices, value)

   Set matrix elements.

   :param indices: Index specification
   :param value: Value(s) to set
   :type value: scalar, array-like, or GraphMatrix

Matrix Operations
-----------------

.. method:: GraphMatrix.transpose()

   Transpose the matrix.

   :returns: Transposed matrix
   :rtype: GraphMatrix

   **Example:**

   .. code-block:: python

      transposed = matrix.transpose()
      # Or using .T property
      transposed = matrix.T

.. method:: GraphMatrix.power(n)

   Raise matrix to integer power (for square matrices).

   :param int n: Power to raise matrix to
   :returns: Matrix raised to power n
   :rtype: GraphMatrix
   :raises ValueError: If matrix is not square

   **Example:**

   .. code-block:: python

      # Matrix multiplication: A² = A @ A
      squared = matrix.power(2)
      cubed = matrix.power(3)

.. method:: GraphMatrix.multiply(other)

   Element-wise multiplication.

   :param other: Matrix or scalar to multiply by
   :type other: GraphMatrix or scalar
   :returns: Element-wise product
   :rtype: GraphMatrix

.. method:: GraphMatrix.matmul(other)

   Matrix multiplication.

   :param other: Matrix to multiply with
   :type other: GraphMatrix
   :returns: Matrix product
   :rtype: GraphMatrix
   :raises ValueError: If dimensions incompatible

   **Example:**

   .. code-block:: python

      # Matrix multiplication
      result = matrix1.matmul(matrix2)
      # Or using @ operator
      result = matrix1 @ matrix2

Axis Operations
---------------

.. method:: GraphMatrix.sum_axis(axis)

   Sum along specified axis.

   :param int axis: Axis to sum along (0=rows, 1=columns)
   :returns: Sum along axis
   :rtype: GraphArray

.. method:: GraphMatrix.mean_axis(axis)

   Mean along specified axis.

   :param int axis: Axis to compute mean along (0=rows, 1=columns)
   :returns: Mean along axis
   :rtype: GraphArray

.. method:: GraphMatrix.std_axis(axis)

   Standard deviation along specified axis.

   :param int axis: Axis to compute std along (0=rows, 1=columns)
   :returns: Standard deviation along axis
   :rtype: GraphArray

.. method:: GraphMatrix.min_axis(axis)

   Minimum along specified axis.

   :param int axis: Axis to find minimum along (0=rows, 1=columns)
   :returns: Minimum along axis
   :rtype: GraphArray

.. method:: GraphMatrix.max_axis(axis)

   Maximum along specified axis.

   :param int axis: Axis to find maximum along (0=rows, 1=columns)
   :returns: Maximum along axis
   :rtype: GraphArray

.. method:: GraphMatrix.argmin_axis(axis)

   Indices of minimum values along axis.

   :param int axis: Axis to find argmin along (0=rows, 1=columns)
   :returns: Indices of minimum values
   :rtype: GraphArray

.. method:: GraphMatrix.argmax_axis(axis)

   Indices of maximum values along axis.

   :param int axis: Axis to find argmax along (0=rows, 1=columns)
   :returns: Indices of maximum values
   :rtype: GraphArray

Statistical Operations
----------------------

.. method:: GraphMatrix.mean()

   Overall mean of all elements.

   :returns: Mean value
   :rtype: float

.. method:: GraphMatrix.std()

   Overall standard deviation of all elements.

   :returns: Standard deviation
   :rtype: float

.. method:: GraphMatrix.sum()

   Sum of all elements.

   :returns: Total sum
   :rtype: float

.. method:: GraphMatrix.min()

   Minimum element in matrix.

   :returns: Minimum value
   :rtype: float

.. method:: GraphMatrix.max()

   Maximum element in matrix.

   :returns: Maximum value
   :rtype: float

.. method:: GraphMatrix.count()

   Count of non-null elements.

   :returns: Number of non-null elements
   :rtype: int

Sparse Operations
-----------------

.. method:: GraphMatrix.to_sparse()

   Convert to sparse representation.

   :returns: Sparse version of matrix
   :rtype: GraphMatrix

.. method:: GraphMatrix.to_dense()

   Convert to dense representation.

   :returns: Dense version of matrix
   :rtype: GraphMatrix

.. method:: GraphMatrix.sparsity()

   Calculate sparsity ratio (fraction of zero elements).

   :returns: Sparsity ratio (0.0 to 1.0)
   :rtype: float

.. method:: GraphMatrix.nnz()

   Number of non-zero elements.

   :returns: Count of non-zero elements
   :rtype: int

Linear Algebra
--------------

.. method:: GraphMatrix.norm(ord='fro')

   Matrix norm.

   :param str ord: Norm type ('fro' for Frobenius, 'nuc' for nuclear)
   :returns: Matrix norm
   :rtype: float

.. method:: GraphMatrix.trace()

   Trace of the matrix (sum of diagonal elements).

   :returns: Trace value
   :rtype: float
   :raises ValueError: If matrix is not square

.. method:: GraphMatrix.diagonal()

   Diagonal elements of the matrix.

   :returns: Diagonal as array
   :rtype: GraphArray

.. method:: GraphMatrix.determinant()

   Determinant of square matrix.

   :returns: Determinant value
   :rtype: float
   :raises ValueError: If matrix is not square

Data Manipulation
-----------------

.. method:: GraphMatrix.reshape(new_shape)

   Reshape matrix to new dimensions.

   :param tuple new_shape: New shape as (rows, columns)
   :returns: Reshaped matrix
   :rtype: GraphMatrix
   :raises ValueError: If total elements don't match

.. method:: GraphMatrix.flatten()

   Flatten matrix to 1D array.

   :returns: Flattened data
   :rtype: GraphArray

.. method:: GraphMatrix.append_row(row)

   Append row to matrix.

   :param row: Row data to append
   :type row: list or GraphArray
   :returns: New matrix with appended row
   :rtype: GraphMatrix

.. method:: GraphMatrix.append_column(column)

   Append column to matrix.

   :param column: Column data to append
   :type column: list or GraphArray
   :returns: New matrix with appended column
   :rtype: GraphMatrix

.. method:: GraphMatrix.remove_row(index)

   Remove row by index.

   :param int index: Row index to remove
   :returns: New matrix without specified row
   :rtype: GraphMatrix

.. method:: GraphMatrix.remove_column(index)

   Remove column by index.

   :param int index: Column index to remove
   :returns: New matrix without specified column
   :rtype: GraphMatrix

Sorting and Filtering
---------------------

.. method:: GraphMatrix.sort_rows_by_column(column_index, ascending=True)

   Sort rows by values in specified column.

   :param int column_index: Index of column to sort by
   :param bool ascending: Sort in ascending order (default: True)
   :returns: Matrix with sorted rows
   :rtype: GraphMatrix

.. method:: GraphMatrix.filter_rows(predicate)

   Filter rows by predicate function.

   :param callable predicate: Function that takes row and returns bool
   :returns: Matrix with filtered rows
   :rtype: GraphMatrix

   **Example:**

   .. code-block:: python

      # Filter rows where first column > 5
      filtered = matrix.filter_rows(lambda row: row[0] > 5)

Comparison Operations
---------------------

.. method:: GraphMatrix.equals(other, tolerance=1e-9)

   Check equality with another matrix.

   :param GraphMatrix other: Matrix to compare with
   :param float tolerance: Tolerance for floating point comparison
   :returns: True if matrices are equal within tolerance
   :rtype: bool

.. method:: GraphMatrix.greater_than(threshold)

   Element-wise greater than comparison.

   :param threshold: Threshold value
   :type threshold: scalar
   :returns: Boolean matrix
   :rtype: GraphMatrix

.. method:: GraphMatrix.less_than(threshold)

   Element-wise less than comparison.

   :param threshold: Threshold value
   :type threshold: scalar
   :returns: Boolean matrix
   :rtype: GraphMatrix

Conversion Methods
------------------

.. method:: GraphMatrix.to_numpy()

   Convert to NumPy array.

   :returns: NumPy array representation
   :rtype: numpy.ndarray

.. method:: GraphMatrix.to_pandas()

   Convert to pandas DataFrame.

   :param list column_names: Optional column names
   :returns: DataFrame representation
   :rtype: pandas.DataFrame

.. method:: GraphMatrix.to_list()

   Convert to list of lists.

   :returns: Nested list representation
   :rtype: list

Utility Methods
---------------

.. method:: GraphMatrix.copy()

   Create a copy of the matrix.

   :returns: New matrix with copied data
   :rtype: GraphMatrix

.. method:: GraphMatrix.memory_usage()

   Get memory usage of the matrix.

   :returns: Memory usage in bytes
   :rtype: int

.. method:: GraphMatrix.is_null(row=None, col=None)

   Check for null values.

   :param int row: Specific row to check, or None for all
   :param int col: Specific column to check, or None for all
   :returns: Boolean value or matrix indicating null positions
   :rtype: bool or GraphMatrix

.. method:: GraphMatrix.fill_null(value)

   Fill null values with specified value.

   :param value: Value to fill nulls with
   :returns: Matrix with nulls filled
   :rtype: GraphMatrix

Display Methods
---------------

.. method:: GraphMatrix.__repr__()

   String representation for terminals.

   :returns: String representation
   :rtype: str

.. method:: GraphMatrix._repr_html_()

   HTML representation for Jupyter notebooks.

   :returns: HTML string
   :rtype: str

.. method:: GraphMatrix.preview(max_rows=10, max_cols=10)

   Preview with limited rows and columns.

   :param int max_rows: Maximum rows to show
   :param int max_cols: Maximum columns to show
   :returns: Preview string
   :rtype: str

.. method:: GraphMatrix.summary()

   Get summary information about the matrix.

   :returns: Summary string
   :rtype: str

Performance Characteristics
---------------------------

- **Storage**: Columnar layout optimized for cache efficiency
- **Sparse Support**: Automatic sparse representation for matrices with many zeros
- **SIMD Operations**: Vectorized operations using modern CPU instructions
- **Memory Mapping**: Large matrices can be memory-mapped for efficiency
- **Lazy Evaluation**: Operations are computed on-demand and cached

Type System
-----------

GraphMatrix supports these data types:

- **Numeric**: int8, int16, int32, int64, float32, float64
- **Boolean**: bool
- **Complex**: complex64, complex128 (for advanced linear algebra)

Type promotion happens automatically during operations with clear rules for maintaining precision.

Graph Integration
-----------------

GraphMatrix integrates seamlessly with graph structures:

.. code-block:: python

   # Adjacency matrix from graph
   adj = g.adjacency()
   
   # Powers for path counting
   paths_2 = adj.power(2)  # 2-hop paths
   paths_3 = adj.power(3)  # 3-hop paths
   
   # Combine with node attributes
   nodes_table = g.nodes.table()
   features = nodes_table[['age', 'salary']]  # Returns GraphMatrix
   
   # Feature similarity
   similarity = features @ features.transpose()

Best Practices
--------------

1. **Use appropriate data types** - int64 vs float64 affects memory and performance
2. **Leverage sparsity** - sparse matrices use significantly less memory for sparse data
3. **Chain operations efficiently** - combine operations to minimize intermediate results
4. **Use axis operations** - prefer sum_axis() over row-by-row iteration
5. **Consider memory layout** - row-major vs column-major affects cache performance

**Example workflow:**

.. code-block:: python

   import groggy as gr
   
   # Create adjacency matrix
   adj = g.adjacency()
   
   # Analyze connectivity patterns
   degree_centrality = adj.sum_axis(axis=1)  # Row sums = out-degrees
   
   # Find 2-hop neighborhoods
   two_hop = adj.power(2)
   
   # Combine with node features
   features = g.nodes.table()[['age', 'salary']]
   feature_similarity = features @ features.transpose()
   
   # Export for external analysis
   similarity_np = feature_similarity.to_numpy()

GraphMatrix provides the foundation for efficient linear algebra operations in Groggy's storage view system.