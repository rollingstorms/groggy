GraphArray API
==============

GraphArray provides high-performance columnar arrays with statistical operations computed in native Rust.

Constructor
-----------

.. function:: groggy.array(values, name=None)

   Create a GraphArray from values.

   :param list values: List of values to store
   :param str name: Optional name for the array
   :returns: New GraphArray instance
   :rtype: GraphArray

   **Example:**

   .. code-block:: python

      import groggy as gr

      ages = gr.array([25, 30, 35, 40, 45], name="ages")

Properties
----------

.. attribute:: GraphArray.values

   Get all values as Python list.

   :type: list

.. attribute:: GraphArray.dtype

   Data type of the array.

   :type: str

.. attribute:: GraphArray.is_sparse

   Whether array is stored sparsely.

   :type: bool

.. method:: GraphArray.__len__()

   Length of the array.

   :returns: Number of elements
   :rtype: int

Indexing and Access
-------------------

.. method:: GraphArray.__getitem__(index)

   Advanced indexing support.

   :param index: Index, slice, list of indices, or boolean mask
   :type index: int, slice, list, or list of bool
   :returns: Single element or new GraphArray
   :rtype: Any or GraphArray

   **Examples:**

   .. code-block:: python

      arr[5]           # Single element
      arr[-1]          # Negative indexing
      arr[1:10:2]      # Slice with step
      arr[[1, 3, 5]]   # Fancy indexing
      arr[mask]        # Boolean indexing

Statistical Operations
----------------------

.. method:: GraphArray.mean()

   Arithmetic mean of values.

   :returns: Mean value, or None if no numeric values
   :rtype: float or None

.. method:: GraphArray.median()

   Median value.

   :returns: Median value, or None if no numeric values
   :rtype: float or None

.. method:: GraphArray.std()

   Standard deviation.

   :returns: Standard deviation, or None if no numeric values
   :rtype: float or None

.. method:: GraphArray.min()

   Minimum value.

   :returns: Minimum value, or None if empty
   :rtype: Any or None

.. method:: GraphArray.max()

   Maximum value.

   :returns: Maximum value, or None if empty
   :rtype: Any or None

.. method:: GraphArray.sum()

   Sum of values.

   :returns: Sum of values, or None if no numeric values
   :rtype: float or None

.. method:: GraphArray.count()

   Count of non-null values.

   :returns: Number of non-null values
   :rtype: int

.. method:: GraphArray.unique()

   Unique values in the array.

   :returns: Array containing unique values
   :rtype: GraphArray

.. method:: GraphArray.value_counts()

   Count of each unique value.

   :returns: Dictionary mapping value -> count
   :rtype: dict

.. method:: GraphArray.describe()

   Comprehensive statistical summary.

   :returns: Dictionary with statistical measures
   :rtype: dict

   **Example:**

   .. code-block:: python

      ages = gr.array([25, 30, 35, 40, 45])
      stats = ages.describe()
      # Returns: {'count': 5, 'mean': 35.0, 'std': 7.91, 'min': 25, 'max': 45, ...}

Data Operations
---------------

.. method:: GraphArray.filter(predicate)

   Filter values by predicate.

   :param callable predicate: Function that takes a value and returns bool
   :returns: New array with filtered values
   :rtype: GraphArray

   **Example:**

   .. code-block:: python

      ages = gr.array([25, 30, 35, 40, 45])
      adults = ages.filter(lambda x: x >= 30)

.. method:: GraphArray.map(transform)

   Transform values using a function.

   :param callable transform: Function that transforms each value
   :returns: New array with transformed values
   :rtype: GraphArray

   **Example:**

   .. code-block:: python

      ages = gr.array([25, 30, 35, 40, 45])
      ages_in_months = ages.map(lambda x: x * 12)

.. method:: GraphArray.sort(ascending=True)

   Sort array values.

   :param bool ascending: Sort in ascending order (default: True)
   :returns: New array with sorted values
   :rtype: GraphArray

.. method:: GraphArray.reverse()

   Reverse array order.

   :returns: New array with reversed values
   :rtype: GraphArray

Conversion Methods
------------------

.. method:: GraphArray.to_numpy()

   Convert to NumPy array.

   :returns: NumPy array with converted values
   :rtype: numpy.ndarray

.. method:: GraphArray.to_pandas()

   Convert to pandas Series.

   :returns: Pandas Series with array data
   :rtype: pandas.Series

.. method:: GraphArray.to_list()

   Convert to Python list.

   :returns: List containing all values
   :rtype: list

Utility Methods
---------------

.. method:: GraphArray.memory_usage()

   Get memory usage of the array.

   :returns: Memory usage in bytes
   :rtype: int

.. method:: GraphArray.copy()

   Create a copy of the array.

   :returns: New GraphArray with copied data
   :rtype: GraphArray

.. method:: GraphArray.is_null(index=None)

   Check for null values.

   :param int index: Specific index to check, or None for all
   :returns: Boolean value or list of booleans
   :rtype: bool or list

.. method:: GraphArray.summary()

   Get summary information.

   :returns: String summary of the array
   :rtype: str

Display Methods
---------------

.. method:: GraphArray.__repr__()

   String representation for terminals.

   :returns: String representation
   :rtype: str

.. method:: GraphArray._repr_html_()

   HTML representation for Jupyter notebooks.

   :returns: HTML string
   :rtype: str

.. method:: GraphArray.preview(limit=10)

   Preview with limited elements.

   :param int limit: Maximum number of elements to show
   :returns: Preview string
   :rtype: str

Performance Characteristics
---------------------------

- **Statistical Operations**: Computed in native Rust for maximum speed
- **Memory Efficiency**: Columnar storage with minimal overhead
- **Caching**: Statistical results cached automatically and invalidated intelligently  
- **Sparse Support**: Automatic sparse representation for data with many zeros/nulls
- **Zero-Copy**: Efficient slicing and indexing without data copying when possible

Type System
-----------

GraphArray supports these data types:

- **Numeric**: int64, float64
- **Text**: String values
- **Boolean**: True/False values  
- **Bytes**: Binary data
- **Null**: Missing values

Type coercion happens automatically when needed, with clear error messages for incompatible operations.

Best Practices
--------------

1. **Use for single-column data** that needs statistical analysis
2. **Leverage caching** - repeated statistical operations are fast
3. **Chain operations** efficiently with method chaining
4. **Convert to NumPy/pandas** only when needed for specific library features
5. **Use appropriate indexing** - boolean masks for filtering, fancy indexing for selection

**Example workflow:**

.. code-block:: python

   import groggy as gr

   # Create array from graph data
   table = g.nodes.table()
   ages = table['age']

   # Statistical analysis (cached)
   print(f"Average age: {ages.mean():.1f}")
   print(f"Age range: {ages.min()}-{ages.max()}")

   # Filtering and analysis
   seniors = ages.filter(lambda x: x >= 65)
   print(f"Seniors: {len(seniors)} ({len(seniors)/len(ages)*100:.1f}%)")

   # Integration with scientific libraries
   import numpy as np
   correlation = np.corrcoef(ages.to_numpy(), salaries.to_numpy())[0,1]

GraphArray provides the foundation for efficient single-column analysis in Groggy's storage view system.