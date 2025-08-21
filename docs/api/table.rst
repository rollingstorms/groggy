GraphTable API
==============

GraphTable provides pandas-like functionality for heterogeneous tabular data with graph integration and relational operations.

Constructor
-----------

.. function:: groggy.table(data, columns=None)

   Create a GraphTable from structured data.

   :param data: Table data as dictionary, list of records, or DataFrame
   :type data: dict, list, or pandas.DataFrame
   :param list columns: Optional column names if data is list of lists
   :returns: New GraphTable instance
   :rtype: GraphTable

   **Examples:**

   .. code-block:: python

      import groggy as gr
      
      # From dictionary
      data = {
          'name': ['alice', 'bob', 'charlie'],
          'age': [30, 25, 35],
          'dept': ['eng', 'design', 'mgmt']
      }
      table = gr.table(data)
      
      # From list of records
      records = [
          {'name': 'alice', 'age': 30, 'dept': 'eng'},
          {'name': 'bob', 'age': 25, 'dept': 'design'}
      ]
      table = gr.table(records)

.. classmethod:: GraphTable.from_pandas(df)

   Create GraphTable from pandas DataFrame.

   :param pandas.DataFrame df: Source DataFrame
   :returns: New GraphTable instance
   :rtype: GraphTable

.. classmethod:: GraphTable.from_csv(filename, **kwargs)

   Create GraphTable from CSV file.

   :param str filename: Path to CSV file
   :param kwargs: Additional CSV parsing options
   :returns: New GraphTable instance
   :rtype: GraphTable

Properties
----------

.. attribute:: GraphTable.shape

   Dimensions of the table as (rows, columns).

   :type: tuple

.. attribute:: GraphTable.columns

   Column names.

   :type: list

.. attribute:: GraphTable.dtypes

   Data types of each column.

   :type: dict

.. attribute:: GraphTable.index

   Row index (if set).

   :type: GraphArray or None

.. method:: GraphTable.__len__()

   Number of rows in the table.

   :returns: Row count
   :rtype: int

Data Access
-----------

Column Access
~~~~~~~~~~~~~

.. method:: GraphTable.__getitem__(key)

   Access columns or subsets of the table.

   :param key: Column name, list of names, or boolean mask
   :type key: str, list, or GraphArray
   :returns: Single column or subset table
   :rtype: GraphArray or GraphTable

   **Examples:**

   .. code-block:: python

      # Single column (returns GraphArray)
      ages = table['age']
      
      # Multiple columns (returns GraphTable)
      subset = table[['name', 'age']]
      
      # Boolean indexing
      mask = table['age'] > 30
      older_people = table[mask]

Row Access
~~~~~~~~~~

.. method:: GraphTable.iloc(index)

   Position-based row access.

   :param index: Row index or slice
   :type index: int or slice
   :returns: Single row or subset table
   :rtype: dict or GraphTable

.. method:: GraphTable.loc(label)

   Label-based row access (requires index).

   :param label: Row label
   :returns: Single row data
   :rtype: dict

.. method:: GraphTable.head(n=5)

   First n rows.

   :param int n: Number of rows to return
   :returns: Subset table
   :rtype: GraphTable

.. method:: GraphTable.tail(n=5)

   Last n rows.

   :param int n: Number of rows to return
   :returns: Subset table
   :rtype: GraphTable

.. method:: GraphTable.sample(n=None, frac=None)

   Random sample of rows.

   :param int n: Number of rows to sample
   :param float frac: Fraction of rows to sample
   :returns: Sampled table
   :rtype: GraphTable

Data Exploration
----------------

.. method:: GraphTable.info()

   Information about the table structure.

   :returns: Table information summary
   :rtype: str

.. method:: GraphTable.describe()

   Statistical summary of numeric columns.

   :returns: Summary statistics
   :rtype: GraphTable

.. method:: GraphTable.nunique()

   Number of unique values per column.

   :returns: Unique counts for each column
   :rtype: dict

.. method:: GraphTable.memory_usage()

   Memory usage per column.

   :returns: Memory usage in bytes
   :rtype: dict

Filtering and Selection
-----------------------

Row Filtering
~~~~~~~~~~~~~

.. method:: GraphTable.filter_rows(predicate)

   Filter rows by predicate function.

   :param callable predicate: Function that takes row dict and returns bool
   :returns: Filtered table
   :rtype: GraphTable

   **Examples:**

   .. code-block:: python

      # Simple filter
      young = table.filter_rows(lambda row: row['age'] < 30)
      
      # Complex filter
      young_engineers = table.filter_rows(
          lambda row: row['age'] < 30 and row['dept'] == 'eng'
      )

.. method:: GraphTable.query(expression)

   Filter using string expression.

   :param str expression: Boolean expression to evaluate
   :returns: Filtered table
   :rtype: GraphTable

   **Example:**

   .. code-block:: python

      # String-based filtering
      result = table.query("age > 25 AND dept == 'eng'")

Column Selection
~~~~~~~~~~~~~~~~

.. method:: GraphTable.select(columns)

   Select specific columns.

   :param list columns: Column names to select
   :returns: Table with selected columns
   :rtype: GraphTable

.. method:: GraphTable.drop(columns)

   Drop specific columns.

   :param columns: Column names to drop
   :type columns: str or list
   :returns: Table without dropped columns
   :rtype: GraphTable

Sorting and Ordering
--------------------

.. method:: GraphTable.sort_by(column, ascending=True)

   Sort table by column values.

   :param str column: Column name to sort by
   :param bool ascending: Sort in ascending order
   :returns: Sorted table
   :rtype: GraphTable

.. method:: GraphTable.sort_values(by, ascending=True)

   Sort by multiple columns.

   :param by: Column name(s) to sort by
   :type by: str or list
   :param ascending: Sort direction(s)
   :type ascending: bool or list
   :returns: Sorted table
   :rtype: GraphTable

.. method:: GraphTable.reverse()

   Reverse row order.

   :returns: Table with reversed row order
   :rtype: GraphTable

Data Cleaning
-------------

.. method:: GraphTable.drop_duplicates(subset=None)

   Remove duplicate rows.

   :param list subset: Columns to consider for duplicates
   :returns: Table without duplicates
   :rtype: GraphTable

.. method:: GraphTable.dropna(how='any', subset=None)

   Remove rows with missing values.

   :param str how: 'any' or 'all' - when to drop
   :param list subset: Columns to consider for nulls
   :returns: Table without null rows
   :rtype: GraphTable

.. method:: GraphTable.fillna(value, method=None)

   Fill missing values.

   :param value: Value to fill with, or dict mapping column->value
   :param str method: Fill method ('forward', 'backward')
   :returns: Table with filled values
   :rtype: GraphTable

.. method:: GraphTable.replace(to_replace, value)

   Replace values in the table.

   :param to_replace: Value(s) to replace
   :param value: Replacement value(s)
   :returns: Table with replaced values
   :rtype: GraphTable

Aggregation and Grouping
------------------------

Group Operations
~~~~~~~~~~~~~~~~

.. method:: GraphTable.group_by(by)

   Group rows by column values.

   :param by: Column name(s) to group by
   :type by: str or list
   :returns: GroupBy object for aggregation
   :rtype: TableGroupBy

   **Example:**

   .. code-block:: python

      # Group and aggregate
      by_dept = table.group_by('dept')
      stats = by_dept.agg({
          'age': ['mean', 'min', 'max'],
          'salary': 'mean'
      })

.. method:: GraphTable.agg(functions)

   Aggregate the entire table.

   :param functions: Aggregation functions by column
   :type functions: dict
   :returns: Aggregated results
   :rtype: dict or GraphTable

   **Example:**

   .. code-block:: python

      # Multiple aggregations
      summary = table.agg({
          'age': ['mean', 'std'],
          'dept': 'nunique',
          'salary': ['min', 'max']
      })

Statistical Operations
~~~~~~~~~~~~~~~~~~~~~~

.. method:: GraphTable.mean(numeric_only=True)

   Mean of numeric columns.

   :param bool numeric_only: Only include numeric columns
   :returns: Column means
   :rtype: dict

.. method:: GraphTable.median(numeric_only=True)

   Median of numeric columns.

   :param bool numeric_only: Only include numeric columns
   :returns: Column medians
   :rtype: dict

.. method:: GraphTable.std(numeric_only=True)

   Standard deviation of numeric columns.

   :param bool numeric_only: Only include numeric columns
   :returns: Column standard deviations
   :rtype: dict

.. method:: GraphTable.min(numeric_only=False)

   Minimum values per column.

   :param bool numeric_only: Only include numeric columns
   :returns: Column minimums
   :rtype: dict

.. method:: GraphTable.max(numeric_only=False)

   Maximum values per column.

   :param bool numeric_only: Only include numeric columns
   :returns: Column maximums
   :rtype: dict

.. method:: GraphTable.count()

   Count of non-null values per column.

   :returns: Non-null counts
   :rtype: dict

Multi-Table Operations
----------------------

Joins
~~~~~

.. method:: GraphTable.join(other, on=None, how='inner', left_on=None, right_on=None)

   Join with another table.

   :param GraphTable other: Table to join with
   :param on: Column(s) to join on (if same in both tables)
   :type on: str or list
   :param str how: Join type ('inner', 'left', 'right', 'outer')
   :param left_on: Column(s) to join on from left table
   :type left_on: str or list
   :param right_on: Column(s) to join on from right table
   :type right_on: str or list
   :returns: Joined table
   :rtype: GraphTable

   **Examples:**

   .. code-block:: python

      # Inner join on common column
      joined = table1.join(table2, on='id')
      
      # Left join with different column names
      joined = table1.join(table2, left_on='user_id', right_on='id', how='left')

.. method:: GraphTable.merge(other, **kwargs)

   Alias for join() with pandas-compatible interface.

Set Operations
~~~~~~~~~~~~~~

.. method:: GraphTable.union(other)

   Union of two tables (combine rows).

   :param GraphTable other: Table to union with
   :returns: Combined table
   :rtype: GraphTable

.. method:: GraphTable.intersect(other)

   Intersection of two tables (common rows).

   :param GraphTable other: Table to intersect with
   :returns: Common rows
   :rtype: GraphTable

.. method:: GraphTable.difference(other)

   Difference of two tables (rows in self but not other).

   :param GraphTable other: Table to subtract
   :returns: Difference table
   :rtype: GraphTable

Reshaping
---------

.. method:: GraphTable.pivot(index, columns, values)

   Pivot table from long to wide format.

   :param str index: Column to use as index
   :param str columns: Column to use for new column names
   :param str values: Column to use for values
   :returns: Pivoted table
   :rtype: GraphTable

.. method:: GraphTable.melt(id_vars=None, value_vars=None, var_name='variable', value_name='value')

   Melt table from wide to long format.

   :param list id_vars: Columns to keep as identifiers
   :param list value_vars: Columns to melt
   :param str var_name: Name for variable column
   :param str value_name: Name for value column
   :returns: Melted table
   :rtype: GraphTable

.. method:: GraphTable.transpose()

   Transpose the table (swap rows and columns).

   :returns: Transposed table
   :rtype: GraphTable

Graph-Aware Operations
----------------------

Neighborhood Analysis
~~~~~~~~~~~~~~~~~~~~~

.. method:: GraphTable.neighborhood_table(graph, node_column, target_node, attributes=None)

   Create table of node's neighbors.

   :param graph: Graph to analyze
   :type graph: Graph
   :param str node_column: Column containing node IDs
   :param target_node: Node to find neighbors for
   :param list attributes: Specific attributes to include
   :returns: Neighbor attributes table
   :rtype: GraphTable

.. method:: GraphTable.k_hop_neighborhood_table(graph, node_column, target_node, k=1, attributes=None)

   Create table of k-hop neighbors.

   :param graph: Graph to analyze
   :type graph: Graph
   :param str node_column: Column containing node IDs
   :param target_node: Node to find neighbors for
   :param int k: Maximum distance for neighbors
   :param list attributes: Specific attributes to include
   :returns: K-hop neighbor attributes table
   :rtype: GraphTable

Graph-Based Filtering
~~~~~~~~~~~~~~~~~~~~~

.. method:: GraphTable.filter_by_degree(graph, node_column, min_degree=None, max_degree=None)

   Filter rows by node degree.

   :param graph: Graph to analyze
   :type graph: Graph
   :param str node_column: Column containing node IDs
   :param int min_degree: Minimum degree threshold
   :param int max_degree: Maximum degree threshold
   :returns: Filtered table
   :rtype: GraphTable

.. method:: GraphTable.filter_by_connectivity(graph, node_column, target_nodes, mode='any')

   Filter by connectivity to target nodes.

   :param graph: Graph to analyze
   :type graph: Graph
   :param str node_column: Column containing node IDs
   :param list target_nodes: Nodes to check connectivity to
   :param str mode: 'any' or 'all' - connectivity requirement
   :returns: Filtered table
   :rtype: GraphTable

.. method:: GraphTable.filter_by_distance(graph, node_column, target_nodes, max_distance)

   Filter by maximum distance to target nodes.

   :param graph: Graph to analyze
   :type graph: Graph
   :param str node_column: Column containing node IDs
   :param list target_nodes: Nodes to measure distance to
   :param int max_distance: Maximum allowed distance
   :returns: Filtered table
   :rtype: GraphTable

Export and Import
-----------------

File Operations
~~~~~~~~~~~~~~~

.. method:: GraphTable.to_csv(filename, **kwargs)

   Export to CSV file.

   :param str filename: Output filename
   :param kwargs: Additional CSV options

.. method:: GraphTable.to_json(filename, orient='records')

   Export to JSON file.

   :param str filename: Output filename
   :param str orient: JSON orientation

.. method:: GraphTable.to_parquet(filename)

   Export to Parquet file.

   :param str filename: Output filename

Library Integration
~~~~~~~~~~~~~~~~~~~

.. method:: GraphTable.to_pandas()

   Convert to pandas DataFrame.

   :returns: DataFrame representation
   :rtype: pandas.DataFrame

.. method:: GraphTable.to_numpy()

   Convert to NumPy array.

   :returns: Array representation (numeric columns only)
   :rtype: numpy.ndarray

.. method:: GraphTable.to_dict(orient='dict')

   Convert to dictionary.

   :param str orient: Dictionary orientation
   :returns: Dictionary representation
   :rtype: dict

.. method:: GraphTable.to_records()

   Convert to list of record dictionaries.

   :returns: List of row dictionaries
   :rtype: list

Indexing
--------

.. method:: GraphTable.set_index(column)

   Set a column as the index.

   :param str column: Column to use as index
   :returns: Table with new index
   :rtype: GraphTable

.. method:: GraphTable.reset_index()

   Reset index to default integer range.

   :returns: Table with reset index
   :rtype: GraphTable

.. method:: GraphTable.reindex(new_index)

   Reorder rows according to new index.

   :param new_index: New index values
   :type new_index: list or GraphArray
   :returns: Reindexed table
   :rtype: GraphTable

Utility Methods
---------------

.. method:: GraphTable.copy()

   Create a copy of the table.

   :returns: New table with copied data
   :rtype: GraphTable

.. method:: GraphTable.equals(other, tolerance=1e-9)

   Check equality with another table.

   :param GraphTable other: Table to compare with
   :param float tolerance: Tolerance for numeric comparison
   :returns: True if tables are equal
   :rtype: bool

.. method:: GraphTable.rename(columns)

   Rename columns.

   :param dict columns: Mapping of old_name -> new_name
   :returns: Table with renamed columns
   :rtype: GraphTable

.. method:: GraphTable.add_column(name, values)

   Add a new column.

   :param str name: Column name
   :param values: Column values
   :type values: list or GraphArray
   :returns: Table with new column
   :rtype: GraphTable

Display Methods
---------------

.. method:: GraphTable.__repr__()

   String representation for terminals.

   :returns: String representation
   :rtype: str

.. method:: GraphTable._repr_html_()

   HTML representation for Jupyter notebooks.

   :returns: HTML string
   :rtype: str

.. method:: GraphTable.preview(max_rows=10, max_cols=None)

   Preview with limited rows and columns.

   :param int max_rows: Maximum rows to show
   :param int max_cols: Maximum columns to show
   :returns: Preview string
   :rtype: str

Performance Characteristics
---------------------------

- **Lazy Evaluation**: Operations are computed on-demand and cached
- **Columnar Storage**: Memory-efficient storage with cache locality
- **Vectorized Operations**: SIMD-optimized operations for numeric data
- **Smart Indexing**: B-tree indexes for fast lookups and joins
- **Memory Mapping**: Large tables can be memory-mapped for efficiency

Type System
-----------

GraphTable supports heterogeneous data:

- **Numeric**: int8, int16, int32, int64, float32, float64
- **String**: Variable-length UTF-8 strings
- **Boolean**: True/False values
- **DateTime**: Timestamp and date types
- **Categorical**: Efficient storage for repeated string values
- **Null**: Missing values with proper null semantics

Best Practices
--------------

1. **Use appropriate data types** - categorical for repeated strings, int32 vs int64
2. **Leverage lazy evaluation** - chain operations without intermediate materialization
3. **Use vectorized operations** - prefer column operations over row iteration
4. **Index for joins** - set indexes on columns used for frequent joins
5. **Consider graph topology** - use graph-aware filtering when available

**Example workflow:**

.. code-block:: python

   import groggy as gr
   
   # Load data
   table = gr.table.from_csv('users.csv')
   
   # Data cleaning
   clean_table = (table
       .dropna(subset=['age'])
       .filter_rows(lambda r: r['age'] > 0)
       .drop_duplicates()
   )
   
   # Analysis
   dept_stats = (clean_table
       .group_by('department')
       .agg({'age': ['mean', 'std'], 'salary': 'mean'})
   )
   
   # Graph integration
   if 'user_id' in table.columns:
       # Filter by graph connectivity
       central_users = table.filter_by_degree(
           g, 'user_id', min_degree=5
       )

GraphTable provides the foundation for relational analysis in Groggy's storage view system.