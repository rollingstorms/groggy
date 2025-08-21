Storage Views
=============

Groggy's storage views provide seamless integration between graph topology and tabular analytics through three unified interfaces: Arrays, Matrices, and Tables.

Overview
--------

The three storage views offer different perspectives on your graph data:

- **GraphArray**: Single-column data with statistical operations
- **GraphMatrix**: Multi-column homogeneous data with linear algebra  
- **GraphTable**: Multi-column heterogeneous data with relational operations

All views support lazy evaluation, meaning operations are computed on-demand and cached for performance.

GraphArray: Statistical Arrays
------------------------------

GraphArray provides NumPy-like functionality for single columns of data with native statistical operations computed in Rust.

Creating Arrays
~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy as gr

   # Create from Python list
   ages = gr.array([25, 30, 35, 40, 45])
   
   # Create from graph data
   g = gr.Graph()
   # ... add nodes with age attribute ...
   table = g.nodes.table()
   age_array = table['age']  # Returns GraphArray

Basic Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Array properties
   print(f"Length: {len(ages)}")
   print(f"Data type: {ages.dtype}")
   print(f"Is sparse: {ages.is_sparse}")

   # Element access
   print(f"First element: {ages[0]}")
   print(f"Last element: {ages[-1]}")
   print(f"Slice: {ages[1:4]}")

   # Iteration
   for age in ages:
       print(age)

Statistical Operations
~~~~~~~~~~~~~~~~~~~~~~

All statistical operations are computed in native Rust for maximum performance:

.. code-block:: python

   # Basic statistics
   print(f"Mean: {ages.mean()}")           # 35.0
   print(f"Median: {ages.median()}")       # 35.0
   print(f"Standard deviation: {ages.std()}")  # 7.91
   print(f"Minimum: {ages.min()}")         # 25
   print(f"Maximum: {ages.max()}")         # 45
   print(f"Sum: {ages.sum()}")             # 175

   # Count operations
   print(f"Count: {ages.count()}")         # 5 (non-null values)
   print(f"Unique values: {len(ages.unique())}")

   # Value distribution
   print(f"Value counts: {ages.value_counts()}")

   # Comprehensive summary
   summary = ages.describe()
   print(summary)  # Dict with count, mean, std, min, max, etc.

Advanced Indexing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Boolean indexing
   mask = [True, False, True, False, True]
   filtered = ages[mask]

   # Fancy indexing
   indices = [0, 2, 4]
   selected = ages[indices]

   # Slice with step
   every_other = ages[::2]

Data Transformations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter with predicate
   adults = ages.filter(lambda x: x >= 30)

   # Transform values
   ages_in_months = ages.map(lambda x: x * 12)

   # Sort
   sorted_ages = ages.sort(ascending=True)
   
   # Reverse
   reversed_ages = ages.reverse()

Integration with Scientific Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to NumPy
   import numpy as np
   ages_numpy = ages.to_numpy()
   correlation = np.corrcoef(ages_numpy, other_array.to_numpy())

   # Convert to pandas
   import pandas as pd
   ages_series = ages.to_pandas()
   ages_series.plot()

GraphMatrix: Matrix Operations
------------------------------

GraphMatrix handles homogeneous multi-column data with linear algebra support.

Creating Matrices
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create from 2D data
   data = [[1, 2], [3, 4], [5, 6]]
   matrix = gr.matrix(data)

   # Create from graph (adjacency matrix)
   adj_matrix = g.adjacency()

   # Create from table columns
   table = g.nodes.table()
   numeric_matrix = table[['age', 'salary']]  # Returns GraphMatrix

Matrix Properties
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Shape and type information
   print(f"Shape: {matrix.shape}")         # (3, 2)
   print(f"Data type: {matrix.dtype}")     # float64
   print(f"Is square: {matrix.is_square}") # False
   print(f"Is sparse: {matrix.is_sparse}") # False

   # Access dimensions
   rows, cols = matrix.shape
   print(f"Rows: {rows}, Columns: {cols}")

Element and Slice Access
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Single element
   element = matrix[0, 1]

   # Row access
   first_row = matrix[0]        # Returns GraphArray
   
   # Column access  
   first_col = matrix[:, 0]     # Returns GraphArray

   # Submatrix
   submatrix = matrix[0:2, :]   # First 2 rows

Matrix Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transpose
   transposed = matrix.transpose()

   # Axis operations
   row_sums = matrix.sum_axis(axis=1)      # Sum each row
   col_means = matrix.mean_axis(axis=0)    # Mean of each column
   col_stds = matrix.std_axis(axis=0)      # Std dev of each column

   # Matrix powers (for adjacency matrices)
   adj_squared = adj_matrix.power(2)       # A²
   adj_cubed = adj_matrix.power(3)         # A³

Format Conversions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sparse/Dense conversion
   sparse_matrix = matrix.to_sparse()
   dense_matrix = sparse_matrix.to_dense()

   # Export to other formats
   numpy_array = matrix.to_numpy()         # 2D NumPy array
   pandas_df = matrix.to_pandas()          # DataFrame with column names

GraphTable: Relational Operations
----------------------------------

GraphTable provides pandas-like functionality for heterogeneous tabular data with graph integration.

Creating Tables
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create from dictionary
   data = {
       'name': ['alice', 'bob', 'charlie'],
       'age': [30, 25, 35],
       'department': ['eng', 'design', 'mgmt']
   }
   table = gr.table(data)

   # Create from graph
   nodes_table = g.nodes.table()
   edges_table = g.edges.table()

   # Create from pandas DataFrame
   import pandas as pd
   df = pd.DataFrame(data)
   table = gr.table.from_pandas(df)

Table Properties
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic information
   print(f"Shape: {table.shape}")          # (3, 3)
   print(f"Columns: {table.columns}")      # ['name', 'age', 'department']
   print(f"Data types: {table.dtypes}")    # {'name': 'string', 'age': 'int', ...}

   # Memory usage
   print(f"Memory usage: {table.memory_usage()} bytes")

Data Access and Slicing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Column access (returns GraphArray)
   ages = table['age']
   names = table['name']

   # Multiple columns (returns GraphTable)
   subset = table[['name', 'age']]

   # Row access
   first_row = table.iloc(0)               # Position-based
   alice_row = table.loc('alice')          # Label-based (if indexed)

   # Head and tail
   print(table.head(2))                    # First 2 rows
   print(table.tail(1))                    # Last 1 row

Data Exploration
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Quick preview
   print(table.head())
   print(table.describe())                 # Statistical summary

   # Random sampling
   sample = table.sample(2)                # 2 random rows

   # Information about the table
   print(f"Non-null values per column:")
   for col in table.columns:
       non_null = table[col].count()
       print(f"  {col}: {non_null}")

Filtering and Selection
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter rows with predicate
   young_people = table.filter_rows(lambda row: row['age'] < 30)
   engineers = table.filter_rows(lambda row: row['department'] == 'eng')

   # Complex filters
   young_engineers = table.filter_rows(
       lambda row: row['age'] < 30 and row['department'] == 'eng'
   )

   # Select specific columns
   names_and_ages = table.select(['name', 'age'])

Sorting and Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sort by column
   sorted_by_age = table.sort_by('age', ascending=True)
   sorted_by_name = table.sort_by('name', ascending=False)

   # Remove duplicates
   unique_table = table.drop_duplicates()

   # Handle missing values
   filled_table = table.fillna('unknown')   # Fill NaN with value
   clean_table = table.dropna()             # Remove rows with NaN

Statistical Operations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Group by operations
   by_dept = table.group_by('department')

   # Aggregations
   dept_stats = by_dept.agg({
       'age': ['mean', 'min', 'max'],
       'name': 'count'
   })

   # Multiple aggregation functions
   summary = table.agg({
       'age': ['mean', 'std'],
       'department': 'nunique'
   })

Multi-Table Operations
----------------------

Joins
~~~~~

.. code-block:: python

   # Create second table
   salaries = gr.table({
       'name': ['alice', 'bob', 'charlie'],
       'salary': [95000, 75000, 120000]
   })

   # Inner join
   combined = table.join(salaries, on='name', how='inner')

   # Left join (keep all rows from left table)
   left_joined = table.join(salaries, on='name', how='left')

   # Right join (keep all rows from right table)  
   right_joined = table.join(salaries, on='name', how='right')

   # Outer join (keep all rows from both tables)
   outer_joined = table.join(salaries, on='name', how='outer')

Set Operations
~~~~~~~~~~~~~~

.. code-block:: python

   # Union (combine rows)
   table1 = gr.table({'id': [1, 2], 'value': [10, 20]})
   table2 = gr.table({'id': [3, 4], 'value': [30, 40]})
   combined = table1.union(table2)

   # Intersection (common rows)
   common = table1.intersect(table2)

Graph-Aware Operations
----------------------

Neighborhood Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get neighborhood as table
   alice_neighbors = gr.GraphTable.neighborhood_table(
       g, "alice", ["age", "department", "salary"]
   )

   # Multi-node neighborhoods
   team_neighbors = gr.GraphTable.multi_neighborhood_table(
       g, ["alice", "bob"], ["age", "role"]
   )

   # K-hop neighborhoods
   extended_network = gr.GraphTable.k_hop_neighborhood_table(
       g, "alice", k=2, ["department", "salary"]
   )

Graph-Aware Filtering
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter by node degree
   influential = table.filter_by_degree(g, 'name', min_degree=3)

   # Filter by connectivity
   connected_to_alice = table.filter_by_connectivity(
       g, 'name', ['alice'], mode='any'
   )

   # Filter by distance
   nearby_nodes = table.filter_by_distance(
       g, 'name', ['alice'], max_distance=2
   )

Export and Integration
----------------------

File Export
~~~~~~~~~~~

.. code-block:: python

   # Export to common formats
   table.to_csv('data.csv')
   table.to_json('data.json')

   # Export specific columns
   table[['name', 'age']].to_csv('names_ages.csv')

Scientific Computing Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to pandas for advanced analysis
   df = table.to_pandas()
   correlation_matrix = df.corr()

   # Convert arrays to NumPy for computation
   ages_np = table['age'].to_numpy()
   salaries_np = table['salary'].to_numpy()
   
   import numpy as np
   correlation = np.corrcoef(ages_np, salaries_np)[0, 1]

Performance Tips
----------------

Lazy Evaluation
~~~~~~~~~~~~~~~

.. code-block:: python

   # Operations are lazy - no computation until needed
   filtered = table.filter_rows(lambda r: r['age'] > 30)  # Instant
   sorted_filtered = filtered.sort_by('age')              # Still lazy
   
   # Computation happens here
   result = sorted_filtered.head(10)                      # Only computes top 10

Memory Efficiency
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use column-specific operations when possible
   mean_age = table['age'].mean()          # Efficient
   
   # Avoid full materialization for large tables
   sample = table.sample(1000)             # Work with subset
   
   # Stream large results
   for chunk in table.iter_chunks(chunk_size=1000):
       process_chunk(chunk)

Caching
~~~~~~~

.. code-block:: python

   # Statistical operations are cached
   ages = table['age']
   mean1 = ages.mean()     # Computed
   mean2 = ages.mean()     # Cached (instant)
   
   # Cache is invalidated when data changes
   ages_modified = ages.filter(lambda x: x > 0)
   mean3 = ages_modified.mean()  # Recomputed

Best Practices
--------------

1. **Choose the right view**: Array for single columns, Matrix for homogeneous data, Table for heterogeneous analysis
2. **Leverage lazy evaluation**: Chain operations without intermediate materialization  
3. **Use graph-aware operations**: Take advantage of topology-based filtering and analysis
4. **Cache expensive operations**: Store results of complex computations
5. **Export strategically**: Convert to pandas/NumPy only when needed for specific libraries

The storage views provide a powerful foundation for graph analytics. Next, explore :doc:`analytics` for advanced graph algorithms and analysis techniques.