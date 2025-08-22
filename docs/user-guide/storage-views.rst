Storage Views
=============

Groggy's storage views provide seamless integration between graph topology and tabular analytics through unified interfaces that work with your graph data.

Overview
--------

Groggy provides integrated storage views for different data analysis needs:

- **GraphTable**: Multi-column data with relational operations and graph integration
- **GraphArray**: Single-column data with statistical operations  
- **GraphMatrix**: Matrix data for adjacency and linear algebra operations

These views work together to provide a unified graph-table analysis experience.

GraphTable: Core Table Operations
---------------------------------

GraphTable is the primary interface for working with tabular graph data, providing both pandas-like operations and graph-aware analysis.

Creating Tables
~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy as gr

   # Create table from dictionary
   data = {
       'name': ['Alice', 'Bob', 'Charlie'],
       'age': [30, 25, 35],
       'department': ['Engineering', 'Design', 'Management']
   }
   table = gr.table(data)
   
   # Get tables from graph data  
   g = gr.Graph()
   alice = g.add_node(name="Alice", age=30, department="Engineering")
   bob = g.add_node(name="Bob", age=25, department="Design")
   
   # Access as tables
   nodes_table = g.nodes.table()
   edges_table = g.edges.table()

Basic Table Properties
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Table information
   print(f"Shape: {table.shape}")          # (rows, columns)
   print(f"Columns: {table.columns}")      # List of column names
   print(f"Length: {len(table)}")          # Number of rows

   # Access data
   print(table.head())                     # First few rows
   print(table.tail())                     # Last few rows

   # Column access (returns GraphArray)
   ages = nodes_table['age']
   names = nodes_table['name']

Statistical Operations  
~~~~~~~~~~~~~~~~~~~~~

Statistical operations are computed efficiently in Rust:

.. code-block:: python

   # Statistical operations on columns (use array operations)
   age_column = table['age']
   avg_age = age_column.mean()
   print(f"Mean age: {avg_age}")
   print(f"Std dev: {age_column.std()}")
   print(f"Min age: {age_column.min()}")
   print(f"Max age: {age_column.max()}")
   
   # Statistical summary
   summary = table.describe()
   print(summary)

Sorting and Filtering
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sort table by column
   sorted_table = table.sort_by('age', ascending=True)
   sorted_desc = table.sort_by('name', ascending=False)

   # Boolean indexing on tables
   young_people = nodes_table[nodes_table['age'] < 30]
   engineers = nodes_table[nodes_table['department'] == 'Engineering']

   # Access individual rows
   first_row = table[0]  # First row as dictionary-like object

Array Operations
~~~~~~~~~~~~~~~

.. code-block:: python

   # Working with individual columns (GraphArray)
   age_column = table['age']
   
   # Basic array statistics
   print(f"Mean: {age_column.mean()}")
   print(f"Std dev: {age_column.std()}")
   print(f"Summary: {age_column.describe()}")

Export and Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to pandas for advanced analysis
   pandas_df = table.to_pandas()
   
   # Export to common formats
   # table.to_csv('data.csv')  # Will be available in future releases
   
   # Work with NetworkX for graph algorithms
   nx_graph = g.to_networkx()
   # Combine with table data for comprehensive analysis

GraphMatrix: Adjacency Matrices
--------------------------------

GraphMatrix primarily handles adjacency matrix operations for graph analysis.

Creating Adjacency Matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get adjacency matrix from graph
   adj_matrix = g.adjacency()
   
   # Check matrix properties
   print(f"Shape: {adj_matrix.shape}")
   print(f"Is sparse: {adj_matrix.is_sparse}")
   
   # Create custom matrix data
   matrix_data = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
   custom_matrix = gr.matrix(matrix_data)

Matrix Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic matrix properties
   print(f"Shape: {adj_matrix.shape}")
   print(f"Is sparse: {adj_matrix.is_sparse}")

   # Axis operations (available based on Rust implementation)
   row_sums = adj_matrix.sum_axis(1)      # Sum each row (out-degrees)
   col_sums = adj_matrix.sum_axis(0)      # Sum each column (in-degrees)
   
   # Statistical operations along axes
   row_means = adj_matrix.mean_axis(1)    # Mean of each row
   col_stds = adj_matrix.std_axis(0)      # Std dev of each column

   # Matrix powers for path analysis
   adj_squared = adj_matrix.power(2)      # 2-step paths
   adj_cubed = adj_matrix.power(3)        # 3-step paths

Matrix Export
~~~~~~~~~~~~~

.. code-block:: python

   # Convert to NumPy for scientific computing
   numpy_array = adj_matrix.to_numpy()
   
   # Use with other libraries
   import numpy as np
   eigenvals = np.linalg.eigvals(numpy_array)

Graph-Aware Table Operations
----------------------------

The real power of Groggy's storage views comes from integrating graph topology with table operations.

Graph-Aware Filtering
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter table rows by graph properties
   nodes_table = g.nodes.table()
   
   # Filter by node degree
   high_degree_nodes = nodes_table.filter_by_degree(g, 'node_id', min_degree=3)
   
   # Filter by connectivity to specific nodes
   connected_to_alice = nodes_table.filter_by_connectivity(
       g, 'node_id', [alice]
   )
   
   # Filter by graph distance
   nearby_nodes = nodes_table.filter_by_distance(
       g, 'node_id', [alice], max_distance=2
   )

Table Joins
~~~~~~~~~~~

.. code-block:: python

   # Create second table for joining
   departments = gr.table({
       'dept_id': ['eng', 'design', 'mgmt'],
       'budget': [500000, 300000, 200000]
   })

   # Join operations (based on Rust implementation)
   inner_result = nodes_table.inner_join(departments, 'department', 'dept_id')
   left_result = nodes_table.left_join(departments, 'department', 'dept_id')
   right_result = nodes_table.right_join(departments, 'department', 'dept_id')
   outer_result = nodes_table.outer_join(departments, 'department', 'dept_id')

Complete Workflow Example
-------------------------

Combining Graph and Table Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete workflow demonstrating storage view integration
   
   # 1. Create graph with data
   g = gr.Graph()
   alice = g.add_node(name="Alice", age=30, department="Engineering", salary=95000)
   bob = g.add_node(name="Bob", age=25, department="Design", salary=75000)
   charlie = g.add_node(name="Charlie", age=35, department="Management", salary=120000)
   
   g.add_edge(alice, bob, relationship="collaborates", strength=0.8)
   g.add_edge(charlie, alice, relationship="manages", strength=0.9)
   
   # 2. Get table view of nodes
   nodes_table = g.nodes.table()
   
   # 3. Analyze with table operations
   salary_column = nodes_table['salary']
   avg_salary = salary_column.mean()
   sorted_by_age = nodes_table.sort_by('age', ascending=False)
   
   # 4. Graph-aware filtering
   high_degree_employees = nodes_table.filter_by_degree(g, 'node_id', min_degree=2)
   
   # 5. Statistical analysis on columns
   salary_column = nodes_table['salary']
   salary_stats = salary_column.describe()
   
   # 6. Matrix operations on graph structure
   adj_matrix = g.adjacency()
   path_matrix = adj_matrix.power(2)  # 2-step connections
   
   # 7. Export for external analysis
   pandas_df = nodes_table.to_pandas()
   
   print(f"Average salary: ${avg_salary:,.2f}")
   print(f"High-degree employees: {len(high_degree_employees)}")

Future Enhancements
------------------

The current storage views provide a solid foundation. Future releases will add:

- **Advanced Table Operations**: Groupby aggregations, pivot tables
- **More Export Formats**: CSV, JSON, Parquet export  
- **Enhanced Array Operations**: More statistical functions and transformations
- **Matrix Operations**: Advanced linear algebra operations
- **Performance Optimizations**: Improved lazy evaluation and caching

Best Practices
--------------

1. **Use appropriate views**: Tables for heterogeneous data, arrays for single columns, matrices for adjacency
2. **Leverage graph-aware operations**: Use topology-based filtering for powerful analysis
3. **Combine views**: Use tables for data prep, arrays for statistics, matrices for structure
4. **Export strategically**: Convert to pandas/NumPy when you need specific external libraries

The storage views provide the foundation for Groggy's unified graph-table analysis approach.