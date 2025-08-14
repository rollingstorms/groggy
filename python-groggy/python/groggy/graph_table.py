"""
GraphTable - DataFrame-like views for graph data

This module provides GraphTable class for tabular views of graph nodes and edges,
similar to pandas DataFrames but optimized for graph data.
"""

from typing import Dict, List, Any, Optional, Set, Union
import json
import csv
from io import StringIO

class GraphTable:
    """DataFrame-like table view for graph nodes or edges."""
    
    def __init__(self, data_source, table_type="nodes", graph=None):
        """
        Initialize GraphTable from a graph data source.
        
        Args:
            data_source: Graph, Subgraph, or list of node/edge IDs
            table_type: "nodes" or "edges"
            graph: Optional graph reference for subgraphs that don't have graph access
        """
        self.data_source = data_source
        self.table_type = table_type
        self.graph_override = graph
        self._cached_data = None
        self._cached_columns = None
    
    def _get_graph(self):
        """Get the underlying graph object."""
        # Use override graph if provided
        if self.graph_override is not None:
            return self.graph_override
        # Check if it's a EnhancedSubgraph - these don't have graph references
        elif hasattr(self.data_source, 'subgraph_type') and hasattr(self.data_source, 'nodes'):
            # This is an EnhancedSubgraph - we need to get the graph from somewhere else
            raise ValueError("EnhancedSubgraph needs graph reference - pass graph parameter to GraphTable")
        elif hasattr(self.data_source, 'graph') and hasattr(self.data_source.graph, 'borrow'):
            # This is a subgraph with a graph reference
            return self.data_source.graph.borrow()
        elif hasattr(self.data_source, 'nodes') and hasattr(self.data_source, 'edges'):
            # This is a graph object
            return self.data_source
        else:
            raise ValueError("Cannot extract graph from data source")
    
    def _extract_value(self, value):
        """Extract the actual value from AttrValue objects."""
        if hasattr(value, 'inner'):
            return value.inner
        else:
            return value
    
    def _get_ids(self):
        """Get the list of node or edge IDs to include in the table."""
        if hasattr(self.data_source, 'nodes') and hasattr(self.data_source, 'edges'):
            # Graph or Subgraph object
            if self.table_type == "nodes":
                if hasattr(self.data_source, 'nodes') and isinstance(self.data_source.nodes, list):
                    # Subgraph with nodes list
                    return self.data_source.nodes
                else:
                    # Graph with node_ids property
                    return list(self.data_source.node_ids)
            else:  # edges
                if hasattr(self.data_source, 'edges') and isinstance(self.data_source.edges, list):
                    # Subgraph with edges list
                    return self.data_source.edges
                else:
                    # Graph with edge_ids property
                    return list(self.data_source.edge_ids)
        elif isinstance(self.data_source, list):
            # Direct list of IDs
            return self.data_source
        else:
            raise ValueError(f"Unsupported data source type: {type(self.data_source)}")
    
    def _discover_attributes(self) -> Set[str]:
        """Discover all attributes present across nodes or edges."""
        graph = self._get_graph()
        ids = self._get_ids()
        attributes = set()
        
        # Comprehensive list of common attributes to check
        if self.table_type == "nodes":
            common_attrs = ['name', 'age', 'dept', 'salary', 'seniority', 'index', 'level', 'component_id', 'influence_score', 'id']
        else:
            common_attrs = ['weight', 'type', 'relationship', 'strength', 'frequency', 'last_contact']
        
        if self.table_type == "nodes":
            for node_id in ids:
                try:
                    node_view = graph.nodes[node_id]
                    # Check all common attributes
                    for attr_name in common_attrs:
                        try:
                            if hasattr(node_view, '__getitem__'):
                                _ = node_view[attr_name]
                                attributes.add(attr_name)
                        except (KeyError, AttributeError):
                            continue
                except Exception:
                    continue
        else:  # edges
            for edge_id in ids:
                try:
                    edge_view = graph.edges[edge_id]
                    # Check all common attributes
                    for attr_name in common_attrs:
                        try:
                            if hasattr(edge_view, '__getitem__'):
                                _ = edge_view[attr_name]
                                attributes.add(attr_name)
                        except (KeyError, AttributeError):
                            continue
                    # Always include source and target for edges
                    attributes.add('source')
                    attributes.add('target')
                except Exception:
                    continue
        
        return attributes
    
    def _build_table_data(self):
        """Build the complete table data with all attributes."""
        if self._cached_data is not None:
            return self._cached_data, self._cached_columns
        
        graph = self._get_graph()
        ids = self._get_ids()
        attributes = self._discover_attributes()
        
        # Always include 'id' as first column
        columns = ['id'] + sorted([attr for attr in attributes if attr != 'id'])
        self._cached_columns = columns
        
        # Build table rows
        rows = []
        for item_id in ids:
            row = {'id': item_id}
            
            if self.table_type == "nodes":
                try:
                    node_view = graph.nodes[item_id]
                    for attr_name in columns[1:]:  # Skip 'id' since we already set it
                        try:
                            if hasattr(node_view, '__getitem__'):
                                value = node_view[attr_name]
                                # Handle AttrValue objects - extract inner value
                                row[attr_name] = self._extract_value(value)
                            else:
                                row[attr_name] = None
                        except (KeyError, AttributeError):
                            row[attr_name] = None
                except Exception:
                    # Fill with None for inaccessible nodes
                    for attr_name in columns[1:]:
                        row[attr_name] = None
            else:  # edges
                try:
                    edge_view = graph.edges[item_id]
                    for attr_name in columns[1:]:  # Skip 'id' since we already set it
                        try:
                            if attr_name == 'source':
                                row[attr_name] = edge_view.source
                            elif attr_name == 'target':
                                row[attr_name] = edge_view.target
                            elif hasattr(edge_view, '__getitem__'):
                                value = edge_view[attr_name]
                                # Handle AttrValue objects - extract inner value
                                row[attr_name] = self._extract_value(value)
                            else:
                                row[attr_name] = None
                        except (KeyError, AttributeError):
                            row[attr_name] = None
                except Exception:
                    # Fill with None for inaccessible edges
                    for attr_name in columns[1:]:
                        row[attr_name] = None
            
            rows.append(row)
        
        self._cached_data = rows
        return rows, columns
    
    def __repr__(self):
        """String representation of the table."""
        rows, columns = self._build_table_data()
        
        if not rows:
            return f"GraphTable({self.table_type}, 0 rows, 0 columns)"
        
        # Create formatted table
        lines = []
        
        # Header
        header = "   " + "  ".join(f"{col:>10}" for col in columns)
        lines.append(header)
        
        # Rows
        for i, row in enumerate(rows):
            formatted_row = f"{i:2d} "
            for col in columns:
                value = row.get(col)
                if value is None:
                    formatted_value = "NaN"
                elif isinstance(value, str):
                    formatted_value = value[:10]  # Truncate long strings
                elif isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                formatted_row += f"  {formatted_value:>10}"
            lines.append(formatted_row)
        
        return "\n".join(lines)
    
    def __len__(self):
        """Number of rows in the table."""
        ids = self._get_ids()
        return len(ids)
    
    def __getitem__(self, key):
        """Access columns or rows like a DataFrame."""
        rows, columns = self._build_table_data()
        
        if isinstance(key, str):
            # Column access
            if key not in columns:
                raise KeyError(f"Column '{key}' not found")
            return [row.get(key) for row in rows]
        elif isinstance(key, int):
            # Row access
            if key < 0 or key >= len(rows):
                raise IndexError(f"Row index {key} out of range")
            return rows[key]
        else:
            raise TypeError("Key must be string (column) or int (row)")
    
    @property
    def columns(self):
        """Get column names."""
        _, columns = self._build_table_data()
        return columns
    
    @property
    def shape(self):
        """Get table shape (rows, columns)."""
        rows, columns = self._build_table_data()
        return (len(rows), len(columns))
    
    def to_dict(self):
        """Convert to dictionary format."""
        rows, columns = self._build_table_data()
        return {
            'columns': columns,
            'data': rows,
            'shape': self.shape,
            'type': self.table_type
        }
    
    def to_pandas(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_pandas(). Install with: pip install pandas")
        
        rows, columns = self._build_table_data()
        return pd.DataFrame(rows, columns=['id'] + [col for col in columns if col != 'id'])
    
    def to_csv(self, filename: str = None, **kwargs):
        """Export to CSV format."""
        rows, columns = self._build_table_data()
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        
        csv_content = output.getvalue()
        output.close()
        
        if filename:
            with open(filename, 'w', newline='') as f:
                f.write(csv_content)
            return f"GraphTable exported to {filename}"
        else:
            return csv_content
    
    def to_json(self, filename: str = None, **kwargs):
        """Export to JSON format with graph metadata."""
        data = self.to_dict()
        
        # Add metadata
        data['metadata'] = {
            'export_type': 'groggy_graph_table',
            'table_type': self.table_type,
            'total_rows': len(data['data']),
            'total_columns': len(data['columns'])
        }
        
        json_content = json.dumps(data, indent=2, default=str)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_content)
            return f"GraphTable exported to {filename}"
        else:
            return json_content
    
    def groupby(self, column: str):
        """Group by a column (simplified groupby)."""
        rows, columns = self._build_table_data()
        
        if column not in columns:
            raise KeyError(f"Column '{column}' not found")
        
        groups = {}
        for row in rows:
            key = row.get(column)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)
        
        return GraphTableGroupBy(groups, column)

class GraphTableGroupBy:
    """Grouped GraphTable for simple aggregations."""
    
    def __init__(self, groups: Dict[Any, List[Dict]], group_column: str):
        self.groups = groups
        self.group_column = group_column
    
    def mean(self, column: str = None):
        """Calculate mean for numeric columns."""
        if column:
            return self._aggregate_column(column, 'mean')
        else:
            return {group_key: self._aggregate_group(group_rows, 'mean') 
                   for group_key, group_rows in self.groups.items()}
    
    def count(self):
        """Count rows in each group."""
        return {group_key: len(group_rows) for group_key, group_rows in self.groups.items()}
    
    def _aggregate_column(self, column: str, agg_func: str):
        """Aggregate a specific column across groups."""
        result = {}
        for group_key, group_rows in self.groups.items():
            values = [row.get(column) for row in group_rows if row.get(column) is not None]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if agg_func == 'mean' and numeric_values:
                result[group_key] = sum(numeric_values) / len(numeric_values)
            else:
                result[group_key] = None
        
        return result
    
    def _aggregate_group(self, rows: List[Dict], agg_func: str):
        """Aggregate all numeric columns in a group."""
        result = {}
        if not rows:
            return result
        
        # Find numeric columns
        for column in rows[0].keys():
            values = [row.get(column) for row in rows if row.get(column) is not None]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if agg_func == 'mean' and numeric_values:
                result[column] = sum(numeric_values) / len(numeric_values)
        
        return result