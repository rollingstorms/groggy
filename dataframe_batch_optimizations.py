"""
DataFrame and Batch Optimization Methods for Groggy Graph Library
================================================================

This file contains DataFrame/batch optimization methods that were extracted from core.py
due to performance regressions in core graph operations. These optimizations should be
integrated back into the library after the core performance issues are resolved.

The methods provide:
- Ultra-fast DataFrame conversion using vectorized Rust backend operations
- Efficient bulk node/edge attribute retrieval and updates
- Chunked processing for large datasets
- Multi-criteria filtering with optimized intersection logic
- Vectorized single-attribute column operations

Performance Benefits:
- Significantly faster than individual get_node_attribute calls
- Optimized for large-scale data extraction and analytics
- Leverages Rust backend's columnar storage optimizations
- Minimal Python overhead for batch operations

TODO: Integrate these methods back into core.py after resolving performance regressions
"""

from typing import Optional, List, Dict, Any, Union
from .data_structures import Node

# Type alias for compatibility
NodeID = Union[str, int]

class DataFrameBatchOptimizations:
    """
    Container class for DataFrame and batch optimization methods.
    These should be integrated back into the Graph class once performance issues are resolved.
    """

    def to_dataframe(self, attr_names: Optional[List[str]] = None, node_ids: Optional[List[NodeID]] = None, library: str = 'pandas'):
        """Convert graph node data to DataFrame format for efficient analysis
        
        Args:
            attr_names: List of attribute names to include (None = all attributes)
            node_ids: List of node IDs to include (None = all nodes)  
            library: DataFrame library to use ('pandas' or 'polars')
            
        Returns:
            DataFrame with node_id as index and attributes as columns
            
        Performance Notes:
            - Uses vectorized operations for maximum performance
            - Significantly faster than individual get_node_attribute calls
            - Optimized for large-scale attribute retrieval
        """
        if not self.use_rust:
            raise NotImplementedError("DataFrame conversion requires Rust backend")
            
        # Use the ultra-fast DataFrame data retrieval
        data_dict = self.get_dataframe_data_fast(attr_names, node_ids)
        
        if library == 'pandas':
            try:
                import pandas as pd
                return pd.DataFrame(data_dict)
            except ImportError:
                raise ImportError("pandas is required for DataFrame conversion. Install with: pip install pandas")
                
        elif library == 'polars':
            try:
                import polars as pl
                return pl.DataFrame(data_dict)
            except ImportError:
                raise ImportError("polars is required for DataFrame conversion. Install with: pip install polars")
        else:
            raise ValueError("library must be 'pandas' or 'polars'")
    
    def get_dataframe_data_fast(self, attr_names: Optional[List[str]] = None, node_ids: Optional[List[NodeID]] = None):
        """Ultra-fast DataFrame data retrieval using vectorized Rust backend
        
        Args:
            attr_names: List of attribute names to include (None = all attributes)
            node_ids: List of node IDs to include (None = all nodes)
            
        Returns:
            Dictionary with attribute names as keys and aligned value lists
            
        Performance Notes:
            - Uses vectorized Rust operations for maximum performance
            - Optimized for large-scale data extraction
            - Much faster than individual attribute retrieval
        """
        if not self.use_rust:
            raise NotImplementedError("Fast DataFrame data requires Rust backend")
            
        node_ids_str = None
        if node_ids is not None:
            node_ids_str = [str(nid) for nid in node_ids]
            
        return self._rust_core.get_dataframe_data(attr_names, node_ids_str)
    
    def get_bulk_node_attribute_vectors(self, attr_names: List[str], node_ids: Optional[List[NodeID]] = None):
        """Ultra-fast vectorized bulk attribute retrieval using columnar store optimization
        
        Args:
            attr_names: List of attribute names to retrieve
            node_ids: Optional list of specific node IDs (None = all nodes)
            
        Returns:
            Dictionary mapping attribute names to (node_indices, values) tuples
            
        Performance Notes:
            - Uses vectorized Rust operations for maximum performance
            - Optimized for large-scale data extraction with minimal Python overhead
            - Much faster than individual attribute retrieval for batch operations
        """
        if not self.use_rust:
            raise NotImplementedError("Vectorized bulk retrieval requires Rust backend")
            
        node_ids_str = None
        if node_ids is not None:
            node_ids_str = [str(nid) for nid in node_ids]
            
        return self._rust_core.get_bulk_node_attribute_vectors(attr_names, node_ids_str)

    def get_single_attribute_vectorized(self, attr_name: str, node_ids: Optional[List[NodeID]] = None):
        """Ultra-fast single attribute column retrieval using vectorized columnar operations
        
        Args:
            attr_name: Name of the attribute to retrieve
            node_ids: Optional list of specific node IDs (None = all nodes with attribute)
            
        Returns:
            Tuple of (node_ids, values) as aligned lists
            
        Performance Notes:
            - Uses vectorized Rust operations for maximum single-attribute performance
            - Optimized for cases like "get all salaries" or "get all weights"
            - Ideal for analytics and statistical operations
        """
        if not self.use_rust:
            raise NotImplementedError("Vectorized attribute retrieval requires Rust backend")
            
        node_ids_str = None
        if node_ids is not None:
            node_ids_str = [str(nid) for nid in node_ids]
            
        return self._rust_core.get_single_attribute_vectorized(attr_name, node_ids_str)

    def export_node_dataframe_optimized(self, attr_names: Optional[List[str]] = None, node_ids: Optional[List[NodeID]] = None):
        """Ultra-fast DataFrame-style bulk data export optimized for pandas/polars
        
        Args:
            attr_names: List of attribute names to include (None = all attributes)
            node_ids: List of node IDs to include (None = all nodes)
            
        Returns:
            Dictionary with attribute names as keys and aligned value lists
            Ready for pandas.DataFrame(data) or polars.DataFrame(data)
            
        Performance Notes:
            - Uses the most optimized Rust DataFrame export method
            - Includes node_id column automatically
            - Optimized for direct DataFrame library consumption
            - Much faster than to_dataframe() for large datasets
        """
        if not self.use_rust:
            raise NotImplementedError("Optimized DataFrame export requires Rust backend")
            
        node_ids_str = None
        if node_ids is not None:
            node_ids_str = [str(nid) for nid in node_ids]
            
        return self._rust_core.export_node_dataframe_optimized(attr_names, node_ids_str)

    def add_nodes_chunked(self, nodes_data: List[Dict[str, Any]], chunk_size: Optional[int] = None):
        """High-performance chunked node creation for very large datasets
        
        Args:
            nodes_data: List of node dictionaries, each containing 'id' and optional attributes
            chunk_size: Optional chunk size for processing (default: 10,000)
            
        Performance Notes:
            - Processes nodes in chunks to optimize memory usage and reduce lock contention
            - Uses optimized bulk operations internally for maximum performance
            - Ideal for loading large datasets (>100k nodes) efficiently
        """
        if not self.use_rust:
            # Fallback to regular add_nodes for Python backend
            self.add_nodes(nodes_data)
            return
            
        # Prepare data for Rust backend
        rust_nodes = []
        for node_data in nodes_data:
            node_id = str(node_data['id'])
            attributes = {k: v for k, v in node_data.items() if k != 'id'}
            rust_nodes.append((node_id, attributes))
        
        # Use bulk operation without frequent cache invalidation
        self._rust_core.add_nodes_chunked(rust_nodes, chunk_size)
        # Only invalidate cache once at the end
        self._invalidate_cache()

    def set_node_attributes_chunked(self, updates: Dict[NodeID, Dict[str, Any]], chunk_size: Optional[int] = None):
        """High-performance chunked attribute updates for large datasets
        
        Args:
            updates: Dictionary mapping node IDs to their attribute updates
            chunk_size: Optional chunk size for processing (default: 10,000)
            
        Performance Notes:
            - Uses chunked processing to optimize memory usage and reduce lock contention
            - Much faster than individual set_node_attribute calls for large datasets
            - Leverages Rust backend's optimized bulk operations
        """
        if self.use_rust:
            # Convert updates to the format expected by Rust
            updates_dict = {}
            for node_id, attrs in updates.items():
                updates_dict[str(node_id)] = attrs
            
            self._rust_core.set_node_attributes_chunked(updates_dict, chunk_size)
            self._invalidate_cache()
        else:
            # Python fallback - batch update with single delta
            self._init_delta()
            effective_nodes, _, _ = self._get_effective_data()
            
            chunk_size = chunk_size or 10000
            items = list(updates.items())
            
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                
                for node_id, attrs in chunk:
                    node_id_str = str(node_id)
                    
                    # Get current node or create if it doesn't exist
                    if node_id_str in effective_nodes:
                        current_node = effective_nodes[node_id_str]
                        updated_attrs = {**current_node.attributes, **attrs}
                    elif node_id_str in self._pending_delta.added_nodes:
                        current_node = self._pending_delta.added_nodes[node_id_str]
                        updated_attrs = {**current_node.attributes, **attrs}
                    else:
                        # Create new node with the attributes
                        updated_attrs = attrs
                    
                    updated_node = Node(node_id_str, updated_attrs)
                    self._pending_delta.added_nodes[node_id_str] = updated_node
                    self._update_cache_for_node_add(node_id_str, updated_node)

    def get_attribute_column(self, attr_name: str, node_ids: Optional[List[NodeID]] = None) -> tuple:
        """Ultra-fast single attribute column retrieval
        
        Args:
            attr_name: Name of the attribute to retrieve
            node_ids: Optional list of specific node IDs (None = all nodes with attribute)
            
        Returns:
            Tuple of (node_ids, values) as aligned lists
            
        Performance Notes:
            - Optimized for single-attribute analytics operations
            - Uses vectorized Rust backend for maximum performance
        """
        if self.use_rust:
            node_ids_str = None
            if node_ids is not None:
                node_ids_str = [str(nid) for nid in node_ids]
            return self._rust_core.get_single_attribute_vectorized(attr_name, node_ids_str)
        else:
            # Python fallback
            result_ids = []
            result_values = []
            effective_nodes, _, _ = self._get_effective_data()
            
            for node_id, node in effective_nodes.items():
                if attr_name in node.attributes:
                    result_ids.append(node_id)
                    result_values.append(node.attributes[attr_name])
            
            return result_ids, result_values

    def filter_nodes_multi_criteria(
        self,
        exact_matches: Optional[Dict[str, Any]] = None,
        numeric_comparisons: Optional[List[tuple]] = None,
        string_comparisons: Optional[List[tuple]] = None
    ) -> List[NodeID]:
        """Advanced multi-criteria node filtering with high performance
        
        Args:
            exact_matches: Dictionary of attribute_name -> expected_value for exact matches
            numeric_comparisons: List of (attribute_name, operator, value) for numeric comparisons
                                 Operators: '>', '>=', '<', '<=', '==', '!='
            string_comparisons: List of (attribute_name, operator, value) for string comparisons
                               Operators: '==', '!=', 'contains', 'startswith', 'endswith'
            
        Returns:
            List of node IDs that match all criteria
            
        Example:
            # Find high-paid engineers in SF
            nodes = g.filter_nodes_multi_criteria(
                exact_matches={'department': 'engineering', 'location': 'SF'},
                numeric_comparisons=[('salary', '>', 100000), ('age', '<', 40)]
            )
            
        Performance Notes:
            - Uses Rust backend with early termination for maximum performance
            - Intersects filter results efficiently to minimize data processing
            - All filtering logic happens in Rust to avoid Python overhead
        """
        if self.use_rust:
            # Convert to the format expected by Rust
            exact_dict = {}
            if exact_matches:
                for k, v in exact_matches.items():
                    exact_dict[k] = str(v)
            
            numeric_list = numeric_comparisons or []
            string_list = string_comparisons or []
            
            # Call Rust backend - this returns node IDs directly, not indices
            node_ids = self._rust_core.filter_nodes_multi_criteria(exact_dict, numeric_list, string_list)
            
            # Return the node IDs directly
            return node_ids
        else:
            # Python fallback - simple implementation
            effective_nodes, _, _ = self._get_effective_data()
            candidates = set(effective_nodes.keys())
            
            # Apply exact matches
            if exact_matches:
                for attr_name, expected_value in exact_matches.items():
                    matching = set()
                    for node_id in candidates:
                        node = effective_nodes[node_id]
                        if node.attributes.get(attr_name) == expected_value:
                            matching.add(node_id)
                    candidates = candidates.intersection(matching)
            
            # Apply numeric comparisons
            if numeric_comparisons:
                for attr_name, operator, value in numeric_comparisons:
                    matching = set()
                    for node_id in candidates:
                        node = effective_nodes[node_id]
                        node_value = node.attributes.get(attr_name)
                        if node_value is not None:
                            try:
                                node_value = float(node_value)
                                if operator == '>':
                                    if node_value > value:
                                        matching.add(node_id)
                                elif operator == '>=':
                                    if node_value >= value:
                                        matching.add(node_id)
                                elif operator == '<':
                                    if node_value < value:
                                        matching.add(node_id)
                                elif operator == '<=':
                                    if node_value <= value:
                                        matching.add(node_id)
                                elif operator == '==':
                                    if node_value == value:
                                        matching.add(node_id)
                                elif operator == '!=':
                                    if node_value != value:
                                        matching.add(node_id)
                            except (ValueError, TypeError):
                                pass
                    candidates = candidates.intersection(matching)
            
            # Apply string comparisons
            if string_comparisons:
                for attr_name, operator, value in string_comparisons:
                    matching = set()
                    for node_id in candidates:
                        node = effective_nodes[node_id]
                        node_value = node.attributes.get(attr_name)
                        if node_value is not None:
                            node_value_str = str(node_value)
                            value_str = str(value)
                            if operator == '==':
                                if node_value_str == value_str:
                                    matching.add(node_id)
                            elif operator == '!=':
                                if node_value_str != value_str:
                                    matching.add(node_id)
                            elif operator == 'contains':
                                if value_str in node_value_str:
                                    matching.add(node_id)
                            elif operator == 'startswith':
                                if node_value_str.startswith(value_str):
                                    matching.add(node_id)
                            elif operator == 'endswith':
                                if node_value_str.endswith(value_str):
                                    matching.add(node_id)
                    candidates = candidates.intersection(matching)
            
            return list(candidates)


# Example usage patterns for when these optimizations are integrated back:

"""
# DataFrame conversion for analysis
df = graph.to_dataframe(['salary', 'department', 'age'])

# Ultra-fast bulk attribute retrieval
salaries = graph.get_attribute_column('salary')
node_ids, salary_values = salaries

# Efficient multi-criteria filtering
high_earners = graph.filter_nodes_multi_criteria(
    exact_matches={'department': 'engineering'},
    numeric_comparisons=[('salary', '>', 100000)]
)

# Chunked bulk operations for large datasets
nodes_data = [{'id': f'user_{i}', 'score': i * 10} for i in range(100000)]
graph.add_nodes_chunked(nodes_data)

# Bulk attribute updates
updates = {f'user_{i}': {'score': i * 20} for i in range(50000)}
graph.set_node_attributes_chunked(updates)

# Direct DataFrame export (fastest for large datasets)
data = graph.export_node_dataframe_optimized(['salary', 'age', 'department'])
df = pd.DataFrame(data)
"""
