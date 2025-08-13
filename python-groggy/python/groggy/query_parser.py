"""
Query parser for converting string queries to filter objects.

Converts strings like "salary > 120000" to NodeFilter objects.
"""

import re
import ast
from typing import Union, Any
from ._groggy import NodeFilter, EdgeFilter, AttributeFilter, AttrValue

class QueryParser:
    """Parse string queries into filter objects."""
    
    # Regex patterns for different query types
    COMPARISON_PATTERN = r'(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)'
    
    def __init__(self):
        pass
    
    def parse_node_filter(self, query: str) -> NodeFilter:
        """Parse a string query into a NodeFilter."""
        query = query.strip()
        
        # Handle simple equality without quotes
        if ' == ' in query:
            return self._parse_comparison(query, NodeFilter)
        elif ' != ' in query:
            return self._parse_comparison(query, NodeFilter)
        elif ' >= ' in query:
            return self._parse_comparison(query, NodeFilter)
        elif ' <= ' in query:
            return self._parse_comparison(query, NodeFilter)
        elif ' > ' in query:
            return self._parse_comparison(query, NodeFilter)
        elif ' < ' in query:
            return self._parse_comparison(query, NodeFilter)
        else:
            raise ValueError(f"Unsupported query format: {query}")
    
    def parse_edge_filter(self, query: str) -> EdgeFilter:
        """Parse a string query into an EdgeFilter."""
        query = query.strip()
        
        # Handle simple equality without quotes
        if ' == ' in query:
            return self._parse_comparison(query, EdgeFilter)
        elif ' != ' in query:
            return self._parse_comparison(query, EdgeFilter)
        elif ' >= ' in query:
            return self._parse_comparison(query, EdgeFilter)
        elif ' <= ' in query:
            return self._parse_comparison(query, EdgeFilter)
        elif ' > ' in query:
            return self._parse_comparison(query, EdgeFilter)
        elif ' < ' in query:
            return self._parse_comparison(query, EdgeFilter)
        else:
            raise ValueError(f"Unsupported query format: {query}")
    
    def _parse_comparison(self, query: str, filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Parse comparison expressions like 'salary > 120000'."""
        match = re.match(self.COMPARISON_PATTERN, query)
        if not match:
            raise ValueError(f"Invalid comparison: {query}")
        
        attr_name, operator, value_str = match.groups()
        
        # Parse the value using Python's AST for safety
        try:
            # Try to safely evaluate the value
            value = ast.literal_eval(value_str.strip())
        except (ValueError, SyntaxError):
            # If it fails, treat as string (remove quotes if present)
            value = value_str.strip().strip('\'"')
        
        # Convert to AttrValue
        attr_value = AttrValue(value)
        
        # Create appropriate AttributeFilter
        if operator == '==':
            attr_filter = AttributeFilter.equals(attr_value)
        elif operator == '!=':
            # Create a NOT equals filter using existing filters
            equals_filter = AttributeFilter.equals(attr_value)
            # For now, we'll need to implement this differently since we don't have NOT
            # Let's use the basic approach
            raise NotImplementedError("!= operator not yet implemented - use explicit filters")
        elif operator == '>':
            attr_filter = AttributeFilter.greater_than(attr_value)
        elif operator == '<':
            attr_filter = AttributeFilter.less_than(attr_value)
        elif operator == '>=':
            # We need >= and <= - let's implement using combinations for now
            raise NotImplementedError(">= operator not yet implemented - use explicit filters")
        elif operator == '<=':
            raise NotImplementedError("<= operator not yet implemented - use explicit filters")
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        
        # Create the appropriate filter
        if filter_class == NodeFilter:
            return NodeFilter.attribute_filter(attr_name, attr_filter)
        else:
            return EdgeFilter.attribute_filter(attr_name, attr_filter)

# Global parser instance
_parser = QueryParser()

def parse_node_query(query: str) -> NodeFilter:
    """Parse a string query into a NodeFilter.
    
    Examples:
        parse_node_query("salary > 120000")
        parse_node_query("department == 'Engineering'")
        parse_node_query("age < 30")
    """
    return _parser.parse_node_filter(query)

def parse_edge_query(query: str) -> EdgeFilter:
    """Parse a string query into an EdgeFilter.
    
    Examples:
        parse_edge_query("weight > 0.5")
        parse_edge_query("relationship == 'collaborates'")
    """
    return _parser.parse_edge_filter(query)