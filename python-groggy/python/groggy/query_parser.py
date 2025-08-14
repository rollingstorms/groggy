"""
Query parser for converting string queries to filter objects.

Converts strings like "salary > 120000" to NodeFilter objects.
"""

import re
import ast
from typing import Union, Any, List
from ._groggy import NodeFilter, EdgeFilter, AttributeFilter, AttrValue

class QueryParser:
    """Parse string queries into filter objects."""
    
    # Regex patterns for different query types
    COMPARISON_PATTERN = r'(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)'
    
    def __init__(self):
        pass
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize a query string, handling quoted strings and operators."""
        tokens = []
        current_token = ""
        in_quotes = False
        quote_char = None
        i = 0
        
        while i < len(query):
            char = query[i]
            
            if char in ['"', "'"] and (not in_quotes or char == quote_char):
                if in_quotes and char == quote_char:
                    # End of quoted string
                    current_token += char
                    tokens.append(current_token)
                    current_token = ""
                    in_quotes = False
                    quote_char = None
                else:
                    # Start of quoted string
                    if current_token.strip():
                        tokens.append(current_token.strip())
                    current_token = char
                    in_quotes = True
                    quote_char = char
            elif in_quotes:
                current_token += char
            elif char.isspace():
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
            elif char == '(':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append('(')
            elif char == ')':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
                tokens.append(')')
            else:
                current_token += char
            
            i += 1
        
        if current_token.strip():
            tokens.append(current_token.strip())
        
        return tokens
    
    def _parse_logical_expression(self, query: str, filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Parse expressions with logical operators (AND, OR, NOT)."""
        
        # Convert to uppercase for case-insensitive matching
        normalized_query = query.upper()
        
        # Check for logical operators
        if ' AND ' in normalized_query:
            return self._parse_and_expression(query, filter_class)
        elif ' OR ' in normalized_query:
            return self._parse_or_expression(query, filter_class)
        elif normalized_query.startswith('NOT '):
            return self._parse_not_expression(query, filter_class)
        else:
            # Single comparison
            return self._parse_comparison(query, filter_class)
    
    def _parse_and_expression(self, query: str, filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Parse AND expressions."""
        # For now, use a simple approach: split by ' AND ' and parse each part
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex AND expressions with more than 2 parts not yet supported: {query}")
        
        left_filter = self._parse_logical_expression(parts[0].strip(), filter_class)
        right_filter = self._parse_logical_expression(parts[1].strip(), filter_class)
        
        # For now, we'll create a composite filter using a helper class
        return CompositeFilter.and_filter([left_filter, right_filter], filter_class)
    
    def _parse_or_expression(self, query: str, filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Parse OR expressions."""
        # For now, use a simple approach: split by ' OR ' and parse each part
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex OR expressions with more than 2 parts not yet supported: {query}")
        
        left_filter = self._parse_logical_expression(parts[0].strip(), filter_class)
        right_filter = self._parse_logical_expression(parts[1].strip(), filter_class)
        
        # For now, we'll create a composite filter using a helper class
        return CompositeFilter.or_filter([left_filter, right_filter], filter_class)
    
    def _parse_not_expression(self, query: str, filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Parse NOT expressions."""
        # Remove 'NOT ' from the beginning
        inner_query = re.sub(r'^NOT\s+', '', query, flags=re.IGNORECASE).strip()
        inner_filter = self._parse_logical_expression(inner_query, filter_class)
        
        # For now, we'll create a composite filter using a helper class
        return CompositeFilter.not_filter(inner_filter, filter_class)
    
    def parse_node_filter(self, query: str) -> NodeFilter:
        """Parse a string query into a NodeFilter with logical operator support."""
        query = query.strip()
        
        # Check for logical operators - if found, use enhanced filtering
        normalized_query = query.upper()
        if ' AND ' in normalized_query or ' OR ' in normalized_query or normalized_query.startswith('NOT '):
            # Use enhanced filtering for logical operations
            from .enhanced_query import enhanced_filter_nodes
            # Return a special marker that the graph can detect
            return EnhancedQueryMarker(query, "nodes")
        else:
            # Use standard parsing for simple queries
            return self._parse_comparison(query, NodeFilter)
    
    def parse_edge_filter(self, query: str) -> EdgeFilter:
        """Parse a string query into an EdgeFilter with logical operator support."""
        query = query.strip()
        
        # Check for logical operators - if found, use enhanced filtering
        normalized_query = query.upper()
        if ' AND ' in normalized_query or ' OR ' in normalized_query or normalized_query.startswith('NOT '):
            # Use enhanced filtering for logical operations
            from .enhanced_query import enhanced_filter_edges
            # Return a special marker that the graph can detect
            return EnhancedQueryMarker(query, "edges")
        else:
            # Use standard parsing for simple queries
            return self._parse_comparison(query, EdgeFilter)
    
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
            attr_filter = AttributeFilter.not_equals(attr_value)
        elif operator == '>':
            attr_filter = AttributeFilter.greater_than(attr_value)
        elif operator == '<':
            attr_filter = AttributeFilter.less_than(attr_value)
        elif operator == '>=':
            attr_filter = AttributeFilter.greater_than_or_equal(attr_value)
        elif operator == '<=':
            attr_filter = AttributeFilter.less_than_or_equal(attr_value)
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
        parse_edge_query("weight > 0.5 AND strength != 'weak'")
    """
    return _parser.parse_edge_filter(query)

class CompositeFilter:
    """Helper class for creating composite filters with logical operations."""
    
    @staticmethod
    def and_filter(filters: List[Union[NodeFilter, EdgeFilter]], filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Create an AND composite filter."""
        # For now, implement by applying filters sequentially
        # This is a simplified approach - in a full implementation, you'd want proper logical evaluation
        if filter_class == NodeFilter:
            return CompositeNodeFilter(filters, 'AND')
        else:
            return CompositeEdgeFilter(filters, 'AND')
    
    @staticmethod
    def or_filter(filters: List[Union[NodeFilter, EdgeFilter]], filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Create an OR composite filter."""
        if filter_class == NodeFilter:
            return CompositeNodeFilter(filters, 'OR')
        else:
            return CompositeEdgeFilter(filters, 'OR')
    
    @staticmethod
    def not_filter(filter_obj: Union[NodeFilter, EdgeFilter], filter_class) -> Union[NodeFilter, EdgeFilter]:
        """Create a NOT composite filter."""
        if filter_class == NodeFilter:
            return CompositeNodeFilter([filter_obj], 'NOT')
        else:
            return CompositeEdgeFilter([filter_obj], 'NOT')

class CompositeNodeFilter:
    """A composite node filter that combines multiple filters with logical operators."""
    
    def __init__(self, filters: List[NodeFilter], operation: str):
        self.filters = filters
        self.operation = operation
    
    def __repr__(self):
        return f"CompositeNodeFilter({self.operation}, {len(self.filters)} filters)"

class CompositeEdgeFilter:
    """A composite edge filter that combines multiple filters with logical operators."""
    
    def __init__(self, filters: List[EdgeFilter], operation: str):
        self.filters = filters
        self.operation = operation
    
    def __repr__(self):
        return f"CompositeEdgeFilter({self.operation}, {len(self.filters)} filters)"