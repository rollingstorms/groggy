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
    
    
    def parse_node_filter(self, query: str) -> NodeFilter:
        """Parse a string query into a NodeFilter with logical operator support."""
        query = query.strip()
        
        # Handle logical operators directly in the parser
        normalized_query = query.upper()
        if ' AND ' in normalized_query:
            return self._parse_and_node_filter(query)
        elif ' OR ' in normalized_query:
            return self._parse_or_node_filter(query)
        elif normalized_query.startswith('NOT '):
            return self._parse_not_node_filter(query)
        else:
            # Use standard parsing for simple queries
            return self._parse_comparison(query, NodeFilter)
    
    def _parse_and_node_filter(self, query: str) -> NodeFilter:
        """Parse AND expressions for nodes."""
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex AND expressions with more than 2 parts not yet supported: {query}")
        
        left_filter = self.parse_node_filter(parts[0].strip())
        right_filter = self.parse_node_filter(parts[1].strip())
        
        return NodeFilter.and_filters([left_filter, right_filter])
    
    def _parse_or_node_filter(self, query: str) -> NodeFilter:
        """Parse OR expressions for nodes."""
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex OR expressions with more than 2 parts not yet supported: {query}")
        
        left_filter = self.parse_node_filter(parts[0].strip())
        right_filter = self.parse_node_filter(parts[1].strip())
        
        return NodeFilter.or_filters([left_filter, right_filter])
    
    def _parse_not_node_filter(self, query: str) -> NodeFilter:
        """Parse NOT expressions for nodes."""
        inner_query = re.sub(r'^NOT\s+', '', query, flags=re.IGNORECASE).strip()
        inner_filter = self.parse_node_filter(inner_query)
        
        return NodeFilter.not_filter(inner_filter)
    
    def parse_edge_filter(self, query: str) -> EdgeFilter:
        """Parse a string query into an EdgeFilter with logical operator support."""
        query = query.strip()
        
        # Handle logical operators directly in the parser
        normalized_query = query.upper()
        if ' AND ' in normalized_query:
            return self._parse_and_edge_filter(query)
        elif ' OR ' in normalized_query:
            return self._parse_or_edge_filter(query)
        elif normalized_query.startswith('NOT '):
            return self._parse_not_edge_filter(query)
        else:
            # Use standard parsing for simple queries
            return self._parse_comparison(query, EdgeFilter)
    
    def _parse_and_edge_filter(self, query: str) -> EdgeFilter:
        """Parse AND expressions for edges."""
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex AND expressions with more than 2 parts not yet supported: {query}")
        
        left_filter = self.parse_edge_filter(parts[0].strip())
        right_filter = self.parse_edge_filter(parts[1].strip())
        
        return EdgeFilter.and_filters([left_filter, right_filter])
    
    def _parse_or_edge_filter(self, query: str) -> EdgeFilter:
        """Parse OR expressions for edges."""
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise ValueError(f"Complex OR expressions with more than 2 parts not yet supported: {query}")
        
        left_filter = self.parse_edge_filter(parts[0].strip())
        right_filter = self.parse_edge_filter(parts[1].strip())
        
        return EdgeFilter.or_filters([left_filter, right_filter])
    
    def _parse_not_edge_filter(self, query: str) -> EdgeFilter:
        """Parse NOT expressions for edges."""
        inner_query = re.sub(r'^NOT\s+', '', query, flags=re.IGNORECASE).strip()
        inner_filter = self.parse_edge_filter(inner_query)
        
        return EdgeFilter.not_filter(inner_filter)
    
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

