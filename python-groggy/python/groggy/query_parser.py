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
        """Parse a string query into a NodeFilter with full logical operator and parentheses support."""
        query = query.strip()
        
        # Handle parentheses with recursive descent parser
        if '(' in query or ')' in query:
            return self._parse_node_expression_with_parentheses(query)
        
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
    
    def _parse_node_expression_with_parentheses(self, query: str) -> NodeFilter:
        """Parse node expressions with parentheses support using recursive descent."""
        tokens = self._tokenize_query(query)
        
        # Convert tokens to a format suitable for expression parsing
        pos = [0]  # Use list for mutable reference in nested functions
        
        def parse_expression() -> NodeFilter:
            """Parse a complete expression with AND/OR precedence."""
            return parse_or_expression()
        
        def parse_or_expression() -> NodeFilter:
            """Parse OR expressions (lowest precedence)."""
            left = parse_and_expression()
            
            while pos[0] < len(tokens) and tokens[pos[0]].upper() == 'OR':
                pos[0] += 1  # consume 'OR'
                right = parse_and_expression()
                left = NodeFilter.or_filters([left, right])
            
            return left
        
        def parse_and_expression() -> NodeFilter:
            """Parse AND expressions (higher precedence than OR)."""
            left = parse_not_expression()
            
            while pos[0] < len(tokens) and tokens[pos[0]].upper() == 'AND':
                pos[0] += 1  # consume 'AND'
                right = parse_not_expression()
                left = NodeFilter.and_filters([left, right])
            
            return left
        
        def parse_not_expression() -> NodeFilter:
            """Parse NOT expressions and primary expressions."""
            if pos[0] < len(tokens) and tokens[pos[0]].upper() == 'NOT':
                pos[0] += 1  # consume 'NOT'
                inner = parse_primary_expression()
                return NodeFilter.not_filter(inner)
            else:
                return parse_primary_expression()
        
        def parse_primary_expression() -> NodeFilter:
            """Parse primary expressions (comparisons or parenthesized expressions)."""
            if pos[0] < len(tokens) and tokens[pos[0]] == '(':
                pos[0] += 1  # consume '('
                result = parse_expression()
                if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                    pos[0] += 1  # consume ')'
                else:
                    raise ValueError(f"Missing closing parenthesis in query: {query}")
                return result
            else:
                # Parse a comparison expression
                comparison_tokens = []
                while (pos[0] < len(tokens) and 
                       tokens[pos[0]] not in ['AND', 'OR', 'NOT', ')', '('] and
                       tokens[pos[0]].upper() not in ['AND', 'OR', 'NOT']):
                    comparison_tokens.append(tokens[pos[0]])
                    pos[0] += 1
                
                if not comparison_tokens:
                    raise ValueError(f"Expected comparison expression in query: {query}")
                
                comparison_str = ' '.join(comparison_tokens)
                return self._parse_comparison(comparison_str, NodeFilter)
        
        return parse_expression()
    
    def _parse_and_node_filter(self, query: str) -> NodeFilter:
        """Parse AND expressions for nodes with 3+ term support."""
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            raise ValueError(f"Invalid AND expression: {query}")
        
        # Handle 3+ terms by recursively combining filters
        filters = []
        for part in parts:
            part = part.strip()
            if part:
                filters.append(self.parse_node_filter(part))
        
        if len(filters) == 0:
            raise ValueError(f"No valid filters found in AND expression: {query}")
        elif len(filters) == 1:
            return filters[0]
        else:
            # Combine all filters with AND
            return NodeFilter.and_filters(filters)
    
    def _parse_or_node_filter(self, query: str) -> NodeFilter:
        """Parse OR expressions for nodes with 3+ term support."""
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            raise ValueError(f"Invalid OR expression: {query}")
        
        # Handle 3+ terms by recursively combining filters
        filters = []
        for part in parts:
            part = part.strip()
            if part:
                filters.append(self.parse_node_filter(part))
        
        if len(filters) == 0:
            raise ValueError(f"No valid filters found in OR expression: {query}")
        elif len(filters) == 1:
            return filters[0]
        else:
            # Combine all filters with OR
            return NodeFilter.or_filters(filters)
    
    def _parse_not_node_filter(self, query: str) -> NodeFilter:
        """Parse NOT expressions for nodes."""
        inner_query = re.sub(r'^NOT\s+', '', query, flags=re.IGNORECASE).strip()
        inner_filter = self.parse_node_filter(inner_query)
        
        return NodeFilter.not_filter(inner_filter)
    
    def _parse_edge_expression_with_parentheses(self, query: str) -> EdgeFilter:
        """Parse edge expressions with parentheses support using recursive descent."""
        tokens = self._tokenize_query(query)
        
        # Convert tokens to a format suitable for expression parsing
        pos = [0]  # Use list for mutable reference in nested functions
        
        def parse_expression() -> EdgeFilter:
            """Parse a complete expression with AND/OR precedence."""
            return parse_or_expression()
        
        def parse_or_expression() -> EdgeFilter:
            """Parse OR expressions (lowest precedence)."""
            left = parse_and_expression()
            
            while pos[0] < len(tokens) and tokens[pos[0]].upper() == 'OR':
                pos[0] += 1  # consume 'OR'
                right = parse_and_expression()
                left = EdgeFilter.or_filters([left, right])
            
            return left
        
        def parse_and_expression() -> EdgeFilter:
            """Parse AND expressions (higher precedence than OR)."""
            left = parse_not_expression()
            
            while pos[0] < len(tokens) and tokens[pos[0]].upper() == 'AND':
                pos[0] += 1  # consume 'AND'
                right = parse_not_expression()
                left = EdgeFilter.and_filters([left, right])
            
            return left
        
        def parse_not_expression() -> EdgeFilter:
            """Parse NOT expressions and primary expressions."""
            if pos[0] < len(tokens) and tokens[pos[0]].upper() == 'NOT':
                pos[0] += 1  # consume 'NOT'
                inner = parse_primary_expression()
                return EdgeFilter.not_filter(inner)
            else:
                return parse_primary_expression()
        
        def parse_primary_expression() -> EdgeFilter:
            """Parse primary expressions (comparisons or parenthesized expressions)."""
            if pos[0] < len(tokens) and tokens[pos[0]] == '(':
                pos[0] += 1  # consume '('
                result = parse_expression()
                if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                    pos[0] += 1  # consume ')'
                else:
                    raise ValueError(f"Missing closing parenthesis in query: {query}")
                return result
            else:
                # Parse a comparison expression
                comparison_tokens = []
                while (pos[0] < len(tokens) and 
                       tokens[pos[0]] not in ['AND', 'OR', 'NOT', ')', '('] and
                       tokens[pos[0]].upper() not in ['AND', 'OR', 'NOT']):
                    comparison_tokens.append(tokens[pos[0]])
                    pos[0] += 1
                
                if not comparison_tokens:
                    raise ValueError(f"Expected comparison expression in query: {query}")
                
                comparison_str = ' '.join(comparison_tokens)
                return self._parse_comparison(comparison_str, EdgeFilter)
        
        return parse_expression()
    
    def parse_edge_filter(self, query: str) -> EdgeFilter:
        """Parse a string query into an EdgeFilter with full logical operator and parentheses support."""
        query = query.strip()
        
        # Handle parentheses with recursive descent parser
        if '(' in query or ')' in query:
            return self._parse_edge_expression_with_parentheses(query)
        
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
        """Parse AND expressions for edges with 3+ term support."""
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            raise ValueError(f"Invalid AND expression: {query}")
        
        # Handle 3+ terms by recursively combining filters
        filters = []
        for part in parts:
            part = part.strip()
            if part:
                filters.append(self.parse_edge_filter(part))
        
        if len(filters) == 0:
            raise ValueError(f"No valid filters found in AND expression: {query}")
        elif len(filters) == 1:
            return filters[0]
        else:
            # Combine all filters with AND
            return EdgeFilter.and_filters(filters)
    
    def _parse_or_edge_filter(self, query: str) -> EdgeFilter:
        """Parse OR expressions for edges with 3+ term support."""
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        
        if len(parts) < 2:
            raise ValueError(f"Invalid OR expression: {query}")
        
        # Handle 3+ terms by recursively combining filters
        filters = []
        for part in parts:
            part = part.strip()
            if part:
                filters.append(self.parse_edge_filter(part))
        
        if len(filters) == 0:
            raise ValueError(f"No valid filters found in OR expression: {query}")
        elif len(filters) == 1:
            return filters[0]
        else:
            # Combine all filters with OR
            return EdgeFilter.or_filters(filters)
    
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
            # Handle boolean literals (true/false vs True/False)
            # Note: Groggy stores Python True as AttrValue(1) and False as AttrValue(0)
            value_normalized = value_str.strip().lower()
            if value_normalized == 'true':
                value = 1  # AttrValue stores True as 1
            elif value_normalized == 'false':
                value = 0  # AttrValue stores False as 0
            else:
                # Try to safely evaluate the value
                value = ast.literal_eval(value_str.strip())
        except (ValueError, SyntaxError):
            # If it fails, treat as string (remove quotes if present)
            value = value_str.strip().strip('\'"')
        
        # Use raw value directly (auto-conversion in AttributeFilter)
        attr_value = value
        
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

