//! Query Parser - Core implementation for parsing query strings into filters.
//!
//! This module provides the core query parsing functionality that eliminates
//! the circular dependency of Rust → Python → Rust calls. All query parsing
//! logic is implemented in pure Rust for universal multi-language support.

use crate::core::query::{AttributeFilter, EdgeFilter, NodeFilter};
use crate::errors::GraphError;
use crate::types::{AttrName, AttrValue};

/// Result type for query parsing operations
pub type QueryResult<T> = Result<T, QueryError>;

/// Errors that can occur during query parsing
#[derive(Debug, Clone, PartialEq)]
pub enum QueryError {
    /// Invalid syntax in query string
    InvalidSyntax { message: String, position: usize },
    /// Unexpected token during parsing
    UnexpectedToken {
        expected: String,
        found: String,
        position: usize,
    },
    /// Missing closing parenthesis
    MissingClosingParen { position: usize },
    /// Invalid attribute value
    InvalidValue { value: String, position: usize },
    /// Empty query string
    EmptyQuery,
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::InvalidSyntax { message, position } => {
                write!(f, "Invalid syntax at position {}: {}", position, message)
            }
            QueryError::UnexpectedToken {
                expected,
                found,
                position,
            } => {
                write!(
                    f,
                    "Expected '{}' but found '{}' at position {}",
                    expected, found, position
                )
            }
            QueryError::MissingClosingParen { position } => {
                write!(f, "Missing closing parenthesis at position {}", position)
            }
            QueryError::InvalidValue { value, position } => {
                write!(
                    f,
                    "Invalid attribute value '{}' at position {}",
                    value, position
                )
            }
            QueryError::EmptyQuery => {
                write!(f, "Query string cannot be empty")
            }
        }
    }
}

impl std::error::Error for QueryError {}

/// Convert QueryError to GraphError for compatibility
impl From<QueryError> for GraphError {
    fn from(error: QueryError) -> Self {
        GraphError::QueryParseError {
            message: error.to_string(),
            query: String::new(), // We'll fill this in at call sites
        }
    }
}

/// Token types for query parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Identifier(String),
    Operator(CompOp),
    LogicalOp(LogicalOp),
    Value(AttrValue),
    LeftParen,
    RightParen,
    Not,
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum CompOp {
    Equals,       // ==
    NotEquals,    // !=
    GreaterThan,  // >
    LessThan,     // <
    GreaterEqual, // >=
    LessEqual,    // <=
}

/// Logical operators
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOp {
    And,
    Or,
}

/// The main query parser
#[derive(Debug)]
pub struct QueryParser {
    tokens: Vec<Token>,
    position: usize,
}

impl QueryParser {
    /// Create a new query parser
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            position: 0,
        }
    }

    /// Parse a node query string into a NodeFilter
    pub fn parse_node_query(&mut self, query: &str) -> QueryResult<NodeFilter> {
        if query.trim().is_empty() {
            return Err(QueryError::EmptyQuery);
        }

        self.tokenize(query)?;
        self.position = 0;
        self.parse_node_expression()
    }

    /// Parse an edge query string into an EdgeFilter
    pub fn parse_edge_query(&mut self, query: &str) -> QueryResult<EdgeFilter> {
        if query.trim().is_empty() {
            return Err(QueryError::EmptyQuery);
        }

        self.tokenize(query)?;
        self.position = 0;
        self.parse_edge_expression()
    }

    /// Tokenize the input query string
    fn tokenize(&mut self, query: &str) -> QueryResult<()> {
        self.tokens.clear();
        let mut chars = query.char_indices().peekable();

        while let Some((pos, ch)) = chars.next() {
            match ch {
                ' ' | '\t' | '\n' => continue, // Skip whitespace
                '(' => self.tokens.push(Token::LeftParen),
                ')' => self.tokens.push(Token::RightParen),
                '=' => {
                    if let Some((_, '=')) = chars.peek() {
                        chars.next(); // consume second =
                        self.tokens.push(Token::Operator(CompOp::Equals));
                    } else {
                        return Err(QueryError::InvalidSyntax {
                            message: "Single '=' not supported, use '=='".to_string(),
                            position: pos,
                        });
                    }
                }
                '!' => {
                    if let Some((_, '=')) = chars.peek() {
                        chars.next(); // consume =
                        self.tokens.push(Token::Operator(CompOp::NotEquals));
                    } else {
                        return Err(QueryError::InvalidSyntax {
                            message: "Single '!' not supported, use '!=' or 'NOT'".to_string(),
                            position: pos,
                        });
                    }
                }
                '>' => {
                    if let Some((_, '=')) = chars.peek() {
                        chars.next(); // consume =
                        self.tokens.push(Token::Operator(CompOp::GreaterEqual));
                    } else {
                        self.tokens.push(Token::Operator(CompOp::GreaterThan));
                    }
                }
                '<' => {
                    if let Some((_, '=')) = chars.peek() {
                        chars.next(); // consume =
                        self.tokens.push(Token::Operator(CompOp::LessEqual));
                    } else {
                        self.tokens.push(Token::Operator(CompOp::LessThan));
                    }
                }
                '\'' | '"' => {
                    // Parse string literal
                    let quote_char = ch;
                    let mut value = String::new();
                    let mut escaped = false;

                    for (_, ch) in chars.by_ref() {
                        if escaped {
                            value.push(ch);
                            escaped = false;
                        } else if ch == '\\' {
                            escaped = true;
                        } else if ch == quote_char {
                            break;
                        } else {
                            value.push(ch);
                        }
                    }

                    self.tokens.push(Token::Value(AttrValue::Text(value)));
                }
                _ if ch.is_ascii_digit() || ch == '-' => {
                    // Parse number
                    let start_pos = pos;
                    let mut number_str = String::new();
                    number_str.push(ch);

                    while let Some(&(_, next_ch)) = chars.peek() {
                        if next_ch.is_ascii_digit() || next_ch == '.' {
                            number_str.push(next_ch);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    let value = if number_str.contains('.') {
                        match number_str.parse::<f32>() {
                            Ok(f) => AttrValue::Float(f),
                            Err(_) => {
                                return Err(QueryError::InvalidValue {
                                    value: number_str,
                                    position: start_pos,
                                })
                            }
                        }
                    } else {
                        match number_str.parse::<i64>() {
                            Ok(i) => {
                                if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                                    AttrValue::SmallInt(i as i32)
                                } else {
                                    AttrValue::Int(i)
                                }
                            }
                            Err(_) => {
                                return Err(QueryError::InvalidValue {
                                    value: number_str,
                                    position: start_pos,
                                })
                            }
                        }
                    };

                    self.tokens.push(Token::Value(value));
                }
                _ if ch.is_ascii_alphabetic() || ch == '_' => {
                    // Parse identifier or keyword
                    let mut ident = String::new();
                    ident.push(ch);

                    while let Some(&(_, next_ch)) = chars.peek() {
                        if next_ch.is_ascii_alphanumeric() || next_ch == '_' {
                            ident.push(next_ch);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    match ident.to_uppercase().as_str() {
                        "AND" => self.tokens.push(Token::LogicalOp(LogicalOp::And)),
                        "OR" => self.tokens.push(Token::LogicalOp(LogicalOp::Or)),
                        "NOT" => self.tokens.push(Token::Not),
                        "TRUE" => self.tokens.push(Token::Value(AttrValue::Bool(true))),
                        "FALSE" => self.tokens.push(Token::Value(AttrValue::Bool(false))),
                        _ => self.tokens.push(Token::Identifier(ident)),
                    }
                }
                _ => {
                    return Err(QueryError::InvalidSyntax {
                        message: format!("Unexpected character '{}'", ch),
                        position: pos,
                    });
                }
            }
        }

        Ok(())
    }

    /// Parse a node filter expression
    fn parse_node_expression(&mut self) -> QueryResult<NodeFilter> {
        self.parse_node_or()
    }

    /// Parse OR expression for nodes
    fn parse_node_or(&mut self) -> QueryResult<NodeFilter> {
        let mut left = self.parse_node_and()?;

        while self.match_token(&Token::LogicalOp(LogicalOp::Or)) {
            let right = self.parse_node_and()?;
            left = NodeFilter::Or(vec![left, right]);
        }

        Ok(left)
    }

    /// Parse AND expression for nodes
    fn parse_node_and(&mut self) -> QueryResult<NodeFilter> {
        let mut left = self.parse_node_not()?;

        while self.match_token(&Token::LogicalOp(LogicalOp::And)) {
            let right = self.parse_node_not()?;
            left = NodeFilter::And(vec![left, right]);
        }

        Ok(left)
    }

    /// Parse NOT expression for nodes
    fn parse_node_not(&mut self) -> QueryResult<NodeFilter> {
        if self.match_token(&Token::Not) {
            let expr = self.parse_node_primary()?;
            Ok(NodeFilter::Not(Box::new(expr)))
        } else {
            self.parse_node_primary()
        }
    }

    /// Parse primary node expression (comparison or parenthesized)
    fn parse_node_primary(&mut self) -> QueryResult<NodeFilter> {
        if self.match_token(&Token::LeftParen) {
            let expr = self.parse_node_expression()?;
            if !self.match_token(&Token::RightParen) {
                return Err(QueryError::MissingClosingParen {
                    position: self.position,
                });
            }
            Ok(expr)
        } else {
            self.parse_node_comparison()
        }
    }

    /// Parse a comparison expression for nodes
    fn parse_node_comparison(&mut self) -> QueryResult<NodeFilter> {
        let attr_name = self.expect_identifier()?;

        if let Some(token) = self.current_token().cloned() {
            match token {
                Token::Operator(op) => {
                    self.advance();
                    let value = self.expect_value()?;

                    match op {
                        CompOp::Equals => Ok(NodeFilter::AttributeEquals {
                            name: attr_name,
                            value,
                        }),
                        CompOp::NotEquals => Ok(NodeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::NotEquals(value),
                        }),
                        CompOp::GreaterThan => Ok(NodeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::GreaterThan(value),
                        }),
                        CompOp::LessThan => Ok(NodeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::LessThan(value),
                        }),
                        CompOp::GreaterEqual => Ok(NodeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::GreaterThanOrEqual(value),
                        }),
                        CompOp::LessEqual => Ok(NodeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::LessThanOrEqual(value),
                        }),
                    }
                }
                _ => {
                    // Just an attribute name without comparison means "has attribute"
                    Ok(NodeFilter::HasAttribute { name: attr_name })
                }
            }
        } else {
            // End of input, treat as "has attribute"
            Ok(NodeFilter::HasAttribute { name: attr_name })
        }
    }

    /// Parse an edge filter expression
    fn parse_edge_expression(&mut self) -> QueryResult<EdgeFilter> {
        self.parse_edge_or()
    }

    /// Parse OR expression for edges
    fn parse_edge_or(&mut self) -> QueryResult<EdgeFilter> {
        let mut left = self.parse_edge_and()?;

        while self.match_token(&Token::LogicalOp(LogicalOp::Or)) {
            let right = self.parse_edge_and()?;
            left = EdgeFilter::Or(vec![left, right]);
        }

        Ok(left)
    }

    /// Parse AND expression for edges
    fn parse_edge_and(&mut self) -> QueryResult<EdgeFilter> {
        let mut left = self.parse_edge_not()?;

        while self.match_token(&Token::LogicalOp(LogicalOp::And)) {
            let right = self.parse_edge_not()?;
            left = EdgeFilter::And(vec![left, right]);
        }

        Ok(left)
    }

    /// Parse NOT expression for edges
    fn parse_edge_not(&mut self) -> QueryResult<EdgeFilter> {
        if self.match_token(&Token::Not) {
            let expr = self.parse_edge_primary()?;
            Ok(EdgeFilter::Not(Box::new(expr)))
        } else {
            self.parse_edge_primary()
        }
    }

    /// Parse primary edge expression
    fn parse_edge_primary(&mut self) -> QueryResult<EdgeFilter> {
        if self.match_token(&Token::LeftParen) {
            let expr = self.parse_edge_expression()?;
            if !self.match_token(&Token::RightParen) {
                return Err(QueryError::MissingClosingParen {
                    position: self.position,
                });
            }
            Ok(expr)
        } else {
            self.parse_edge_comparison()
        }
    }

    /// Parse a comparison expression for edges
    fn parse_edge_comparison(&mut self) -> QueryResult<EdgeFilter> {
        let attr_name = self.expect_identifier()?;

        if let Some(token) = self.current_token().cloned() {
            match token {
                Token::Operator(op) => {
                    self.advance();
                    let value = self.expect_value()?;

                    match op {
                        CompOp::Equals => Ok(EdgeFilter::AttributeEquals {
                            name: attr_name,
                            value,
                        }),
                        CompOp::NotEquals => Ok(EdgeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::NotEquals(value),
                        }),
                        CompOp::GreaterThan => Ok(EdgeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::GreaterThan(value),
                        }),
                        CompOp::LessThan => Ok(EdgeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::LessThan(value),
                        }),
                        CompOp::GreaterEqual => Ok(EdgeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::GreaterThanOrEqual(value),
                        }),
                        CompOp::LessEqual => Ok(EdgeFilter::AttributeFilter {
                            name: attr_name,
                            filter: AttributeFilter::LessThanOrEqual(value),
                        }),
                    }
                }
                _ => {
                    // Just an attribute name without comparison means "has attribute"
                    Ok(EdgeFilter::HasAttribute { name: attr_name })
                }
            }
        } else {
            // End of input, treat as "has attribute"
            Ok(EdgeFilter::HasAttribute { name: attr_name })
        }
    }

    /// Helper methods for parsing
    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    fn match_token(&mut self, expected: &Token) -> bool {
        if let Some(token) = self.current_token() {
            if token == expected {
                self.advance();
                return true;
            }
        }
        false
    }

    fn expect_identifier(&mut self) -> QueryResult<AttrName> {
        if let Some(Token::Identifier(name)) = self.current_token() {
            let name = name.clone();
            self.advance();
            Ok(name)
        } else {
            Err(QueryError::UnexpectedToken {
                expected: "identifier".to_string(),
                found: format!("{:?}", self.current_token()),
                position: self.position,
            })
        }
    }

    fn expect_value(&mut self) -> QueryResult<AttrValue> {
        if let Some(Token::Value(value)) = self.current_token() {
            let value = value.clone();
            self.advance();
            Ok(value)
        } else {
            Err(QueryError::UnexpectedToken {
                expected: "value".to_string(),
                found: format!("{:?}", self.current_token()),
                position: self.position,
            })
        }
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_comparison() {
        let mut parser = QueryParser::new();
        let filter = parser.parse_node_query("salary > 120000").unwrap();

        match filter {
            NodeFilter::AttributeFilter { name, filter } => {
                assert_eq!(name, "salary");
                match filter {
                    AttributeFilter::GreaterThan(AttrValue::SmallInt(120000)) => (),
                    _ => panic!(
                        "Expected GreaterThan with SmallInt(120000), got {:?}",
                        filter
                    ),
                }
            }
            _ => panic!("Expected AttributeFilter, got {:?}", filter),
        }
    }

    #[test]
    fn test_string_equals() {
        let mut parser = QueryParser::new();
        let filter = parser
            .parse_node_query("department == 'Engineering'")
            .unwrap();

        match filter {
            NodeFilter::AttributeEquals { name, value } => {
                assert_eq!(name, "department");
                assert_eq!(value, AttrValue::Text("Engineering".to_string()));
            }
            _ => panic!("Expected AttributeEquals"),
        }
    }

    #[test]
    fn test_complex_logical() {
        let mut parser = QueryParser::new();
        let filter = parser
            .parse_node_query("(salary > 120000 AND department == 'Engineering') OR age < 25")
            .unwrap();

        match filter {
            NodeFilter::Or(_) => (), // Structure is correct
            _ => panic!("Expected Or filter, got {:?}", filter),
        }
    }

    #[test]
    fn test_has_attribute() {
        let mut parser = QueryParser::new();
        let filter = parser.parse_node_query("email").unwrap();

        match filter {
            NodeFilter::HasAttribute { name } => {
                assert_eq!(name, "email");
            }
            _ => panic!("Expected HasAttribute"),
        }
    }

    #[test]
    fn test_not_filter() {
        let mut parser = QueryParser::new();
        let filter = parser.parse_node_query("NOT department == 'HR'").unwrap();

        match filter {
            NodeFilter::Not(inner) => match inner.as_ref() {
                NodeFilter::AttributeEquals { name, value } => {
                    assert_eq!(name, "department");
                    assert_eq!(value, &AttrValue::Text("HR".to_string()));
                }
                _ => panic!("Expected AttributeEquals inside Not"),
            },
            _ => panic!("Expected Not filter"),
        }
    }

    #[test]
    fn test_error_cases() {
        let mut parser = QueryParser::new();

        // Empty query
        assert!(parser.parse_node_query("").is_err());

        // Missing value
        assert!(parser.parse_node_query("salary >").is_err());

        // Invalid syntax
        assert!(parser.parse_node_query("salary > > 100").is_err());

        // Missing closing paren
        assert!(parser.parse_node_query("(salary > 100").is_err());
    }

    #[test]
    fn test_edge_queries() {
        let mut parser = QueryParser::new();
        let filter = parser.parse_edge_query("weight > 0.5").unwrap();

        match filter {
            EdgeFilter::AttributeFilter { name, filter } => {
                assert_eq!(name, "weight");
                match filter {
                    AttributeFilter::GreaterThan(AttrValue::Float(f)) => {
                        assert!((f - 0.5).abs() < f32::EPSILON);
                    }
                    _ => panic!("Expected GreaterThan with Float(0.5)"),
                }
            }
            _ => panic!("Expected AttributeFilter"),
        }
    }
}
