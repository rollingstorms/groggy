//! Compiled scalar array predicates using the graph query lexer and boolean grammar.
use crate::{
    errors::{GraphError, GraphResult},
    query::{
        query_parser::{CompOp, QueryParser, Token},
        AttributeFilter, NodeFilter,
    },
    types::AttrValue,
};

/// An array predicate compiled once, with errors surfaced by evaluate/validate.
/// Infallible construction is retained for native source compatibility.
pub struct QueryEvaluator {
    predicate: GraphResult<Predicate>,
}

#[derive(Debug)]
enum Operand {
    Value,
    Length,
    Even,
    Constant,
}
#[derive(Debug)]
enum Predicate {
    Compare(Operand, AttributeFilter),
    Exists,
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}

impl QueryEvaluator {
    /// Compile an expression. Call validate to surface errors before reading rows.
    pub fn new(query: &str) -> Self {
        Self {
            predicate: compile(query),
        }
    }
    /// Validate even when the input array is empty.
    pub fn validate(&self) -> GraphResult<()> {
        self.predicate.as_ref().map(|_| ()).map_err(Clone::clone)
    }
    /// Evaluate with exact mixed integer/float comparisons and no row parsing.
    pub fn evaluate(&self, value: &AttrValue) -> GraphResult<bool> {
        self.predicate
            .as_ref()
            .map_err(Clone::clone)?
            .evaluate(value)
    }
}

fn operand(name: &str) -> GraphResult<Operand> {
    match name {
        "value" => Ok(Operand::Value),
        "length" => Ok(Operand::Length),
        "__array_even" => Ok(Operand::Even),
        "__array_constant" => Ok(Operand::Constant),
        _ => Err(invalid(format!(
            "Unsupported array query operand '{name}'; use value or length"
        ))),
    }
}
fn lower(filter: NodeFilter) -> GraphResult<Predicate> {
    Ok(match filter {
        NodeFilter::HasAttribute { name } => {
            operand(&name)?;
            Predicate::Exists
        }
        NodeFilter::AttributeEquals { name, value } => {
            Predicate::Compare(operand(&name)?, AttributeFilter::Equals(value))
        }
        NodeFilter::AttributeFilter { name, filter } => Predicate::Compare(operand(&name)?, filter),
        NodeFilter::And(parts) => {
            Predicate::And(parts.into_iter().map(lower).collect::<GraphResult<_>>()?)
        }
        NodeFilter::Or(parts) => {
            Predicate::Or(parts.into_iter().map(lower).collect::<GraphResult<_>>()?)
        }
        NodeFilter::Not(part) => Predicate::Not(Box::new(lower(*part)?)),
        _ => {
            return Err(invalid(
                "Graph traversal predicates are unsupported on scalar arrays",
            ))
        }
    })
}
fn compile(query: &str) -> GraphResult<Predicate> {
    let tokens = QueryParser::scalar_tokens(query).map_err(|e| invalid(e.to_string()))?;
    // Adapt only token patterns, never raw text: quoted AND/OR/operator text is
    // left intact. Preserve the documented parity shorthand and bare text RHS.
    let mut adapted = Vec::with_capacity(tokens.len());
    let mut position = 0;
    while position < tokens.len() {
        let rest = &tokens[position..];
        if let [Token::Identifier(name), Token::Remainder, Token::Value(divisor), Token::Operator(CompOp::Equals), Token::Value(remainder), ..] =
            rest
        {
            if name == "value" && *divisor == AttrValue::Int(2) && *remainder == AttrValue::Int(0) {
                adapted.extend([
                    Token::Identifier("__array_even".into()),
                    Token::Operator(CompOp::Equals),
                    Token::Value(AttrValue::Bool(true)),
                ]);
                position += 5;
                continue;
            }
        }
        let token = &tokens[position];
        let begins_atom = adapted.last().is_none_or(|previous| {
            matches!(
                previous,
                Token::LeftParen | Token::LogicalOp(_) | Token::Not
            )
        });
        match token {
            Token::Value(AttrValue::Bool(value)) if begins_atom => {
                adapted.extend([
                    Token::Identifier("__array_constant".into()),
                    Token::Operator(CompOp::Equals),
                    Token::Value(AttrValue::Bool(*value)),
                ]);
            }
            Token::Identifier(text)
                if matches!(adapted.last(), Some(Token::Operator(_)))
                    && !text.eq_ignore_ascii_case("null") =>
            {
                adapted.push(Token::Value(AttrValue::Text(text.clone())));
            }
            Token::Identifier(name) if name.starts_with("__array_") => {
                return Err(invalid("Reserved internal array query operand"))
            }
            _ => adapted.push(token.clone()),
        }
        position += 1;
    }
    lower(QueryParser::scalar_predicate(adapted).map_err(|e| invalid(e.to_string()))?)
}

impl Predicate {
    fn evaluate(&self, value: &AttrValue) -> GraphResult<bool> {
        Ok(match self {
            Self::Exists => true,
            Self::Compare(Operand::Value, filter) => scalar_matches(value, filter),
            Self::Compare(Operand::Length, filter) => {
                let length = match value {
                    AttrValue::Text(_)
                    | AttrValue::CompactText(_)
                    | AttrValue::CompressedText(_) => value.text_content().map_err(invalid)?.len(),
                    AttrValue::FloatVec(values) => values.len(),
                    AttrValue::Float64Vec(values) => values.len(),
                    AttrValue::IntVec(values) => values.len(),
                    AttrValue::TextVec(values) => values.len(),
                    AttrValue::BoolVec(values) => values.len(),
                    AttrValue::NodeArray(values) | AttrValue::EdgeArray(values) => values.len(),
                    AttrValue::CompressedFloatVec(values) => {
                        values.decompress_float_vec().map_err(invalid)?.len()
                    }
                    AttrValue::Bytes(values) => values.len(),
                    _ => 0,
                };
                let length =
                    i64::try_from(length).map_err(|_| invalid("Array value length exceeds i64"))?;
                scalar_matches(&AttrValue::Int(length), filter)
            }
            Self::Compare(Operand::Even, filter) => {
                let even = match value {
                    AttrValue::Int(value) => value % 2 == 0,
                    AttrValue::SmallInt(value) => value % 2 == 0,
                    AttrValue::Float(value) => value.is_finite() && f64::from(*value) % 2.0 == 0.0,
                    AttrValue::Float64(value) => value.is_finite() && *value % 2.0 == 0.0,
                    _ => false,
                };
                scalar_matches(&AttrValue::Bool(even), filter)
            }
            Self::Compare(Operand::Constant, filter) => {
                scalar_matches(&AttrValue::Bool(true), filter)
            }
            Self::And(parts) => {
                for part in parts {
                    if !part.evaluate(value)? {
                        return Ok(false);
                    }
                }
                true
            }
            Self::Or(parts) => {
                for part in parts {
                    if part.evaluate(value)? {
                        return Ok(true);
                    }
                }
                false
            }
            Self::Not(part) => !part.evaluate(value)?,
        })
    }
}

fn scalar_matches(value: &AttrValue, filter: &AttributeFilter) -> bool {
    use crate::storage::numeric::scalar::{attribute_number, compare_operator};
    use std::cmp::Ordering;
    let target = match filter {
        AttributeFilter::Equals(target)
        | AttributeFilter::NotEquals(target)
        | AttributeFilter::GreaterThan(target)
        | AttributeFilter::LessThan(target)
        | AttributeFilter::GreaterThanOrEqual(target)
        | AttributeFilter::LessThanOrEqual(target) => target,
    };
    if let Some((left, right)) = attribute_number(value).zip(attribute_number(target)) {
        let operator = match filter {
            AttributeFilter::Equals(_) => "==",
            AttributeFilter::NotEquals(_) => "!=",
            AttributeFilter::GreaterThan(_) => ">",
            AttributeFilter::LessThan(_) => "<",
            AttributeFilter::GreaterThanOrEqual(_) => ">=",
            AttributeFilter::LessThanOrEqual(_) => "<=",
        };
        return compare_operator(left, right, operator);
    }
    if let Some((left, right)) = value.text_content().ok().zip(target.text_content().ok()) {
        let compared = left.cmp(&right);
        return match filter {
            AttributeFilter::Equals(_) => compared == Ordering::Equal,
            AttributeFilter::NotEquals(_) => compared != Ordering::Equal,
            AttributeFilter::GreaterThan(_) => compared == Ordering::Greater,
            AttributeFilter::LessThan(_) => compared == Ordering::Less,
            AttributeFilter::GreaterThanOrEqual(_) => compared != Ordering::Less,
            AttributeFilter::LessThanOrEqual(_) => compared != Ordering::Greater,
        };
    }
    // Preserve array filtering's separate boolean and mismatched-domain policy.
    match (value, target, filter) {
        (
            AttrValue::Bool(_),
            AttrValue::Bool(_),
            AttributeFilter::Equals(_) | AttributeFilter::NotEquals(_),
        )
        | (
            AttrValue::Null,
            AttrValue::Null,
            AttributeFilter::Equals(_) | AttributeFilter::NotEquals(_),
        ) => filter.matches(value),
        _ => false,
    }
}

/// Batch query evaluator for efficient filtering of multiple values
pub struct BatchQueryEvaluator {
    evaluator: QueryEvaluator,
}

impl BatchQueryEvaluator {
    /// Create a new batch evaluator
    pub fn new(query: &str) -> Self {
        Self {
            evaluator: QueryEvaluator::new(query),
        }
    }

    /// Filter a vector of values using the query
    pub fn filter_values(&self, values: Vec<AttrValue>) -> GraphResult<Vec<AttrValue>> {
        self.evaluator.validate()?;
        let mut result = Vec::new();

        for value in values {
            if self.evaluator.evaluate(&value)? {
                result.push(value);
            }
        }

        Ok(result)
    }

    /// Get indices of values that pass the filter
    pub fn filter_indices(&self, values: &[AttrValue]) -> GraphResult<Vec<usize>> {
        self.evaluator.validate()?;
        let mut result = Vec::new();

        for (i, value) in values.iter().enumerate() {
            if self.evaluator.evaluate(value)? {
                result.push(i);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_comparisons() {
        let evaluator = QueryEvaluator::new("value > 100");

        assert!(evaluator.evaluate(&AttrValue::Int(150)).unwrap());
        assert!(!evaluator.evaluate(&AttrValue::Int(50)).unwrap());
    }

    #[test]
    fn test_or_expressions() {
        let evaluator = QueryEvaluator::new("value > 100 OR value < 10");

        assert!(evaluator.evaluate(&AttrValue::Int(150)).unwrap());
        assert!(evaluator.evaluate(&AttrValue::Int(5)).unwrap());
        assert!(!evaluator.evaluate(&AttrValue::Int(50)).unwrap());
    }

    #[test]
    fn test_length_expressions() {
        let evaluator = QueryEvaluator::new("length > 5");

        assert!(evaluator
            .evaluate(&AttrValue::Text("hello world".to_string()))
            .unwrap());
        assert!(!evaluator
            .evaluate(&AttrValue::Text("hi".to_string()))
            .unwrap());
    }

    #[test]
    fn test_batch_filtering() {
        let evaluator = BatchQueryEvaluator::new("value > 50");
        let values = vec![
            AttrValue::Int(25),
            AttrValue::Int(75),
            AttrValue::Int(30),
            AttrValue::Int(100),
        ];

        let filtered = evaluator.filter_values(values).unwrap();
        assert_eq!(filtered.len(), 2);
    }
}
