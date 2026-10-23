//! Simple query parser for lazy array filtering operations
//! Supports basic filtering expressions like "value > 100", "length > 5", etc.

use crate::errors::GraphResult;
use crate::types::AttrValue;

/// Simple query evaluator for AttrValue filtering
pub struct QueryEvaluator {
    query: String,
}

impl QueryEvaluator {
    /// Create a new query evaluator
    pub fn new(query: &str) -> Self {
        Self {
            query: query.to_string(),
        }
    }

    /// Evaluate the query against a single AttrValue
    pub fn evaluate(&self, value: &AttrValue) -> GraphResult<bool> {
        // Parse simple expressions like "value > 100", "length > 5", etc.
        if self.query.contains("OR") {
            return self.evaluate_or_expression(value);
        }

        if self.query.contains("AND") {
            return self.evaluate_and_expression(value);
        }

        self.evaluate_simple_expression(value)
    }

    /// Evaluate OR expression (any condition can be true)
    fn evaluate_or_expression(&self, value: &AttrValue) -> GraphResult<bool> {
        let parts: Vec<&str> = self.query.split("OR").map(|s| s.trim()).collect();

        for part in parts {
            let evaluator = QueryEvaluator::new(part);
            if evaluator.evaluate(value)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Evaluate AND expression (all conditions must be true)
    fn evaluate_and_expression(&self, value: &AttrValue) -> GraphResult<bool> {
        let parts: Vec<&str> = self.query.split("AND").map(|s| s.trim()).collect();

        for part in parts {
            let evaluator = QueryEvaluator::new(part);
            if !evaluator.evaluate(value)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Evaluate simple expressions like "value > 100"
    fn evaluate_simple_expression(&self, value: &AttrValue) -> GraphResult<bool> {
        // Handle special cases first
        if self.query == "true" {
            return Ok(true);
        }
        if self.query == "false" {
            return Ok(false);
        }

        // Match the longest token first so >= and <= are not consumed as > or <.
        for operator in [">=", "<=", "==", "!=", ">", "<"] {
            if let Some(result) = self.try_parse_comparison(value, operator)? {
                return Ok(result);
            }
        }

        // Handle special expressions
        if self.query.starts_with("length >") {
            return self.evaluate_length_expression(value);
        }
        if self.query.contains("% 2 == 0") {
            return self.evaluate_even_expression(value);
        }

        // Default: return true (pass-through filter)
        Ok(true)
    }

    /// Try to parse and evaluate a comparison expression
    fn try_parse_comparison(&self, value: &AttrValue, operator: &str) -> GraphResult<Option<bool>> {
        if !self.query.contains(operator) {
            return Ok(None);
        }

        let parts: Vec<&str> = self.query.split(operator).map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Ok(None);
        }

        let left = parts[0];
        let right = parts[1];

        // Handle "value" comparisons
        if left == "value" {
            let target_value = self.parse_value(right)?;
            return Ok(Some(self.compare_values(value, &target_value, operator)?));
        }

        Ok(None)
    }

    /// Parse a string into an AttrValue
    fn parse_value(&self, s: &str) -> GraphResult<AttrValue> {
        // Try to parse as different types
        if let Ok(int_val) = s.parse::<i64>() {
            return Ok(AttrValue::Int(int_val));
        }

        if let Ok(float_val) = s.parse::<f64>() {
            return Ok(AttrValue::Float64(float_val));
        }

        if s == "true" {
            return Ok(AttrValue::Bool(true));
        }

        if s == "false" {
            return Ok(AttrValue::Bool(false));
        }

        // Default to string (remove quotes if present)
        let cleaned = s.trim_matches('"').trim_matches('\'');
        Ok(AttrValue::Text(cleaned.to_string()))
    }

    /// Compare two AttrValues using the specified operator
    fn compare_values(
        &self,
        left: &AttrValue,
        right: &AttrValue,
        operator: &str,
    ) -> GraphResult<bool> {
        use crate::storage::numeric::scalar::{attribute_number, compare_operator};
        if let Some((a, b)) = attribute_number(left).zip(attribute_number(right)) {
            return Ok(compare_operator(a, b, operator));
        }
        match (left, right) {
            (AttrValue::Text(a), AttrValue::Text(b)) => Ok(match operator {
                "==" => a == b,
                "!=" => a != b,
                ">" => a > b,
                "<" => a < b,
                ">=" => a >= b,
                "<=" => a <= b,
                _ => false,
            }),
            (AttrValue::Bool(a), AttrValue::Bool(b)) => Ok(match operator {
                "==" => a == b,
                "!=" => a != b,
                _ => false,
            }),
            _ => Ok(false), // Type mismatch
        }
    }

    /// Evaluate length-based expressions like "length > 5"
    fn evaluate_length_expression(&self, value: &AttrValue) -> GraphResult<bool> {
        let length = match value {
            AttrValue::Text(s) => s.len(),
            AttrValue::FloatVec(v) => v.len(),
            AttrValue::Float64Vec(v) => v.len(),
            AttrValue::Bytes(b) => b.len(),
            _ => 0,
        };

        // Parse the threshold
        let parts: Vec<&str> = self.query.split(">").collect();
        if parts.len() == 2 {
            if let Ok(threshold) = parts[1].trim().parse::<usize>() {
                return Ok(length > threshold);
            }
        }

        Ok(false)
    }

    /// Evaluate even/odd expressions like "value % 2 == 0"
    fn evaluate_even_expression(&self, value: &AttrValue) -> GraphResult<bool> {
        match value {
            AttrValue::Int(i) => Ok(*i % 2 == 0),
            AttrValue::SmallInt(i) => Ok(*i % 2 == 0),
            AttrValue::Float(f) => Ok(f.is_finite() && f64::from(*f) % 2.0 == 0.0),
            AttrValue::Float64(f) => Ok(f.is_finite() && *f % 2.0 == 0.0),
            _ => Ok(false),
        }
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
