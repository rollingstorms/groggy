//! Type conversion utilities for GraphMatrix
//!
//! This module provides conversion traits between Groggy's AttrValue system
//! and the NumericType trait system used in the advanced matrix backend.

use crate::errors::{GraphError, GraphResult};
use crate::storage::advanced_matrix::NumericType;
use crate::types::AttrValue;

/// Trait for converting AttrValue to numeric types
pub trait FromAttrValue<T: NumericType> {
    fn from_attr_value(value: &AttrValue) -> GraphResult<T>;
}

impl FromAttrValue<f64> for f64 {
    fn from_attr_value(value: &AttrValue) -> GraphResult<f64> {
        match value {
            AttrValue::Float(f) => Ok(*f as f64),
            AttrValue::Int(i) => Ok(*i as f64),
            AttrValue::SmallInt(i) => Ok(*i as f64),
            AttrValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            AttrValue::Text(s) => s
                .parse()
                .map_err(|_| GraphError::InvalidInput(format!("Cannot convert '{}' to f64", s))),
            AttrValue::Null => Ok(0.0),
            _ => Err(GraphError::InvalidInput(
                "Unsupported attribute type for f64".into(),
            )),
        }
    }
}

impl FromAttrValue<f32> for f32 {
    fn from_attr_value(value: &AttrValue) -> GraphResult<f32> {
        <f64 as FromAttrValue<f64>>::from_attr_value(value).map(|v| v as f32)
    }
}

impl FromAttrValue<i64> for i64 {
    fn from_attr_value(value: &AttrValue) -> GraphResult<i64> {
        match value {
            AttrValue::Int(i) => Ok(*i),
            AttrValue::SmallInt(i) => Ok(*i as i64),
            AttrValue::Float(f) => Ok(*f as i64),
            AttrValue::Bool(b) => Ok(if *b { 1 } else { 0 }),
            AttrValue::Text(s) => s
                .parse()
                .map_err(|_| GraphError::InvalidInput(format!("Cannot convert '{}' to i64", s))),
            AttrValue::Null => Ok(0),
            _ => Err(GraphError::InvalidInput(
                "Unsupported attribute type for i64".into(),
            )),
        }
    }
}

impl FromAttrValue<i32> for i32 {
    fn from_attr_value(value: &AttrValue) -> GraphResult<i32> {
        <i64 as FromAttrValue<i64>>::from_attr_value(value).map(|v| v as i32)
    }
}

impl FromAttrValue<bool> for bool {
    fn from_attr_value(value: &AttrValue) -> GraphResult<bool> {
        match value {
            AttrValue::Bool(b) => Ok(*b),
            AttrValue::Int(i) => Ok(*i != 0),
            AttrValue::SmallInt(i) => Ok(*i != 0),
            AttrValue::Float(f) => Ok(*f != 0.0),
            AttrValue::Text(s) => match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Ok(true),
                "false" | "0" | "no" | "off" => Ok(false),
                _ => Err(GraphError::InvalidInput(format!(
                    "Cannot convert '{}' to bool",
                    s
                ))),
            },
            AttrValue::Null => Ok(false),
            _ => Err(GraphError::InvalidInput(
                "Unsupported attribute type for bool".into(),
            )),
        }
    }
}

/// Extension trait to add conversion methods to NumericType
pub trait NumericTypeExt<T: NumericType> {
    /// Convert from AttrValue using the FromAttrValue trait
    fn from_attr_value(value: &AttrValue) -> GraphResult<T>;
}

impl<T: NumericType> NumericTypeExt<T> for T
where
    T: FromAttrValue<T>,
{
    fn from_attr_value(value: &AttrValue) -> GraphResult<T> {
        <T as FromAttrValue<T>>::from_attr_value(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_conversions() {
        assert_eq!(
            <f64 as FromAttrValue<f64>>::from_attr_value(&AttrValue::Float(3.5)).unwrap(),
            3.5
        );
        assert_eq!(
            <f64 as FromAttrValue<f64>>::from_attr_value(&AttrValue::Int(42)).unwrap(),
            42.0
        );
        assert_eq!(
            <f64 as FromAttrValue<f64>>::from_attr_value(&AttrValue::Bool(true)).unwrap(),
            1.0
        );
        assert_eq!(
            <f64 as FromAttrValue<f64>>::from_attr_value(&AttrValue::Bool(false)).unwrap(),
            0.0
        );
        assert_eq!(
            <f64 as FromAttrValue<f64>>::from_attr_value(&AttrValue::Text("2.5".to_string()))
                .unwrap(),
            2.5
        );
        assert_eq!(
            <f64 as FromAttrValue<f64>>::from_attr_value(&AttrValue::Null).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_i64_conversions() {
        assert_eq!(
            <i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Int(42)).unwrap(),
            42
        );
        assert_eq!(
            <i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Float(3.5)).unwrap(),
            3
        );
        assert_eq!(
            <i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Bool(true)).unwrap(),
            1
        );
        assert_eq!(
            <i64 as FromAttrValue<i64>>::from_attr_value(&AttrValue::Bool(false)).unwrap(),
            0
        );
    }

    #[test]
    fn test_bool_conversions() {
        assert!(<bool as FromAttrValue<bool>>::from_attr_value(&AttrValue::Bool(true)).unwrap());
        assert!(<bool as FromAttrValue<bool>>::from_attr_value(&AttrValue::Int(1)).unwrap());
        assert!(!<bool as FromAttrValue<bool>>::from_attr_value(&AttrValue::Int(0)).unwrap());
        assert!(
            <bool as FromAttrValue<bool>>::from_attr_value(&AttrValue::Text("true".to_string()))
                .unwrap()
        );
        assert!(
            !<bool as FromAttrValue<bool>>::from_attr_value(&AttrValue::Text("false".to_string()))
                .unwrap()
        );
    }
}
