//! Integration tests for BaseArray ↔ BaseTable core functionality
//!
//! This module tests the integration between BaseArray and BaseTable systems
//! to ensure they work together seamlessly.

// TODO: These tests use outdated BaseArray::with_name() API that no longer exists.
// Update to use current BaseArray::new() API.
#[cfg(all(test, feature = "integration_tests_disabled"))]
mod tests {
    use super::super::base::BaseTable;
    use super::super::traits::Table;
    use crate::storage::array::{ArrayOps, BaseArray};
    use crate::types::{AttrValue, AttrValueType};
    use std::collections::HashMap;

    /// Test BaseTable creation from BaseArray columns
    #[test]
    #[ignore] // TODO: Update to use current BaseArray API
    fn test_basetable_from_basearray_columns() {
        // Create BaseArray columns
        let ages = BaseArray::with_name(
            vec![AttrValue::Int(25), AttrValue::Int(30), AttrValue::Int(35)],
            AttrValueType::Int,
            "age".to_string(),
        );

        let names = BaseArray::with_name(
            vec![
                AttrValue::Text("Alice".to_string()),
                AttrValue::Text("Bob".to_string()),
                AttrValue::Text("Charlie".to_string()),
            ],
            AttrValueType::Text,
            "name".to_string(),
        );

        // Create BaseTable from columns
        let mut columns = HashMap::new();
        columns.insert("age".to_string(), ages);
        columns.insert("name".to_string(), names);

        let table = BaseTable::from_columns(columns).expect("Table creation should succeed");

        // Test table properties
        assert_eq!(table.nrows(), 3);
        assert_eq!(table.ncols(), 2);
        assert!(table.has_column("age"));
        assert!(table.has_column("name"));
        assert!(!table.has_column("nonexistent"));
    }

    /// Test table.column().iter() chaining
    #[test]
    #[ignore] // TODO: Update to use current BaseArray API
    fn test_table_column_iter_chaining() {
        // Create test data
        let ages = BaseArray::with_name(
            vec![
                AttrValue::Int(25),
                AttrValue::Int(30),
                AttrValue::Int(35),
                AttrValue::Int(20),
            ],
            AttrValueType::Int,
            "age".to_string(),
        );

        let names = BaseArray::with_name(
            vec![
                AttrValue::Text("Alice".to_string()),
                AttrValue::Text("Bob".to_string()),
                AttrValue::Text("Charlie".to_string()),
                AttrValue::Text("David".to_string()),
            ],
            AttrValueType::Text,
            "name".to_string(),
        );

        let mut columns = HashMap::new();
        columns.insert("age".to_string(), ages);
        columns.insert("name".to_string(), names);

        let table = BaseTable::from_columns(columns).expect("Table creation should succeed");

        // Test column access and iteration
        let age_column = table.column("age").expect("Age column should exist");
        assert_eq!(age_column.len(), 4);

        // Test that we can get the iterator (basic chaining)
        let age_iter = age_column.iter();
        let collected: Vec<AttrValue> = age_iter.into_vec();
        assert_eq!(collected.len(), 4);

        // Verify the data
        match &collected[0] {
            AttrValue::Int(val) => assert_eq!(*val, 25),
            _ => panic!("Expected Int value"),
        }
    }

    /// Test BaseArray iterator operations through table
    #[test]
    #[ignore] // TODO: Update to use current BaseArray API
    fn test_basearray_iter_operations() {
        // Create test array
        let ages = BaseArray::with_name(
            vec![
                AttrValue::Int(25),
                AttrValue::Int(30),
                AttrValue::Int(35),
                AttrValue::Int(20),
            ],
            AttrValueType::Int,
            "age".to_string(),
        );

        // Test direct array iteration
        let iter = ages.iter();
        let collected: Vec<AttrValue> = iter.into_vec();
        assert_eq!(collected.len(), 4);

        // Test that we can access individual elements
        assert_eq!(ages.get(0), Some(&AttrValue::Int(25)));
        assert_eq!(ages.get(1), Some(&AttrValue::Int(30)));
        assert_eq!(ages.get(4), None); // Out of bounds
    }

    /// Test table operations with BaseArray integration
    #[test]
    #[ignore] // TODO: Update to use current BaseArray API
    fn test_table_operations() {
        // Create test table
        let ages = BaseArray::with_name(
            vec![AttrValue::Int(25), AttrValue::Int(30), AttrValue::Int(35)],
            AttrValueType::Int,
            "age".to_string(),
        );

        let salaries = BaseArray::with_name(
            vec![
                AttrValue::Float(50000.0),
                AttrValue::Float(60000.0),
                AttrValue::Float(70000.0),
            ],
            AttrValueType::Float,
            "salary".to_string(),
        );

        let mut columns = HashMap::new();
        columns.insert("age".to_string(), ages);
        columns.insert("salary".to_string(), salaries);

        let table = BaseTable::from_columns(columns).expect("Table creation should succeed");

        // Test head operation
        let head_table = table.head(2);
        assert_eq!(head_table.nrows(), 2);
        assert_eq!(head_table.ncols(), 2);

        // Test tail operation
        let tail_table = table.tail(1);
        assert_eq!(tail_table.nrows(), 1);
        assert_eq!(tail_table.ncols(), 2);

        // Test slice operation
        let slice_table = table.slice(1, 3);
        assert_eq!(slice_table.nrows(), 2);
        assert_eq!(slice_table.ncols(), 2);
    }

    /// Test column access and properties
    #[test]
    #[ignore] // TODO: Update to use current BaseArray API
    fn test_column_access_properties() {
        // Create mixed-type columns
        let ids = BaseArray::with_name(
            vec![AttrValue::Int(1), AttrValue::Int(2), AttrValue::Int(3)],
            AttrValueType::Int,
            "id".to_string(),
        );

        let active = BaseArray::with_name(
            vec![
                AttrValue::Bool(true),
                AttrValue::Bool(false),
                AttrValue::Bool(true),
            ],
            AttrValueType::Bool,
            "active".to_string(),
        );

        let mut columns = HashMap::new();
        columns.insert("id".to_string(), ids);
        columns.insert("active".to_string(), active);

        let table = BaseTable::from_columns(columns).expect("Table creation should succeed");

        // Test column properties
        let id_column = table.column("id").expect("ID column should exist");
        assert_eq!(id_column.dtype(), &AttrValueType::Int);
        assert_eq!(id_column.name(), Some(&"id".to_string()));

        let active_column = table.column("active").expect("Active column should exist");
        assert_eq!(active_column.dtype(), &AttrValueType::Bool);
        assert_eq!(active_column.name(), Some(&"active".to_string()));

        // Test column_by_index
        let first_column = table.column_by_index(0).expect("First column should exist");
        let second_column = table
            .column_by_index(1)
            .expect("Second column should exist");
        assert!(table.column_by_index(2).is_none()); // Should not exist

        // Verify we have both columns
        assert!(
            first_column.dtype() == &AttrValueType::Int
                || first_column.dtype() == &AttrValueType::Bool
        );
        assert!(
            second_column.dtype() == &AttrValueType::Int
                || second_column.dtype() == &AttrValueType::Bool
        );
        assert_ne!(first_column.dtype(), second_column.dtype());
    }
}
