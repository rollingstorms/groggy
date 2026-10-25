//! Strict structural-column readers shared by materialization and writeback.
use super::{BaseTable, Table};
use crate::{storage::array::BaseArray, AttrValue, GraphError, GraphResult};

pub(super) fn column<'a>(
    table: &'a BaseTable,
    name: &str,
) -> GraphResult<&'a BaseArray<AttrValue>> {
    let column = table
        .column(name)
        .ok_or_else(|| GraphError::InvalidInput(format!("Missing structural column {name}")))?;
    if column.len() != table.nrows() {
        return Err(GraphError::InvalidInput(format!(
            "{name}: inconsistent column length"
        )));
    }
    Ok(column)
}

pub(super) fn id(value: &AttrValue, name: &str, row: usize) -> GraphResult<usize> {
    let value = match value {
        AttrValue::Int(value) => Some(*value),
        AttrValue::SmallInt(value) => Some(i64::from(*value)),
        _ => None,
    };
    value
        .and_then(|value| usize::try_from(value).ok())
        .filter(|id| *id != usize::MAX)
        .ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "{name} row {row}: expected a nonnegative integer ID"
            ))
        })
}

pub(super) fn ids(table: &BaseTable, name: &str) -> GraphResult<Vec<usize>> {
    column(table, name)?
        .iter()
        .enumerate()
        .map(|(row, value)| id(value, name, row))
        .collect()
}

pub(super) fn materialized_edge_ids(table: &BaseTable) -> GraphResult<Vec<usize>> {
    let values: Vec<_> = column(table, "edge_id")?
        .iter()
        .enumerate()
        .map(|(row, value)| {
            if matches!(value, AttrValue::Null) {
                Ok(None)
            } else {
                id(value, "edge_id", row).map(Some)
            }
        })
        .collect::<GraphResult<_>>()?;
    // Preserve legacy null-ID assignment after the maximum declared ID, but check
    // arithmetic and integer representability rather than overflowing or wrapping.
    let mut next = values.iter().flatten().copied().max().unwrap_or(0);
    values
        .into_iter()
        .enumerate()
        .map(|(row, value)| match value {
            Some(id) => Ok(id),
            None => {
                next = next
                    .checked_add(1)
                    .filter(|id| i64::try_from(*id).is_ok())
                    .ok_or_else(|| {
                        GraphError::InvalidInput(format!(
                            "edge_id row {row}: automatic ID exceeds the table integer range"
                        ))
                    })?;
                Ok(next)
            }
        })
        .collect()
}

/// A specialized reshape may not discard structural columns or duplicate IDs.
/// General long/aggregated tables are available through the underlying BaseTable.
pub(super) fn validate_reshape(table: &BaseTable, required: &[&str]) -> GraphResult<()> {
    let mut seen = std::collections::HashSet::new();
    for (column_index, name) in required.iter().enumerate() {
        let values = ids(table, name).map_err(|error| {
            GraphError::InvalidInput(format!(
            "Reshape must retain entity identity; use base_table() for a general table: {error}"
        ))
        })?;
        if column_index == 0 && values.iter().any(|id| !seen.insert(*id)) {
            return Err(GraphError::InvalidInput(
                "Reshape duplicates entity IDs; use base_table() for a general table".into(),
            ));
        }
    }
    Ok(())
}
