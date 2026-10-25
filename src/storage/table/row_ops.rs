//! Typed row keys, stable grouping, and explicit table combination.
use super::{AttrValue, BaseArray, BaseTable, GraphResult, HashMap};
use crate::{
    storage::table::{provenance::TableProvenance, TableArray},
    GraphError,
};
use std::collections::HashSet;

type Groups = Vec<(Vec<AttrValue>, Vec<usize>)>;
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}

pub(super) fn row_key(columns: &[&BaseArray<AttrValue>], row: usize) -> Vec<AttrValue> {
    columns
        .iter()
        .map(|c| c.get(row).cloned().unwrap_or(AttrValue::Null))
        .collect()
}
impl BaseTable {
    /// Checked bulk identity lookup; scan IDs once and copy only selected first rows.
    pub(in crate::storage::table) fn identity_rows(
        &self,
        id_column: &str,
        requested: &[usize],
    ) -> GraphResult<HashMap<usize, HashMap<String, AttrValue>>> {
        let ids = crate::storage::table::structural::column(self, id_column)?;
        let columns = self
            .column_order
            .iter()
            .map(|name| Ok((name, crate::storage::table::structural::column(self, name)?)))
            .collect::<GraphResult<Vec<_>>>()?;
        let requested: HashSet<_> = requested.iter().copied().collect();
        let mut result = HashMap::new();
        for (row, value) in ids.iter().enumerate() {
            let id = crate::storage::table::structural::id(value, id_column, row)?;
            if requested.contains(&id) {
                result.entry(id).or_insert_with(|| {
                    columns
                        .iter()
                        .map(|(name, column)| ((*name).clone(), column.data()[row].clone()))
                        .collect()
                });
            }
        }
        Ok(result)
    }

    /// Select rows by logical property equality without constructing a query string.
    pub fn filter_by_value(&self, name: &str, value: &AttrValue) -> GraphResult<Self> {
        let column = self
            .columns
            .get(name)
            .ok_or_else(|| invalid(format!("Column '{name}' not found")))?;
        let rows = column
            .iter()
            .enumerate()
            .filter_map(|(row, candidate)| (candidate == value).then_some(row))
            .collect::<Vec<_>>();
        self.select_rows(&rows)
    }
    pub(super) fn key_columns(&self, names: &[&str]) -> GraphResult<Vec<&BaseArray<AttrValue>>> {
        let mut seen = HashSet::new();
        names
            .iter()
            .map(|&name| {
                if !seen.insert(name) {
                    return Err(invalid(format!("Key column '{name}' repeated")));
                }
                self.columns
                    .get(name)
                    .ok_or_else(|| invalid(format!("Key column '{name}' not found")))
            })
            .collect()
    }
    pub(super) fn stable_groups(&self, names: &[&str]) -> GraphResult<Groups> {
        let columns = self.key_columns(names)?;
        let mut index: HashMap<Vec<AttrValue>, usize> = HashMap::new();
        let mut groups: Groups = Vec::new();
        for row in 0..self.nrows {
            let key = row_key(&columns, row);
            let i = *index.entry(key.clone()).or_insert_with(|| {
                groups.push((key, Vec::new()));
                groups.len() - 1
            });
            groups[i].1.push(row);
        }
        Ok(groups)
    }
    /// Stable groups in first-key occurrence order; each group retains direct row identity.
    pub fn groupby(&self, by: &[&str]) -> GraphResult<TableArray> {
        let groups = self.stable_groups(by)?;
        let mut tables = Vec::with_capacity(groups.len());
        let mut keys = Vec::with_capacity(groups.len());
        for (key, rows) in groups {
            tables.push(self.select_rows(&rows)?);
            keys.push(by.iter().map(|s| s.to_string()).zip(key).collect());
        }
        Ok(TableArray::from_tables_with_keys(tables, keys).remember_origin(self, by))
    }
    /// Keep the first row for each logical key, retaining its exact source identity.
    /// An empty subset considers all rows equal, including zero-column rows.
    pub fn drop_duplicates(&self, subset: Option<&[String]>) -> GraphResult<Self> {
        let names = subset
            .unwrap_or(&self.column_order)
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let columns = self.key_columns(&names)?;
        let mut seen = HashSet::new();
        let rows = (0..self.nrows)
            .filter(|&i| seen.insert(row_key(&columns, i)))
            .collect::<Vec<_>>();
        self.select_rows(&rows)
    }
    fn same_row_schema(&self, other: &Self) -> GraphResult<()> {
        if self.column_order != other.column_order {
            return Err(invalid(
                "Row combination requires identical ordered column schemas",
            ));
        }
        self.key_columns(
            &self
                .column_order
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )?;
        other.key_columns(
            &other
                .column_order
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )?;
        Ok(())
    }
    fn combine_rows(&self, other: &Self, operation: &str) -> GraphResult<Self> {
        self.same_row_schema(other)?;
        let names = self
            .column_order
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let left = self.key_columns(&names)?;
        let right = other.key_columns(&names)?;
        let other_keys: HashSet<_> = if operation == "intersect" {
            (0..other.nrows).map(|i| row_key(&right, i)).collect()
        } else {
            HashSet::new()
        };
        let count = self
            .nrows
            .checked_add(other.nrows)
            .ok_or_else(|| invalid("Row combination count overflow"))?;
        let mut rows = Vec::new();
        rows.try_reserve(if operation == "intersect" {
            self.nrows
        } else {
            count
        })
        .map_err(|_| invalid("Row combination allocation exceeds capacity"))?;
        let mut seen = HashSet::new();
        for (is_right, table, columns) in [(false, self, &left), (true, other, &right)] {
            if is_right && operation == "intersect" {
                break;
            }
            for row in 0..table.nrows {
                let keep = if operation == "stack" {
                    true
                } else {
                    let key = row_key(columns, row);
                    (operation != "intersect" || other_keys.contains(&key)) && seen.insert(key)
                };
                if keep {
                    rows.push((is_right, row));
                }
            }
        }
        let mut columns = HashMap::new();
        for (i, name) in self.column_order.iter().enumerate() {
            let values = rows
                .iter()
                .map(|&(r, row)| {
                    if r { right[i] } else { left[i] }
                        .get(row)
                        .cloned()
                        .unwrap_or(AttrValue::Null)
                })
                .collect();
            columns.insert(name.clone(), BaseArray::from_attr_values(values));
        }
        let mut result = Self::from_columns(columns)?;
        result.column_order = self.column_order.clone();
        result.nrows = rows.len();
        result.provenance =
            TableProvenance::joined(self.provenance.as_ref(), other.provenance.as_ref());
        Ok(result)
    }
    /// Stable set union: first occurrence in the left input, then unseen right rows.
    pub fn union(&self, other: &Self) -> GraphResult<Self> {
        self.combine_rows(other, "union")
    }
    /// Stable set intersection: first matching occurrence in the left input.
    pub fn intersect(&self, other: &Self) -> GraphResult<Self> {
        self.combine_rows(other, "intersect")
    }
    /// Append every row, requiring identical ordered column schemas.
    pub fn stack(&self, other: &Self) -> GraphResult<Self> {
        self.combine_rows(other, "stack")
    }
    /// Combine columns by row position with null padding. Right conflicts receive `_y`;
    /// collisions with existing/generated names reject before materialization.
    pub fn concat(&self, other: &Self) -> GraphResult<Self> {
        let mut columns = HashMap::new();
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let count = self.nrows.max(other.nrows);
        let mut plan = Vec::new();
        for (right, table) in [(false, self), (true, other)] {
            for name in &table.column_order {
                let output = if right && self.columns.contains_key(name) {
                    format!("{name}_y")
                } else {
                    name.clone()
                };
                if !seen.insert(output.clone()) {
                    return Err(invalid(format!(
                        "Concat output column '{output}' collides; rename input columns explicitly"
                    )));
                }
                let column = table
                    .columns
                    .get(name)
                    .ok_or_else(|| invalid("Missing concat column"))?;
                plan.push((output, column));
            }
        }
        for (name, column) in plan {
            let mut values = column.data().clone();
            values.resize(count, AttrValue::Null);
            order.push(name.clone());
            columns.insert(name, BaseArray::from_attr_values(values));
        }
        let mut result = Self::from_columns(columns)?;
        result.column_order = order;
        result.nrows = count;
        result.provenance =
            TableProvenance::joined(self.provenance.as_ref(), other.provenance.as_ref());
        Ok(result)
    }
}
