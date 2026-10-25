//! Shared typed hash join planning and columnar materialization.
use super::{AttrValue, BaseArray, BaseTable, GraphResult, HashMap};
use crate::{storage::table::provenance::TableProvenance, GraphError};
use std::collections::HashSet;

type Row = (Option<usize>, Option<usize>);

fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}

impl BaseTable {
    /// Inner join using logical property identity and left-major stable row order.
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        self.join_on(other, &[left_on], &[right_on], "inner")
    }
    /// Left join, including unmatched left rows with null right cells.
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        self.join_on(other, &[left_on], &[right_on], "left")
    }
    /// Right join retains the historical swapped-left schema and right-major row order.
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        self.join_on(other, &[left_on], &[right_on], "right")
    }
    /// Full outer join: left-major matches/unmatched rows, then unmatched right rows.
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        self.join_on(other, &[left_on], &[right_on], "outer")
    }
    /// Cartesian product in left-major order, retaining read-only source contexts.
    pub fn cross_join(&self, other: &Self) -> GraphResult<Self> {
        self.join_on(other, &[], &[], "cross")
    }

    /// Join on one or more ordered pairs of columns. Null matches null, and keys
    /// use AttrValue equality (including canonical NaNs/storage representations).
    /// Generated column-name collisions reject instead of discarding data.
    /// Joined rows retain sources but have no automatic writeback identity.
    pub fn join_on(
        &self,
        other: &Self,
        left_on: &[&str],
        right_on: &[&str],
        how: &str,
    ) -> GraphResult<Self> {
        let how = how.to_ascii_lowercase();
        if !matches!(how.as_str(), "inner" | "left" | "right" | "outer" | "cross") {
            return Err(invalid(format!("Unsupported join type: {how}")));
        }
        if how != "cross" {
            if left_on.is_empty() || left_on.len() != right_on.len() {
                return Err(invalid("Join keys must be nonempty lists of equal length"));
            }
            for (table, names, side) in [(self, left_on, "Left"), (other, right_on, "Right")] {
                let mut seen = HashSet::new();
                for &name in names {
                    if !seen.insert(name) {
                        return Err(invalid(format!("{side} join column '{name}' repeated")));
                    }
                    if !table.columns.contains_key(name) {
                        return Err(invalid(format!("{side} join column '{name}' not found")));
                    }
                }
            }
        }
        if how == "right" {
            return other.join_on(self, right_on, left_on, "left");
        }
        // Plan names before expanding rows. Preserve existing public naming policies.
        let mut names = HashSet::new();
        let mut plan = Vec::new();
        for (right, table) in [(false, self), (true, other)] {
            for name in &table.column_order {
                let shared_key = left_on
                    .iter()
                    .zip(right_on)
                    .any(|(a, b)| a == b && *a == name);
                if right
                    && how != "cross"
                    && right_on.contains(&name.as_str())
                    && (how != "outer" || shared_key)
                {
                    continue;
                }
                let output = if matches!(how.as_str(), "inner" | "left") {
                    if right && self.columns.contains_key(name) {
                        format!("right_{name}")
                    } else if !right
                        && other.columns.contains_key(name)
                        && !left_on.contains(&name.as_str())
                    {
                        format!("left_{name}")
                    } else {
                        name.clone()
                    }
                } else if right && self.columns.contains_key(name) {
                    format!("{name}_right")
                } else {
                    name.clone()
                };
                if !names.insert(output.clone()) {
                    return Err(invalid(format!(
                        "Join output column '{output}' collides; rename input columns explicitly"
                    )));
                }
                plan.push((
                    right,
                    name.as_str(),
                    output,
                    how == "outer" && !right && shared_key,
                ));
            }
        }
        let rows = if how == "cross" {
            let count = self
                .nrows
                .checked_mul(other.nrows)
                .ok_or_else(|| invalid("Cross join row count overflow"))?;
            let mut rows = Vec::new();
            rows.try_reserve_exact(count)
                .map_err(|_| invalid("Cross join row allocation exceeds capacity"))?;
            for left in 0..self.nrows {
                for right in 0..other.nrows {
                    rows.push((Some(left), Some(right)));
                }
            }
            rows
        } else {
            fn columns<'a>(
                table: &'a BaseTable,
                keys: &[&str],
            ) -> GraphResult<Vec<&'a BaseArray<AttrValue>>> {
                keys.iter()
                    .map(|key| {
                        table
                            .columns
                            .get(*key)
                            .ok_or_else(|| invalid("Missing join key"))
                    })
                    .collect()
            }
            let left = columns(self, left_on)?;
            let right = columns(other, right_on)?;
            let key = |columns: &[&BaseArray<AttrValue>], row| -> Vec<AttrValue> {
                columns
                    .iter()
                    .map(|col| col.get(row).cloned().unwrap_or(AttrValue::Null))
                    .collect()
            };
            let mut index: HashMap<Vec<AttrValue>, Vec<usize>> = HashMap::new();
            for row in 0..other.nrows {
                index.entry(key(&right, row)).or_default().push(row);
            }
            let mut matched = vec![false; other.nrows];
            let mut rows: Vec<Row> = Vec::new();
            for row in 0..self.nrows {
                if let Some(matches) = index.get(&key(&left, row)) {
                    rows.try_reserve(matches.len())
                        .map_err(|_| invalid("Join row allocation exceeds capacity"))?;
                    for &r in matches {
                        rows.push((Some(row), Some(r)));
                        matched[r] = true;
                    }
                } else if how != "inner" {
                    rows.push((Some(row), None));
                }
            }
            if how == "outer" {
                for (row, matched) in matched.into_iter().enumerate() {
                    if !matched {
                        rows.push((None, Some(row)));
                    }
                }
            }
            rows
        };
        let mut columns = HashMap::new();
        let mut order = Vec::new();
        for (right, name, output, coalesce) in plan {
            let table = if right { other } else { self };
            let column = table
                .columns
                .get(name)
                .ok_or_else(|| invalid("Missing join output column"))?;
            let values = rows
                .iter()
                .map(|&(left, r)| {
                    let row = if right { r } else { left };
                    if let Some(row) = row {
                        column.get(row).cloned().unwrap_or(AttrValue::Null)
                    } else if coalesce {
                        r.and_then(|r| other.columns.get(name)?.get(r))
                            .cloned()
                            .unwrap_or(AttrValue::Null)
                    } else {
                        AttrValue::Null
                    }
                })
                .collect();
            order.push(output.clone());
            columns.insert(output, BaseArray::from_attr_values(values));
        }
        let mut result = Self::from_columns(columns)?;
        result.column_order = order;
        result.nrows = rows.len(); // A Cartesian product can have rows and no columns.
        result.provenance =
            TableProvenance::joined(self.provenance.as_ref(), other.provenance.as_ref());
        Ok(result)
    }
}
