//! Checked collection metadata, shared numeric reductions and source-aware outputs.
use super::{AttrValue as V, BaseTable, GraphResult, HashMap, Table, TableArray};
use crate::{
    storage::array::{reductions, BaseArray},
    GraphError,
};
use std::collections::HashSet;
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reduction {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Nunique,
    All,
}
impl Reduction {
    fn parse(name: &str) -> GraphResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sum" => Ok(Self::Sum),
            "mean" | "avg" => Ok(Self::Mean),
            "count" => Ok(Self::Count),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "nunique" => Ok(Self::Nunique),
            "all" => Ok(Self::All),
            _ => Err(invalid(format!("Unknown aggregation function: {name}"))),
        }
    }
    fn apply<'a>(self, values: impl Iterator<Item = &'a V> + Clone) -> GraphResult<V> {
        let values = values.filter(|v| !matches!(v, V::Null));
        if self == Self::Count {
            return count(values.count());
        }
        if self == Self::Nunique {
            return count(values.collect::<HashSet<_>>().len());
        }
        if values.clone().next().is_none() {
            return Ok(V::Null);
        }
        match self {
            Self::Sum => reductions::sum_iter(values),
            Self::Mean => reductions::mean_iter(values).map(V::Float64),
            Self::Min => reductions::numeric_extremum_iter(values, true),
            Self::Max => reductions::numeric_extremum_iter(values, false),
            Self::All => {
                let mut all = true;
                for v in values {
                    if let V::Bool(v) = v {
                        all &= *v;
                    } else {
                        return Err(invalid("'all' aggregation only supports boolean columns"));
                    }
                }
                Ok(V::Bool(all))
            }
            _ => Err(invalid("Invalid reduction")),
        }
    }
}
fn count(n: usize) -> GraphResult<V> {
    i64::try_from(n)
        .map(V::Int)
        .map_err(|_| invalid("Table collection count exceeds i64"))
}
impl TableArray {
    /// Build a per-table grouped collection using explicit constant key columns.
    /// Empty tables carry null key values; all nonempty key columns must be constant.
    pub fn from_group_columns(tables: Vec<BaseTable>, columns: &[String]) -> GraphResult<Self> {
        let mut seen = HashSet::new();
        for c in columns {
            if !seen.insert(c) {
                return Err(invalid(format!("Group key '{c}' repeated")));
            }
        }
        let keys = tables
            .iter()
            .map(|t| {
                columns
                    .iter()
                    .map(|c| {
                        let column = t
                            .column(c)
                            .ok_or_else(|| invalid(format!("Group key column '{c}' missing")))?;
                        Ok((c.clone(), column.get(0).cloned().unwrap_or(V::Null)))
                    })
                    .collect::<GraphResult<HashMap<_, _>>>()
            })
            .collect::<GraphResult<Vec<_>>>()?;
        let result = Self::from_tables_with_keys(tables, keys).declare_keys(columns);
        result.key_names()?;
        Ok(result)
    }
    pub(super) fn key_names(&self) -> GraphResult<Vec<String>> {
        if let Some(error) = self.context.as_ref().and_then(|c| c.error.as_ref()) {
            return Err(invalid(error.clone()));
        }
        let Some(keys) = &self.group_keys else {
            return Ok(Vec::new());
        };
        if keys.len() != self.tables.len() {
            return Err(invalid("Group key count must equal table count"));
        }
        let mut names = keys
            .first()
            .map(|k| k.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                self.context
                    .as_ref()
                    .map(|c| c.group_names.clone())
                    .unwrap_or_default()
            });
        names.sort();
        for (table, key) in self.tables.iter().zip(keys) {
            if key.len() != names.len() || names.iter().any(|n| !key.contains_key(n)) {
                return Err(invalid("Group key schemas differ between tables"));
            }
            for name in &names {
                if let Some(column) = table.column(name) {
                    let expected = key.get(name).ok_or_else(|| invalid("Missing group key"))?;
                    if column.iter().any(|value| value != expected) {
                        return Err(invalid(format!(
                            "Group key '{name}' disagrees with table values"
                        )));
                    }
                }
            }
        }
        Ok(names)
    }
    pub(super) fn column_names_checked(&self) -> GraphResult<Vec<String>> {
        let mut names = Vec::new();
        let mut seen = HashSet::new();
        for table in &self.tables {
            let mut local = HashSet::new();
            for name in table.column_names() {
                if !local.insert(name) || table.column(name).is_none() {
                    return Err(invalid("Invalid collection table column schema"));
                }
                if seen.insert(name.clone()) {
                    names.push(name.clone());
                }
            }
        }
        if self.tables.is_empty() {
            if let Some(context) = &self.context {
                names = context.prototype.column_names().to_vec();
            }
        }
        Ok(names)
    }
    fn validate_specs(&self, specs: &[(String, Reduction)], keys: &[String]) -> GraphResult<()> {
        let names = self.column_names_checked()?;
        for (name, _) in specs {
            if keys.contains(name) {
                return Err(invalid(format!(
                    "Aggregation column '{name}' collides with group key"
                )));
            }
            if (!self.tables.is_empty() || self.context.is_some()) && !names.contains(name) {
                return Err(invalid(format!("Aggregation column '{name}' missing")));
            }
        }
        Ok(())
    }
    fn aggregate_specs(&self, specs: Vec<(String, Reduction)>) -> GraphResult<BaseTable> {
        let keys = self.key_names()?;
        self.validate_specs(&specs, &keys)?;
        let mut order = keys.clone();
        order.extend(specs.iter().map(|(n, _)| n.clone()));
        let rows = if self.group_keys.is_some() {
            self.tables.len()
        } else {
            usize::from(!self.tables.is_empty())
        };
        let mut columns = HashMap::new();
        for name in keys {
            let values = self
                .group_keys
                .as_ref()
                .into_iter()
                .flatten()
                .map(|k| {
                    k.get(&name)
                        .cloned()
                        .ok_or_else(|| invalid("Missing group key"))
                })
                .collect::<GraphResult<Vec<_>>>()?;
            columns.insert(name, BaseArray::from_attr_values(values));
        }
        for (name, operation) in specs {
            let values = if self.group_keys.is_some() {
                self.tables
                    .iter()
                    .map(|table| {
                        operation.apply(
                            table
                                .column(&name)
                                .into_iter()
                                .flat_map(|c| c.data().iter()),
                        )
                    })
                    .collect::<GraphResult<Vec<_>>>()?
            } else if self.tables.is_empty() {
                Vec::new()
            } else {
                vec![operation.apply(self.tables.iter().flat_map(|table| {
                    table
                        .column(&name)
                        .into_iter()
                        .flat_map(|c| c.data().iter())
                }))?]
            };
            columns.insert(name, BaseArray::from_attr_values(values));
        }
        self.output(columns, order, rows)
    }
    fn default_reduction(&self, operation: Reduction) -> GraphResult<BaseTable> {
        let keys = self.key_names()?;
        let names = self.column_names_checked()?;
        let mut specs = Vec::new();
        for name in names {
            if keys.contains(&name) {
                continue;
            }
            let cells = self
                .tables
                .iter()
                .flat_map(|t| t.column(&name).into_iter().flat_map(|c| c.data().iter()))
                .filter(|v| !matches!(v, V::Null));
            let eligible = if self.tables.is_empty() {
                match operation {
                    Reduction::Count | Reduction::Nunique => true,
                    Reduction::All => self
                        .context
                        .as_ref()
                        .is_some_and(|c| c.boolean.contains(&name)),
                    _ => self
                        .context
                        .as_ref()
                        .is_some_and(|c| c.numeric.contains(&name)),
                }
            } else {
                match operation {
                    Reduction::Count | Reduction::Nunique => true,
                    Reduction::All => cells.clone().any(|v| matches!(v, V::Bool(_))),
                    _ => {
                        (cells.clone().next().is_none()
                            && self
                                .context
                                .as_ref()
                                .filter(|c| c.prototype.has_column(&name))
                                .map(|c| c.numeric.contains(&name))
                                .unwrap_or(true))
                            || cells.clone().any(|v| {
                                crate::storage::numeric::scalar::attribute_number(v).is_some()
                            })
                    }
                }
            };
            if eligible {
                specs.push((name, operation));
            }
        }
        self.aggregate_specs(specs)
    }
    /// Sum numeric columns, excluding group keys; ungrouped collections reduce all rows.
    pub fn sum(&self) -> GraphResult<BaseTable> {
        self.default_reduction(Reduction::Sum)
    }
    /// Mean of all non-null numeric observations; ungrouped means are weighted by row count.
    pub fn mean(&self) -> GraphResult<BaseTable> {
        self.default_reduction(Reduction::Mean)
    }
    /// Count non-null cells in every non-key column.
    pub fn count(&self) -> GraphResult<BaseTable> {
        self.default_reduction(Reduction::Count)
    }
    /// Minimum original numeric scalar in each non-key numeric column.
    pub fn min(&self) -> GraphResult<BaseTable> {
        self.default_reduction(Reduction::Min)
    }
    /// Maximum original numeric scalar in each non-key numeric column.
    pub fn max(&self) -> GraphResult<BaseTable> {
        self.default_reduction(Reduction::Max)
    }
    /// Logical all over non-null boolean cells, validating every observation.
    pub fn all(&self) -> GraphResult<BaseTable> {
        self.default_reduction(Reduction::All)
    }
    /// Explicit column reductions; keys first (sorted), then sorted aggregation columns.
    /// Missing columns in some tables are null observations; columns absent everywhere reject.
    pub fn agg(&self, functions: HashMap<String, String>) -> GraphResult<BaseTable> {
        let mut specs = functions
            .into_iter()
            .map(|(n, op)| Reduction::parse(&op).map(|op| (n, op)))
            .collect::<GraphResult<Vec<_>>>()?;
        specs.sort_by(|a, b| a.0.cmp(&b.0));
        self.aggregate_specs(specs)
    }
    /// Flatten rows with a union of column schemas in first-occurrence order and null padding.
    /// All source captures are retained without automatic row writeback identity.
    pub fn concat(&self) -> GraphResult<BaseTable> {
        self.key_names()?;
        let names = self.column_names_checked()?;
        let rows = self.tables.iter().try_fold(0usize, |n, t| {
            n.checked_add(t.nrows())
                .ok_or_else(|| invalid("Collection row count overflow"))
        })?;
        let mut columns = HashMap::new();
        for name in &names {
            let mut values = Vec::new();
            values
                .try_reserve_exact(rows)
                .map_err(|_| invalid("Collection flatten allocation exceeds capacity"))?;
            for table in &self.tables {
                if let Some(c) = table.column(name) {
                    values.extend(c.data().iter().cloned());
                } else {
                    values.resize(values.len() + table.nrows(), V::Null);
                }
            }
            columns.insert(name.clone(), BaseArray::from_attr_values(values));
        }
        self.output(columns, names, rows)
    }
}
