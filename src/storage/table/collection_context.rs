//! Retained empty schema and frozen source ownership for collection transformations.
use super::{AttrValue as V, BaseTable, GraphResult, HashMap, Table, TableArray};
use crate::{storage::array::BaseArray, GraphError};
use std::{collections::HashSet, sync::Arc};
#[derive(Debug, Clone)]
pub(super) struct CollectionContext {
    pub prototype: BaseTable,
    pub group_names: Vec<String>,
    pub numeric: HashSet<String>,
    pub boolean: HashSet<String>,
    pub error: Option<String>,
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}
fn roles<'a>(
    tables: impl Iterator<Item = &'a BaseTable> + Clone,
    names: &[String],
) -> (HashSet<String>, HashSet<String>) {
    let mut numeric = HashSet::new();
    let mut boolean = HashSet::new();
    for name in names {
        let mut nonnull = false;
        let mut number = false;
        let mut flag = false;
        for value in tables
            .clone()
            .flat_map(|t| t.column(name).into_iter().flat_map(|c| c.data().iter()))
        {
            if matches!(value, V::Null) {
                continue;
            }
            nonnull = true;
            number |= crate::storage::numeric::scalar::attribute_number(value).is_some();
            flag |= matches!(value, V::Bool(_));
        }
        if number || !nonnull {
            numeric.insert(name.clone());
        }
        if flag {
            boolean.insert(name.clone());
        }
    }
    (numeric, boolean)
}
impl TableArray {
    pub(super) fn output(
        &self,
        columns: HashMap<String, BaseArray<V>>,
        order: Vec<String>,
        rows: usize,
    ) -> GraphResult<BaseTable> {
        BaseTable::collection_result(
            columns,
            order,
            rows,
            self.context
                .iter()
                .map(|c| &c.prototype)
                .chain(self.tables.iter()),
        )
    }
    pub(super) fn capture_context(&self) -> Arc<CollectionContext> {
        let mut error = self
            .context
            .as_ref()
            .and_then(|c| c.error.clone())
            .or_else(|| self.key_names().err().map(|e| e.to_string()))
            .or_else(|| self.column_names_checked().err().map(|e| e.to_string()));
        let names = self.column_names_checked().unwrap_or_default();
        let group_names = self.key_names().unwrap_or_default();
        let columns = names
            .iter()
            .map(|n| (n.clone(), BaseArray::from_attr_values(Vec::new())))
            .collect();
        let prototype = self
            .output(columns, names.clone(), 0)
            .unwrap_or_else(|failure| {
                error.get_or_insert_with(|| failure.to_string());
                BaseTable::new()
            });
        let (mut numeric, mut boolean) = if self.tables.is_empty() {
            self.context
                .as_ref()
                .map(|c| (c.numeric.clone(), c.boolean.clone()))
                .unwrap_or_default()
        } else {
            roles(self.tables.iter(), &names)
        };
        if !self.tables.is_empty() {
            if let Some(previous) = &self.context {
                for name in &names {
                    if previous.prototype.has_column(name)
                        && !self.tables.iter().any(|t| {
                            t.column(name)
                                .is_some_and(|c| c.iter().any(|v| !matches!(v, V::Null)))
                        })
                    {
                        if previous.numeric.contains(name) {
                            numeric.insert(name.clone());
                        } else {
                            numeric.remove(name);
                        }
                        if previous.boolean.contains(name) {
                            boolean.insert(name.clone());
                        } else {
                            boolean.remove(name);
                        }
                    }
                }
            }
        }
        Arc::new(CollectionContext {
            prototype,
            group_names,
            numeric,
            boolean,
            error,
        })
    }
    pub(super) fn empty_schema(&self) -> BaseTable {
        self.capture_context().prototype.clone()
    }
    pub(super) fn inherit(
        &self,
        tables: Vec<BaseTable>,
        group_keys: Option<Vec<HashMap<String, V>>>,
    ) -> Self {
        Self {
            tables,
            group_keys,
            context: Some(self.capture_context()),
        }
    }
    pub(in crate::storage::table) fn remember_origin(
        mut self,
        source: &BaseTable,
        keys: &[&str],
    ) -> Self {
        let mut group_names = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>();
        group_names.sort();
        let (numeric, boolean) = roles(std::iter::once(source), source.column_names());
        self.context = Some(Arc::new(CollectionContext {
            prototype: source.head(0),
            group_names,
            numeric,
            boolean,
            error: None,
        }));
        self
    }
    /// Retain an empty source schema/capture for adapters whose grouped result has no tables.
    /// Supplying property rows is rejected; this is metadata, not another group.
    pub fn with_empty_source(mut self, source: &BaseTable) -> GraphResult<Self> {
        if source.nrows() != 0 {
            return Err(invalid("Collection source schema must have zero rows"));
        }
        let mut context = (*self.capture_context()).clone();
        if self.tables.is_empty() {
            for key in &context.group_names {
                if !source.has_column(key) {
                    return Err(invalid("Declared group key absent from source schema"));
                }
            }
            context.prototype = source.clone();
            (context.numeric, context.boolean) =
                roles(std::iter::once(source), source.column_names());
        } else {
            let columns = context.prototype.columns().clone();
            let names = context.prototype.column_names().to_vec();
            context.prototype =
                BaseTable::collection_result(columns, names, 0, [&context.prototype, source])?;
        }
        self.context = Some(Arc::new(context));
        Ok(self)
    }
    pub(super) fn declare_keys(mut self, keys: &[String]) -> Self {
        let mut context = (*self.capture_context()).clone();
        context.group_names = keys.to_vec();
        context.group_names.sort();
        self.context = Some(Arc::new(context));
        self
    }
    /// Select table positions while preserving retained schema and source captures.
    pub fn select_tables(&self, indices: &[usize]) -> GraphResult<Self> {
        self.key_names()?;
        if indices.iter().any(|&i| i >= self.tables.len()) {
            return Err(invalid("Collection table index out of bounds"));
        }
        let tables = indices.iter().map(|&i| self.tables[i].clone()).collect();
        let keys = self
            .group_keys
            .as_ref()
            .map(|keys| indices.iter().map(|&i| keys[i].clone()).collect());
        Ok(self.inherit(tables, keys))
    }
    pub(super) fn select_schema(&self, columns: &[String]) -> GraphResult<Self> {
        let mut seen = HashSet::new();
        if columns.iter().any(|n| !seen.insert(n)) {
            return Err(invalid("Repeated collection projection column"));
        }
        let mut context = (*self.capture_context()).clone();
        context.prototype = context.prototype.select(columns)?;
        context.numeric.retain(|n| seen.contains(n));
        context.boolean.retain(|n| seen.contains(n));
        let tables = self
            .tables
            .iter()
            .map(|t| t.select(columns))
            .collect::<GraphResult<Vec<_>>>()?;
        Ok(Self {
            tables,
            group_keys: self.group_keys.clone(),
            context: Some(Arc::new(context)),
        })
    }
    pub(super) fn drop_schema(&self, columns: &[String]) -> GraphResult<Self> {
        let mut context = (*self.capture_context()).clone();
        context.prototype = context.prototype.drop_columns(columns)?;
        context.numeric.retain(|n| !columns.contains(n));
        context.boolean.retain(|n| !columns.contains(n));
        let tables = self
            .tables
            .iter()
            .map(|t| t.drop_columns(columns))
            .collect::<GraphResult<Vec<_>>>()?;
        Ok(Self {
            tables,
            group_keys: self.group_keys.clone(),
            context: Some(Arc::new(context)),
        })
    }
    pub(super) fn rename_schema(&self, mapping: &HashMap<String, String>) -> GraphResult<Self> {
        let mut context = (*self.capture_context()).clone();
        let rename = |name: &String| mapping.get(name).cloned().unwrap_or_else(|| name.clone());
        let names = context
            .prototype
            .column_names()
            .iter()
            .map(rename)
            .collect::<Vec<_>>();
        let keys = context.group_names.iter().map(rename).collect::<Vec<_>>();
        if names.iter().collect::<HashSet<_>>().len() != names.len()
            || keys.iter().collect::<HashSet<_>>().len() != keys.len()
        {
            return Err(invalid(
                "Collection rename creates colliding column/group key names",
            ));
        }
        // Renaming an external metadata key onto a physical non-key column is also ambiguous.
        for (old, new) in context.group_names.iter().zip(&keys) {
            if old != new && context.prototype.has_column(new) && !context.group_names.contains(new)
            {
                return Err(invalid(
                    "Collection rename collides with a physical non-key column",
                ));
            }
        }
        for name in context.prototype.column_names() {
            if !context.group_names.contains(name) && keys.contains(&rename(name)) {
                return Err(invalid(
                    "Collection rename collides with an external group key",
                ));
            }
        }
        context.prototype = context.prototype.rename(mapping.clone())?;
        context.group_names = keys;
        context.group_names.sort();
        context.numeric = context.numeric.iter().map(rename).collect();
        context.boolean = context.boolean.iter().map(rename).collect();
        let tables = self
            .tables
            .iter()
            .map(|t| t.rename(mapping.clone()))
            .collect::<GraphResult<Vec<_>>>()?;
        let group_keys = self.group_keys.as_ref().map(|keys| {
            keys.iter()
                .map(|key| key.iter().map(|(n, v)| (rename(n), v.clone())).collect())
                .collect()
        });
        Ok(Self {
            tables,
            group_keys,
            context: Some(Arc::new(context)),
        })
    }
    pub(super) fn joined_collection(
        &self,
        other: &Self,
        on: &str,
        how: &str,
        tables: Vec<BaseTable>,
    ) -> GraphResult<Self> {
        let left = self.capture_context();
        let right = other.capture_context();
        let prototype = left
            .prototype
            .join_on(&right.prototype, &[on], &[on], how)?;
        let (a, b, plan_how) = if how == "right" {
            (&right, &left, "left")
        } else {
            (&left, &right, how)
        };
        let mut numeric = HashSet::new();
        let mut boolean = HashSet::new();
        let plan = a
            .prototype
            .join_plan(&b.prototype, &[on], &[on], plan_how)?;
        for &(r, name, ref output, coalesce) in &plan {
            let source = if r { b } else { a };
            if source.numeric.contains(name) || (coalesce && b.numeric.contains(name)) {
                numeric.insert(output.clone());
            }
            if source.boolean.contains(name) || (coalesce && b.boolean.contains(name)) {
                boolean.insert(output.clone());
            }
        }
        let both = self.group_keys.is_some() && other.group_keys.is_some();
        let key_name = |right_side: bool, name: &String| {
            let plan_side = if how == "right" {
                !right_side
            } else {
                right_side
            };
            plan.iter()
                .find(|(r, n, _, _)| *r == plan_side && *n == name)
                .map(|(_, _, out, _)| out.clone())
                .unwrap_or_else(|| {
                    if both && right_side {
                        format!("right_{name}")
                    } else {
                        name.clone()
                    }
                })
        };
        let mut group_names = Vec::new();
        for (right_side, context, grouped) in [
            (false, &left, self.group_keys.is_some()),
            (true, &right, other.group_keys.is_some()),
        ] {
            if grouped {
                for name in &context.group_names {
                    let output = key_name(right_side, name);
                    if group_names.contains(&output) {
                        return Err(invalid("Collection join group key names collide"));
                    }
                    group_names.push(output);
                }
            }
        }
        group_names.sort();
        let group_keys = if self.group_keys.is_some() || other.group_keys.is_some() {
            let mut groups = Vec::new();
            for i in 0..tables.len() {
                let mut key = HashMap::new();
                for (right_side, keys) in [(false, &self.group_keys), (true, &other.group_keys)] {
                    if let Some(keys) = keys {
                        for (name, value) in &keys[i] {
                            key.insert(key_name(right_side, name), value.clone());
                        }
                    }
                }
                groups.push(key);
            }
            Some(groups)
        } else {
            None
        };
        Ok(Self {
            tables,
            group_keys,
            context: Some(Arc::new(CollectionContext {
                prototype,
                group_names,
                numeric,
                boolean,
                error: None,
            })),
        })
    }
}
