//! Exact portable-value compaction with caller-supplied optional reclamation roots.
use super::*;
use crate::storage::retained_group;
use std::hash::Hasher;
use std::io::Write;

pub(crate) type Columns = HashMap<AttrName, Vec<AttrValue>>;
pub(crate) type Offsets = HashMap<AttrName, Vec<Option<usize>>>;

pub(crate) struct CompactedColumns {
    pub(crate) columns: Columns,
    pub(crate) offsets: Offsets,
    pub(crate) previous: usize,
    pub(crate) current: usize,
    pub(crate) reclaimed: usize,
}
/// Roots supplied by the graph coordinator. Pool storage does not interpret history.
pub(crate) struct RetainedRoots<'a> {
    pub(crate) mappings: &'a HashMap<AttrName, HashMap<usize, usize>>,
    pub(crate) values: Vec<(&'a AttrName, &'a AttrValue)>,
}
impl CompactedColumns {
    pub(crate) fn offset(&self, name: &AttrName, index: usize) -> GraphResult<usize> {
        self.offsets
            .get(name)
            .and_then(|v| v.get(index))
            .copied()
            .flatten()
            .ok_or_else(|| {
                GraphError::InvalidInput(format!(
                    "Missing retained compaction offset in {name:?} at {index}"
                ))
            })
    }
}
struct HashWriter(std::collections::hash_map::DefaultHasher);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.write(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn fingerprint(encoded: &serde_json::Value) -> GraphResult<u64> {
    let mut writer = HashWriter(Default::default());
    serde_json::to_writer(&mut writer, encoded)
        .map_err(|e| GraphError::InvalidInput(format!("Cannot fingerprint property: {e}")))?;
    Ok(writer.0.finish())
}
pub(crate) fn compact(
    columns: Columns,
    roots: Option<RetainedRoots<'_>>,
) -> GraphResult<CompactedColumns> {
    let mut root_values: HashMap<AttrName, HashMap<u64, Vec<&AttrValue>>> = HashMap::new();
    if let Some(roots) = &roots {
        for (name, value) in &roots.values {
            let encoded = retained_group::encode(value)?;
            let candidates = root_values
                .entry((*name).clone())
                .or_default()
                .entry(fingerprint(&encoded)?)
                .or_default();
            let mut present = false;
            for candidate in candidates.iter() {
                if retained_group::encode(candidate)? == encoded {
                    present = true;
                    break;
                }
            }
            if !present {
                candidates.push(value);
            }
        }
    }

    let mut result = CompactedColumns {
        columns: HashMap::new(),
        offsets: HashMap::new(),
        previous: 0,
        current: 0,
        reclaimed: 0,
    };
    for (name, values) in columns {
        result.previous += values.len();
        let mut unique = Vec::new();
        let mut offsets = Vec::with_capacity(values.len());
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        let mapped: std::collections::HashSet<_> = roots
            .as_ref()
            .and_then(|r| r.mappings.get(&name))
            .into_iter()
            .flat_map(|rows| rows.values().copied())
            .collect();
        for (old_index, value) in values.into_iter().enumerate() {
            let encoded = retained_group::encode(&value).map_err(|error| {
                GraphError::InvalidInput(format!(
                    "Cannot compact column {name:?} value {old_index}: {error}"
                ))
            })?;
            let hash = fingerprint(&encoded)?;
            if roots.is_some() && !mapped.contains(&old_index) {
                let mut retained = false;
                if let Some(candidates) =
                    root_values.get(&name).and_then(|values| values.get(&hash))
                {
                    for candidate in candidates {
                        if retained_group::encode(candidate)? == encoded {
                            retained = true;
                            break;
                        }
                    }
                }
                if !retained {
                    offsets.push(None);
                    result.reclaimed += 1;
                    continue;
                }
            }
            let bucket = buckets.entry(hash).or_default();
            let mut found = None;
            for &index in bucket.iter() {
                // Hash collisions never decide equality. Portable equality retains
                // tags, signed zero, NaN payloads and exact JSON/text spelling.
                if retained_group::encode(&unique[index])? == encoded {
                    found = Some(index);
                    break;
                }
            }
            let index = match found {
                Some(index) => index,
                None => {
                    let index = unique.len();
                    unique.push(value);
                    bucket.push(index);
                    index
                }
            };
            offsets.push(Some(index));
        }
        unique.shrink_to_fit();
        result.current += unique.len();
        result.columns.insert(name.clone(), unique);
        result.offsets.insert(name, offsets);
    }
    Ok(result)
}
impl GraphPool {
    pub(crate) fn compaction_columns(&self, is_node: bool) -> Columns {
        let columns = if is_node {
            &self.node_attributes
        } else {
            &self.edge_attributes
        };
        columns
            .iter()
            .map(|(name, column)| (name.clone(), column.values.clone()))
            .collect()
    }
    pub(crate) fn compaction_lengths(&self, is_node: bool) -> HashMap<AttrName, usize> {
        let columns = if is_node {
            &self.node_attributes
        } else {
            &self.edge_attributes
        };
        columns
            .iter()
            .map(|(name, column)| (name.clone(), column.len()))
            .collect()
    }
    pub(crate) fn install_compacted_columns(&mut self, is_node: bool, columns: Columns) {
        let target = if is_node {
            &mut self.node_attributes
        } else {
            &mut self.edge_attributes
        };
        *target = columns
            .into_iter()
            .map(|(name, values)| {
                let mut column = AttributeColumn::new();
                column.values = values;
                (name, column)
            })
            .collect();
    }
}
