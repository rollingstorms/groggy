//! Exact portable-value deduplication. Distinct and unmapped values remain retained.
use super::*;
use crate::storage::retained_group;
use std::hash::Hasher;
use std::io::Write;

pub(crate) type Columns = HashMap<AttrName, Vec<AttrValue>>;
pub(crate) type Offsets = HashMap<AttrName, Vec<usize>>;

pub(crate) struct CompactedColumns {
    pub(crate) columns: Columns,
    pub(crate) offsets: Offsets,
    pub(crate) previous: usize,
    pub(crate) current: usize,
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
pub(crate) fn compact(columns: Columns) -> GraphResult<CompactedColumns> {
    let mut result = CompactedColumns {
        columns: HashMap::new(),
        offsets: HashMap::new(),
        previous: 0,
        current: 0,
    };
    for (name, values) in columns {
        result.previous += values.len();
        let mut unique = Vec::new();
        let mut offsets = Vec::with_capacity(values.len());
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        for value in values {
            let encoded = retained_group::encode(&value)?;
            let mut writer = HashWriter(Default::default());
            serde_json::to_writer(&mut writer, &encoded).map_err(|e| {
                GraphError::InvalidInput(format!("Cannot compact column {name:?}: {e}"))
            })?;
            let bucket = buckets.entry(writer.0.finish()).or_default();
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
            offsets.push(index);
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
