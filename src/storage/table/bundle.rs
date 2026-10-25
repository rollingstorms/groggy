//! GraphTable-owned bundle codec. One atomic authoritative file; v2 read compatibility.
use super::{BundleMetadata, EnhancedBundleMetadata, GraphTable};
use crate::{
    storage::{
        array::BaseArray,
        table::{BaseTable, EdgesTable, NodesTable, Table},
        value_codec,
    },
    GraphError, GraphResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Cursor, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"GROGTB03";
const FILE: &str = "bundle.cbor";
const HEADER: usize = 24;
fn invalid(error: impl std::fmt::Display) -> GraphError {
    GraphError::InvalidInput(format!("Invalid graph-table bundle: {error}"))
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Column {
    name: String,
    values: Vec<Value>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredTable {
    rows: usize,
    columns: Vec<Column>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    version: u32,
    created_at: String,
    groggy_version: String,
    validation_policy: super::ValidationPolicy,
    nodes: StoredTable,
    edges: StoredTable,
}
impl StoredTable {
    fn encode(table: &BaseTable) -> GraphResult<Self> {
        let unique: HashSet<_> = table.column_names().iter().collect();
        if unique.len() != table.column_names().len() || unique.len() != table.columns().len() {
            return Err(invalid("column order must name every column exactly once"));
        }
        let mut columns = Vec::new();
        for name in table.column_names() {
            let column = crate::storage::table::structural::column(table, name)?;
            let values = column
                .iter()
                .enumerate()
                .map(|(row, value)| {
                    value_codec::encode(value)
                        .map_err(|error| invalid(format!("{name} row {row}: {error}")))
                })
                .collect::<GraphResult<_>>()?;
            columns.push(Column {
                name: name.clone(),
                values,
            });
        }
        Ok(Self {
            rows: table.nrows(),
            columns,
        })
    }
    fn decode(self) -> GraphResult<BaseTable> {
        let mut seen = HashSet::new();
        let mut columns = HashMap::new();
        let mut order = Vec::new();
        for column in self.columns {
            if !seen.insert(column.name.clone()) || column.values.len() != self.rows {
                return Err(invalid("duplicate column or inconsistent row count"));
            }
            let values = column
                .values
                .into_iter()
                .map(value_codec::decode)
                .collect::<GraphResult<Vec<_>>>()?;
            order.push(column.name.clone());
            columns.insert(column.name, BaseArray::from_attr_values(values));
        }
        // Graph-table components must have structural columns even when empty.
        let table = BaseTable::with_column_order(columns, order)?;
        if table.nrows() != self.rows {
            return Err(invalid("stored row count does not match columns"));
        }
        Ok(table)
    }
}

fn require_columns(nodes: &BaseTable, edges: &BaseTable) -> GraphResult<()> {
    if !nodes.has_column("node_id")
        || ["edge_id", "source", "target"]
            .iter()
            .any(|name| !edges.has_column(name))
    {
        return Err(invalid("missing canonical structural columns"));
    }
    Ok(())
}

pub(super) fn save(table: &GraphTable, path: &Path) -> GraphResult<()> {
    // Fully encode first: unsupported pooled refs must not disturb an existing save.
    require_columns(table.nodes.base_table(), table.edges.base_table())?;
    let payload = Payload {
        version: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        groggy_version: env!("CARGO_PKG_VERSION").into(),
        validation_policy: table.policy.clone(),
        nodes: StoredTable::encode(table.nodes.base_table())?,
        edges: StoredTable::encode(table.edges.base_table())?,
    };
    let mut bytes = Vec::new();
    ciborium::into_writer(&payload, &mut bytes).map_err(invalid)?;
    let length = u64::try_from(bytes.len()).map_err(invalid)?;
    let checksum = xxhash_rust::xxh3::xxh3_64(&bytes);
    std::fs::create_dir_all(path).map_err(invalid)?;
    let mut temporary = tempfile::NamedTempFile::new_in(path).map_err(invalid)?;
    temporary
        .write_all(MAGIC)
        .and_then(|_| temporary.write_all(&length.to_le_bytes()))
        .and_then(|_| temporary.write_all(&checksum.to_le_bytes()))
        .and_then(|_| temporary.write_all(&bytes))
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(invalid)?;
    temporary
        .persist(path.join(FILE))
        .map_err(|error| invalid(error.error))?;
    // Failure after rename can leave a complete published file; callers can verify/reopen.
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(invalid)?;
    Ok(())
}

fn current(path: &Path) -> GraphResult<(GraphTable, BundleMetadata, Value)> {
    let bytes = std::fs::read(path.join(FILE)).map_err(invalid)?;
    if bytes.len() < HEADER || &bytes[..8] != MAGIC {
        return Err(invalid("unsupported header/version"));
    }
    let number = |start| -> GraphResult<u64> {
        Ok(u64::from_le_bytes(
            bytes[start..start + 8].try_into().map_err(invalid)?,
        ))
    };
    if number(8)? != (bytes.len() - HEADER) as u64 {
        return Err(invalid("payload length mismatch or trailing data"));
    }
    let checksum = xxhash_rust::xxh3::xxh3_64(&bytes[HEADER..]);
    if checksum != number(16)? {
        return Err(invalid("checksum mismatch"));
    }
    let mut cursor = Cursor::new(&bytes[HEADER..]);
    let payload: Payload = ciborium::from_reader(&mut cursor).map_err(invalid)?;
    if cursor.position() != (bytes.len() - HEADER) as u64 || payload.version != 3 {
        return Err(invalid("unsupported payload version or trailing data"));
    }
    let nodes = payload.nodes.decode()?;
    let edges = payload.edges.decode()?;
    require_columns(&nodes, &edges)?;
    let nodes = NodesTable::from_base_table(nodes)?;
    let edges = EdgesTable::from_base_table(edges)?;
    let table = GraphTable::with_policy(nodes, edges, payload.validation_policy.clone());
    let checksums = HashMap::from([("xxh3_64".into(), format!("{checksum:016x}"))]);
    let metadata = BundleMetadata {
        version: "3.0".into(),
        created_at: payload.created_at,
        node_count: table.nodes.nrows(),
        edge_count: table.edges.nrows(),
        validation_policy: payload.validation_policy,
        checksums,
    };
    let report = table.validate();
    let value = json!({
        "version": metadata.version, "created_at": metadata.created_at,
        "groggy_version": payload.groggy_version,
        "node_count": metadata.node_count, "edge_count": metadata.edge_count,
        "validation_policy": metadata.validation_policy, "checksums": metadata.checksums,
        "schema_info": {"node_columns": table.nodes.column_names(), "edge_columns": table.edges.column_names(),
            "has_node_attributes": table.nodes.ncols() > 1, "has_edge_attributes": table.edges.ncols() > 3},
        "validation_summary": {"is_valid":report.is_valid(), "error_count":report.errors.len(),
            "warning_count":report.warnings.len(), "info_count":report.info.len()},
        "storage_file": FILE, "checksum_algorithm":"xxh3_64",
    });
    Ok((table, metadata, value))
}
fn legacy_checksum(data: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut hash);
    format!("{:x}", hash.finish())
}
fn legacy(path: &Path) -> GraphResult<(GraphTable, BundleMetadata, Value)> {
    let text = std::fs::read_to_string(path.join("metadata.json")).map_err(invalid)?;
    let mut stored: EnhancedBundleMetadata = serde_json::from_str(&text).map_err(invalid)?;
    if stored.version != "2.0" {
        return Err(invalid(format!(
            "unsupported legacy version {}",
            stored.version
        )));
    }
    let recorded = std::mem::take(&mut stored.checksums.metadata_sha256);
    let normalized = serde_json::to_string_pretty(&stored).map_err(invalid)?;
    if legacy_checksum(&normalized) != recorded {
        return Err(invalid("legacy metadata checksum mismatch"));
    }
    stored.checksums.metadata_sha256 = recorded;
    let nodes = std::fs::read_to_string(path.join("nodes.csv")).map_err(invalid)?;
    let edges = std::fs::read_to_string(path.join("edges.csv")).map_err(invalid)?;
    if legacy_checksum(&nodes) != stored.checksums.nodes_sha256
        || legacy_checksum(&edges) != stored.checksums.edges_sha256
    {
        return Err(invalid("legacy data checksum mismatch"));
    }
    let manifest: super::BundleManifest =
        serde_json::from_slice(&std::fs::read(path.join("MANIFEST.json")).map_err(invalid)?)
            .map_err(invalid)?;
    let expected = HashMap::from([
        ("metadata.json", stored.checksums.metadata_sha256.as_str()),
        ("nodes.csv", stored.checksums.nodes_sha256.as_str()),
        ("edges.csv", stored.checksums.edges_sha256.as_str()),
    ]);
    let mut seen = HashSet::new();
    if manifest.format_version != "2.0"
        || manifest.created_at != stored.created_at
        || manifest.files.len() != expected.len()
        || manifest.files.iter().any(|file| {
            !seen.insert(file.name.as_str())
                || expected.get(file.name.as_str()).copied() != Some(file.checksum.as_str())
        })
    {
        return Err(invalid("legacy manifest mismatch"));
    }
    let table = GraphTable::with_policy(
        GraphTable::parse_nodes_from_csv(&nodes)?,
        GraphTable::parse_edges_from_csv(&edges)?,
        stored.validation_policy.clone(),
    );
    if table.nodes.nrows() != stored.node_count
        || table.edges.nrows() != stored.edge_count
        || table.nodes.column_names() != stored.schema_info.node_columns
        || table.edges.column_names() != stored.schema_info.edge_columns
    {
        return Err(invalid("legacy schema or row count mismatch"));
    }
    let metadata = BundleMetadata {
        version: stored.version.clone(),
        created_at: stored.created_at.clone(),
        node_count: stored.node_count,
        edge_count: stored.edge_count,
        validation_policy: stored.validation_policy.clone(),
        // These historical field names do not imply SHA-256: v2 used DefaultHasher.
        checksums: HashMap::from([
            ("nodes_sha256".into(), stored.checksums.nodes_sha256.clone()),
            ("edges_sha256".into(), stored.checksums.edges_sha256.clone()),
            (
                "metadata_sha256".into(),
                stored.checksums.metadata_sha256.clone(),
            ),
        ]),
    };
    let mut value = serde_json::to_value(stored).map_err(invalid)?;
    value["checksum_algorithm"] = json!("legacy_default_hasher");
    Ok((table, metadata, value))
}
fn read(path: &Path) -> GraphResult<(GraphTable, BundleMetadata, Value)> {
    if !path.is_dir() {
        return Err(invalid(format!(
            "not a bundle directory: {}",
            path.display()
        )));
    }
    if path.join(FILE).try_exists().map_err(invalid)? {
        current(path)
    } else {
        legacy(path)
    }
}
pub(super) fn load(path: &Path) -> GraphResult<(GraphTable, BundleMetadata)> {
    read(path).map(|(table, metadata, _)| (table, metadata))
}
pub(super) fn metadata_value(path: &Path) -> GraphResult<Value> {
    read(path).map(|(_, _, value)| value)
}
