//! Explicit non-repairing file verification with detached, locked ownership.
use super::checkpoint::{check_path, same_inode};
use super::*;

/// Verified physical journal and full retained graph counts. This reads bytes
/// rather than assuming the in-memory owner still matches an intact file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageIntegrityReport {
    /// Complete verified file size; no incomplete trailing bytes are accepted.
    pub file_bytes: u64,
    /// Logical journal records, independent of large-baseline fragment count.
    pub logical_records: u64,
    /// Verified retained graph/current schema, reconstructed from the file.
    pub graph: crate::IntegrityReport,
}
/// Detached verification input. Holds the original inode lock and a separate
/// read-only file cursor. Always finish the outcome on the original owner.
pub struct StorageVerificationPlan {
    source: File,
    reader: File,
    path: PathBuf,
    length: u64,
    state: StateId,
    sequence: u64,
    schema_revision: u64,
}
/// Verification result retaining the source lock until consumed/dropped.
pub struct StorageVerificationOutcome {
    source: File,
    path: PathBuf,
    state: StateId,
    sequence: u64,
    result: GraphResult<StorageIntegrityReport>,
}
impl DurableGraph {
    /// Verify current on-disk framing/checksums, replay and all retained versions,
    /// without repairing/truncating bytes. Errors require reopen before writes.
    pub fn verify_storage(&mut self) -> GraphResult<StorageIntegrityReport> {
        let plan = self.prepare_storage_verification()?;
        self.finish_storage_verification(plan.execute())
    }
    /// Detach explicit full-file verification. Owner operations remain blocked
    /// until the outcome is finished; abandoning it requires reopening the owner.
    #[doc(hidden)]
    pub fn prepare_storage_verification(&mut self) -> GraphResult<StorageVerificationPlan> {
        self.ready()?;
        if let Err(error) = check_path(&self.file, &self.path) {
            self.poisoned = true;
            return Err(error);
        }
        let reader = File::open(&self.path)
            .map_err(|e| io_error(&self.path, "open_verification_reader", e))?;
        if !same_inode(&self.file, &reader)? {
            self.poisoned = true;
            return Err(invalid("journal path changed before verification"));
        }
        let source = self
            .file
            .try_clone()
            .map_err(|e| io_error(&self.path, "clone_verification_lock", e))?;
        let length = source
            .metadata()
            .map_err(|e| io_error(&self.path, "stat_verification_source", e))?
            .len();
        let plan = StorageVerificationPlan {
            source,
            reader,
            path: self.path.clone(),
            length,
            state: self.graph.current_state_id(),
            sequence: self.next_sequence,
            schema_revision: self.schema_revision,
        };
        self.poisoned = true;
        Ok(plan)
    }
    /// Reenable the original owner only after successful complete verification.
    /// A corrupt/incomplete/changed file leaves it blocked until explicit reopen.
    #[doc(hidden)]
    pub fn finish_storage_verification(
        &mut self,
        outcome: StorageVerificationOutcome,
    ) -> GraphResult<StorageIntegrityReport> {
        if outcome.path != self.path
            || outcome.state != self.graph.current_state_id()
            || outcome.sequence != self.next_sequence
            || !same_inode(&self.file, &outcome.source)?
        {
            self.poisoned = true;
            return Err(invalid(
                "storage verification outcome belongs to a different writer",
            ));
        }
        self.poisoned = outcome.result.is_err();
        outcome.result
    }
}
impl StorageVerificationPlan {
    /// Read and fully validate the source on a worker, without writing bytes.
    /// Reconstruction/retained verification owns memory proportional to the graph.
    pub fn execute(mut self) -> StorageVerificationOutcome {
        let result = self.verify();
        StorageVerificationOutcome {
            source: self.source,
            path: self.path,
            state: self.state,
            sequence: self.sequence,
            result,
        }
    }
    fn verify(&mut self) -> GraphResult<StorageIntegrityReport> {
        check_path(&self.source, &self.path)?;
        if !same_inode(&self.source, &self.reader)?
            || self.reader.metadata().map_err(invalid)?.len() != self.length
        {
            return Err(invalid("verification source changed before reading"));
        }
        let decoded = read_journal(&mut self.reader, &self.path)?;
        if decoded.position != decoded.length {
            return Err(invalid(format!(
                "journal has {} incomplete trailing bytes; verification does not repair them",
                decoded.length - decoded.position
            )));
        }
        if decoded.length != self.length
            || decoded.sequence != self.sequence
            || decoded.graph.current_state_id() != self.state
            || decoded.schema_revision != self.schema_revision
        {
            return Err(invalid(
                "verified journal metadata differs from the captured owner",
            ));
        }
        if let Some(schema) = &decoded.schema {
            decoded.graph.check_schema_policy(schema)?;
        }
        let mut graph = decoded.graph.check_integrity()?;
        graph.schema_revision = Some(decoded.schema_revision);
        graph.current_schema_checked = decoded.schema.is_some();
        check_path(&self.source, &self.path)?;
        if self.source.metadata().map_err(invalid)?.len() != self.length {
            return Err(invalid("verification source changed while reading"));
        }
        Ok(StorageIntegrityReport {
            file_bytes: decoded.length,
            logical_records: decoded.sequence,
            graph,
        })
    }
}
