//! Streaming physical framing for large initialization records. Logical sequence zero
//! remains one transaction; no partial baseline is a recoverable graph.
use super::*;
use std::io;

pub(super) const BEGIN: &[u8; 8] = b"GBEGIN01";
const PART: &[u8; 8] = b"GPART001";
const END: &[u8; 8] = b"GEND0001";
const CHUNK_BYTES: usize = 8 * 1024 * 1024;

fn write_frame(writer: &mut impl Write, marker: &[u8; 8], payload: &[u8]) -> io::Result<()> {
    let length = payload.len() as u64;
    writer.write_all(marker)?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(&(!length).to_le_bytes())?;
    writer.write_all(&xxhash_rust::xxh3::xxh3_64(payload).to_le_bytes())?;
    writer.write_all(payload)
}

struct ChunkWriter<'a, W> {
    writer: &'a mut W,
    buffer: Vec<u8>,
    chunk_bytes: usize,
    parts: u64,
    bytes: u64,
    hash: xxhash_rust::xxh3::Xxh3,
}
impl<W: Write> ChunkWriter<'_, W> {
    fn emit(&mut self) -> io::Result<()> {
        let marker = if self.parts == 0 { BEGIN } else { PART };
        let mut payload = Vec::with_capacity(8 + self.buffer.len());
        payload.extend_from_slice(&self.parts.to_le_bytes());
        payload.extend_from_slice(&self.buffer);
        write_frame(self.writer, marker, &payload)?;
        #[cfg(all(test, unix))]
        if self.parts == 0 {
            super::checkpoint::pause_for_process_test("AfterFirstFragment");
        }
        self.parts = self
            .parts
            .checked_add(1)
            .ok_or_else(|| io::Error::other("baseline chunk count exhausted"))?;
        self.buffer.clear();
        Ok(())
    }
    fn finish(mut self) -> io::Result<()> {
        if self.parts == 0 {
            // Small baselines retain the original format and can be read by older builds.
            return write_frame(self.writer, FRAME, &self.buffer);
        }
        if !self.buffer.is_empty() {
            self.emit()?;
        }
        let mut manifest = Vec::with_capacity(24);
        manifest.extend_from_slice(&self.parts.to_le_bytes());
        manifest.extend_from_slice(&self.bytes.to_le_bytes());
        manifest.extend_from_slice(&self.hash.digest().to_le_bytes());
        write_frame(self.writer, END, &manifest)
    }
}
impl<W: Write> Write for ChunkWriter<'_, W> {
    fn write(&mut self, mut bytes: &[u8]) -> io::Result<usize> {
        let count = bytes.len();
        self.bytes = self
            .bytes
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::other("baseline byte count exhausted"))?;
        self.hash.update(bytes);
        while !bytes.is_empty() {
            let take = bytes.len().min(self.chunk_bytes - self.buffer.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buffer.len() == self.chunk_bytes {
                self.emit()?;
            }
        }
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

pub(super) fn write_initial_record(writer: &mut impl Write, record: &Record) -> GraphResult<()> {
    if !matches!(record, Record::InitArchive { .. }) {
        return writer
            .write_all(&encode_record(0, record)?)
            .map_err(invalid);
    }
    write_chunked(writer, record, CHUNK_BYTES)
}
fn write_chunked(writer: &mut impl Write, record: &Record, chunk_bytes: usize) -> GraphResult<()> {
    let mut chunks = ChunkWriter {
        writer,
        buffer: Vec::with_capacity(chunk_bytes),
        chunk_bytes,
        parts: 0,
        bytes: 0,
        hash: xxhash_rust::xxh3::Xxh3::new(),
    };
    #[derive(Serialize)]
    struct EntryRef<'a> {
        sequence: u64,
        record: &'a Record,
    }
    ciborium::ser::into_writer(
        &EntryRef {
            sequence: 0,
            record,
        },
        &mut chunks,
    )
    .map_err(invalid)?;
    chunks.finish().map_err(invalid)
}

/// Validate all physical fragments into a scratch file, then decode directly from
/// it. Serialized baseline bytes do not accumulate in RAM. Archive restoration
/// still owns the decoded retained graph and validates it before publication.
pub(super) fn read_chunked(
    file: &mut File,
    first: [u8; HEADER_SIZE],
    start: u64,
    file_length: u64,
) -> GraphResult<(Entry, u64)> {
    let mut scratch = tempfile::tempfile().map_err(invalid)?;
    let mut header = first;
    let mut position = start;
    let mut parts = 0u64;
    let mut total = 0u64;
    let mut hash = xxhash_rust::xxh3::Xxh3::new();
    loop {
        let number = |offset: usize| {
            let mut bytes = [0; 8];
            bytes.copy_from_slice(&header[offset..offset + 8]);
            u64::from_le_bytes(bytes)
        };
        let count = number(8);
        if number(16) != !count || count > MAX_RECORD as u64 {
            return Err(invalid("invalid baseline fragment length"));
        }
        if count > file_length.saturating_sub(position + HEADER_SIZE as u64) {
            return Err(invalid("incomplete baseline fragment"));
        }
        let is_end = &header[..8] == END;
        let expected = if parts == 0 { BEGIN } else { PART };
        if (!is_end && &header[..8] != expected) || (is_end && parts == 0) {
            return Err(invalid("invalid baseline fragment order"));
        }
        if (is_end && count != 24) || (!is_end && !(9..=(CHUNK_BYTES as u64 + 8)).contains(&count))
        {
            return Err(invalid("invalid baseline fragment size"));
        }
        let mut payload = vec![0; count as usize];
        file.read_exact(&mut payload).map_err(invalid)?;
        if xxhash_rust::xxh3::xxh3_64(&payload) != number(24) {
            return Err(invalid("baseline fragment checksum mismatch"));
        }
        position = position
            .checked_add(HEADER_SIZE as u64 + count)
            .ok_or_else(|| invalid("baseline length overflow"))?;
        let number = |offset: usize| {
            let mut bytes = [0; 8];
            bytes.copy_from_slice(&payload[offset..offset + 8]);
            u64::from_le_bytes(bytes)
        };
        if is_end {
            if number(0) != parts || number(8) != total || number(16) != hash.digest() {
                return Err(invalid("baseline manifest mismatch"));
            }
            break;
        }
        if number(0) != parts {
            return Err(invalid("noncontiguous baseline fragments"));
        }
        scratch.write_all(&payload[8..]).map_err(invalid)?;
        hash.update(&payload[8..]);
        total = total
            .checked_add(count - 8)
            .ok_or_else(|| invalid("baseline length overflow"))?;
        parts = parts
            .checked_add(1)
            .ok_or_else(|| invalid("baseline fragment count overflow"))?;
        if file_length.saturating_sub(position) < HEADER_SIZE as u64 {
            return Err(invalid(
                "journal initialization is incomplete: missing baseline terminator",
            ));
        }
        file.read_exact(&mut header).map_err(invalid)?;
    }
    scratch.seek(SeekFrom::Start(0)).map_err(invalid)?;
    let mut reader = std::io::BufReader::with_capacity(64 * 1024, scratch);
    let entry: Entry = ciborium::de::from_reader(&mut reader).map_err(invalid)?;
    if reader.stream_position().map_err(invalid)? != total {
        return Err(invalid("trailing bytes in baseline"));
    }
    if entry.sequence != 0 || !matches!(entry.record, Record::InitArchive { .. }) {
        return Err(invalid(
            "chunked initialization must contain one retained archive",
        ));
    }
    Ok((entry, position - start))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record() -> Record {
        let mut graph = Graph::new();
        graph.add_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph
            .set_node_attr(
                0,
                "payload".into(),
                crate::AttrValue::Text("αβ\0payload".repeat(20)),
            )
            .unwrap();
        graph.create_branch("saved".into()).unwrap();
        graph.remove_node(1).unwrap();
        Record::InitArchive {
            archive: Box::new(graph.archive_document().unwrap()),
            schema: None,
            schema_revision: 7,
        }
    }
    fn encoded(record: &Record, chunk: usize) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        write_chunked(&mut bytes, record, chunk).unwrap();
        bytes
    }
    fn boundaries(bytes: &[u8]) -> Vec<(usize, usize)> {
        let mut rows = Vec::new();
        let mut pos = MAGIC.len();
        while pos < bytes.len() {
            let count = u64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap()) as usize;
            rows.push((pos, pos + HEADER_SIZE + count));
            pos += HEADER_SIZE + count;
        }
        rows
    }
    fn reject_unchanged(bytes: &[u8]) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad");
        std::fs::write(&path, bytes).unwrap();
        assert!(DurableGraph::open(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
    #[test]
    fn small_initialization_is_byte_compatible_and_chunked_one_replays_then_appends() {
        let record = record();
        let small = encoded(&record, CHUNK_BYTES);
        assert_eq!(&small[8..], encode_record(0, &record).unwrap());
        let bytes = encoded(&record, 97);
        assert_eq!(&bytes[8..16], BEGIN);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal");
        std::fs::write(&path, &bytes).unwrap();
        let mut store = DurableGraph::open(&path).unwrap();
        assert_eq!(store.schema_revision(), 7);
        assert_eq!(store.next_sequence, 1);
        assert_eq!(store.graph().unwrap().node_count(), 1);
        assert_eq!(store.graph().unwrap().pool().statistics().node_count, 2);
        assert_eq!(store.graph().unwrap().list_branches().len(), 2);
        let state = store.graph().unwrap().current_state_id();
        let added = store
            .apply_batch(
                GraphBatch {
                    add_nodes: 1,
                    ..Default::default()
                },
                state,
                "next".into(),
                "test".into(),
            )
            .unwrap();
        assert_eq!(added.added_nodes, vec![2]);
        drop(store);
        let recovered = DurableGraph::open(&path).unwrap();
        assert_eq!(recovered.next_sequence, 2);
        assert_eq!(recovered.graph().unwrap().node_count(), 2);
    }
    #[test]
    fn every_truncated_fragment_or_missing_terminator_rejects_without_rewriting() {
        let bytes = encoded(&record(), 97);
        for (start, end) in boundaries(&bytes) {
            for cut in [
                start,
                start + 1,
                start + 8,
                start + HEADER_SIZE - 1,
                start + HEADER_SIZE,
                end - 1,
            ] {
                if cut < bytes.len() {
                    reject_unchanged(&bytes[..cut]);
                }
            }
        }
    }
    #[test]
    fn corrupted_reordered_duplicated_and_spliced_chunks_reject_without_rewriting() {
        let bytes = encoded(&record(), 97);
        let frames = boundaries(&bytes);
        for (start, end) in &frames {
            for offset in [start + 8, start + 16, start + 24, end - 1] {
                let mut bad = bytes.clone();
                bad[offset] ^= 1;
                reject_unchanged(&bad);
            }
        }
        let mut removed = bytes.clone();
        removed.drain(frames[1].0..frames[1].1);
        reject_unchanged(&removed);
        let mut duplicate = bytes.clone();
        duplicate.splice(
            frames[1].0..frames[1].0,
            bytes[frames[1].0..frames[1].1].iter().copied(),
        );
        reject_unchanged(&duplicate);
        let mut reordered = bytes.clone();
        let (a, b) = frames[1];
        let (c, d) = frames[2];
        reordered[a..b].copy_from_slice(&bytes[c..d]);
        reordered[c..d].copy_from_slice(&bytes[a..b]);
        reject_unchanged(&reordered);
        // Valid per-frame checksums do not bypass the full-payload digest.
        let mut spliced = bytes.clone();
        let (a, b) = frames[1];
        spliced[b - 1] ^= 1;
        let digest = xxhash_rust::xxh3::xxh3_64(&spliced[a + HEADER_SIZE..b]);
        spliced[a + 24..a + 32].copy_from_slice(&digest.to_le_bytes());
        reject_unchanged(&spliced);
    }
    #[test]
    fn chunked_duplicate_initialization_and_nonarchive_records_reject() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(
            encode_record(
                0,
                &Record::Init {
                    config: GraphConfig::new(),
                    schema: None,
                },
            )
            .unwrap(),
        );
        bytes.extend_from_slice(&encoded(&record(), 97)[8..]);
        reject_unchanged(&bytes);
        let mut bytes = MAGIC.to_vec();
        write_chunked(
            &mut bytes,
            &Record::Init {
                config: GraphConfig::new(),
                schema: None,
            },
            17,
        )
        .unwrap();
        reject_unchanged(&bytes);
    }
    #[test]
    fn valid_chunked_baseline_still_discards_only_a_partial_ordinary_tail() {
        let mut bytes = encoded(&record(), 97);
        let clean = bytes.clone();
        bytes.extend_from_slice(b"GFRAME01\x01");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal");
        std::fs::write(&path, &bytes).unwrap();
        let store = DurableGraph::open(&path).unwrap();
        assert_eq!(store.recovered_tail_bytes(), 9);
        drop(store);
        assert_eq!(std::fs::read(path).unwrap(), clean);
    }
}
