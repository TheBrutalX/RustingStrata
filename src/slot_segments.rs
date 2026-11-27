//! Content-addressed KV segment store (the storage core of phase-2 deduplication).
//!
//! See `docs/SLOT-ARCHIVE-PHASE2.md` §2.3. A segment is a fixed token range of a conversation history plus an
//! opaque payload (the K/V rows for those positions). Segments are identified by a stable hash of the model
//! fingerprint, the parent segment id and the token list, so two histories that share a prefix produce the same
//! segment chain and share bytes on disk.
//!
//! This module is deliberately standalone: it does not know about `GpuModel`, `ArchiveEntry` or the `STRATASLOT`
//! format. The archive v2 integration stores the ordered segment ids in the entry and calls [`SegmentStore::get`]
//! to restore them. The parent chain is stored (and included in the id) but never resolved here: the caller keeps
//! the order and knows where each segment starts in the history.
//!
//! Layout under [`SegmentStore::open`]'s directory:
//!
//! ```text
//! <dir>/segments.json                  compacted manifest, one row per segment
//! <dir>/segments.log                   append-only journal since the last compaction
//! <dir>/segments.json.tmp.<pid>.<n>    write temp, removed at startup
//! <dir>/segments/<2 hex>/<id>.seg      one `STRATSEG` v1 file per segment
//! <dir>/segments/<2 hex>/.<id>.*.tmp   write temp, removed at startup
//! ```
//!
//! `segments.json` is a cache of the directory contents and the base of the journal: when it is missing or
//! corrupt it is moved aside and rebuilt from the `.seg` files, and the journal is discarded. A journal whose
//! final record is truncated by a crash is replayed up to the last complete line and then compacted. Reference
//! changes (`put` dedupe, `unref`) are journaled and fsynced before the in-memory update, so a compacted manifest
//! plus a truncated journal cannot silently lose a reference count. Orphan segment files are re-adopted with
//! `refs = 1`, and the archive reconciles the true counts from the surviving entries at startup.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// File name of the segment manifest inside the store directory.
pub const MANIFEST: &str = "segments.json";
/// File name of the append-only segment journal, holding mutations since the last compaction.
pub const JOURNAL: &str = "segments.log";
/// Subdirectory holding the content-addressed segment files.
pub const SEGMENTS_DIR: &str = "segments";
/// Header magic: first 8 bytes of every segment file.
pub const MAGIC: [u8; 8] = *b"STRATSEG";
/// Footer magic: written after the payload.
pub const FOOTER_MAGIC: [u8; 8] = *b"STRATEND";
/// Segment format version understood by this module.
pub const VERSION: u32 = 1;
/// Largest accepted segment payload (1 GiB). A 1024-token q8 segment is a few MB, so this is a sanity bound,
/// not a sizing target.
pub const MAX_PAYLOAD_BYTES: usize = 1 << 30;
/// Largest accepted token count (1M ids, 4 MiB). Real segments are 256-2048 tokens.
pub const MAX_TOKENS: usize = 1 << 20;
/// Journal record schema version.
const JOURNAL_VERSION: u32 = 1;
/// Compact after this many journal records even when the journal is smaller than the snapshot.
const JOURNAL_COMPACT_RECORDS: usize = 4096;

/// Fixed header length: magic, version, flags, fingerprint, n_tokens, parent length, payload length.
const HEADER_LEN: u64 = 40;
/// Footer length: magic plus the total number of bytes written before it.
const FOOTER_LEN: u64 = 16;
/// Longest accepted parent id (a segment id is 16 hex chars; this leaves room for a future scheme).
const MAX_PARENT_LEN: usize = 4096;
/// Manifest schema version written and understood by this module.
const MANIFEST_VERSION: u32 = 1;

/// One content-addressed segment and its manifest row.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Stable content id: 16 lowercase hex characters.
    pub id: String,
    /// Absolute path to the `.seg` file. Derived from the id, never accepted from a client.
    pub file: PathBuf,
    /// Apparent file size.
    pub bytes: u64,
    /// Number of token ids in the segment.
    pub n_tokens: usize,
    /// Parent segment id, when the segment extends another one. Not resolved by the store.
    pub parent: Option<String>,
    /// Number of entries that reference this segment.
    pub refs: u64,
    /// `STRATSEG` version (1).
    pub format: u32,
    pub created_ms: u64,
    pub last_used_ms: u64,
}

/// What [`SegmentStore::gc`] removed.
#[derive(Clone, Debug, Default)]
pub struct GcReport {
    /// Number of manifest rows removed.
    pub removed: usize,
    /// Logical bytes freed by removing zero-ref files.
    pub freed_bytes: u64,
}

/// A content-addressed segment store rooted at one directory.
pub struct SegmentStore {
    dir: PathBuf,
    segments: BTreeMap<String, Segment>,
    journal_bytes: u64,
    journal_records: usize,
    manifest_bytes: u64,
}

impl SegmentStore {
    /// Open (or create) `dir`, load `segments.json`, replay `segments.log`, and rebuild the manifest from
    /// `segments/<2 hex>/<id>.seg` when the compacted manifest is missing or corrupt. Missing/corrupt rows are
    /// dropped, stale temp files are removed, and the directory and manifest are created `0700`/`0600` on Unix.
    pub fn open(dir: PathBuf) -> Result<Self> {
        let dir = if dir.is_absolute() {
            dir
        } else {
            std::env::current_dir()
                .context("resolving the relative segment store directory")?
                .join(dir)
        };
        ensure_dir(&dir)?;
        ensure_dir(&dir.join(SEGMENTS_DIR))?;
        remove_stale_temps(&dir);

        let manifest = dir.join(MANIFEST);
        let (segments, changed, journal_bytes, journal_records) = match load_manifest(&dir) {
            Ok((segments, mut changed)) => {
                let (records, repaired, bytes, count) = load_journal(&dir);
                let mut segments = segments;
                for op in records {
                    if let Err(err) = apply_journal(&dir, &mut segments, op) {
                        crate::say!("segment store: ignoring a bad journal record: {err:#}");
                        changed = true;
                    }
                }
                (segments, changed || repaired, bytes, count)
            }
            Err(err) => {
                if manifest.exists() {
                    let bad = dir.join(format!("{MANIFEST}.bad.{}", now_ms()));
                    match fs::rename(&manifest, &bad) {
                        Ok(()) => crate::say!(
                            "segment store: {err:#}; moved the manifest to {} and rebuilding",
                            bad.display()
                        ),
                        Err(rename_err) => crate::say!(
                            "segment store: {err:#}; could not move the manifest aside: {rename_err:#}; rebuilding"
                        ),
                    }
                } else {
                    crate::say!("segment store: {err:#}; rebuilding the manifest from the segment files");
                }
                discard_journal(&dir, "the compacted segment manifest is unusable");
                (rebuild_manifest(&dir)?, true, 0, 0)
            }
        };

        let manifest_bytes = fs::metadata(&manifest).map(|meta| meta.len()).unwrap_or(0);
        let mut store = SegmentStore { dir, segments, journal_bytes, journal_records, manifest_bytes };
        if changed {
            store.compact()?;
        }
        Ok(store)
    }

    /// Store `tokens` (and their opaque payload) as a segment.
    ///
    /// The id is `hash(fingerprint, parent, tokens)`; `parent` is the segment this one extends, or `None` for a
    /// chain root. Storing the same `(fingerprint, parent, tokens, payload)` again does not rewrite the file: it
    /// increments the existing segment's `refs` and touches `last_used_ms`.
    pub fn put(&mut self, fingerprint: u64, parent: Option<&str>, tokens: &[u32], payload: &[u8]) -> Result<Segment> {
        check_bounds(tokens.len(), payload.len())?;
        if let Some(parent) = parent {
            ensure!(valid_id(parent), "invalid parent segment id {parent:?}");
        }

        let id = segment_id(fingerprint, parent, tokens);
        if self.segments.contains_key(&id) {
            match self.get(&id) {
                Ok(stored) if stored == payload => {
                    let existing = self.segments.get(&id).expect("checked above").clone();
                    let refs = existing.refs.saturating_add(1);
                    let used = now_ms();
                    let op = SegmentJournalOp::Refs {
                        v: JOURNAL_VERSION,
                        id: id.clone(),
                        refs,
                        last_used_ms: used,
                    };
                    self.append_journal(&op)?;
                    let segment = self.segments.get_mut(&id).expect("checked above");
                    segment.refs = refs;
                    segment.last_used_ms = used;
                    let segment = segment.clone();
                    self.maybe_compact();
                    return Ok(segment);
                }
                Ok(_) => bail!(
                    "segment {id} already exists with different content; refusing to overwrite it"
                ),
                Err(err) => {
                    // A stale or corrupt file at the content address: replace it.
                    crate::say!("segment store: {id} is unusable ({err:#}); rewriting it");
                    self.segments.remove(&id);
                    let _ = fs::remove_file(segment_path(&self.dir, &id));
                }
            }
        }

        let file = segment_path(&self.dir, &id);
        let parent_dir = file
            .parent()
            .context("segment path has no parent directory")?;
        ensure_dir(parent_dir)?;

        let (tmp, mut handle) = create_temp(parent_dir, &id)?;
        let written = write_segment_file(&mut handle, fingerprint, parent, tokens, payload);
        drop(handle);
        if let Err(err) = written {
            let _ = fs::remove_file(&tmp);
            return Err(err);
        }
        let bytes = match fs::metadata(&tmp) {
            Ok(meta) => meta.len(),
            Err(err) => {
                let _ = fs::remove_file(&tmp);
                return Err(err).with_context(|| format!("stat {}", tmp.display()));
            }
        };
        if let Err(err) = fs::rename(&tmp, &file) {
            let _ = fs::remove_file(&tmp);
            return Err(err)
                .with_context(|| format!("renaming {} to {}", tmp.display(), file.display()));
        }
        fsync_dir(parent_dir);

        let now = now_ms();
        let segment = Segment {
            id: id.clone(),
            file,
            bytes,
            n_tokens: tokens.len(),
            parent: parent.map(str::to_string),
            refs: 1,
            format: VERSION,
            created_ms: now,
            last_used_ms: now,
        };
        self.segments.insert(id.clone(), segment.clone());
        let op = SegmentJournalOp::Put { v: JOURNAL_VERSION, row: segment_row(&segment) };
        if let Err(err) = self.append_journal(&op) {
            // Keep memory consistent with the manifest on disk; the file stays as an orphan and is adopted by
            // the next rebuild.
            self.segments.remove(&id);
            return Err(err);
        }
        self.maybe_compact();
        Ok(segment)
    }

    /// True when `id` names a row in the manifest. The row was validated against its file at open/put time; this
    /// does not read the file, so it is cheap enough for startup entry validation and GC.
    pub fn contains(&self, id: &str) -> bool {
        valid_id(id) && self.segments.contains_key(id)
    }

    /// The manifest row for `id`, without reading its file.
    pub fn segment(&self, id: &str) -> Option<&Segment> {
        self.segments.get(id)
    }

    /// Read and validate a segment file, returning its payload. Does not change `refs`; updates the in-memory
    /// `last_used_ms` only (persisted with the next structural change).
    pub fn get(&mut self, id: &str) -> Result<Vec<u8>> {
        Ok(self.load(id)?.1)
    }

    /// [`SegmentStore::get`] with an additional proof that the segment holds exactly `tokens`. The archive v2 entry
    /// stores the expected range, so a segment that content-addresses to a different token list (a colliding id, a
    /// hand-edited manifest or a swapped file) is rejected before its payload is installed.
    pub fn get_checked(&mut self, id: &str, tokens: &[u32]) -> Result<Vec<u8>> {
        let (info, payload) = self.load(id)?;
        ensure!(
            info.tokens == tokens,
            "segment {id} does not contain the expected token range"
        );
        Ok(payload)
    }

    /// Read, validate and return the parsed header plus payload of one segment.
    fn load(&mut self, id: &str) -> Result<(FileInfo, Vec<u8>)> {
        ensure!(valid_id(id), "invalid segment id {id:?}");
        let Some(segment) = self.segments.get(id).cloned() else {
            bail!("unknown segment {id}");
        };
        let (info, payload) = read_full(&segment.file)
            .with_context(|| format!("reading segment {id}"))?;
        ensure!(
            info.tokens.len() == segment.n_tokens,
            "segment {id} has {} tokens, the manifest says {}",
            info.tokens.len(),
            segment.n_tokens
        );
        ensure!(
            info.parent == segment.parent,
            "segment {id} has a different parent than the manifest"
        );
        let computed = segment_id(info.fingerprint, info.parent.as_deref(), &info.tokens);
        ensure!(
            computed == id,
            "segment file {} contains segment {computed}, not {id}",
            segment.file.display()
        );
        if let Some(segment) = self.segments.get_mut(id) {
            segment.last_used_ms = now_ms();
        }
        Ok((info, payload))
    }

    /// Drop one reference. When the count reaches zero the file and row are removed; a missing id is a no-op.
    pub fn unref(&mut self, id: &str) -> Result<()> {
        ensure!(valid_id(id), "invalid segment id {id:?}");
        let Some(segment) = self.segments.get(id) else { return Ok(()) };
        if segment.refs > 1 {
            let refs = segment.refs - 1;
            let used = now_ms();
            let op = SegmentJournalOp::Refs {
                v: JOURNAL_VERSION,
                id: id.to_string(),
                refs,
                last_used_ms: used,
            };
            self.append_journal(&op)?;
            let segment = self.segments.get_mut(id).expect("checked above");
            segment.refs = refs;
            segment.last_used_ms = used;
            self.maybe_compact();
            return Ok(());
        }

        let file = segment.file.clone();
        if let Some(segment) = self.segments.get_mut(id) {
            segment.refs = 0;
        }
        match fs::remove_file(&file) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                // Keep the zero-ref row so `gc` can retry the unlink; do not lose track of the file.
                crate::say!(
                    "segment store: cannot remove {}: {err:#}; leaving it for GC",
                    file.display()
                );
                let op = SegmentJournalOp::Refs {
                    v: JOURNAL_VERSION,
                    id: id.to_string(),
                    refs: 0,
                    last_used_ms: now_ms(),
                };
                self.append_journal(&op)?;
                self.maybe_compact();
                return Ok(());
            }
        }
        let op = SegmentJournalOp::Remove { v: JOURNAL_VERSION, id: id.to_string() };
        if let Err(err) = self.append_journal(&op) {
            // The file is already gone; a stale row is dropped by the next open/GC validation.
            crate::say!("segment store: journaling the removal of {id}: {err:#}");
        }
        self.segments.remove(id);
        self.maybe_compact();
        Ok(())
    }

    /// Remove zero-ref segments and rows whose file disappeared. A row whose file cannot be unlinked is kept for
    /// a later attempt.
    pub fn gc(&mut self) -> Result<GcReport> {
        let ids: Vec<String> = self.segments.keys().cloned().collect();
        let mut report = GcReport::default();
        let mut changed = false;
        for id in ids {
            let Some(segment) = self.segments.get(&id) else { continue };
            let present = fs::metadata(&segment.file).map(|meta| meta.is_file()).unwrap_or(false);
            if segment.refs > 0 && present {
                continue;
            }
            let bytes = segment.bytes;
            let file = segment.file.clone();
            if present {
                match fs::remove_file(&file) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                    Err(err) => {
                        crate::say!(
                            "segment store: cannot remove {}: {err:#}; keeping the row for a later GC",
                            file.display()
                        );
                        continue;
                    }
                }
            }
            self.segments.remove(&id);
            report.removed += 1;
            if present {
                report.freed_bytes = report.freed_bytes.saturating_add(bytes);
            }
            changed = true;
        }
        if changed || self.journal_records > 0 {
            self.compact()?;
        }
        Ok(report)
    }

    /// All segments, sorted by id.
    pub fn list(&self) -> Vec<Segment> {
        self.segments.values().cloned().collect()
    }

    /// Replace every row's `refs` with the count in `counts` (ids absent from `counts` become zero).
    ///
    /// The archive entries are the source of truth for v2 references, while a rebuilt manifest can only adopt
    /// orphan files with `refs = 1`. Reconciling at startup makes two entries that share a segment keep it after
    /// either is deleted, and lets `gc` drop orphan segments. Ids unknown to the store are ignored (the caller
    /// validates that each entry's segments are present).
    pub fn sync_refs(&mut self, counts: &BTreeMap<String, u64>) -> Result<()> {
        let mut changed = false;
        for id in self.segments.keys().cloned().collect::<Vec<_>>() {
            let want = counts.get(&id).copied().unwrap_or(0);
            if let Some(segment) = self.segments.get_mut(&id) {
                if segment.refs != want {
                    segment.refs = want;
                    changed = true;
                }
            }
        }
        if changed || self.journal_records > 0 {
            self.compact()?;
        }
        Ok(())
    }

    /// Append one journal record and fsync it. A failed write is truncated back to the previous length so the
    /// journal never contains a malformed record in its middle.
    fn append_journal(&mut self, op: &SegmentJournalOp) -> Result<()> {
        ensure_dir(&self.dir)?;
        let path = self.dir.join(JOURNAL);
        let mut line = serde_json::to_vec(op).context("serializing the segment journal record")?;
        line.push(b'\n');
        let existed = path.exists();
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let before = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        let write = (|| -> Result<()> {
            file.write_all(&line).context("writing the segment journal")?;
            file.sync_data().context("syncing the segment journal")?;
            Ok(())
        })();
        if let Err(err) = write {
            let _ = file.set_len(before);
            return Err(err);
        }
        drop(file);
        if !existed {
            fsync_dir(&self.dir);
        }
        self.journal_bytes = before.saturating_add(line.len() as u64);
        self.journal_records += 1;
        Ok(())
    }

    fn clear_journal(&mut self) {
        let path = self.dir.join(JOURNAL);
        match fs::OpenOptions::new().write(true).truncate(true).open(&path) {
            Ok(file) => {
                if let Err(err) = file.sync_all() {
                    crate::say!("segment store: syncing the cleared journal: {err:#}");
                }
                fsync_dir(&self.dir);
                self.journal_bytes = 0;
                self.journal_records = 0;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                self.journal_bytes = 0;
                self.journal_records = 0;
            }
            Err(err) => {
                crate::say!("segment store: clearing the journal {}: {err:#}", path.display());
            }
        }
    }

    /// Write the compacted `segments.json` and clear the journal. Also called by the archive when it compacts,
    /// so a direct reader normally sees both manifests in step.
    pub(crate) fn compact(&mut self) -> Result<()> {
        ensure_dir(&self.dir)?;
        ensure_dir(&self.dir.join(SEGMENTS_DIR))?;
        let manifest = SegmentsFile {
            version: MANIFEST_VERSION,
            segments: self.segments.values().map(segment_row).collect(),
        };
        let data = serde_json::to_vec(&manifest).context("serializing the segment manifest")?;
        write_atomic(&self.dir, &data)?;
        self.manifest_bytes = data.len() as u64;
        self.clear_journal();
        Ok(())
    }

    /// Compact when the journal reaches the size of the snapshot or the record cap. Errors are logged because the
    /// mutation is already durable in the journal.
    fn maybe_compact(&mut self) {
        if self.journal_bytes >= self.manifest_bytes || self.journal_records >= JOURNAL_COMPACT_RECORDS {
            if let Err(err) = self.compact() {
                crate::say!("segment store: manifest compaction failed: {err:#}");
            }
        }
    }
}


/// The stable segment id: FNV-1a over a domain tag, the fingerprint, the parent id (length-prefixed) and the
/// token ids (length-prefixed). `start` is implied by the parent chain and `end - start` by the token count, so
/// the store does not need to resolve the chain to compute an id.
pub(crate) fn segment_id(fingerprint: u64, parent: Option<&str>, tokens: &[u32]) -> String {
    let mut hash = Fnv::new();
    hash.write(b"STRATSEG-segment-v1");
    hash.write(&fingerprint.to_le_bytes());
    match parent {
        Some(parent) => {
            hash.write(&[1]);
            hash.write(&(parent.len() as u64).to_le_bytes());
            hash.write(parent.as_bytes());
        }
        None => hash.write(&[0]),
    }
    hash.write(&(tokens.len() as u64).to_le_bytes());
    for &token in tokens {
        hash.write(&token.to_le_bytes());
    }
    format!("{:016x}", hash.finish())
}

/// A 16-character lowercase hex id. Ids are the only client/manifest-supplied path component.
fn valid_id(id: &str) -> bool {
    id.len() == 16
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn check_bounds(n_tokens: usize, payload_len: usize) -> Result<()> {
    ensure!(n_tokens > 0, "a segment must contain at least one token");
    ensure!(
        n_tokens <= MAX_TOKENS,
        "segment has {n_tokens} tokens, the limit is {MAX_TOKENS}"
    );
    ensure!(payload_len > 0, "a segment payload cannot be empty");
    ensure!(
        payload_len <= MAX_PAYLOAD_BYTES,
        "segment payload of {payload_len} bytes exceeds the {MAX_PAYLOAD_BYTES}-byte limit"
    );
    Ok(())
}

/// The absolute file path for an id. Only called with a [`valid_id`].
fn segment_path(dir: &Path, id: &str) -> PathBuf {
    debug_assert!(valid_id(id));
    dir.join(SEGMENTS_DIR)
        .join(&id[..2])
        .join(format!("{id}.seg"))
}

/// Milliseconds since the Unix epoch, saturating to 0 before it.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn mtime_ms(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as u64)
}

// ---------------------------------------------------------------------------------------------------------
// File format
// ---------------------------------------------------------------------------------------------------------

/// The parsed header plus token list of a segment file.
struct FileInfo {
    fingerprint: u64,
    parent: Option<String>,
    tokens: Vec<u32>,
    payload_len: u64,
}

impl FileInfo {
    fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Bytes from the start of the file through the footer.
    fn total_len(&self) -> u64 {
        HEADER_LEN
            + self.parent.as_ref().map_or(0, |parent| parent.len() as u64)
            + 4 * self.tokens.len() as u64
            + self.payload_len
            + FOOTER_LEN
    }
}

fn read_preamble<R: Read>(r: &mut R) -> Result<(u64, Option<String>, Vec<u32>, u64)> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).context("reading the segment magic")?;
    ensure!(
        magic == MAGIC,
        "bad segment magic {magic:02x?}, expected {MAGIC:02x?}"
    );

    let version = crate::slot_file::read_u32(r).context("reading the segment version")?;
    ensure!(
        version == VERSION,
        "unsupported segment version {version}, expected {VERSION}"
    );
    let flags = crate::slot_file::read_u32(r).context("reading the segment flags")?;
    ensure!(flags == 0, "unknown segment flags {flags:#x}");

    let fingerprint = crate::slot_file::read_u64(r)?;
    let n_tokens = crate::slot_file::read_u32(r)? as usize;
    let parent_len = crate::slot_file::read_u32(r)? as usize;
    let payload_len = crate::slot_file::read_u64(r)?;

    ensure!(
        n_tokens > 0 && n_tokens <= MAX_TOKENS,
        "segment has {n_tokens} tokens, the expected range is 1..={MAX_TOKENS}"
    );
    ensure!(
        parent_len <= MAX_PARENT_LEN,
        "segment parent id is {parent_len} bytes, the limit is {MAX_PARENT_LEN}"
    );
    ensure!(
        payload_len > 0 && payload_len <= MAX_PAYLOAD_BYTES as u64,
        "segment payload is {payload_len} bytes, the expected range is 1..={MAX_PAYLOAD_BYTES}"
    );

    let parent = if parent_len == 0 {
        None
    } else {
        let mut buf = vec![0u8; parent_len];
        r.read_exact(&mut buf).context("reading the segment parent id")?;
        let parent = String::from_utf8(buf).context("segment parent id is not UTF-8")?;
        ensure!(valid_id(&parent), "invalid parent segment id {parent:?}");
        Some(parent)
    };

    let mut tokens = Vec::with_capacity(n_tokens);
    for index in 0..n_tokens {
        tokens.push(
            crate::slot_file::read_u32(r)
                .with_context(|| format!("reading segment token {index} of {n_tokens}"))?,
        );
    }
    Ok((fingerprint, parent, tokens, payload_len))
}

fn validate_footer(footer: &[u8; FOOTER_LEN as usize], expected: u64) -> Result<()> {
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&footer[..8]);
    ensure!(
        magic == FOOTER_MAGIC,
        "bad segment footer magic {magic:02x?}, expected {FOOTER_MAGIC:02x?}"
    );
    let stored = u64::from_le_bytes(footer[8..16].try_into().expect("8 bytes"));
    ensure!(
        stored == expected,
        "segment footer length {stored} does not match the {expected} bytes before it"
    );
    Ok(())
}

/// Validate a segment file without reading its payload. Used at startup and on rebuild.
fn read_info(path: &Path) -> Result<FileInfo> {
    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let meta = file
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());

    let (fingerprint, parent, tokens, payload_len) = {
        let mut reader = BufReader::new(&mut file);
        read_preamble(&mut reader).with_context(|| format!("reading {}", path.display()))?
    };
    let info = FileInfo { fingerprint, parent, tokens, payload_len };
    let total = info.total_len();
    ensure!(
        meta.len() == total,
        "segment file {} is {} bytes, the header describes {total}",
        path.display(),
        meta.len()
    );

    let footer_pos = total - FOOTER_LEN;
    let mut reader = BufReader::new(file);
    reader
        .seek(SeekFrom::Start(footer_pos))
        .with_context(|| format!("seeking to the footer of {}", path.display()))?;
    let mut footer = [0u8; FOOTER_LEN as usize];
    reader
        .read_exact(&mut footer)
        .with_context(|| format!("reading the footer of {}", path.display()))?;
    validate_footer(&footer, footer_pos)?;
    Ok(info)
}

/// Read and validate a full segment file, including the payload and the footer.
fn read_full(path: &Path) -> Result<(FileInfo, Vec<u8>)> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let meta = file
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());

    let mut reader = BufReader::new(file);
    let (fingerprint, parent, tokens, payload_len) = read_preamble(&mut reader)?;
    let info = FileInfo { fingerprint, parent, tokens, payload_len };
    let total = info.total_len();
    ensure!(
        meta.len() == total,
        "segment file {} is {} bytes, the header describes {total}",
        path.display(),
        meta.len()
    );

    let mut payload = vec![0u8; payload_len as usize];
    reader
        .read_exact(&mut payload)
        .with_context(|| format!("reading the payload of {}", path.display()))?;
    let mut footer = [0u8; FOOTER_LEN as usize];
    reader
        .read_exact(&mut footer)
        .with_context(|| format!("reading the footer of {}", path.display()))?;
    validate_footer(&footer, total - FOOTER_LEN)?;
    Ok((info, payload))
}

fn write_segment_file<W: Write>(w: &mut W, fingerprint: u64, parent: Option<&str>, tokens: &[u32], payload: &[u8]) -> Result<()> {
    let parent_bytes = parent.unwrap_or("").as_bytes();
    w.write_all(&MAGIC).context("writing the segment magic")?;
    crate::slot_file::write_u32(w, VERSION)?;
    crate::slot_file::write_u32(w, 0)?; // flags
    crate::slot_file::write_u64(w, fingerprint)?;
    crate::slot_file::write_u32(w, tokens.len() as u32)?;
    crate::slot_file::write_u32(w, parent_bytes.len() as u32)?;
    crate::slot_file::write_u64(w, payload.len() as u64)?;
    w.write_all(parent_bytes).context("writing the segment parent id")?;
    for &token in tokens {
        crate::slot_file::write_u32(w, token)?;
    }
    w.write_all(payload).context("writing the segment payload")?;
    w.write_all(&FOOTER_MAGIC).context("writing the segment footer")?;
    let before_footer = HEADER_LEN
        + parent_bytes.len() as u64
        + 4 * tokens.len() as u64
        + payload.len() as u64;
    crate::slot_file::write_u64(w, before_footer)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct SegmentsFile {
    version: u32,
    segments: Vec<SegmentRow>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SegmentRow {
    id: String,
    bytes: u64,
    n_tokens: usize,
    parent: Option<String>,
    refs: u64,
    format: u32,
    created_ms: u64,
    last_used_ms: u64,
}

fn segment_row(segment: &Segment) -> SegmentRow {
    SegmentRow {
        id: segment.id.clone(),
        bytes: segment.bytes,
        n_tokens: segment.n_tokens,
        parent: segment.parent.clone(),
        refs: segment.refs,
        format: segment.format,
        created_ms: segment.created_ms,
        last_used_ms: segment.last_used_ms,
    }
}

// ---------------------------------------------------------------------------------------------------------
// Segment journal
// ---------------------------------------------------------------------------------------------------------
//
// `segments.log` is newline-delimited JSON, one record per mutation since the last `segments.json`
// compaction. A `put` carries the full row so a crash right after a segment file write is replayable; `refs`
// updates an existing row; `remove` drops it. Truncated final lines are ignored. The journal never carries the
// archive's entry refs, but every `refs` record is replayable onto the compacted snapshot, so a compacted
// manifest + truncated journal cannot lose a reference count (the archive reconciles the counts from surviving
// entries at startup as the final authority).

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase")]
enum SegmentJournalOp {
    Put { v: u32, row: SegmentRow },
    Refs { v: u32, id: String, refs: u64, last_used_ms: u64 },
    Remove { v: u32, id: String },
}

fn load_journal(dir: &Path) -> (Vec<SegmentJournalOp>, bool, u64, usize) {
    let path = dir.join(JOURNAL);
    let data = match fs::read(&path) {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return (Vec::new(), false, 0, 0),
        Err(err) => {
            crate::say!("segment store: cannot read the journal {}: {err:#}", path.display());
            return (Vec::new(), true, 0, 0);
        }
    };
    let bytes = data.len() as u64;
    let mut records = Vec::new();
    let mut repaired = false;
    for line in data.split_inclusive(|byte| *byte == b'\n') {
        if !line.ends_with(b"\n") {
            repaired = true;
            break;
        }
        match serde_json::from_slice::<SegmentJournalOp>(line) {
            Ok(op) => {
                let version = match &op {
                    SegmentJournalOp::Put { v, .. }
                    | SegmentJournalOp::Refs { v, .. }
                    | SegmentJournalOp::Remove { v, .. } => *v,
                };
                if version != JOURNAL_VERSION {
                    crate::say!(
                        "segment store: journal record version {version}, expected {JOURNAL_VERSION}; ignoring the rest"
                    );
                    repaired = true;
                    break;
                }
                records.push(op);
            }
            Err(err) => {
                crate::say!("segment store: corrupt journal record in {}: {err}", path.display());
                repaired = true;
                break;
            }
        }
    }
    (records, repaired, bytes, data.iter().filter(|byte| **byte == b'\n').count())
}

/// Apply one segment journal record. A `put` row is revalidated against its file; a row whose file disappeared
/// or is corrupt is dropped rather than re-adopted.
fn apply_journal(
    dir: &Path,
    segments: &mut BTreeMap<String, Segment>,
    op: SegmentJournalOp,
) -> Result<()> {
    match op {
        SegmentJournalOp::Put { row, .. } => match validate_row(dir, &row) {
            Ok(mut segment) => {
                if let Some(existing) = segments.get(&segment.id) {
                    segment.last_used_ms = segment.last_used_ms.max(existing.last_used_ms);
                }
                segments.insert(segment.id.clone(), segment);
            }
            Err(err) => {
                crate::say!("segment store: dropping journal segment {}: {err:#}", row.id);
                return Err(err);
            }
        },
        SegmentJournalOp::Refs { id, refs, last_used_ms, .. } => {
            if let Some(segment) = segments.get_mut(&id) {
                segment.refs = refs;
                segment.last_used_ms = segment.last_used_ms.max(last_used_ms);
            }
        }
        SegmentJournalOp::Remove { id, .. } => {
            segments.remove(&id);
        }
    }
    Ok(())
}

fn discard_journal(dir: &Path, reason: &str) {
    let path = dir.join(JOURNAL);
    if !path.exists() {
        return;
    }
    let bad = dir.join(format!("{JOURNAL}.bad.{}", now_ms()));
    match fs::rename(&path, &bad) {
        Ok(()) => crate::say!("segment store: {reason}; moved the journal to {}", bad.display()),
        Err(err) => {
            crate::say!("segment store: {reason}; removing the unreadable journal: {err:#}");
            let _ = fs::remove_file(&path);
        }
    }
}


/// Parse `segments.json` and validate every row against its file. Returns the rows and whether anything was
/// dropped or corrected (so [`SegmentStore::open`] can rewrite the manifest).
fn load_manifest(dir: &Path) -> Result<(BTreeMap<String, Segment>, bool)> {
    let path = dir.join(MANIFEST);
    let data = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let manifest: SegmentsFile =
        serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))?;
    if manifest.version != MANIFEST_VERSION {
        bail!(
            "unsupported segment manifest version {}, expected {MANIFEST_VERSION}",
            manifest.version
        );
    }

    let mut segments = BTreeMap::new();
    let mut changed = false;
    for row in manifest.segments {
        match validate_row(dir, &row) {
            Ok(segment) => {
                if segment.bytes != row.bytes {
                    changed = true;
                }
                if segments.contains_key(&segment.id) {
                    crate::say!(
                        "segment store: duplicate manifest row {}; keeping the first",
                        segment.id
                    );
                    changed = true;
                } else {
                    segments.insert(segment.id.clone(), segment);
                }
            }
            Err(err) => {
                crate::say!("segment store: dropping segment {}: {err:#}", row.id);
                // Only touch the path when the id is safe; a traversal-looking row is dropped unread.
                if valid_id(&row.id) {
                    let _ = fs::remove_file(segment_path(dir, &row.id));
                }
                changed = true;
            }
        }
    }
    Ok((segments, changed))
}

fn validate_row(dir: &Path, row: &SegmentRow) -> Result<Segment> {
    ensure!(valid_id(&row.id), "invalid segment id {:?}", row.id);
    ensure!(
        row.format == VERSION,
        "unsupported segment format {}, expected {VERSION}",
        row.format
    );
    ensure!(
        row.n_tokens > 0 && row.n_tokens <= MAX_TOKENS,
        "segment has {} tokens, the expected range is 1..={MAX_TOKENS}",
        row.n_tokens
    );
    if let Some(parent) = &row.parent {
        ensure!(valid_id(parent), "invalid parent segment id {parent:?}");
    }

    let file = segment_path(dir, &row.id);
    let info = read_info(&file)?;
    ensure!(
        info.n_tokens() == row.n_tokens,
        "file has {} tokens, the manifest says {}",
        info.n_tokens(),
        row.n_tokens
    );
    ensure!(
        info.parent == row.parent,
        "file parent {:?} does not match the manifest {:?}",
        info.parent,
        row.parent
    );
    let computed = segment_id(info.fingerprint, info.parent.as_deref(), &info.tokens);
    ensure!(
        computed == row.id,
        "file content belongs to segment {computed}, not {}",
        row.id
    );
    let meta = fs::metadata(&file).with_context(|| format!("stat {}", file.display()))?;
    ensure!(meta.is_file(), "{} is not a regular file", file.display());

    Ok(Segment {
        id: row.id.clone(),
        file,
        bytes: meta.len(),
        n_tokens: row.n_tokens,
        parent: row.parent.clone(),
        refs: row.refs,
        format: row.format,
        created_ms: row.created_ms,
        last_used_ms: row.last_used_ms,
    })
}

/// Scan `segments/<2 hex>/<id>.seg`, adopt every valid file with `refs = 1`, and return them keyed by id.
fn rebuild_manifest(dir: &Path) -> Result<BTreeMap<String, Segment>> {
    let mut segments = BTreeMap::new();
    let root = match fs::read_dir(dir.join(SEGMENTS_DIR)) {
        Ok(root) => root,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(segments),
        Err(err) => return Err(err).with_context(|| format!("reading {}", dir.join(SEGMENTS_DIR).display())),
    };
    for shard in root.flatten() {
        if !shard.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
            continue;
        }
        let shard_name = shard.file_name().to_string_lossy().into_owned();
        let files = match fs::read_dir(shard.path()) {
            Ok(files) => files,
            Err(err) => {
                crate::say!("segment store: cannot scan {}: {err:#}", shard.path().display());
                continue;
            }
        };
        for file in files.flatten() {
            let name = file.file_name().to_string_lossy().into_owned();
            let Some(id) = name.strip_suffix(".seg") else { continue };
            if !valid_id(id) || &id[..2] != shard_name {
                crate::say!("segment store: ignoring unexpected file {}", file.path().display());
                continue;
            }
            if !file.file_type().map(|kind| kind.is_file()).unwrap_or(false) {
                continue;
            }
            let path = file.path();
            let info = match read_info(&path) {
                Ok(info) => info,
                Err(err) => {
                    crate::say!("segment store: cannot recover {}: {err:#}", path.display());
                    continue;
                }
            };
            let computed = segment_id(info.fingerprint, info.parent.as_deref(), &info.tokens);
            if computed != id {
                crate::say!(
                    "segment store: {} contains segment {computed}, not {id}; skipping",
                    path.display()
                );
                continue;
            }
            let meta = match fs::metadata(&path) {
                Ok(meta) => meta,
                Err(err) => {
                    crate::say!("segment store: cannot stat {}: {err:#}", path.display());
                    continue;
                }
            };
            if segments.contains_key(id) {
                crate::say!(
                    "segment store: duplicate segment {id} at {}; keeping the first",
                    path.display()
                );
                continue;
            }
            let used = mtime_ms(&meta);
            segments.insert(
                id.to_string(),
                Segment {
                    id: id.to_string(),
                    file: path,
                    bytes: meta.len(),
                    n_tokens: info.n_tokens(),
                    parent: info.parent.clone(),
                    refs: 1,
                    format: VERSION,
                    created_ms: used,
                    last_used_ms: used,
                },
            );
        }
    }
    Ok(segments)
}

// ---------------------------------------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------------------------------------

/// Create `dir` (and parents) when missing, with mode 0700 on Unix. An existing directory is left untouched.
fn ensure_dir(dir: &Path) -> Result<()> {
    match fs::metadata(dir) {
        Ok(meta) if meta.is_dir() => Ok(()),
        Ok(_) => bail!("segment store path {} exists but is not a directory", dir.display()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(dir)
                .with_context(|| format!("creating the segment directory {}", dir.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).with_context(|| {
                    format!("setting mode 0700 on the segment directory {}", dir.display())
                })?;
            }
            Ok(())
        }
        Err(err) => Err(err).with_context(|| format!("stat {}", dir.display())),
    }
}

fn create_exclusive(path: &Path) -> Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .with_context(|| format!("creating {}", path.display()))
}

/// Create a uniquely named temp file in `dir`, returning its path and an open handle.
fn create_temp(dir: &Path, id: &str) -> Result<(PathBuf, fs::File)> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    for attempt in 0..32u32 {
        let path = dir.join(format!(
            ".{id}.{}.{nanos}.{attempt}.tmp",
            std::process::id()
        ));
        match create_exclusive(&path) {
            Ok(file) => return Ok((path, file)),
            Err(err) => {
                if err
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::AlreadyExists)
                {
                    continue;
                }
                return Err(err);
            }
        }
    }
    bail!("could not create a temporary segment file in {}", dir.display())
}

fn fsync_dir(dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(file) = fs::File::open(dir) {
            let _ = file.sync_all();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
}

/// Write `data` to `segments.json` via a temp file + `sync_all` + rename, then fsync the directory (best
/// effort).
fn write_atomic(dir: &Path, data: &[u8]) -> Result<()> {
    let manifest = dir.join(MANIFEST);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut tmp = dir.join(format!("{MANIFEST}.tmp.{}.{nanos}", std::process::id()));
    let mut file = {
        let mut attempt = 0u32;
        loop {
            match create_exclusive(&tmp) {
                Ok(file) => break file,
                Err(err)
                    if attempt < 16
                        && err
                            .downcast_ref::<std::io::Error>()
                            .is_some_and(|io| io.kind() == std::io::ErrorKind::AlreadyExists) =>
                {
                    attempt += 1;
                    tmp = dir.join(format!(
                        "{MANIFEST}.tmp.{}.{nanos}.{attempt}",
                        std::process::id()
                    ));
                }
                Err(err) => return Err(err),
            }
        }
    };

    let mut write = || -> Result<()> {
        file.write_all(data).context("writing the segment manifest")?;
        file.sync_all().context("syncing the segment manifest")?;
        Ok(())
    };
    if let Err(err) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    drop(file);

    if let Err(err) = fs::rename(&tmp, &manifest) {
        let _ = fs::remove_file(&tmp);
        return Err(err)
            .with_context(|| format!("renaming {} to {}", tmp.display(), manifest.display()));
    }
    fsync_dir(dir);
    Ok(())
}

/// Remove temp files left by a crashed writer: `segments.json.tmp.*` at the top and `*.tmp` in the shards.
fn remove_stale_temps(dir: &Path) {
    if let Ok(root) = fs::read_dir(dir) {
        for entry in root.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("segments.json.tmp.") {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    let segments = dir.join(SEGMENTS_DIR);
    let Ok(root) = fs::read_dir(segments) else { return };
    for shard in root.flatten() {
        if !shard.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
            continue;
        }
        if let Ok(files) = fs::read_dir(shard.path()) {
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                if name.ends_with(".tmp") {
                    let _ = fs::remove_file(file.path());
                }
            }
        }
    }
}

/// A minimal FNV-1a (64-bit) writer. Kept local so segment ids are stable across processes, restarts and Rust
/// versions; `DefaultHasher` makes no such guarantee.
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Fnv(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP: u64 = 0x0123_4567_89ab_cdef;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            TempDir(std::env::temp_dir().join(format!(
                "strata-segments-{name}-{}-{nanos}",
                std::process::id()
            )))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn open(dir: &Path) -> SegmentStore {
        SegmentStore::open(dir.to_path_buf()).unwrap()
    }

    fn tokens(start: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| start + i).collect()
    }

    fn payload(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    fn segment_files(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(shards) = fs::read_dir(dir.join(SEGMENTS_DIR)) {
            for shard in shards.flatten() {
                if !shard.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
                    continue;
                }
                if let Ok(files) = fs::read_dir(shard.path()) {
                    for file in files.flatten() {
                        if file.file_name().to_string_lossy().ends_with(".seg") {
                            out.push(file.path());
                        }
                    }
                }
            }
        }
        out.sort();
        out
    }

    fn temp_names(dir: &Path) -> Vec<String> {
        let mut out: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp"))
            .collect();
        out.sort();
        out
    }

    #[test]
    fn segment_store_is_send() {
        fn is_send<T: Send>() {}
        is_send::<SegmentStore>();
    }

    #[test]
    fn put_get_round_trip() {
        let tmp = TempDir::new("roundtrip");
        let dir = tmp.path();
        let mut store = open(dir);
        let toks = tokens(100, 16);
        let data = payload(3, 300);

        let segment = store.put(FP, None, &toks, &data).unwrap();
        assert!(valid_id(&segment.id));
        assert_eq!(segment.n_tokens, toks.len());
        assert_eq!(segment.refs, 1);
        assert_eq!(segment.format, VERSION);
        assert!(segment.file.is_absolute());
        assert_eq!(
            segment.file,
            dir.join(SEGMENTS_DIR)
                .join(&segment.id[..2])
                .join(format!("{}.seg", segment.id))
        );
        assert_eq!(fs::metadata(&segment.file).unwrap().len(), segment.bytes);
        assert!(dir.join(MANIFEST).exists());

        assert_eq!(store.get(&segment.id).unwrap(), data);
        assert_eq!(store.list().len(), 1);
        assert_eq!(store.list()[0].refs, 1, "get must not change refs");

        let raw = fs::read(&segment.file).unwrap();
        assert_eq!(&raw[..8], &MAGIC);
        assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), VERSION);
        assert_eq!(&raw[raw.len() - 16..raw.len() - 8], &FOOTER_MAGIC);

        assert!(store.get("0000000000000000").is_err());
    }

    #[test]
    fn identical_put_dedupes_and_increments_refs() {
        let tmp = TempDir::new("dedupe");
        let dir = tmp.path();
        let mut store = open(dir);
        let toks = tokens(10, 32);
        let data = payload(7, 512);

        let first = store.put(FP, None, &toks, &data).unwrap();
        let second = store.put(FP, None, &toks, &data).unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(second.refs, 2);
        assert_eq!(store.list().len(), 1);
        assert_eq!(segment_files(dir).len(), 1);
        assert_eq!(store.list()[0].refs, 2);

        // Same tokens but a different payload is a contradiction, not a silent overwrite.
        let err = store
            .put(FP, None, &toks, &payload(9, data.len()))
            .unwrap_err();
        assert!(err.to_string().contains("different content"), "{err}");
        assert_eq!(store.list()[0].refs, 2);
        assert_eq!(store.get(&first.id).unwrap(), data);
    }

    #[test]
    fn parent_token_and_fingerprint_are_part_of_the_id() {
        let tmp = TempDir::new("ids");
        let dir = tmp.path();
        let mut store = open(dir);
        let toks = tokens(1, 12);
        let data = payload(2, 64);

        let root = store.put(FP, None, &toks, &data).unwrap();
        let child = store.put(FP, Some(&root.id), &toks, &data).unwrap();
        let other_tokens = store
            .put(FP, None, &tokens(500, 12), &data)
            .unwrap();
        let other_model = store.put(FP ^ 1, None, &toks, &data).unwrap();

        let mut ids = vec![
            root.id.clone(),
            child.id.clone(),
            other_tokens.id.clone(),
            other_model.id.clone(),
        ];
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 4);
        assert_eq!(store.list().len(), 4);
        assert_eq!(segment_files(dir).len(), 4);
        assert_eq!(child.parent.as_deref(), Some(root.id.as_str()));
        assert!(root.parent.is_none());
        assert_eq!(store.get(&child.id).unwrap(), data);
    }

    #[test]
    fn unref_and_gc_remove_files() {
        let tmp = TempDir::new("refs");
        let dir = tmp.path();
        let mut store = open(dir);
        let toks = tokens(20, 8);
        let data = payload(4, 100);

        let segment = store.put(FP, None, &toks, &data).unwrap();
        store.put(FP, None, &toks, &data).unwrap();
        assert_eq!(store.list()[0].refs, 2);

        store.unref(&segment.id).unwrap();
        assert_eq!(store.list()[0].refs, 1);
        assert!(segment.file.exists());
        store.unref(&segment.id).unwrap();
        assert!(store.list().is_empty());
        assert!(!segment.file.exists());
        store.unref(&segment.id).unwrap(); // idempotent

        // A row whose file disappeared is dropped by GC.
        let missing = store.put(FP, None, &tokens(300, 8), &payload(5, 40)).unwrap();
        fs::remove_file(&missing.file).unwrap();
        let report = store.gc().unwrap();
        assert_eq!(report.removed, 1);
        assert_eq!(report.freed_bytes, 0);
        assert!(store.list().is_empty());

        // A zero-ref row (e.g. a previous unlink failure) is removed with its file.
        let zero = store.put(FP, None, &tokens(400, 8), &payload(6, 40)).unwrap();
        store.segments.get_mut(&zero.id).unwrap().refs = 0;
        let report = store.gc().unwrap();
        assert_eq!(report.removed, 1);
        assert_eq!(report.freed_bytes, zero.bytes);
        assert!(!zero.file.exists());
        assert!(store.list().is_empty());
    }

    #[test]
    fn corrupt_and_missing_files_are_dropped_at_open() {
        let tmp = TempDir::new("corrupt");
        let dir = tmp.path();
        let mut store = open(dir);
        let corrupt = store
            .put(FP, None, &tokens(1, 8), &payload(1, 80))
            .unwrap();
        let missing = store
            .put(FP, None, &tokens(100, 8), &payload(2, 80))
            .unwrap();

        // Corrupt the footer length: the file still has the right size, so only validation catches it.
        let mut raw = fs::read(&corrupt.file).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        fs::write(&corrupt.file, &raw).unwrap();
        assert!(store.get(&corrupt.id).is_err());

        fs::remove_file(&missing.file).unwrap();
        drop(store);

        let store = open(dir);
        assert!(store.list().is_empty(), "{:?}", store.list());
        assert!(!corrupt.file.exists(), "a corrupt segment is removed at startup");
        assert!(!missing.file.exists());
    }

    #[test]
    fn manifest_is_rebuilt_from_segment_files() {
        let tmp = TempDir::new("rebuild");
        let dir = tmp.path();
        let mut store = open(dir);
        let root_tokens = tokens(1, 8);
        let root_data = payload(1, 90);
        let root = store.put(FP, None, &root_tokens, &root_data).unwrap();
        let child_data = payload(2, 70);
        let child = store
            .put(FP, Some(&root.id), &tokens(9, 4), &child_data)
            .unwrap();
        drop(store);

        fs::remove_file(dir.join(MANIFEST)).unwrap();
        let mut store = open(dir);
        let list = store.list();
        assert_eq!(list.len(), 2);
        assert!(list.iter().all(|segment| segment.refs == 1));
        assert!(list.iter().all(|segment| segment.parent.is_none() || segment.id == child.id));
        assert_eq!(store.get(&root.id).unwrap(), root_data);
        assert_eq!(store.get(&child.id).unwrap(), child_data);

        // Reusing a recovered segment increments its ref instead of rewriting it.
        let again = store.put(FP, None, &root_tokens, &root_data).unwrap();
        assert_eq!(again.id, root.id);
        assert_eq!(store.segments.get(&root.id).unwrap().refs, 2);
        assert_eq!(segment_files(dir).len(), 2);
    }

    #[test]
    fn corrupt_manifest_is_moved_aside_and_rebuilt() {
        let tmp = TempDir::new("bad-manifest");
        let dir = tmp.path();
        let mut store = open(dir);
        let segment = store
            .put(FP, None, &tokens(1, 6), &payload(3, 60))
            .unwrap();
        drop(store);

        fs::write(dir.join(MANIFEST), b"{ this is not json").unwrap();
        let store = open(dir);
        assert_eq!(store.list().len(), 1);
        assert_eq!(store.list()[0].id, segment.id);
        let bad = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().starts_with("segments.json.bad."));
        assert!(bad);
    }

    #[test]
    fn stale_temp_files_are_removed() {
        let tmp = TempDir::new("temps");
        let dir = tmp.path();
        fs::create_dir_all(dir.join(SEGMENTS_DIR).join("ab")).unwrap();
        let top = dir.join("segments.json.tmp.1.2");
        let shard = dir.join(SEGMENTS_DIR).join("ab").join(".dead.1.2.tmp");
        fs::write(&top, b"junk").unwrap();
        fs::write(&shard, b"junk").unwrap();

        let store = open(dir);
        assert!(!top.exists());
        assert!(!shard.exists());
        assert!(store.list().is_empty());
        assert!(dir.join(MANIFEST).exists());
    }

    #[test]
    fn ids_and_manifest_rows_cannot_escape_the_store() {
        let tmp = TempDir::new("traversal");
        let dir = tmp.path();
        let mut store = open(dir);
        for bad in ["", "../evil", "aa/../../evil", "zzzzzzzzzzzzzzzz", "0123456789abcdeZ"] {
            assert!(store.get(bad).is_err(), "{bad:?} must not be readable");
            assert!(store.unref(bad).is_err(), "{bad:?} must not be unref-able");
        }
        let toks = tokens(1, 4);
        let data = payload(1, 8);
        assert!(store.put(FP, Some("../evil"), &toks, &data).is_err());
        assert!(store.list().is_empty());

        // A traversal-looking manifest row is dropped before its path is touched.
        let outside = dir.join("..").join("strata-segment-evil");
        let _ = fs::remove_file(&outside);
        let json = serde_json::json!({
            "version": 1,
            "segments": [{
                "id": "../strata-segment-evil",
                "bytes": 1,
                "n_tokens": 1,
                "parent": null,
                "refs": 1,
                "format": 1,
                "created_ms": 0,
                "last_used_ms": 0
            }]
        });
        fs::write(dir.join(MANIFEST), serde_json::to_vec(&json).unwrap()).unwrap();
        let store = open(dir);
        assert!(store.list().is_empty());
        assert!(!outside.exists());
    }

    #[test]
    fn manifest_is_valid_and_leaves_no_temps() {
        let tmp = TempDir::new("atomic");
        let dir = tmp.path();
        let mut store = open(dir);
        let segment = store
            .put(FP, None, &tokens(1, 8), &payload(2, 128))
            .unwrap();

        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["segments"][0]["id"], segment.id);
        assert_eq!(value["segments"][0]["refs"], 1);
        assert_eq!(value["segments"][0]["n_tokens"], 8);
        assert!(temp_names(dir).is_empty(), "{:?}", temp_names(dir));
        let shard_temps: Vec<String> = fs::read_dir(segment.file.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(shard_temps.is_empty(), "{shard_temps:?}");

        // A second structural write replaces the manifest atomically and stays parseable.
        store.unref(&segment.id).unwrap();
        store.compact().unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(value["segments"].as_array().unwrap().len(), 0);
        assert!(temp_names(dir).is_empty());
    }

    #[test]
    fn empty_and_oversized_inputs_are_rejected() {
        let tmp = TempDir::new("bounds");
        let dir = tmp.path();
        let mut store = open(dir);
        let toks = tokens(1, 4);
        let data = payload(1, 8);

        let err = store.put(FP, None, &[], &data).unwrap_err();
        assert!(err.to_string().contains("at least one token"), "{err}");
        let err = store.put(FP, None, &toks, &[]).unwrap_err();
        assert!(err.to_string().contains("cannot be empty"), "{err}");

        assert!(check_bounds(0, 1).is_err());
        assert!(check_bounds(1, 0).is_err());
        assert!(check_bounds(MAX_TOKENS + 1, 1).is_err());
        assert!(check_bounds(1, MAX_PAYLOAD_BYTES + 1).is_err());
        assert!(check_bounds(1, 1).is_ok());
        assert!(store.list().is_empty());
    }

    #[test]
    fn journal_replays_refs_and_remove() {
        let tmp = TempDir::new("journal-replay");
        let dir = tmp.path();
        let mut store = open(dir);
        let segment = store
            .put(FP, None, &tokens(1, 8), &payload(2, 64))
            .unwrap();
        store
            .append_journal(&SegmentJournalOp::Refs {
                v: JOURNAL_VERSION,
                id: segment.id.clone(),
                refs: 5,
                last_used_ms: segment.last_used_ms + 1000,
            })
            .unwrap();
        drop(store);

        let mut store = open(dir);
        assert_eq!(store.segment(&segment.id).unwrap().refs, 5);
        assert!(store.segment(&segment.id).unwrap().last_used_ms >= segment.last_used_ms + 1000);
        store
            .append_journal(&SegmentJournalOp::Remove { v: JOURNAL_VERSION, id: segment.id.clone() })
            .unwrap();
        drop(store);

        let store = open(dir);
        assert!(store.list().is_empty());
        assert!(!store.contains(&segment.id));
    }

    #[test]
    fn truncated_journal_tail_is_ignored_and_compacted() {
        let tmp = TempDir::new("journal-truncated");
        let dir = tmp.path();
        let mut store = open(dir);
        let segment = store
            .put(FP, None, &tokens(1, 8), &payload(3, 80))
            .unwrap();
        store
            .append_journal(&SegmentJournalOp::Refs {
                v: JOURNAL_VERSION,
                id: segment.id.clone(),
                refs: 7,
                last_used_ms: segment.last_used_ms,
            })
            .unwrap();
        drop(store);

        let mut file = fs::OpenOptions::new().append(true).open(dir.join(JOURNAL)).unwrap();
        file.write_all(br#"{"op":"refs","v":1,"id":"#).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let store = open(dir);
        assert_eq!(store.segment(&segment.id).unwrap().refs, 7);
        assert_eq!(fs::metadata(dir.join(JOURNAL)).unwrap().len(), 0);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(value["segments"][0]["refs"], 7);
    }

    #[test]
    fn compacted_manifest_plus_truncated_journal_keeps_refs() {
        let tmp = TempDir::new("journal-compacted");
        let dir = tmp.path();
        let mut store = open(dir);
        let segment = store
            .put(FP, None, &tokens(1, 8), &payload(4, 96))
            .unwrap();
        store.compact().unwrap();
        let snapshot: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(snapshot["segments"][0]["refs"], 1);

        store
            .append_journal(&SegmentJournalOp::Refs {
                v: JOURNAL_VERSION,
                id: segment.id.clone(),
                refs: 2,
                last_used_ms: segment.last_used_ms,
            })
            .unwrap();
        drop(store);
        let mut file = fs::OpenOptions::new().append(true).open(dir.join(JOURNAL)).unwrap();
        file.write_all(br#"{"op":"put","v":1,"row":{"id":"#).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let store = open(dir);
        assert_eq!(store.segment(&segment.id).unwrap().refs, 2);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(value["segments"][0]["refs"], 2);
        assert_eq!(fs::metadata(dir.join(JOURNAL)).unwrap().len(), 0);
    }
}
