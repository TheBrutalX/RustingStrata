//! Disk-backed archive of idle slot state (see `docs/SLOT-ARCHIVE.md`).
//!
//! This module owns the archive directory, the manifest and the prefix index. It does not touch the GPU: the
//! server asks for a path with [`Archive::prepare_save`], writes the slot with
//! `GpuModel::save_slot_file_as`, then commits the file with [`Archive::commit_save`]. A lookup returns the
//! longest entry whose token history is an exact prefix of the request; the server restores it with the existing
//! `restore_slot_file`.
//!
//! The files are ordinary `STRATASLOT` v1 slot files (see `src/slot_file.rs`); the manifest adds the index and
//! the budget. Owner is the conversation owner (`GpuModel::slot_owner`), not the request that triggered a
//! checkpoint.

use crate::gpu_model::GpuModel;
use crate::slot_segments::{self, SegmentStore};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

/// File name of the compacted manifest inside [`ArchiveConfig::dir`].
pub const MANIFEST: &str = "manifest.json";
/// File name of the append-only journal holding manifest mutations since the last compaction.
pub const JOURNAL: &str = "manifest.log";
/// Headroom kept free below the logical budget, so a save can always write a temp file and the manifest.
pub const RESERVE_BYTES: u64 = 1 << 30;
/// Manifest schema version written and understood by this module.
const MANIFEST_VERSION: u32 = 1;
/// Journal record schema version.
const JOURNAL_VERSION: u32 = 1;
/// Maximum number of pending LRU touches before they are appended to the journal.
const TOUCH_FLUSH_MAX: usize = 256;
/// Maximum interval between two forced touch flushes (60 s).
const TOUCH_FLUSH_INTERVAL_MS: u64 = 60_000;
/// Compact after this many journal records even when the journal is smaller than the snapshot.
const JOURNAL_COMPACT_RECORDS: usize = 4096;
/// Small archives compact on every structural write, so the compacted JSON stays useful to operators and
/// external tooling. Larger archives amortize the snapshot write through the journal.
const EAGER_COMPACT_ENTRIES: usize = 256;

/// Tokens per content-addressed KV segment (docs/SLOT-ARCHIVE-PHASE2.md §2.3 and §15).
pub const SEGMENT_TOKENS: usize = 1024;
/// Magic of a v2 entry state file (`STRSTATE`): the token list, segment refs and the recurrent snapshot.
pub const STATE_MAGIC: [u8; 8] = *b"STRSTATE";
/// Version of the `STRSTATE` format written by this module. Version 1 files (no segment refs) stay readable.
pub const STATE_VERSION: u32 = 2;
/// Fixed v1 `STRSTATE` header length: magic, version, fingerprint, owner, rope offset, token count.
const STATE_V1_HEADER_LEN: u64 = 8 + 4 + 8 + 8 + 4 + 4;
/// Fixed v2 `STRSTATE` header length: the v1 fields plus the segment-ref count.
const STATE_HEADER_LEN: u64 = STATE_V1_HEADER_LEN + 4;
/// Wire length of one ordered segment reference: 16-byte id, start u64, end u64.
const STATE_SEGMENT_REF_LEN: u64 = 16 + 8 + 8;
/// Sanity bound on the token list in a `STRSTATE` file (16M ids, 64 MiB), far above any `--ctx`.
const MAX_STATE_TOKENS: usize = 1 << 24;
/// Sanity bound on the segment-ref list (1M refs, 32 MiB). Real 1024-token histories use far fewer.
const MAX_STATE_SEGMENTS: usize = 1 << 20;

/// Archive policy and location. Built from the `--slot-archive-*` flags.
#[derive(Clone, Debug)]
pub struct ArchiveConfig {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    pub min_tokens: usize,
    pub min_delta: usize,
    pub versions: usize,
}

/// One ordered KV segment of a v2 entry: the content-addressed segment id and the token range it covers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRef {
    pub id: String,
    pub start: usize,
    pub end: usize,
}

/// One immutable archive entry and its manifest row.
///
/// Format 1 is a single `STRATASLOT` v1 file in [`ArchiveEntry::file`]. Format 2 is a `STRSTATE` state file in
/// [`ArchiveEntry::state_file`] plus an ordered list of content-addressed KV segments in
/// [`ArchiveEntry::segments`]; `file` points at the same state file so a v1-only reader still has a path, but it is
/// not a `STRATASLOT` file and must only be restored with [`Archive::restore_segmented`].
#[derive(Clone, Debug)]
pub struct ArchiveEntry {
    /// Stable content identity; also the dedupe key.
    pub id: String,
    /// Absolute path to the `.slot` file (v1) or to the `.state` file (v2). Never accepted from a client.
    pub file: PathBuf,
    /// v2 only: the `STRSTATE` file holding the token list and the recurrent snapshot.
    pub state_file: Option<PathBuf>,
    /// v2 only: the ordered KV segments, covering `[0, n_tokens)` contiguously.
    pub segments: Vec<SegmentRef>,
    /// Apparent file size, used for the logical budget.
    pub bytes: u64,
    /// Number of tokens in the saved history (also the number of restored KV rows).
    pub n_tokens: usize,
    /// Conversation owner (API key id), not the owner of the request that triggered the checkpoint.
    pub owner: u64,
    /// Stable model/KV/YaRN/split fingerprint from `GpuModel::slot_fingerprint`.
    pub fingerprint: u64,
    /// 64-bit hash of the full token history; a candidate filter, never the final proof.
    pub prefix_hash: u64,
    pub created_ms: u64,
    pub last_used_ms: u64,
    /// Longest entry whose history is a prefix of this one, when known.
    pub parent: Option<String>,
    /// Admin protection from GC.
    pub pinned: bool,
    /// `divergence`, `evict`, `checkpoint`, ...; for logs and diagnostics.
    pub reason: String,
    /// `STRATASLOT` version (1) or the segmented entry version (2).
    pub format: u32,
}

/// Aggregate counters for `/metrics` and `/api/stats`.
#[derive(Clone, Debug, Default)]
pub struct ArchiveStats {
    pub entries: usize,
    pub bytes: u64,
    pub budget_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub restores: u64,
    pub evictions: u64,
    /// Entries removed by the TTL policy (also counted in `evictions`).
    pub expired: u64,
    pub failures: u64,
}

/// What [`Archive::gc`] removed.
#[derive(Clone, Debug, Default)]
pub struct GcReport {
    pub evicted: usize,
    /// Entries removed by the TTL policy (a subset of `evicted`).
    pub expired: usize,
    pub freed_bytes: u64,
}

/// Result of [`Archive::prepare_save`].
pub enum SaveSlot {
    /// An entry with the same content already exists; touch it and skip the GPU copy.
    Existing(ArchiveEntry),
    /// A new entry: write it to `path`, then call [`Archive::commit_save`] with `id`.
    New { id: String, path: PathBuf },
}

/// The archive: directory, manifest, prefix index and budget.
pub struct Archive {
    cfg: ArchiveConfig,
    entries: BTreeMap<String, ArchiveEntry>,
    /// Content-addressed KV segments shared by v2 entries; `<dir>/segments.json` + `<dir>/segments/`.
    segments: SegmentStore,
    stats: ArchiveStats,
    /// Transient in-flight references, keyed by entry id. Shared with [`Lease`] guards so a guard can release
    /// the lease after the `Archive` borrow has ended.
    leases: Arc<Mutex<BTreeMap<String, usize>>>,
    /// Lifecycle policy; all zero means off. Set with [`Archive::set_policy`], not persisted in the manifest.
    ttl_ms: u64,
    per_key_bytes: u64,
    pin_bytes: u64,
    /// LRU touches not yet appended to the journal. `touch` only updates memory and this map; the update is
    /// durable after [`Archive::flush`] (also called from `gc`, the next compaction and `Drop`).
    pending_touches: BTreeMap<String, u64>,
    /// Bytes and records currently in `manifest.log`, used to decide when to compact.
    journal_bytes: u64,
    journal_records: usize,
    /// Size of the last compacted snapshot, the amortization target for journal compaction.
    manifest_bytes: u64,
    /// When the pending touches were last flushed.
    last_touch_flush_ms: u64,
}

/// A transient reference to an archive entry, held while a restore or save is in flight.
///
/// [`Archive::delete`] refuses to remove a leased entry and [`Archive::gc`] skips it, so the file cannot
/// disappear between [`Archive::lookup_leased`] and the restore that reads it. Dropping the guard releases the
/// lease; a guard whose entry has already disappeared is harmless.
#[must_use = "the lease is released when the guard is dropped"]
pub struct Lease {
    id: String,
    leases: Arc<Mutex<BTreeMap<String, usize>>>,
}

impl Lease {
    /// The id of the leased entry.
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        // A poisoned lock only happens after a panic while the table was locked; Drop cannot panic, so recover
        // the guard and still release the lease.
        let mut table = self
            .leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(count) = table.get_mut(&self.id) else {
            // The entry disappeared while leased (or the lease was never registered): nothing to do.
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            table.remove(&self.id);
        }
    }
}

impl Archive {
    /// Open (or create) `cfg.dir`, load `manifest.json`, replay `manifest.log`, and rebuild the index from the
    /// shard files when the compacted manifest is missing or corrupt. Creates the directory with mode 0700 on Unix
    /// when it does not exist.
    ///
    /// Recovery order:
    /// 1. a valid compacted manifest is loaded, then the journal is replayed over it (a truncated last record is
    ///    ignored and the journal is compacted);
    /// 2. a missing/corrupt manifest is rebuilt from the surviving v1 `.slot` files and v2 `.state` files, using
    ///    `segments.json` (itself rebuilt from `.seg` files) to validate every v2 segment reference. The journal is
    ///    discarded in this case because the files, not the lost snapshot, are the source of truth.
    pub fn open(cfg: ArchiveConfig) -> Result<Self> {
        let mut cfg = cfg;
        if !cfg.dir.is_absolute() {
            cfg.dir = std::env::current_dir()
                .context("resolving the relative slot archive directory")?
                .join(&cfg.dir);
        }
        ensure_dir(&cfg.dir)?;
        remove_stale_temps(&cfg.dir);

        // The segment store is needed to validate rebuilt v2 entries, so open it first. Its own manifest is
        // likewise loaded or rebuilt from the `.seg` files.
        let mut segments = SegmentStore::open(cfg.dir.clone())?;

        let manifest = cfg.dir.join(MANIFEST);
        let mut journal_bytes = 0u64;
        let mut journal_records = 0usize;
        let mut journal_repaired = false;
        let (mut entries, rebuilt) = match load_manifest(&cfg.dir) {
            Ok(entries) => {
                let (records, repaired, bytes, count) = load_journal(&cfg.dir);
                journal_bytes = bytes;
                journal_records = count;
                journal_repaired = repaired;
                let mut entries = entries;
                for record in records {
                    if let Err(err) = apply_journal(&cfg.dir, &mut entries, record) {
                        crate::say!("slot archive: ignoring a bad manifest journal record: {err:#}");
                        journal_repaired = true;
                    }
                }
                (entries, false)
            }
            Err(err) => {
                if manifest.exists() {
                    let bad = cfg.dir.join(format!("{MANIFEST}.bad.{}", now_ms()));
                    match fs::rename(&manifest, &bad) {
                        Ok(()) => crate::say!(
                            "slot archive: {err:#}; moved the manifest to {} and rebuilding",
                            bad.display()
                        ),
                        Err(rename_err) => crate::say!(
                            "slot archive: {err:#}; could not move the manifest aside: {rename_err:#}; rebuilding"
                        ),
                    }
                } else {
                    crate::say!("slot archive: {err:#}; rebuilding the manifest from the archive files");
                }
                // The compacted manifest is the base of the journal; without it the journal cannot be replayed
                // safely, and the surviving files are authoritative.
                discard_journal(&cfg.dir, "the compacted manifest is unusable");
                (rebuild_manifest(&cfg.dir, &segments)?, true)
            }
        };

        // Startup validation: stat every file and compare the header with the manifest row. Missing or
        // inconsistent entries are dropped (the file, if any, stays on disk for a later rebuild). A v2 entry is
        // dropped when its state file or any referenced segment is missing/inconsistent.
        let mut changed = rebuilt || journal_repaired;
        let ids: Vec<String> = entries.keys().cloned().collect();
        for id in ids {
            let entry = entries[&id].clone();
            if let Err(err) = validate_entry(&entry, &segments) {
                crate::say!("slot archive: dropping {id}: {err:#}");
                entries.remove(&id);
                changed = true;
            }
        }

        // The surviving v2 entries are the source of truth for segment refcounts: a rebuilt manifest adopts orphan
        // files with `refs = 1`, and a dropped entry may have released or lost references. Reconciling makes a
        // shared segment survive the deletion of one of its entries.
        let mut refs: BTreeMap<String, u64> = BTreeMap::new();
        for entry in entries.values().filter(|entry| entry.format == 2) {
            for segment in &entry.segments {
                *refs.entry(segment.id.clone()).or_insert(0) += 1;
            }
        }
        if let Err(err) = segments.sync_refs(&refs) {
            crate::say!("slot archive: segment refcount reconciliation failed: {err:#}");
        }
        // Orphan segment files from a crashed save or a dropped entry are not referenced by any valid entry.
        if let Err(err) = segments.gc() {
            crate::say!("slot archive: segment cleanup at open failed: {err:#}");
        }

        let manifest_bytes = fs::metadata(&manifest).map(|meta| meta.len()).unwrap_or(0);
        let mut archive = Archive {
            cfg,
            entries,
            segments,
            stats: ArchiveStats::default(),
            leases: Arc::new(Mutex::new(BTreeMap::new())),
            ttl_ms: 0,
            per_key_bytes: 0,
            pin_bytes: 0,
            pending_touches: BTreeMap::new(),
            journal_bytes,
            journal_records,
            manifest_bytes,
            last_touch_flush_ms: now_ms(),
        };
        if changed {
            archive.compact()?;
        }
        Ok(archive)
    }

    pub fn config(&self) -> &ArchiveConfig {
        &self.cfg
    }

    /// The content-addressed KV segment store (inspection/GC; never exposed by the HTTP API).
    pub fn segments(&self) -> &SegmentStore {
        &self.segments
    }

    /// Set the lifecycle policy; zero disables each limit. Not persisted: the server re-applies its flags on
    /// startup. `(ttl_ms, per_key_bytes, pin_bytes)`.
    pub fn set_policy(&mut self, ttl_ms: u64, per_key_bytes: u64, pin_bytes: u64) {
        self.ttl_ms = ttl_ms;
        self.per_key_bytes = per_key_bytes;
        self.pin_bytes = pin_bytes;
    }

    /// The current `(ttl_ms, per_key_bytes, pin_bytes)` policy.
    pub fn policy(&self) -> (u64, u64, u64) {
        (self.ttl_ms, self.per_key_bytes, self.pin_bytes)
    }

    /// Pin or unpin an entry for GC/TTL protection and persist the manifest.
    ///
    /// When `pin_bytes > 0`, pinning is refused if the pinned total would exceed the cap; unpinning always
    /// works. Pinning an already-pinned (or unpinning a non-pinned) entry is a no-op that still succeeds.
    pub fn set_pinned(&mut self, id: &str, pinned: bool) -> Result<()> {
        let entry = self
            .entries
            .get(id)
            .with_context(|| format!("slot archive: no entry {id} to pin"))?;
        if entry.pinned == pinned {
            return Ok(());
        }
        if pinned && self.pin_bytes > 0 {
            let after = self.pinned_bytes().saturating_add(entry.bytes);
            if after > self.pin_bytes {
                bail!(
                    "slot archive: pinning {id} would use {after} pinned bytes, over the {}-byte pin cap",
                    self.pin_bytes
                );
            }
        }
        let entry = self.entries.get_mut(id).context("entry disappeared while pinning")?;
        entry.pinned = pinned;
        let record = JournalOp::Pin { v: JOURNAL_VERSION, id: id.to_string(), pinned };
        if let Err(err) = self.append_journal(&record) {
            // Revert the in-memory change when it could not be journaled; the compacted snapshot still has the old
            // value, so the archive stays consistent.
            if let Some(entry) = self.entries.get_mut(id) {
                entry.pinned = !pinned;
            }
            return Err(err);
        }
        self.maybe_compact();
        Ok(())
    }

    /// Logical bytes held by pinned entries.
    pub fn pinned_bytes(&self) -> u64 {
        self.entries.values().filter(|e| e.pinned).map(|e| e.bytes).sum()
    }

    pub fn stats(&self) -> ArchiveStats {
        let mut s = self.stats.clone();
        s.entries = self.entries.len();
        s.bytes = self.entries.values().map(|e| e.bytes).sum();
        s.budget_bytes = self.cfg.budget_bytes;
        s
    }

    /// Entries newest-used first. `owner` filters by conversation owner; `limit == 0` means no limit.
    pub fn list(&self, owner: Option<u64>, limit: usize) -> Vec<ArchiveEntry> {
        let mut out: Vec<ArchiveEntry> = self
            .entries
            .values()
            .filter(|e| owner.is_none_or(|o| e.owner == o))
            .cloned()
            .collect();
        out.sort_by(|a, b| b.last_used_ms.cmp(&a.last_used_ms).then_with(|| a.id.cmp(&b.id)));
        if limit > 0 {
            out.truncate(limit);
        }
        out
    }

    /// Reserve an entry id and a final path. Returns [`SaveSlot::Existing`] when the same content is already
    /// archived (the caller should only [`Archive::touch`] it).
    pub fn prepare_save(&mut self, owner: u64, tokens: &[u32], fingerprint: u64, reason: &str) -> Result<SaveSlot> {
        let _ = reason;
        if tokens.len() < self.cfg.min_tokens {
            bail!(
                "slot archive: history of {} tokens is below the min_tokens limit {}",
                tokens.len(),
                self.cfg.min_tokens
            );
        }

        let id = entry_id(fingerprint, owner, tokens);
        if let Some(entry) = self.entries.get_mut(&id) {
            if entry.file.exists() {
                // Dedupe: the same content is already on disk. Update the in-memory LRU position; `touch`
                // (or the next manifest write) persists it.
                let used = now_ms();
                entry.last_used_ms = used;
                self.note_touch(&id, used);
                return self.entries.get(&id).cloned().map(SaveSlot::Existing).ok_or_else(|| {
                    anyhow::anyhow!("slot archive entry disappeared while preparing a save")
                });
            }
            // A stale row whose file disappeared: forget it (releasing v2 segment references) and let the caller
            // write a new file.
            let stale = entry.clone();
            self.entries.remove(&id);
            let ids: Vec<String> = stale.segments.into_iter().map(|segment| segment.id).collect();
            self.release_segments(&ids);
            let missing = stale.state_file.as_deref().unwrap_or(&stale.file).to_path_buf();
            let _ = fs::remove_file(missing);
        }

        let shard = id.get(..2).context("archive entry id is too short")?;
        let shard_dir = self.cfg.dir.join(shard);
        ensure_dir(&shard_dir)?;
        let path = shard_dir.join(format!("{id}.slot"));
        Ok(SaveSlot::New { id, path })
    }

    /// Register the file written at `path`, enforce the budget/lineage policy, and persist the manifest.
    ///
    /// Lineage pruning is deliberately lazy: a commit never deletes an ancestor just because `versions`
    /// is 1. Ancestors are only removed by [`Archive::gc`] (or by the budget-pressure branch below), so a
    /// divergence entry and its continuation can coexist while the archive has room.
    pub fn commit_save(&mut self, id: &str, path: &Path, owner: u64, tokens: &[u32], fingerprint: u64, reason: &str) -> Result<ArchiveEntry> {
        if !path.starts_with(&self.cfg.dir) {
            bail!(
                "slot archive: refusing to register {} outside the archive directory {}",
                path.display(),
                self.cfg.dir.display()
            );
        }
        let meta = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
        if !meta.is_file() {
            bail!("slot archive: {} is not a regular file", path.display());
        }
        let (header, file_tokens) =
            read_slot(path).with_context(|| format!("reading the header of {}", path.display()))?;
        if header.fingerprint != fingerprint {
            bail!(
                "slot archive: {} has fingerprint {:016x}, expected {:016x}",
                path.display(),
                header.fingerprint,
                fingerprint
            );
        }
        if header.owner != owner {
            bail!(
                "slot archive: {} has owner {}, expected {}",
                path.display(),
                header.owner,
                owner
            );
        }
        if header.n_tokens != tokens.len() as u64 {
            bail!(
                "slot archive: {} has {} tokens, expected {}",
                path.display(),
                header.n_tokens,
                tokens.len()
            );
        }
        if file_tokens != tokens {
            bail!("slot archive: {} token history does not match the archived tokens", path.display());
        }

        let prefix_hash = prefix_hash(tokens);
        let expected = id_for(fingerprint, owner, tokens.len(), prefix_hash);
        if id != expected {
            bail!("slot archive: entry id {id} does not match its content ({expected})");
        }

        let bytes = meta.len();
        if let Err(err) = self.reserve_budget(id, owner, bytes, fingerprint) {
            let _ = fs::remove_file(path);
            return Err(err);
        }

        // Lineage is computed after budget enforcement, so it never points at an entry evicted above.
        let parent = self
            .longest_prefix(owner, tokens, fingerprint)
            .filter(|e| e.n_tokens < tokens.len() && e.id != id)
            .map(|e| e.id);

        let now = now_ms();
        let (created_ms, pinned) = self
            .entries
            .get(id)
            .map_or((now, false), |e| (e.created_ms, e.pinned));
        let entry = ArchiveEntry {
            id: id.to_string(),
            file: path.to_path_buf(),
            state_file: None,
            segments: Vec::new(),
            bytes,
            n_tokens: tokens.len(),
            owner,
            fingerprint,
            prefix_hash,
            created_ms,
            last_used_ms: now,
            parent,
            pinned,
            reason: reason.to_string(),
            format: crate::slot_file::VERSION,
        };
        self.entries.insert(id.to_string(), entry.clone());
        let record = JournalOp::Add { v: JOURNAL_VERSION, entry: self.manifest_row(&entry) };
        if let Err(err) = self.append_journal(&record) {
            self.entries.remove(id);
            let _ = fs::remove_file(path);
            return Err(err);
        }
        self.maybe_compact();
        Ok(entry)
    }

    /// Enforce the per-owner quota and the global budget for a new entry of `bytes`.
    ///
    /// The caller owns the data cleanup (remove the state file, release the segments); this only decides whether the
    /// entry fits and evicts other entries as the policy allows.
    fn reserve_budget(&mut self, id: &str, owner: u64, bytes: u64, fingerprint: u64) -> Result<()> {
        let limit = self.cfg.budget_bytes.saturating_sub(RESERVE_BYTES);
        if bytes > limit {
            bail!(
                "slot archive: a {bytes}-byte entry does not fit in the {limit}-byte budget ({} bytes reserved)",
                RESERVE_BYTES
            );
        }

        // Per-key quota: evict this owner's own LRU entries first so one conversation cannot crowd every
        // other owner out. Pinned and leased entries are protected, and another owner is never touched here.
        if self.per_key_bytes > 0 {
            if bytes > self.per_key_bytes {
                bail!(
                    "slot archive: a {bytes}-byte entry exceeds the {}-byte per-owner quota",
                    self.per_key_bytes
                );
            }
            let mut owner_used: u64 = self
                .entries
                .values()
                .filter(|e| e.owner == owner && e.id != id)
                .map(|e| e.bytes)
                .sum();
            if owner_used.saturating_add(bytes) > self.per_key_bytes {
                let mut candidates: Vec<(u64, String)> = self
                    .entries
                    .values()
                    .filter(|e| e.owner == owner && e.id != id && !e.pinned)
                    .map(|e| (e.last_used_ms, e.id.clone()))
                    .collect();
                candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                for (_, candidate) in candidates {
                    if owner_used.saturating_add(bytes) <= self.per_key_bytes {
                        break;
                    }
                    if let Some(freed) = self.evict_entry(&candidate) {
                        owner_used = owner_used.saturating_sub(freed);
                    }
                }
                if owner_used.saturating_add(bytes) > self.per_key_bytes {
                    bail!(
                        "slot archive: owner {owner} is over its {}-byte quota and only pinned or leased \
                         entries remain to evict",
                        self.per_key_bytes
                    );
                }
            }
        }

        // Global budget: evict foreign (unusable for this fingerprint) files first, then expired entries,
        // then the oldest unpinned entries.
        let mut used = self.used_bytes();
        if used.saturating_add(bytes) > limit {
            // Only under pressure: apply the `versions` policy before the LRU pass.
            self.prune_all_lineages();
            let now = now_ms();
            used = self.used_bytes();
            let mut candidates: Vec<(u8, u64, String)> = self
                .entries
                .values()
                .filter(|e| !e.pinned && e.id != id)
                .map(|e| {
                    let rank = if e.fingerprint != fingerprint {
                        0
                    } else if self.ttl_expired(e, now) {
                        1
                    } else {
                        2
                    };
                    (rank, e.last_used_ms, e.id.clone())
                })
                .collect();
            candidates.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            for (rank, _, candidate) in candidates {
                if used.saturating_add(bytes) <= limit {
                    break;
                }
                if let Some(freed) = self.evict_entry(&candidate) {
                    used = used.saturating_sub(freed);
                    if rank == 1 {
                        self.stats.expired = self.stats.expired.saturating_add(1);
                    }
                }
            }
            if used.saturating_add(bytes) > limit {
                bail!(
                    "slot archive is full: a {bytes}-byte entry does not fit under the {limit}-byte limit \
                     even after evicting every unpinned entry"
                );
            }
        }
        Ok(())
    }

    /// Save the idle slot `slot` as a v2 segmented entry: `SEGMENT_TOKENS`-sized content-addressed KV segments plus
    /// one `STRSTATE` file holding the token list and the recurrent snapshot.
    ///
    /// `owner` is the conversation owner (`GpuModel::slot_owner`), as in [`Archive::commit_save`]. The caller is
    /// responsible for the slot being idle and for holding whatever lock keeps it so; this method only validates the
    /// history (`pos == history.len()`, non-empty). The same content already archived is touched, not rewritten.
    ///
    /// On failure every reference taken by this call is released and the state file is removed. Segment bytes are
    /// counted per new file only; an entry that shares segments with an older entry does not pay for them again.
    pub fn save_segmented(
        &mut self,
        g: &mut GpuModel<'_>,
        slot: usize,
        owner: u64,
        reason: &str,
    ) -> Result<ArchiveEntry> {
        if slot >= g.slots() {
            bail!("slot archive: slot {slot} of {}", g.slots());
        }
        let tokens = g.slot_tokens(slot).to_vec();
        if tokens.is_empty() {
            bail!("slot archive: slot {slot} has no token history to save");
        }
        if g.slot_pos(slot) != tokens.len() {
            bail!(
                "slot archive: slot {slot}: pos {} but {} tokens in its history",
                g.slot_pos(slot),
                tokens.len()
            );
        }
        if tokens.len() < self.cfg.min_tokens {
            bail!(
                "slot archive: history of {} tokens is below the min_tokens limit {}",
                tokens.len(),
                self.cfg.min_tokens
            );
        }
        let fingerprint = g.slot_fingerprint();
        let prefix_hash = prefix_hash(&tokens);
        let id = id_for(fingerprint, owner, tokens.len(), prefix_hash);

        // Same content is already archived: touch it. Only a complete v2 entry counts; a stale or v1 row with the
        // same id is removed and replaced (a lease blocks the replacement, never the touch of a live v2 entry).
        if let Some(existing) = self.entries.get(&id).cloned() {
            let reusable = existing.format == 2
                && existing.n_tokens == tokens.len()
                && self.entry_data_present(&existing)
                && existing.state_file.as_deref().is_some_and(|path| {
                    read_state_header(path).is_ok_and(|(header, saved)| {
                        header.fingerprint == fingerprint
                            && header.owner == owner
                            && header.n_tokens == tokens.len()
                            && saved == tokens
                    })
                });
            if reusable {
                self.touch(&id);
                return self
                    .entries
                    .get(&id)
                    .cloned()
                    .context("slot archive entry disappeared while touching it");
            }
            if self.leases(&id) > 0 {
                bail!("slot archive: entry {id} is leased; cannot replace it with a segmented save");
            }
            self.delete(&id)?;
        }

        // Write the segments first: a crash leaves only orphan segment files, never an entry whose segments are
        // missing. Track every segment this entry takes a reference on so a later failure can release it again.
        let mut refs = Vec::new();
        let mut taken: Vec<String> = Vec::new();
        let mut new_bytes = 0u64;
        let mut parent: Option<String> = None;
        let mut write_segments = || -> Result<()> {
            for (index, chunk) in tokens.chunks(SEGMENT_TOKENS).enumerate() {
                let start = index * SEGMENT_TOKENS;
                let end = start + chunk.len();
                let segment_id = slot_segments::segment_id(fingerprint, parent.as_deref(), chunk);
                let existed = self.segments.contains(&segment_id);
                let mut payload = Vec::new();
                g.save_kv_segment(slot, start, end, &mut payload)?;
                let segment = self.segments.put(fingerprint, parent.as_deref(), chunk, &payload)?;
                if segment.id != segment_id {
                    bail!(
                        "slot archive: segment store returned id {}, expected {segment_id}",
                        segment.id
                    );
                }
                if !existed {
                    new_bytes = new_bytes.saturating_add(segment.bytes);
                }
                taken.push(segment.id.clone());
                refs.push(SegmentRef { id: segment.id.clone(), start, end });
                parent = Some(segment.id);
            }
            Ok(())
        };
        if let Err(err) = write_segments() {
            self.release_segments(&taken);
            return Err(err);
        }

        // The recurrent snapshot is the per-entry part; the token list and the header live in the same file.
        let mut state_payload = Vec::new();
        let state_written = (|| -> Result<PathBuf> {
            g.save_recurrent_to(slot, &mut state_payload)?;
            let path = state_path(&self.cfg.dir, &id)?;
            write_state_file_atomic(
                &path,
                &StateHeader {
                    fingerprint,
                    owner,
                    rope_offset: g.slot_rope_offset(slot),
                    n_tokens: tokens.len(),
                },
                &refs,
                &tokens,
                &state_payload,
            )?;
            Ok(path)
        })();
        let state_path = match state_written {
            Ok(path) => path,
            Err(err) => {
                self.release_segments(&taken);
                return Err(err);
            }
        };
        let bytes = match fs::metadata(&state_path) {
            Ok(meta) if meta.is_file() => meta.len().saturating_add(new_bytes),
            Ok(_) => {
                let _ = fs::remove_file(&state_path);
                self.release_segments(&taken);
                bail!("slot archive: {} is not a regular file", state_path.display());
            }
            Err(err) => {
                let _ = fs::remove_file(&state_path);
                self.release_segments(&taken);
                return Err(err).with_context(|| format!("stat {}", state_path.display()));
            }
        };

        if let Err(err) = self.reserve_budget(&id, owner, bytes, fingerprint) {
            let _ = fs::remove_file(&state_path);
            self.release_segments(&taken);
            return Err(err);
        }

        // Lineage is computed after budget enforcement, so it never points at an entry evicted above.
        let parent_entry = self
            .longest_prefix(owner, &tokens, fingerprint)
            .filter(|e| e.n_tokens < tokens.len() && e.id != id)
            .map(|e| e.id);
        let now = now_ms();
        let entry = ArchiveEntry {
            id: id.clone(),
            file: state_path.clone(),
            state_file: Some(state_path.clone()),
            segments: refs,
            bytes,
            n_tokens: tokens.len(),
            owner,
            fingerprint,
            prefix_hash,
            created_ms: now,
            last_used_ms: now,
            parent: parent_entry,
            pinned: false,
            reason: reason.to_string(),
            format: 2,
        };
        self.entries.insert(id.clone(), entry.clone());
        let record = JournalOp::Add { v: JOURNAL_VERSION, entry: self.manifest_row(&entry) };
        if let Err(err) = self.append_journal(&record) {
            self.entries.remove(&id);
            let _ = fs::remove_file(&state_path);
            self.release_segments(&taken);
            return Err(err);
        }
        self.maybe_compact();
        // The slot is reusable at this history, exactly as after a v1 synchronous save (`save_state` cleared the
        // marker while writing the recurrent payload).
        g.mark_slot_saved(slot, &tokens, owner);
        Ok(entry)
    }

    /// Restore a v2 entry into the idle slot `slot` and return the number of restored tokens.
    ///
    /// The entry's recorded `owner` (the conversation owner) is installed; the caller's request owner is not used.
    /// `allow_evict` is deliberately not a parameter: an automatic restore never evicts a non-empty slot under a
    /// unified budget. Any error clears the target slot before returning, so the server can fall back to prefill.
    pub fn restore_segmented(
        &mut self,
        g: &mut GpuModel<'_>,
        slot: usize,
        entry: &ArchiveEntry,
    ) -> Result<usize> {
        let result = self.restore_segmented_inner(g, slot, entry);
        if result.is_err() {
            // A partially installed state must never be used, and an automatic restore failure means prefill.
            if let Err(err) = g.erase_slot(slot) {
                crate::say!("slot archive: clearing slot {slot} after a failed segmented restore: {err:#}");
            }
        }
        result
    }

    fn restore_segmented_inner(
        &mut self,
        g: &mut GpuModel<'_>,
        slot: usize,
        entry: &ArchiveEntry,
    ) -> Result<usize> {
        if entry.format != 2 {
            bail!(
                "slot archive: entry {} is format {}, not a segmented (format 2) entry",
                entry.id,
                entry.format
            );
        }
        if entry.n_tokens == 0 {
            bail!("slot archive: entry {} has no tokens", entry.id);
        }
        if entry.n_tokens > g.ctx() {
            bail!(
                "slot archive: entry {} has {} tokens, --ctx is {}",
                entry.id,
                entry.n_tokens,
                g.ctx()
            );
        }
        let fingerprint = g.slot_fingerprint();
        if entry.fingerprint != fingerprint {
            bail!(
                "slot archive: entry {} fingerprint {:016x} does not match this model {:016x}",
                entry.id,
                entry.fingerprint,
                fingerprint
            );
        }
        let state_file = entry
            .state_file
            .as_deref()
            .context("segmented entry has no state file")?;
        let (header, state_segments, tokens, state_payload) = read_state_file(state_file)?;
        if header.fingerprint != entry.fingerprint {
            bail!(
                "slot archive: {} header fingerprint {:016x} does not match the entry {:016x}",
                state_file.display(),
                header.fingerprint,
                entry.fingerprint
            );
        }
        if header.owner != entry.owner {
            bail!(
                "slot archive: {} header owner {} does not match the entry owner {}",
                state_file.display(),
                header.owner,
                entry.owner
            );
        }
        if header.n_tokens != entry.n_tokens || tokens.len() != entry.n_tokens {
            bail!(
                "slot archive: {} has {} tokens, the entry has {}",
                state_file.display(),
                tokens.len(),
                entry.n_tokens
            );
        }
        // A v2 state file carries the ordered refs; a legacy v1 state file does not, so the manifest row is used.
        if !state_segments.is_empty() && state_segments != entry.segments {
            bail!(
                "slot archive: {} segment refs do not match the manifest row",
                state_file.display()
            );
        }
        if !segments_cover(&entry.segments, entry.n_tokens) {
            bail!("slot archive: entry {} segment ranges are not a contiguous cover of its history", entry.id);
        }
        validate_segment_chain(entry.fingerprint, &entry.segments, &tokens)
            .with_context(|| format!("validating the segment chain of {}", entry.id))?;

        // Capacity/unified growth first; never evicts (automatic restore). This does not touch the file.
        g.prepare_segmented_restore(slot, tokens.len(), false)?;
        for segment in &entry.segments {
            let payload = self.segments.get_checked(&segment.id, &tokens[segment.start..segment.end])?;
            let mut reader = &payload[..];
            g.restore_kv_segment(slot, segment.start, segment.end, &mut reader)?;
        }
        let mut payload = &state_payload[..];
        let consumed = g.restore_recurrent_from(slot, &mut payload)?;
        if consumed != state_payload.len() as u64 {
            bail!(
                "slot archive: {} recurrent payload is {} bytes, {} were consumed",
                state_file.display(),
                state_payload.len(),
                consumed
            );
        }
        g.finish_segmented_restore(slot, &tokens, entry.owner, header.rope_offset)?;
        Ok(tokens.len())
    }

    /// Drop every reference in `ids`, logging (but not failing) a manifest write error: an over-counted ref is
    /// safe, a dangling entry is not.
    fn release_segments(&mut self, ids: &[String]) {
        for id in ids.iter().rev() {
            if let Err(err) = self.segments.unref(id) {
                crate::say!("slot archive: releasing segment {id}: {err:#}");
            }
        }
    }

    /// Longest entry whose token history is an exact prefix of `tokens[..keep]`, with the same owner and
    /// fingerprint and `n_tokens <= ctx`. Increments the hit/miss counters.
    pub fn lookup(&mut self, owner: u64, tokens: &[u32], keep: usize, fingerprint: u64, ctx: usize) -> Result<Option<ArchiveEntry>> {
        let found = self.lookup_match(owner, tokens, keep, fingerprint, ctx)?;
        if found.is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        Ok(found)
    }

    /// Same exact lookup as [`Archive::lookup`], but also takes a transient [`Lease`] on the match before
    /// returning it. The lease protects the entry from `DELETE` and GC until the guard is dropped.
    ///
    /// A miss increments the same counter as [`Archive::lookup`].
    pub fn lookup_leased(&mut self, owner: u64, tokens: &[u32], keep: usize, fingerprint: u64, ctx: usize) -> Result<Option<(ArchiveEntry, Lease)>> {
        let Some(entry) = self.lookup_match(owner, tokens, keep, fingerprint, ctx)? else {
            self.stats.misses += 1;
            return Ok(None);
        };
        self.stats.hits += 1;
        let lease = self.add_lease(&entry.id);
        Ok(Some((entry, lease)))
    }

    /// Take a lease on an existing entry. `None` when no entry with `id` is currently indexed.
    pub fn lease(&mut self, id: &str) -> Option<Lease> {
        if !self.entries.contains_key(id) {
            return None;
        }
        Some(self.add_lease(id))
    }

    /// Number of live leases on `id` (0 when the entry is unleased or unknown).
    pub fn leases(&self, id: &str) -> usize {
        self.lease_table().get(id).copied().unwrap_or(0)
    }

    fn lease_table(&self) -> MutexGuard<'_, BTreeMap<String, usize>> {
        self.leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Register one lease. Drop ignores missing ids, so registering before the entry is inserted is harmless.
    fn add_lease(&self, id: &str) -> Lease {
        *self.lease_table().entry(id.to_string()).or_insert(0) += 1;
        Lease { id: id.to_string(), leases: Arc::clone(&self.leases) }
    }

    /// The shared body of [`Archive::lookup`] and [`Archive::lookup_leased`]: exact prefix match and LRU
    /// touch, without hit/miss accounting.
    fn lookup_match(&mut self, owner: u64, tokens: &[u32], keep: usize, fingerprint: u64, ctx: usize) -> Result<Option<ArchiveEntry>> {
        let n_max = keep.min(ctx).min(tokens.len());
        let mut candidates: Vec<(usize, String)> = self
            .entries
            .values()
            .filter(|e| e.owner == owner && e.fingerprint == fingerprint && e.n_tokens <= n_max)
            .map(|e| (e.n_tokens, e.id.clone()))
            .collect();
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        // Hash each distinct prefix length once; the hash is only a cheap candidate filter.
        let mut hashes: BTreeMap<usize, u64> = BTreeMap::new();
        for &(n, _) in &candidates {
            hashes.entry(n).or_insert_with(|| prefix_hash(&tokens[..n]));
        }

        for (n, id) in candidates {
            let Some(entry) = self.entries.get(&id) else { continue };
            if entry.prefix_hash != hashes[&n] {
                continue;
            }
            // Hash match: prove it with the saved token list (the file header for v1, the state file for v2).
            let exact = match read_entry_tokens(entry) {
                Ok(saved) => saved.len() == n && saved == tokens[..n],
                Err(err) => {
                    crate::say!("slot archive: cannot read the tokens of {}: {err:#}", entry.id);
                    false
                }
            };
            if !exact {
                continue;
            }
            let used = now_ms();
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.last_used_ms = used;
                let cloned = entry.clone();
                self.note_touch(&id, used);
                return Ok(Some(cloned));
            }
        }

        Ok(None)
    }

    /// Longest existing entry whose history is a hash-prefix of `tokens` (same owner/fingerprint). Hash only:
    /// used for the `min-delta` heuristic, never to authorize a restore.
    pub fn longest_prefix(&self, owner: u64, tokens: &[u32], fingerprint: u64) -> Option<ArchiveEntry> {
        let mut hashes: BTreeMap<usize, u64> = BTreeMap::new();
        let mut best: Option<&ArchiveEntry> = None;
        for entry in self
            .entries
            .values()
            .filter(|e| e.owner == owner && e.fingerprint == fingerprint && e.n_tokens <= tokens.len())
        {
            let hash = *hashes
                .entry(entry.n_tokens)
                .or_insert_with(|| prefix_hash(&tokens[..entry.n_tokens]));
            if entry.prefix_hash == hash
                && best.is_none_or(|b| entry.n_tokens > b.n_tokens)
            {
                best = Some(entry);
            }
        }
        best.cloned()
    }

    /// Mark an entry as used now (LRU). No-op when it disappeared.
    ///
    /// This is cheap: it updates memory and a bounded pending-touch map; the journal is written only when the
    /// batch fills, the flush interval elapses, or [`Archive::flush`]/[`Archive::compact`]/GC runs. A crash can
    /// therefore lose recent LRU hints, never an entry.
    pub fn touch(&mut self, id: &str) {
        let used = now_ms();
        let Some(entry) = self.entries.get_mut(id) else { return };
        entry.last_used_ms = used;
        self.note_touch(id, used);
    }

    /// Delete an entry, its state/slot file and (for v2) its segment references. Returns the freed logical bytes.
    ///
    /// An entry with live [`Lease`]s is not deleted: the caller gets an error and should retry after the restore
    /// or save that holds it completes. Missing files are tolerated; a state file that cannot be unlinked keeps the
    /// entry so a later GC can retry.
    pub fn delete(&mut self, id: &str) -> Result<u64> {
        let Some(entry) = self.entries.get(id).cloned() else { return Ok(0) };
        let held = self.leases(id);
        if held > 0 {
            bail!(
                "slot archive: entry {id} is leased by {held} in-flight operation(s); retry after it completes"
            );
        }
        self.remove_entry_data(&entry, true)?;
        let record = JournalOp::Delete { v: JOURNAL_VERSION, id: id.to_string() };
        if let Err(err) = self.append_journal(&record) {
            // The data is already gone; a stale journal row is harmless because startup validation drops it.
            crate::say!("slot archive: journaling the deletion of {id}: {err:#}");
        }
        self.entries.remove(id);
        self.maybe_compact();
        Ok(entry.bytes)
    }

    /// Evict unusable entries first, then (only when over the target) prune lineages per `versions` and evict
    /// unpinned LRU entries until `used_bytes <= budget - RESERVE_BYTES`.
    pub fn gc(&mut self) -> Result<GcReport> {
        let limit = self.cfg.budget_bytes.saturating_sub(RESERVE_BYTES);
        let mut report = GcReport::default();
        let mut changed = false;

        // TTL: remove unpinned idle entries before any budget work. Leased entries are skipped by
        // `evict_entry`; a future `last_used_ms` saturates to a zero age.
        if self.ttl_ms > 0 {
            let now = now_ms();
            let expired: Vec<String> = self
                .entries
                .values()
                .filter(|e| self.ttl_expired(e, now))
                .map(|e| e.id.clone())
                .collect();
            for id in expired {
                if let Some(freed) = self.evict_entry(&id) {
                    report.evicted += 1;
                    report.expired += 1;
                    report.freed_bytes = report.freed_bytes.saturating_add(freed);
                    self.stats.expired = self.stats.expired.saturating_add(1);
                    changed = true;
                }
            }
        }

        // Unusable: the manifest row points at a missing file (or, for v2, a missing segment). (Foreign
        // fingerprints are not identifiable without a current fingerprint argument, so they are only evicted by
        // the LRU pass.)
        let ids: Vec<String> = self.entries.keys().cloned().collect();
        for id in ids {
            if !self.entry_data_present(&self.entries[&id]) {
                if let Some(freed) = self.evict_entry(&id) {
                    report.evicted += 1;
                    report.freed_bytes = report.freed_bytes.saturating_add(freed);
                    changed = true;
                }
            }
        }

        let mut used = self.used_bytes();
        if used > limit {
            // Lazy lineage policy: `versions` is applied only when the archive is over its target.
            let (pruned, pruned_bytes) = self.prune_all_lineages();
            if pruned > 0 {
                report.evicted += pruned;
                report.freed_bytes = report.freed_bytes.saturating_add(pruned_bytes);
                changed = true;
            }
            used = self.used_bytes();
            let mut candidates: Vec<(u64, String)> = self
                .entries
                .values()
                .filter(|e| !e.pinned)
                .map(|e| (e.last_used_ms, e.id.clone()))
                .collect();
            candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            for (_, id) in candidates {
                if used <= limit {
                    break;
                }
                if let Some(freed) = self.evict_entry(&id) {
                    report.evicted += 1;
                    report.freed_bytes = report.freed_bytes.saturating_add(freed);
                    used = used.saturating_sub(freed);
                    changed = true;
                }
            }
        }

        if changed || !self.pending_touches.is_empty() || self.journal_records > 0 {
            // GC is a natural compaction point: it settles the LRU touches and folds the journal into the snapshot.
            self.compact()?;
        }
        // Entry references are gone; drop the zero-ref segment files in the same pass. Rows still referenced by a
        // v2 entry stay, so a shared segment survives until its last entry goes.
        if let Err(err) = self.segments.gc() {
            crate::say!("slot archive: segment GC failed: {err:#}");
        }
        Ok(report)
    }

    pub fn note_restore(&mut self) {
        self.stats.restores += 1;
    }

    pub fn note_failure(&mut self) {
        self.stats.failures += 1;
    }

    fn used_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.bytes).sum()
    }

    /// True when `entry` is subject to an enabled TTL and has been idle longer than it. Pinned entries never
    /// expire; a `last_used_ms` in the future (clock skew) is treated as just used.
    fn ttl_expired(&self, entry: &ArchiveEntry, now: u64) -> bool {
        self.ttl_ms > 0
            && !entry.pinned
            && now.saturating_sub(entry.last_used_ms) > self.ttl_ms
    }

    /// Remove an entry, its file(s) and its v2 segment references. `None` when the state/slot file could not be
    /// unlinked (the entry is kept) or when the entry is leased (GC must never pull a file out from under an
    /// in-flight restore). Segment reference releases are best effort: an over-counted ref leaks disk but never
    /// leaves an entry pointing at a missing segment.
    fn evict_entry(&mut self, id: &str) -> Option<u64> {
        if self.leases(id) > 0 {
            return None;
        }
        let entry = self.entries.get(id)?.clone();
        if let Err(err) = self.remove_entry_data(&entry, false) {
            crate::say!("slot archive: keeping {id}: {err:#}");
            return None;
        }
        self.entries.remove(id);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        Some(entry.bytes)
    }

    /// Release a v2 entry's segment references, remove its state file, and then remove a v1 entry's slot file.
    ///
    /// For v2 the order is segments first, then the state file: a crash between the two leaves a state file
    /// without references (harmless, GC'd/adopted later), never an entry whose segments are already gone. A state
    /// file that cannot be unlinked is logged and left as an orphan; the entry is still removed, because it has
    /// already released its segment references and must not survive pointing at them. A v1 unlink error in `strict`
    /// mode (explicit DELETE) keeps the entry, as before.
    fn remove_entry_data(&mut self, entry: &ArchiveEntry, strict: bool) -> Result<()> {
        if entry.format == 2 {
            for segment in entry.segments.iter().rev() {
                if let Err(err) = self.segments.unref(&segment.id) {
                    crate::say!("slot archive: releasing segment {} of {}: {err:#}", segment.id, entry.id);
                }
            }
            let path = entry.state_file.as_deref().unwrap_or(&entry.file);
            remove_file_if_present(path, false)?;
        } else {
            remove_file_if_present(&entry.file, strict)?;
        }
        Ok(())
    }

    /// True when every file an entry needs is present: its slot/state file and, for v2, each segment row and file.
    fn entry_data_present(&self, entry: &ArchiveEntry) -> bool {
        let file = if entry.format == 2 {
            entry.state_file.as_deref().unwrap_or(&entry.file)
        } else {
            &entry.file
        };
        if !fs::metadata(file).map(|meta| meta.is_file()).unwrap_or(false) {
            return false;
        }
        if entry.format == 2 {
            return entry.segments.iter().all(|segment| {
                self.segments
                    .segment(&segment.id)
                    .and_then(|row| fs::metadata(&row.file).ok())
                    .is_some_and(|meta| meta.is_file())
            });
        }
        true
    }

    /// Apply the `versions` policy to the lineage of `leaf_id`.
    ///
    /// `versions <= 1` keeps only the leaf: walk up the parent chain and delete every ancestor that has no
    /// other child. For `versions > 1`, the newest `versions` entries of the lineage are kept and the chain
    /// above them is pruned the same way. The walk stops at the first ancestor that is shared with another
    /// branch, because deleting it would corrupt that branch's lineage.
    ///
    /// This is the lazy lineage policy: it runs only from [`Archive::gc`] and the budget-pressure branch of
    /// [`Archive::commit_save`], never from a normal commit. Returns the number of entries removed and their
    /// logical bytes.
    fn prune_lineage(&mut self, leaf_id: &str) -> (usize, u64) {
        let versions = self.cfg.versions.max(1);
        let mut evicted = 0usize;
        let mut freed = 0u64;
        let mut child = leaf_id.to_string();
        let mut next = self.entries.get(leaf_id).and_then(|e| e.parent.clone());
        let mut depth = 1usize;
        while let Some(parent_id) = next {
            let Some(parent) = self.entries.get(&parent_id) else { break };
            next = parent.parent.clone();
            let shared = self
                .entries
                .values()
                .any(|e| e.id != child && e.parent.as_deref() == Some(parent_id.as_str()));
            if shared {
                break;
            }
            if depth >= versions {
                if let Some(bytes) = self.evict_entry(&parent_id) {
                    evicted += 1;
                    freed = freed.saturating_add(bytes);
                }
            }
            child = parent_id;
            depth += 1;
        }
        (evicted, freed)
    }

    /// Apply the `versions` policy to every lineage. Only used when the budget is under pressure.
    fn prune_all_lineages(&mut self) -> (usize, u64) {
        let ids: Vec<String> = self.entries.keys().cloned().collect();
        let mut evicted = 0usize;
        let mut freed = 0u64;
        for id in ids {
            if self.entries.contains_key(&id) {
                let (count, bytes) = self.prune_lineage(&id);
                evicted += count;
                freed = freed.saturating_add(bytes);
            }
        }
        (evicted, freed)
    }

    /// Build one manifest row from an in-memory entry.
    fn manifest_row(&self, e: &ArchiveEntry) -> ManifestEntry {
        ManifestEntry {
            id: e.id.clone(),
            file: relative_string(&self.cfg.dir, &e.file),
            state_file: e
                .state_file
                .as_deref()
                .map(|path| relative_string(&self.cfg.dir, path)),
            segments: e.segments.clone(),
            bytes: e.bytes,
            n_tokens: e.n_tokens,
            owner: e.owner,
            fingerprint: format!("{:016x}", e.fingerprint),
            prefix_hash: format!("{:016x}", e.prefix_hash),
            created_ms: e.created_ms,
            last_used_ms: e.last_used_ms,
            parent: e.parent.clone(),
            pinned: e.pinned,
            reason: e.reason.clone(),
            format: e.format,
        }
    }

    /// Append one record to `manifest.log` and fsync it. A partial write is truncated back to the previous
    /// length, so a reader never sees a malformed record in the middle of the journal.
    fn append_journal(&mut self, op: &JournalOp) -> Result<()> {
        ensure_dir(&self.cfg.dir)?;
        let path = self.cfg.dir.join(JOURNAL);
        let mut line = serde_json::to_vec(op).context("serializing the archive journal record")?;
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
            file.write_all(&line).context("writing the archive journal")?;
            file.sync_data().context("syncing the archive journal")?;
            Ok(())
        })();
        if let Err(err) = write {
            let _ = file.set_len(before);
            return Err(err);
        }
        drop(file);
        if !existed {
            fsync_dir(&self.cfg.dir);
        }
        self.journal_bytes = before.saturating_add(line.len() as u64);
        self.journal_records += 1;
        Ok(())
    }

    /// Truncate `manifest.log` after its records have been folded into `manifest.json`. Best effort: a failure
    /// leaves an idempotent journal behind, which a later open replays onto the new snapshot.
    fn clear_journal(&mut self) {
        let path = self.cfg.dir.join(JOURNAL);
        match fs::OpenOptions::new().write(true).truncate(true).open(&path) {
            Ok(file) => {
                if let Err(err) = file.sync_all() {
                    crate::say!("slot archive: syncing the cleared journal: {err:#}");
                }
                fsync_dir(&self.cfg.dir);
                self.journal_bytes = 0;
                self.journal_records = 0;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                self.journal_bytes = 0;
                self.journal_records = 0;
            }
            Err(err) => {
                crate::say!("slot archive: clearing the journal {}: {err:#}", path.display());
            }
        }
    }

    /// Compact when the journal has grown to the size of the snapshot, at a record cap, or eagerly for a small
    /// archive so operators and external tools see a current `manifest.json`. Errors are logged: the mutation is
    /// already durable in the journal, so a failed compaction must not fail the caller.
    fn maybe_compact(&mut self) {
        if self.entries.len() <= EAGER_COMPACT_ENTRIES || self.journal_is_large() {
            if let Err(err) = self.compact() {
                crate::say!("slot archive: manifest compaction failed: {err:#}");
            }
        }
    }

    /// True when the journal has grown to the snapshot size or the record cap. Used by the touch path, which
    /// must not rewrite the snapshot just because the archive is small.
    fn journal_is_large(&self) -> bool {
        self.journal_bytes >= self.manifest_bytes || self.journal_records >= JOURNAL_COMPACT_RECORDS
    }

    /// Make pending LRU touches durable by appending them to the journal. This does not rewrite `manifest.json`;
    /// the next structural compaction folds them into the snapshot.
    pub fn flush(&mut self) -> Result<()> {
        self.flush_touches(true)
    }

    /// Write the full compacted snapshot and clear the journal. This is the explicit "compact now" operation; it
    /// is also used by GC and by the adaptive policy in [`Archive::maybe_compact`].
    pub fn compact(&mut self) -> Result<()> {
        ensure_dir(&self.cfg.dir)?;
        let manifest = ManifestFile {
            version: MANIFEST_VERSION,
            entries: self
                .entries
                .values()
                .map(|entry| self.manifest_row(entry))
                .collect(),
        };
        let data = serde_json::to_vec(&manifest).context("serializing the archive manifest")?;
        write_atomic(&self.cfg.dir, &data)?;
        self.manifest_bytes = data.len() as u64;
        self.pending_touches.clear();
        self.last_touch_flush_ms = now_ms();
        self.clear_journal();
        // Keep the second manifest in step when the archive is compacted, so a direct reader sees recent segment
        // rows too. A failure here is logged: the segment journal is still replayable.
        if let Err(err) = self.segments.compact() {
            crate::say!("slot archive: segment manifest compaction failed: {err:#}");
        }
        Ok(())
    }

    /// Append the pending touch batch. `force` ignores the interval/count thresholds (used by `flush`, GC and
    /// `Drop`).
    fn flush_touches(&mut self, force: bool) -> Result<()> {
        if self.pending_touches.is_empty() {
            return Ok(());
        }
        let now = now_ms();
        if !force
            && self.pending_touches.len() < TOUCH_FLUSH_MAX
            && now.saturating_sub(self.last_touch_flush_ms) < TOUCH_FLUSH_INTERVAL_MS
        {
            return Ok(());
        }
        let touches: Vec<TouchRow> = self
            .pending_touches
            .iter()
            .map(|(id, last_used_ms)| TouchRow { id: id.clone(), last_used_ms: *last_used_ms })
            .collect();
        let op = JournalOp::Touch { v: JOURNAL_VERSION, touches };
        self.append_journal(&op)?;
        self.pending_touches.clear();
        self.last_touch_flush_ms = now;
        // A pure-touch workload must also have a bound: fold the journal in when it reaches the snapshot size.
        if self.journal_is_large() {
            if let Err(err) = self.compact() {
                crate::say!("slot archive: manifest compaction after touch flush failed: {err:#}");
            }
        }
        Ok(())
    }

    /// Cheap in-memory LRU update plus the bounded batch policy.
    fn note_touch(&mut self, id: &str, used: u64) {
        self.pending_touches.insert(id.to_string(), used);
        let due = self.pending_touches.len() >= TOUCH_FLUSH_MAX
            || used.saturating_sub(self.last_touch_flush_ms) >= TOUCH_FLUSH_INTERVAL_MS;
        if due {
            if let Err(err) = self.flush_touches(true) {
                crate::say!("slot archive: flushing LRU touches: {err:#}");
            }
        }
    }
}

impl Drop for Archive {
    fn drop(&mut self) {
        // A clean shutdown should not lose recent LRU hints. Entries themselves are already durable.
        if let Err(err) = self.flush_touches(true) {
            crate::say!("slot archive: flushing LRU touches at shutdown: {err:#}");
        }
    }
}

/// Milliseconds since the Unix epoch, saturating to 0 before it.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// Join a manifest-relative file name to the archive directory, rejecting escapes.
pub(crate) fn entry_path(dir: &Path, file: &str) -> Result<PathBuf> {
    let p = Path::new(file);
    if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir)) {
        bail!("archive manifest contains an invalid path: {file}");
    }
    Ok(dir.join(p))
}

/// Read the token list at the start of a slot file without reading the KV. Used by lookup to turn a hash
/// candidate into an exact match.
pub(crate) fn read_tokens(path: &Path) -> Result<Vec<u32>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::new(f);
    let (_header, tokens) = crate::slot_file::read_header(&mut r)?;
    Ok(tokens)
}

/// The token history of an entry: the `STRATASLOT` header for v1, the `STRSTATE` header for v2.
pub(crate) fn read_entry_tokens(entry: &ArchiveEntry) -> Result<Vec<u32>> {
    match entry.format {
        1 => read_tokens(&entry.file),
        2 => {
            let path = entry
                .state_file
                .as_deref()
                .context("segmented entry has no state file")?;
            read_state_tokens(path)
        }
        other => bail!("slot archive: entry {} has unsupported format {other}", entry.id),
    }
}

// ---------------------------------------------------------------------------------------------------------
// STRSTATE: the v2 entry's token list, segment refs and recurrent snapshot
// ---------------------------------------------------------------------------------------------------------
//
// Version 2 (written here):
// ```text
// magic[8]  = "STRSTATE"
// version: u32 = 2
// fingerprint: u64
// owner: u64
// rope_offset: i32
// n_tokens: u32
// n_segments: u32
// segments: n_segments × { id[16], start u64, end u64 }
// tokens: n_tokens × u32
// state_len: u64
// state: state_len bytes (the `STRRSEG` payload from GpuModel::save_recurrent_to)
// ```
//
// Version 1 (read-only compatibility) has the same header through `n_tokens`, then `tokens` and the payload; it
// carries no segment refs. A v1 state file is still restorable while its manifest row supplies the refs, but it
// cannot be reconstructed after manifest loss (the refs would have to be invented).
//
// There is no footer: the exact file length is checked, so a truncated or extended file is rejected. The segment
// refs are validated against the content-addressed chain over `tokens`, which lets `open` rebuild v2 entries from
// the state file alone.

/// The parsed `STRSTATE` header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StateHeader {
    pub fingerprint: u64,
    pub owner: u64,
    pub rope_offset: i32,
    pub n_tokens: usize,
}

fn read_state_preamble<R: Read>(
    r: &mut R,
) -> Result<(u32, StateHeader, Vec<SegmentRef>, Vec<u32>, u64)> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).context("reading the state file magic")?;
    if magic != STATE_MAGIC {
        bail!(
            "bad state file magic {magic:02x?}, expected {STATE_MAGIC:02x?}"
        );
    }
    let version = crate::slot_file::read_u32(r).context("reading the state file version")?;
    if version != 1 && version != STATE_VERSION {
        bail!("unsupported state file version {version}, expected 1 or {STATE_VERSION}");
    }
    let fingerprint = crate::slot_file::read_u64(r).context("reading the state file fingerprint")?;
    let owner = crate::slot_file::read_u64(r).context("reading the state file owner")?;
    let mut rope = [0u8; 4];
    r.read_exact(&mut rope).context("reading the state file rope offset")?;
    let rope_offset = i32::from_le_bytes(rope);
    let n_tokens = crate::slot_file::read_u32(r).context("reading the state file token count")? as usize;
    if n_tokens == 0 || n_tokens > MAX_STATE_TOKENS {
        bail!("state file has {n_tokens} tokens, the expected range is 1..={MAX_STATE_TOKENS}");
    }

    let mut segments = Vec::new();
    if version >= 2 {
        let n_segments =
            crate::slot_file::read_u32(r).context("reading the state segment count")? as usize;
        // Each segment covers at least one token, so the count cannot exceed the token count. The cap also
        // bounds the allocation for a malicious header.
        if n_segments == 0 || n_segments > n_tokens || n_segments > MAX_STATE_SEGMENTS {
            bail!(
                "state file has {n_segments} segment refs for {n_tokens} tokens, the expected range is 1..={}",
                n_tokens.min(MAX_STATE_SEGMENTS)
            );
        }
        segments.reserve(n_segments);
        for index in 0..n_segments {
            let mut id = [0u8; 16];
            r.read_exact(&mut id)
                .with_context(|| format!("reading state segment id {index} of {n_segments}"))?;
            let id = String::from_utf8(id.to_vec())
                .with_context(|| format!("state segment id {index} is not UTF-8"))?;
            if !valid_segment_id(&id) {
                bail!("invalid segment id {id:?} in the state file");
            }
            let start = crate::slot_file::read_u64(r)
                .with_context(|| format!("reading state segment {index} start"))?;
            let end = crate::slot_file::read_u64(r)
                .with_context(|| format!("reading state segment {index} end"))?;
            let start = usize::try_from(start).context("state segment start does not fit usize")?;
            let end = usize::try_from(end).context("state segment end does not fit usize")?;
            segments.push(SegmentRef { id, start, end });
        }
    }

    let mut tokens = Vec::with_capacity(n_tokens);
    for index in 0..n_tokens {
        tokens.push(
            crate::slot_file::read_u32(r)
                .with_context(|| format!("reading state file token {index} of {n_tokens}"))?,
        );
    }
    let payload_len = crate::slot_file::read_u64(r).context("reading the state payload length")?;
    let header = StateHeader { fingerprint, owner, rope_offset, n_tokens };
    if version >= 2 {
        validate_segment_chain(fingerprint, &segments, &tokens)
            .context("validating the segment chain in the state file")?;
    }
    Ok((version, header, segments, tokens, payload_len))
}

/// Validate the file length against the header and return the expected total size.
fn state_expected_len(
    version: u32,
    n_tokens: usize,
    n_segments: usize,
    payload_len: u64,
) -> Result<u64> {
    let tokens = (n_tokens as u64)
        .checked_mul(4)
        .context("state file token list length overflow")?;
    let refs = (n_segments as u64)
        .checked_mul(STATE_SEGMENT_REF_LEN)
        .context("state segment ref list length overflow")?;
    let base = if version >= 2 { STATE_HEADER_LEN } else { STATE_V1_HEADER_LEN };
    base.checked_add(refs)
        .and_then(|n| n.checked_add(tokens))
        .and_then(|n| n.checked_add(8))
        .and_then(|n| n.checked_add(payload_len))
        .context("state file length overflow")
}

/// Read and validate a `STRSTATE` header plus token list, without reading the recurrent payload.
pub(crate) fn read_state_tokens(path: &Path) -> Result<Vec<u32>> {
    Ok(read_state_manifest(path)?.2)
}

/// Read and validate a `STRSTATE` header plus token list. This is the startup/lookup path: it never reads the
/// (potentially tens of MB) recurrent payload.
pub(crate) fn read_state_header(path: &Path) -> Result<(StateHeader, Vec<u32>)> {
    let (header, _, tokens) = read_state_manifest(path)?;
    Ok((header, tokens))
}

/// Read and validate a `STRSTATE` header, its ordered segment refs and its token list.
pub(crate) fn read_state_manifest(path: &Path) -> Result<(StateHeader, Vec<SegmentRef>, Vec<u32>)> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let meta = f.metadata().with_context(|| format!("stat {}", path.display()))?;
    ensure_regular(&meta, path)?;
    let (version, header, segments, tokens, payload_len) = {
        let mut r = BufReader::new(f);
        read_state_preamble(&mut r).with_context(|| format!("reading {}", path.display()))?
    };
    let expected = state_expected_len(version, header.n_tokens, segments.len(), payload_len)?;
    if meta.len() != expected {
        bail!(
            "state file {} is {} bytes, the header describes {expected}",
            path.display(),
            meta.len()
        );
    }
    Ok((header, segments, tokens))
}

/// Read and validate a whole `STRSTATE` file.
pub(crate) fn read_state_file(path: &Path) -> Result<(StateHeader, Vec<SegmentRef>, Vec<u32>, Vec<u8>)> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let meta = f.metadata().with_context(|| format!("stat {}", path.display()))?;
    ensure_regular(&meta, path)?;
    let mut r = BufReader::new(f);
    let (version, header, segments, tokens, payload_len) = read_state_preamble(&mut r)
        .with_context(|| format!("reading {}", path.display()))?;
    let expected = state_expected_len(version, header.n_tokens, segments.len(), payload_len)?;
    if meta.len() != expected {
        bail!(
            "state file {} is {} bytes, the header describes {expected}",
            path.display(),
            meta.len()
        );
    }
    let payload_len = usize::try_from(payload_len).context("state payload does not fit in memory")?;
    let mut payload = vec![0u8; payload_len];
    r.read_exact(&mut payload)
        .with_context(|| format!("reading the state payload of {}", path.display()))?;
    Ok((header, segments, tokens, payload))
}

fn ensure_regular(meta: &fs::Metadata, path: &Path) -> Result<()> {
    if !meta.is_file() {
        bail!("{} is not a regular file", path.display());
    }
    Ok(())
}

/// The state file path for entry `id`: `<dir>/<2 hex>/<id>.state`.
fn state_path(dir: &Path, id: &str) -> Result<PathBuf> {
    let shard = id.get(..2).context("archive entry id is too short")?;
    Ok(dir.join(shard).join(format!("{id}.state")))
}

/// Write a `STRSTATE` v2 file via a temp file + fsync + rename (0600), returning its length.
fn write_state_file_atomic(
    path: &Path,
    header: &StateHeader,
    segments: &[SegmentRef],
    tokens: &[u32],
    payload: &[u8],
) -> Result<u64> {
    if tokens.is_empty() {
        bail!("refusing to write an empty state file");
    }
    if tokens.len() != header.n_tokens {
        bail!(
            "state file header says {} tokens, the list has {}",
            header.n_tokens,
            tokens.len()
        );
    }
    validate_segment_chain(header.fingerprint, segments, tokens)
        .context("refusing to write an inconsistent segment chain")?;
    let n_tokens = u32::try_from(tokens.len()).context("state token count does not fit in u32")?;
    let n_segments = u32::try_from(segments.len()).context("segment count does not fit in u32")?;
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    ensure_dir(dir)?;
    let (tmp, mut file) = create_temp(dir, "state")?;

    let write = (|| -> Result<()> {
        file.write_all(&STATE_MAGIC).context("writing the state file magic")?;
        crate::slot_file::write_u32(&mut file, STATE_VERSION)?;
        crate::slot_file::write_u64(&mut file, header.fingerprint)?;
        crate::slot_file::write_u64(&mut file, header.owner)?;
        file.write_all(&header.rope_offset.to_le_bytes())
            .context("writing the state rope offset")?;
        crate::slot_file::write_u32(&mut file, n_tokens)?;
        crate::slot_file::write_u32(&mut file, n_segments)?;
        for segment in segments {
            if segment.id.len() != 16 {
                bail!("segment id {:?} is not 16 bytes", segment.id);
            }
            file.write_all(segment.id.as_bytes())
                .context("writing the state segment id")?;
            crate::slot_file::write_u64(&mut file, segment.start as u64)?;
            crate::slot_file::write_u64(&mut file, segment.end as u64)?;
        }
        for &token in tokens {
            crate::slot_file::write_u32(&mut file, token)?;
        }
        crate::slot_file::write_u64(&mut file, payload.len() as u64)?;
        file.write_all(payload).context("writing the state payload")?;
        file.sync_all().context("syncing the state file")?;
        Ok(())
    })();
    if let Err(err) = write {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    drop(file);
    let bytes = fs::metadata(&tmp)
        .with_context(|| format!("stat {}", tmp.display()))?
        .len();
    if let Err(err) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(err)
            .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()));
    }
    fsync_dir(dir);
    Ok(bytes)
}


/// Create a uniquely named 0600 temp file in `dir`, returning its path and handle.
fn create_temp(dir: &Path, tag: &str) -> Result<(PathBuf, fs::File)> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    for attempt in 0..32u32 {
        let path = dir.join(format!(
            ".{tag}.{}.{nanos}.{attempt}.tmp",
            std::process::id()
        ));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists && attempt < 31 => continue,
            Err(err) => {
                return Err(err).with_context(|| format!("creating {}", path.display()));
            }
        }
    }
    bail!("could not create a temporary state file in {}", dir.display())
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

/// Remove `path` if present. With `strict`, an unlink error other than "not found" is returned.
fn remove_file_if_present(path: &Path, strict: bool) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) if strict => {
            Err(err).with_context(|| format!("removing {}", path.display()))
        }
        Err(err) => {
            crate::say!("slot archive: cannot remove {}: {err:#}", path.display());
            Ok(())
        }
    }
}

/// True when `segments` is an ordered, contiguous cover of `[0, n_tokens)`.
fn segments_cover(segments: &[SegmentRef], n_tokens: usize) -> bool {
    let mut next = 0usize;
    for segment in segments {
        if segment.start != next || segment.end <= segment.start || segment.end > n_tokens {
            return false;
        }
        next = segment.end;
    }
    next == n_tokens
}

/// Prove that the ordered refs are exactly the content-addressed chain for `tokens`: every id is the 16-hex
/// segment id of `(fingerprint, previous id, tokens[start..end])`. This is what makes a ref list reconstructible
/// from a state file without inventing segments.
fn validate_segment_chain(fingerprint: u64, segments: &[SegmentRef], tokens: &[u32]) -> Result<()> {
    if !segments_cover(segments, tokens.len()) {
        bail!("segment ranges are not a contiguous cover of the history");
    }
    let mut parent: Option<String> = None;
    for segment in segments {
        if !valid_segment_id(&segment.id) {
            bail!("invalid segment id {:?}", segment.id);
        }
        let expected = slot_segments::segment_id(
            fingerprint,
            parent.as_deref(),
            &tokens[segment.start..segment.end],
        );
        if expected != segment.id {
            bail!(
                "segment {} at [{}, {}) is not the content-addressed chain id {expected}",
                segment.id,
                segment.start,
                segment.end
            );
        }
        parent = Some(segment.id.clone());
    }
    Ok(())
}

/// A segment id is 16 lowercase hex characters (the same rule as [`slot_segments`]).
fn valid_segment_id(id: &str) -> bool {
    id.len() == 16
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}


/// Stable 64-bit FNV-1a hash of a token history.
///
/// The algorithm is FNV-1a with the 64-bit offset basis `0xcbf29ce484222325` and prime
/// `0x100000001b3`. Each token contributes its four little-endian bytes, in order; the empty history hashes
/// to the offset basis. FNV-1a is used instead of `DefaultHasher` because the manifest id must be identical
/// across processes, restarts and Rust versions.
pub(crate) fn prefix_hash(tokens: &[u32]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

/// The stable content id: `fingerprint-owner-n_tokens-prefix_hash`, all hex except `n_tokens`.
pub(crate) fn id_for(fingerprint: u64, owner: u64, n_tokens: usize, prefix_hash: u64) -> String {
    format!("{fingerprint:016x}-{owner:016x}-{n_tokens}-{prefix_hash:016x}")
}

fn entry_id(fingerprint: u64, owner: u64, tokens: &[u32]) -> String {
    id_for(fingerprint, owner, tokens.len(), prefix_hash(tokens))
}

// ---------------------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct ManifestFile {
    version: u32,
    entries: Vec<ManifestEntry>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ManifestEntry {
    id: String,
    file: String,
    /// v2 only: the `STRSTATE` file, relative to the archive directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    state_file: Option<String>,
    /// v2 only: the ordered KV segments. Absent in a v1 manifest (and empty for a v1 entry).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    segments: Vec<SegmentRef>,
    bytes: u64,
    n_tokens: usize,
    owner: u64,
    fingerprint: String,
    prefix_hash: String,
    created_ms: u64,
    last_used_ms: u64,
    parent: Option<String>,
    pinned: bool,
    reason: String,
    format: u32,
}

fn parse_hex(value: &str, field: &str) -> Result<u64> {
    u64::from_str_radix(value.trim(), 16)
        .with_context(|| format!("invalid {field} {value:?} in the archive manifest"))
}

/// Convert one JSON manifest row into an in-memory entry, rejecting unsafe paths.
fn archive_entry_from_manifest(dir: &Path, row: ManifestEntry) -> Result<ArchiveEntry> {
    let file = entry_path(dir, &row.file)?;
    let state_file = row
        .state_file
        .as_deref()
        .map(|state| entry_path(dir, state))
        .transpose()?;
    Ok(ArchiveEntry {
        id: row.id,
        file,
        state_file,
        segments: row.segments,
        bytes: row.bytes,
        n_tokens: row.n_tokens,
        owner: row.owner,
        fingerprint: parse_hex(&row.fingerprint, "fingerprint")?,
        prefix_hash: parse_hex(&row.prefix_hash, "prefix_hash")?,
        created_ms: row.created_ms,
        last_used_ms: row.last_used_ms,
        parent: row.parent,
        pinned: row.pinned,
        reason: row.reason,
        format: row.format,
    })
}

/// Parse `manifest.json`. Any error (missing file, bad JSON, bad version, unsafe path) is handled by
/// [`Archive::open`] as a corrupt manifest: it is moved aside and the index is rebuilt from the shards.
fn load_manifest(dir: &Path) -> Result<BTreeMap<String, ArchiveEntry>> {
    let path = dir.join(MANIFEST);
    let data = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let manifest: ManifestFile =
        serde_json::from_slice(&data).with_context(|| format!("parsing {}", path.display()))?;
    if manifest.version != MANIFEST_VERSION {
        bail!(
            "unsupported archive manifest version {}, expected {MANIFEST_VERSION}",
            manifest.version
        );
    }

    let mut entries = BTreeMap::new();
    for row in manifest.entries {
        let entry = archive_entry_from_manifest(dir, row)?;
        entries.insert(entry.id.clone(), entry);
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------------------------------------
// Manifest journal
// ---------------------------------------------------------------------------------------------------------
//
// `manifest.log` is newline-delimited JSON, one record per mutation since the last compaction. Records are
// `add` (a full manifest row, so an entry written just before a crash is replayable), `delete`, `pin` and a
// batched `touch` list. A record is appended together with its trailing newline and fsynced; a crash can leave
// at most a partial final line, which is ignored. A complete but malformed line is treated as corruption: the
// valid prefix is applied and the journal is compacted.

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase")]
enum JournalOp {
    Add { v: u32, entry: ManifestEntry },
    Delete { v: u32, id: String },
    Pin { v: u32, id: String, pinned: bool },
    Touch { v: u32, touches: Vec<TouchRow> },
}

#[derive(Serialize, Deserialize, Debug)]
struct TouchRow {
    id: String,
    last_used_ms: u64,
}

/// Read `manifest.log`, returning the complete records, whether a truncated/corrupt tail was found, the byte
/// length and the number of complete records. A missing journal is empty.
fn load_journal(dir: &Path) -> (Vec<JournalOp>, bool, u64, usize) {
    let path = dir.join(JOURNAL);
    let data = match fs::read(&path) {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return (Vec::new(), false, 0, 0),
        Err(err) => {
            crate::say!("slot archive: cannot read the journal {}: {err:#}", path.display());
            return (Vec::new(), true, 0, 0);
        }
    };
    let bytes = data.len() as u64;
    let mut records = Vec::new();
    let mut repaired = false;
    for line in data.split_inclusive(|byte| *byte == b'\n') {
        // Only newline-terminated lines are complete. The final fragment after a crash is ignored.
        if !line.ends_with(b"\n") {
            repaired = true;
            break;
        }
        match serde_json::from_slice::<JournalOp>(line) {
            Ok(op) => {
                let version = match &op {
                    JournalOp::Add { v, .. }
                    | JournalOp::Delete { v, .. }
                    | JournalOp::Pin { v, .. }
                    | JournalOp::Touch { v, .. } => *v,
                };
                if version != JOURNAL_VERSION {
                    crate::say!(
                        "slot archive: journal record version {version}, expected {JOURNAL_VERSION}; ignoring the rest"
                    );
                    repaired = true;
                    break;
                }
                records.push(op);
            }
            Err(err) => {
                crate::say!("slot archive: corrupt journal record in {}: {err}", path.display());
                repaired = true;
                break;
            }
        }
    }
    (records, repaired, bytes, data.iter().filter(|byte| **byte == b'\n').count())
}

/// Apply one journal record over a loaded snapshot.
fn apply_journal(dir: &Path, entries: &mut BTreeMap<String, ArchiveEntry>, op: JournalOp) -> Result<()> {
    match op {
        JournalOp::Add { entry, .. } => {
            let mut entry = archive_entry_from_manifest(dir, entry)?;
            if let Some(existing) = entries.get(&entry.id) {
                // A touch may have been compacted into the snapshot but not journaled (touches are batched);
                // never move the LRU clock backwards on replay.
                entry.last_used_ms = entry.last_used_ms.max(existing.last_used_ms);
            }
            entries.insert(entry.id.clone(), entry);
        }
        JournalOp::Delete { id, .. } => {
            entries.remove(&id);
        }
        JournalOp::Pin { id, pinned, .. } => {
            if let Some(entry) = entries.get_mut(&id) {
                entry.pinned = pinned;
            }
        }
        JournalOp::Touch { touches, .. } => {
            for touch in touches {
                if let Some(entry) = entries.get_mut(&touch.id) {
                    // A replayed touch may be older than the compacted snapshot after a crash between the
                    // snapshot rename and the journal truncate; never move the LRU clock backwards.
                    entry.last_used_ms = entry.last_used_ms.max(touch.last_used_ms);
                }
            }
        }
    }
    Ok(())
}

/// Discard a journal that cannot be replayed (its base snapshot is gone or corrupt), preserving it for
/// diagnostics under a `.bad.<ts>` name so it is never replayed again.
fn discard_journal(dir: &Path, reason: &str) {
    let path = dir.join(JOURNAL);
    if !path.exists() {
        return;
    }
    let bad = dir.join(format!("{JOURNAL}.bad.{}", now_ms()));
    match fs::rename(&path, &bad) {
        Ok(()) => crate::say!("slot archive: {reason}; moved the journal to {}", bad.display()),
        Err(err) => {
            crate::say!("slot archive: {reason}; removing the unreadable journal: {err:#}");
            let _ = fs::remove_file(&path);
        }
    }
}


/// Stat the file(s) and compare the on-disk header with the manifest row. Format 1 is a `STRATASLOT` file; format 2
/// is a `STRSTATE` file plus a contiguous segment chain. An unsupported format is invalid, never silently accepted.
fn validate_entry(entry: &ArchiveEntry, segments: &SegmentStore) -> Result<()> {
    match entry.format {
        1 => {
            let meta = fs::metadata(&entry.file)
                .with_context(|| format!("stat {}", entry.file.display()))?;
            ensure_regular(&meta, &entry.file)?;
            let (header, tokens) = read_slot(&entry.file)?;
            if header.fingerprint != entry.fingerprint {
                bail!(
                    "header fingerprint {:016x} does not match the manifest {:016x}",
                    header.fingerprint,
                    entry.fingerprint
                );
            }
            if header.owner != entry.owner {
                bail!(
                    "header owner {} does not match the manifest {}",
                    header.owner,
                    entry.owner
                );
            }
            if header.n_tokens != entry.n_tokens as u64 || tokens.len() != entry.n_tokens {
                bail!(
                    "header has {} tokens, the manifest says {}",
                    header.n_tokens,
                    entry.n_tokens
                );
            }
        }
        2 => {
            let path = entry
                .state_file
                .as_deref()
                .context("segmented entry has no state file")?;
            let (header, state_segments, tokens) = read_state_manifest(path)?;
            if header.fingerprint != entry.fingerprint {
                bail!(
                    "state file fingerprint {:016x} does not match the manifest {:016x}",
                    header.fingerprint,
                    entry.fingerprint
                );
            }
            if header.owner != entry.owner {
                bail!(
                    "state file owner {} does not match the manifest {}",
                    header.owner,
                    entry.owner
                );
            }
            if header.n_tokens != entry.n_tokens || tokens.len() != entry.n_tokens {
                bail!(
                    "state file has {} tokens, the manifest says {}",
                    tokens.len(),
                    entry.n_tokens
                );
            }
            if !state_segments.is_empty() && state_segments != entry.segments {
                bail!("state file segment refs do not match the manifest row");
            }
            if !segments_cover(&entry.segments, entry.n_tokens) {
                bail!("segment ranges are not a contiguous cover of the history");
            }
            validate_segment_chain(entry.fingerprint, &entry.segments, &tokens)?;
            for segment in &entry.segments {
                let row = segments
                    .segment(&segment.id)
                    .with_context(|| format!("segment {} is missing", segment.id))?;
                if row.n_tokens != segment.end - segment.start {
                    bail!(
                        "segment {} has {} tokens, the entry range [{}, {}) needs {}",
                        segment.id,
                        row.n_tokens,
                        segment.start,
                        segment.end,
                        segment.end - segment.start
                    );
                }
                let meta = fs::metadata(&row.file)
                    .with_context(|| format!("stat {}", row.file.display()))?;
                ensure_regular(&meta, &row.file)?;
            }
        }
        other => bail!("unsupported entry format {other}"),
    }
    Ok(())
}

/// Rebuild the entry index after `manifest.json` loss from the surviving files:
/// * v1 `*.slot` files are adopted with their header/token list;
/// * v2 `*.state` files carry the ordered segment refs, so their entries can be reconstructed. Every ref is
///   validated against the token list (content-addressed chain) and against the segment store; invalid data is
///   dropped, never invented.
///
/// Segment bytes are charged to the first (by id) entry that references them, so the reconstructed logical total
/// counts each segment once. Pin/lineage state is not recoverable from the files and defaults to unpinned/no
/// parent, as with the v1 rebuild.
fn rebuild_manifest(dir: &Path, segments: &SegmentStore) -> Result<BTreeMap<String, ArchiveEntry>> {
    let mut candidates: Vec<ArchiveEntry> = Vec::new();
    let root = fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    for shard in root.flatten() {
        let shard_path = shard.path();
        if !shard.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let shard_name = shard.file_name().to_string_lossy().into_owned();
        let files = match fs::read_dir(&shard_path) {
            Ok(files) => files,
            Err(err) => {
                crate::say!("slot archive: cannot scan {}: {err:#}", shard_path.display());
                continue;
            }
        };
        for file in files.flatten() {
            let path = file.path();
            let name = file.file_name().to_string_lossy().into_owned();
            if !file.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let meta = match fs::metadata(&path) {
                Ok(meta) => meta,
                Err(err) => {
                    crate::say!("slot archive: cannot stat {}: {err:#}", path.display());
                    continue;
                }
            };
            let used = mtime_ms(&meta);

            if name.ends_with(".slot") {
                let (header, tokens) = match read_slot(&path) {
                    Ok(parsed) => parsed,
                    Err(err) => {
                        crate::say!("slot archive: cannot recover {}: {err:#}", path.display());
                        continue;
                    }
                };
                let id = id_for(header.fingerprint, header.owner, tokens.len(), prefix_hash(&tokens));
                candidates.push(ArchiveEntry {
                    id,
                    file: path,
                    state_file: None,
                    segments: Vec::new(),
                    bytes: meta.len(),
                    n_tokens: tokens.len(),
                    owner: header.owner,
                    fingerprint: header.fingerprint,
                    prefix_hash: prefix_hash(&tokens),
                    created_ms: used,
                    last_used_ms: used,
                    parent: None,
                    pinned: false,
                    reason: "recovered".to_string(),
                    format: crate::slot_file::VERSION,
                });
                continue;
            }

            let Some(id) = name.strip_suffix(".state") else { continue };
            let id = id.to_string();
            if id.len() < 2 || &id[..2] != shard_name {
                crate::say!(
                    "slot archive: ignoring state file {} in the wrong shard",
                    path.display()
                );
                continue;
            }
            let (header, state_segments, tokens) = match read_state_manifest(&path) {
                Ok(parsed) => parsed,
                Err(err) => {
                    crate::say!("slot archive: cannot recover {}: {err:#}", path.display());
                    continue;
                }
            };
            if state_segments.is_empty() {
                // A legacy v1 `STRSTATE` file has no refs and the manifest that held them is gone. Reconstructing
                // it would invent segment references, so it is left on disk and reported instead.
                crate::say!(
                    "slot archive: {} is a v1 state file without segment refs and cannot be rebuilt",
                    path.display()
                );
                continue;
            }
            let expected = id_for(header.fingerprint, header.owner, tokens.len(), prefix_hash(&tokens));
            if id != expected {
                crate::say!(
                    "slot archive: state file {} names entry {id}, content says {expected}; skipping",
                    path.display()
                );
                continue;
            }
            let entry = ArchiveEntry {
                id,
                file: path.clone(),
                state_file: Some(path.clone()),
                segments: state_segments,
                bytes: meta.len(),
                n_tokens: tokens.len(),
                owner: header.owner,
                fingerprint: header.fingerprint,
                prefix_hash: prefix_hash(&tokens),
                created_ms: used,
                last_used_ms: used,
                parent: None,
                pinned: false,
                reason: "recovered".to_string(),
                format: 2,
            };
            if let Err(err) = validate_entry(&entry, segments) {
                crate::say!("slot archive: dropping rebuilt v2 entry {}: {err:#}", entry.id);
                continue;
            }
            candidates.push(entry);
        }
    }

    // Deterministic order; a v2 row wins over a v1 row with the same content id (it is the richer format).
    candidates.sort_by(|a, b| a.id.cmp(&b.id).then_with(|| b.format.cmp(&a.format)));
    let mut claimed: BTreeSet<String> = BTreeSet::new();
    let mut entries: BTreeMap<String, ArchiveEntry> = BTreeMap::new();
    for mut entry in candidates {
        if entries.contains_key(&entry.id) {
            crate::say!(
                "slot archive: duplicate recovered entry {}; keeping the first",
                entry.id
            );
            continue;
        }
        if entry.format == 2 {
            for segment in &entry.segments {
                if claimed.insert(segment.id.clone()) {
                    if let Some(row) = segments.segment(&segment.id) {
                        entry.bytes = entry.bytes.saturating_add(row.bytes);
                    }
                }
            }
        }
        entries.insert(entry.id.clone(), entry);
    }
    Ok(entries)
}

fn mtime_ms(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as u64)
}

/// Read a full slot header and its token list. The KV payload and footer are not read.
fn read_slot(path: &Path) -> Result<(crate::slot_file::SlotHeader, Vec<u32>)> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::new(f);
    crate::slot_file::read_header(&mut r).with_context(|| format!("reading {}", path.display()))
}

// ---------------------------------------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------------------------------------

/// Create `dir` (and parents) when missing, with mode 0700 on Unix. An existing directory is left untouched.
fn ensure_dir(dir: &Path) -> Result<()> {
    match fs::metadata(dir) {
        Ok(meta) if meta.is_dir() => Ok(()),
        Ok(_) => bail!("slot archive path {} exists but is not a directory", dir.display()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(dir)
                .with_context(|| format!("creating the archive directory {}", dir.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).with_context(|| {
                    format!("setting mode 0700 on the archive directory {}", dir.display())
                })?;
            }
            Ok(())
        }
        Err(err) => Err(err).with_context(|| format!("stat {}", dir.display())),
    }
}

/// Build the manifest-relative `file` value with forward slashes.
fn relative_string(dir: &Path, file: &Path) -> String {
    let Ok(rel) = file.strip_prefix(dir) else {
        return file.to_string_lossy().into_owned();
    };
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Write `data` to `manifest.json` via a temp file + `sync_all` + rename, then fsync the directory
/// (best effort).
fn write_atomic(dir: &Path, data: &[u8]) -> Result<()> {
    let manifest = dir.join(MANIFEST);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut tmp = dir.join(format!("{MANIFEST}.tmp.{}.{nanos}", std::process::id()));
    let mut file = {
        let mut attempt = 0u32;
        loop {
            let mut options = fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&tmp) {
                Ok(file) => break file,
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists && attempt < 16 => {
                    attempt += 1;
                    tmp = dir.join(format!(
                        "{MANIFEST}.tmp.{}.{nanos}.{attempt}",
                        std::process::id()
                    ));
                }
                Err(err) => {
                    return Err(err).with_context(|| format!("creating {}", tmp.display()));
                }
            }
        }
    };

    let mut write = || -> Result<()> {
        file.write_all(data).context("writing the archive manifest")?;
        file.sync_all().context("syncing the archive manifest")?;
        Ok(())
    };
    if let Err(err) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    drop(file);

    if let Err(err) = fs::rename(&tmp, &manifest) {
        let _ = fs::remove_file(&tmp);
        return Err(err).with_context(|| {
            format!("renaming {} to {}", tmp.display(), manifest.display())
        });
    }

    #[cfg(unix)]
    {
        if let Ok(dir_file) = fs::File::open(dir) {
            let _ = dir_file.sync_all();
        }
    }
    Ok(())
}

/// Remove temp files left by a crashed writer: `manifest.json.tmp.*` at the top and `*.tmp*` in the shards.
fn remove_stale_temps(dir: &Path) {
    let Ok(root) = fs::read_dir(dir) else { return };
    for entry in root.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        match entry.file_type() {
            Ok(kind) if kind.is_dir() => {
                if let Ok(files) = fs::read_dir(&path) {
                    for file in files.flatten() {
                        let file_name = file.file_name().to_string_lossy().into_owned();
                        if file_name.contains(".tmp") {
                            let _ = fs::remove_file(file.path());
                        }
                    }
                }
            }
            _ if name.starts_with("manifest.json.tmp.") || name.ends_with(".tmp") => {
                let _ = fs::remove_file(&path);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot_file::{self, SlotHeader};
    use std::time::Duration;

    const FP: u64 = 0xa1b2_c3d4_e5f6_0718;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "strata-slot-archive-{name}-{}-{nanos}",
                std::process::id()
            ));
            TempDir(path)
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

    fn config(dir: &Path) -> ArchiveConfig {
        ArchiveConfig {
            dir: dir.to_path_buf(),
            budget_bytes: RESERVE_BYTES + 1_000_000,
            min_tokens: 1,
            min_delta: 0,
            versions: 1,
        }
    }

    fn write_slot(path: &Path, owner: u64, fingerprint: u64, tokens: &[u32]) -> u64 {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut file = fs::File::create(path).unwrap();
        let header = SlotHeader {
            fingerprint,
            n_tokens: tokens.len() as u64,
            rope_offset: 0,
            owner,
            flags: 0,
            n_parts: 0,
        };
        slot_file::write_header(&mut file, &header, tokens).unwrap();
        let payload = slot_file::HEADER_LEN + 4 * tokens.len() as u64;
        slot_file::write_footer(&mut file, payload).unwrap();
        file.sync_all().unwrap();
        fs::metadata(path).unwrap().len()
    }

    fn save(archive: &mut Archive, owner: u64, tokens: &[u32]) -> ArchiveEntry {
        match archive.prepare_save(owner, tokens, FP, "test").unwrap() {
            SaveSlot::Existing(entry) => entry,
            SaveSlot::New { id, path } => {
                write_slot(&path, owner, FP, tokens);
                archive
                    .commit_save(&id, &path, owner, tokens, FP, "test")
                    .unwrap()
            }
        }
    }

    fn manifest_json(dir: &Path) -> serde_json::Value {
        serde_json::from_slice(&fs::read(dir.join(MANIFEST)).unwrap()).unwrap()
    }

    fn write_manifest_json(dir: &Path, value: &serde_json::Value) {
        fs::write(dir.join(MANIFEST), serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn set_last_used(archive: &mut Archive, id: &str, used: u64) {
        archive.entries.get_mut(id).unwrap().last_used_ms = used;
    }

    #[test]
    fn archive_is_send_and_has_no_gpu_dependency() {
        fn is_send<T: Send>() {}
        is_send::<Archive>();
    }

    #[test]
    fn manifest_round_trip_and_relative_paths() {
        let tmp = TempDir::new("roundtrip");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (100..140).collect();
        let entry = save(&mut archive, 7, &tokens);
        assert_eq!(entry.n_tokens, tokens.len());
        assert_eq!(entry.owner, 7);
        assert_eq!(entry.fingerprint, FP);
        assert_eq!(entry.prefix_hash, prefix_hash(&tokens));
        assert_eq!(entry.format, 1);
        assert!(entry.file.is_absolute());
        assert_eq!(archive.stats().entries, 1);
        assert_eq!(archive.stats().bytes, entry.bytes);

        let json = manifest_json(dir);
        assert_eq!(json["version"], 1);
        let file = json["entries"][0]["file"].as_str().unwrap();
        assert!(!file.starts_with('/'), "relative path expected, got {file}");
        assert_eq!(
            json["entries"][0]["fingerprint"].as_str().unwrap(),
            format!("{FP:016x}")
        );
        drop(archive);

        let archive = Archive::open(config(dir)).unwrap();
        let entries = archive.list(None, 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, entry.id);
        assert_eq!(entries[0].bytes, entry.bytes);
        assert_eq!(entries[0].prefix_hash, entry.prefix_hash);
        assert_eq!(entries[0].file, entry.file);
        assert_eq!(entries[0].reason, "test");
        assert!(!entries[0].pinned);
    }

    #[test]
    fn open_rebuilds_from_orphan_files() {
        let tmp = TempDir::new("rebuild");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=10).collect::<Vec<_>>());
        let b = save(&mut archive, 2, &(100..=120).collect::<Vec<_>>());
        // An orphan file that never made it into the manifest.
        let orphan_tokens: Vec<u32> = (500..=509).collect();
        let orphan_path = dir.join("fe").join("orphan.slot");
        write_slot(&orphan_path, 3, FP, &orphan_tokens);
        // A stale temp file, which startup must delete.
        let stale = dir.join("fe").join(".old.123.456.tmp");
        fs::write(&stale, b"junk").unwrap();

        fs::remove_file(dir.join(MANIFEST)).unwrap();
        let archive = Archive::open(config(dir)).unwrap();
        let mut entries = archive.list(None, 0);
        entries.sort_by(|x, y| x.n_tokens.cmp(&y.n_tokens));
        assert_eq!(entries.len(), 3, "{entries:?}");
        assert!(entries.iter().all(|e| e.reason == "recovered"));
        assert!(entries.iter().all(|e| e.parent.is_none()));
        assert!(entries.iter().any(|e| e.id == a.id));
        assert!(entries.iter().any(|e| e.id == b.id));
        assert!(entries.iter().any(|e| e.n_tokens == orphan_tokens.len()));
        assert!(!stale.exists());
        // The rebuilt manifest exists and lists all three.
        assert_eq!(manifest_json(dir)["entries"].as_array().unwrap().len(), 3);
        assert!(dir.join(MANIFEST).exists());
    }

    #[test]
    fn corrupt_manifest_is_moved_aside_and_rebuilt() {
        let tmp = TempDir::new("corrupt");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = save(&mut archive, 1, &(1..=8).collect::<Vec<_>>());
        drop(archive);

        fs::write(dir.join(MANIFEST), b"{ this is not json").unwrap();
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 1);
        assert_eq!(archive.list(None, 0)[0].id, entry.id);
        let bad: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("manifest.json.bad."))
            .collect();
        assert_eq!(bad.len(), 1, "{bad:?}");
        // The rebuilt manifest parses again.
        assert_eq!(manifest_json(dir)["version"], 1);
    }

    #[test]
    fn dedupe_returns_existing_and_writes_one_file() {
        let tmp = TempDir::new("dedupe");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (10..30).collect();
        let entry = save(&mut archive, 5, &tokens);

        match archive.prepare_save(5, &tokens, FP, "again").unwrap() {
            SaveSlot::Existing(existing) => {
                assert_eq!(existing.id, entry.id);
                assert_eq!(existing.file, entry.file);
            }
            SaveSlot::New { .. } => panic!("expected Existing"),
        }
        assert_eq!(archive.stats().entries, 1);
        let files = fs::read_dir(entry.file.parent().unwrap())
            .unwrap()
            .flatten()
            .count();
        assert_eq!(files, 1);
    }

    #[test]
    fn prepare_save_path_and_min_tokens() {
        let tmp = TempDir::new("prepare");
        let dir = tmp.path();
        let mut cfg = config(dir);
        cfg.min_tokens = 5;
        let mut archive = Archive::open(cfg).unwrap();
        let err = match archive.prepare_save(1, &[1, 2, 3, 4], FP, "test") {
            Err(err) => err,
            Ok(_) => panic!("a history below min_tokens must be rejected"),
        };
        assert!(err.to_string().contains("min_tokens"), "{err}");
        assert_eq!(archive.stats().entries, 0);

        let tokens: Vec<u32> = (1..=5).collect();
        match archive.prepare_save(1, &tokens, FP, "test").unwrap() {
            SaveSlot::New { id, path } => {
                let shard = &id[..2];
                assert_eq!(path, dir.join(shard).join(format!("{id}.slot")));
                assert!(path.parent().unwrap().is_dir());
                assert!(id.starts_with(&format!("{FP:016x}-0000000000000001-5-")));
            }
            SaveSlot::Existing(_) => panic!("expected New"),
        }
    }

    #[test]
    fn lookup_finds_the_longest_exact_prefix() {
        let tmp = TempDir::new("lookup");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=8).collect();
        let short = save(&mut archive, 1, &tokens[..4]);
        let long = save(&mut archive, 1, &tokens[..6]);
        // Lazy lineage: the ancestor survives the commit, so both lengths are lookup candidates.
        assert_eq!(archive.stats().entries, 2);

        let hit = archive.lookup(1, &tokens, 8, FP, 100).unwrap().unwrap();
        assert_eq!(hit.id, long.id);
        assert_eq!(hit.n_tokens, 6);
        let stats = archive.stats();
        assert_eq!((stats.hits, stats.misses), (1, 0));

        // `keep` and `ctx` cap the match; a shorter entry still wins.
        let hit = archive.lookup(1, &tokens, 4, FP, 100).unwrap().unwrap();
        assert_eq!(hit.id, short.id);
        assert!(archive.lookup(1, &tokens, 2, FP, 100).unwrap().is_none());
        let hit = archive.lookup(1, &tokens, 8, FP, 5).unwrap().unwrap();
        assert_eq!(hit.id, short.id, "ctx must exclude the length-6 entry");
        assert!(archive.lookup(1, &tokens, 8, FP, 3).unwrap().is_none());
        assert_eq!(archive.stats().misses, 2);

        // Exact token equality, not only equality of the shared prefix: a change inside the saved prefix
        // rejects the entry; a change after it still matches the shorter prefix.
        let mut inside = tokens.clone();
        inside[3] = 999;
        assert!(archive.lookup(1, &inside, 8, FP, 100).unwrap().is_none());
        let mut after = tokens.clone();
        after[5] = 999;
        assert_eq!(
            archive.lookup(1, &after, 8, FP, 100).unwrap().unwrap().id,
            short.id
        );
    }

    #[test]
    fn lookup_rejects_a_hash_only_false_candidate() {
        let tmp = TempDir::new("collision");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let request: Vec<u32> = (1..=8).collect();
        let real = save(&mut archive, 1, &request[..4]);

        // Plant an entry whose manifest hash matches the request prefix of length 6 but whose file holds
        // different tokens. `lookup` must fall through to the real shorter entry.
        let bogus_tokens: Vec<u32> = vec![91, 92, 93, 94, 95, 96];
        let bogus_hash = prefix_hash(&request[..6]);
        let bogus_id = id_for(FP, 1, 6, bogus_hash);
        let bogus_rel = format!("ff/{bogus_id}.slot");
        let bogus_path = dir.join(&bogus_rel);
        let bogus_bytes = write_slot(&bogus_path, 1, FP, &bogus_tokens);

        let mut json = manifest_json(dir);
        let now = now_ms();
        json["entries"].as_array_mut().unwrap().push(serde_json::json!({
            "id": bogus_id,
            "file": bogus_rel,
            "bytes": bogus_bytes,
            "n_tokens": 6,
            "owner": 1,
            "fingerprint": format!("{FP:016x}"),
            "prefix_hash": format!("{bogus_hash:016x}"),
            "created_ms": now,
            "last_used_ms": now,
            "parent": null,
            "pinned": false,
            "reason": "test",
            "format": 1
        }));
        write_manifest_json(dir, &json);

        let mut archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 2);
        // Hash-only matching sees the length-6 candidate...
        assert_eq!(
            archive.longest_prefix(1, &request, FP).unwrap().n_tokens,
            6
        );
        // ...but the exact lookup rejects it and returns the length-4 entry.
        let hit = archive.lookup(1, &request, 8, FP, 100).unwrap().unwrap();
        assert_eq!(hit.id, real.id);
        assert_eq!(hit.n_tokens, 4);
    }

    #[test]
    fn owner_and_fingerprint_isolation() {
        let tmp = TempDir::new("owner");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=10).collect();
        save(&mut archive, 1, &tokens);

        assert!(archive.lookup(2, &tokens, 10, FP, 100).unwrap().is_none());
        assert!(archive.longest_prefix(2, &tokens, FP).is_none());
        assert!(archive.list(Some(2), 0).is_empty());
        assert_eq!(archive.list(Some(1), 0).len(), 1);
        assert!(archive.lookup(1, &tokens, 10, FP ^ 1, 100).unwrap().is_none());
        assert!(archive.list(None, 0).len() == 1);
        assert!(archive.lookup(1, &tokens, 10, FP, 100).unwrap().is_some());
    }

    #[test]
    fn list_is_newest_used_first_with_limit() {
        let tmp = TempDir::new("list");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        std::thread::sleep(Duration::from_millis(5));
        let b = save(&mut archive, 1, &(10..=16).collect::<Vec<_>>());
        std::thread::sleep(Duration::from_millis(5));
        archive.touch(&a.id);

        let all = archive.list(None, 0);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, a.id);
        assert_eq!(all[1].id, b.id);
        assert_eq!(archive.list(None, 1).len(), 1);
        assert_eq!(archive.list(None, 1)[0].id, a.id);
        // `touch` persisted the new order.
        drop(archive);
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.list(None, 0)[0].id, a.id);
    }

    #[test]
    fn budget_evicts_old_unpinned_entries() {
        let tmp = TempDir::new("budget");
        let dir = tmp.path();
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + 400;
        let mut archive = Archive::open(cfg).unwrap();
        let mut saved = Vec::new();
        for i in 0..6u32 {
            // Distinct first token: no lineage relationship, so `versions` pruning does not interfere.
            let tokens: Vec<u32> = (0..10).map(|j| i * 1000 + j + 1).collect();
            std::thread::sleep(Duration::from_millis(2));
            saved.push(save(&mut archive, 1, &tokens));
        }
        let limit = RESERVE_BYTES + 400 - RESERVE_BYTES;
        assert!(archive.stats().bytes <= limit, "{:?}", archive.stats());
        assert!(archive.stats().evictions > 0);
        let remaining: Vec<String> = archive.list(None, 0).into_iter().map(|e| e.id).collect();
        assert!(remaining.contains(&saved[5].id), "the newest entry must survive");
        assert!(!remaining.contains(&saved[0].id), "the oldest entry must be evicted");
    }

    #[test]
    fn budget_rejects_an_entry_that_cannot_fit_and_deletes_it() {
        let tmp = TempDir::new("toobig");
        let dir = tmp.path();
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + 50;
        let mut archive = Archive::open(cfg).unwrap();
        let tokens: Vec<u32> = (1..=10).collect();
        let (id, path) = match archive.prepare_save(1, &tokens, FP, "test").unwrap() {
            SaveSlot::New { id, path } => (id, path),
            SaveSlot::Existing(_) => panic!("expected New"),
        };
        write_slot(&path, 1, FP, &tokens);
        assert!(path.exists());
        let err = archive
            .commit_save(&id, &path, 1, &tokens, FP, "test")
            .unwrap_err();
        assert!(err.to_string().contains("does not fit"), "{err}");
        assert!(!path.exists(), "the rejected file must be deleted");
        assert_eq!(archive.stats().entries, 0);
    }

    #[test]
    fn versions_prune_lineage_lazily() {
        let tmp = TempDir::new("versions");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=40).collect();
        let a = save(&mut archive, 1, &tokens[..10]);
        let b = save(&mut archive, 1, &tokens[..20]);
        let c = save(&mut archive, 1, &tokens[..30]);
        assert_eq!(b.parent.as_deref(), Some(a.id.as_str()));
        assert_eq!(c.parent.as_deref(), Some(b.id.as_str()));
        // Commits are lazy: all three survive even with the default `versions = 1`.
        assert_eq!(
            archive.list(None, 0).len(),
            3,
            "commit_save must not prune ancestors"
        );
        assert!(a.file.exists() && b.file.exists());

        // Only budget pressure applies the versions policy.
        drop(archive);
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + c.bytes;
        let mut archive = Archive::open(cfg).unwrap();
        let report = archive.gc().unwrap();
        assert_eq!(report.evicted, 2);
        let left = archive.list(None, 0);
        assert_eq!(left.len(), 1, "{left:?}");
        assert_eq!(left[0].id, c.id);
        assert!(!a.file.exists());
        assert!(!b.file.exists());

        // `versions = 2` keeps the two newest entries of a lineage.
        drop(archive);
        archive = Archive::open(config(dir)).unwrap();
        archive.delete(&c.id).unwrap();
        let base: Vec<u32> = (100..=140).collect();
        let _e1 = save(&mut archive, 1, &base[..10]);
        let _e2 = save(&mut archive, 1, &base[..20]);
        let e3 = save(&mut archive, 1, &base[..30]);
        let e4 = save(&mut archive, 1, &base[..40]);
        assert_eq!(archive.list(None, 0).len(), 4);
        drop(archive);

        let mut cfg = config(dir);
        cfg.versions = 2;
        cfg.budget_bytes = RESERVE_BYTES + e3.bytes + e4.bytes;
        let mut archive = Archive::open(cfg).unwrap();
        let report = archive.gc().unwrap();
        assert_eq!(report.evicted, 2);
        let mut left = archive.list(None, 0);
        left.sort_by_key(|e| e.n_tokens);
        assert_eq!(left.len(), 2, "{left:?}");
        assert_eq!(left[0].id, e3.id);
        assert_eq!(left[1].id, e4.id);
    }

    #[test]
    fn delete_removes_the_file_and_persists() {
        let tmp = TempDir::new("delete");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=7).collect::<Vec<_>>());
        let b = save(&mut archive, 1, &(20..=27).collect::<Vec<_>>());
        let freed = archive.delete(&a.id).unwrap();
        assert_eq!(freed, a.bytes);
        assert!(!a.file.exists());
        assert_eq!(archive.stats().entries, 1);
        assert_eq!(archive.delete("no-such-id").unwrap(), 0);
        assert_eq!(archive.list(None, 0)[0].id, b.id);
        drop(archive);
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 1);
        assert_eq!(archive.stats().bytes, b.bytes);
    }

    #[test]
    fn gc_evicts_missing_files_then_lru_to_target() {
        let tmp = TempDir::new("gc");
        let dir = tmp.path();
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + 10_000;
        let mut archive = Archive::open(cfg).unwrap();
        let mut saved = Vec::new();
        for i in 0..4u32 {
            let tokens: Vec<u32> = (0..10).map(|j| i * 1000 + j + 1).collect();
            std::thread::sleep(Duration::from_millis(2));
            saved.push(save(&mut archive, 1, &tokens));
        }
        // An entry whose file disappeared is unusable and goes first.
        fs::remove_file(&saved[1].file).unwrap();
        let report = archive.gc().unwrap();
        assert_eq!(report.evicted, 1);
        assert!(report.freed_bytes > 0);
        assert!(!archive.list(None, 0).iter().any(|e| e.id == saved[1].id));
        assert_eq!(archive.stats().entries, 3);
        // Nothing else needs eviction at this budget.
        assert_eq!(archive.gc().unwrap().evicted, 0);

        // Shrink the budget: GC evicts oldest-first down to budget - reserve.
        drop(archive);
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + 250;
        let mut archive = Archive::open(cfg).unwrap();
        let report = archive.gc().unwrap();
        assert!(report.evicted >= 1);
        assert!(archive.stats().bytes <= 250, "{:?}", archive.stats());
        assert_eq!(archive.stats().entries, 2);
    }

    #[test]
    fn entry_path_rejects_escapes_and_open_rebuilds_over_them() {
        let dir = Path::new("/tmp/archive");
        assert!(entry_path(dir, "/etc/passwd").is_err());
        assert!(entry_path(dir, "../evil.slot").is_err());
        assert!(entry_path(dir, "a/../../evil.slot").is_err());
        assert_eq!(
            entry_path(dir, "ab/x.slot").unwrap(),
            PathBuf::from("/tmp/archive/ab/x.slot")
        );

        // A manifest with a traversal path is treated as corrupt and rebuilt from the shards.
        let tmp = TempDir::new("paths");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        drop(archive);
        let mut json = manifest_json(dir);
        json["entries"][0]["file"] = serde_json::Value::String("../evil.slot".to_string());
        write_manifest_json(dir, &json);

        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 1);
        assert_eq!(archive.list(None, 0)[0].id, entry.id);
        let bad = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().starts_with("manifest.json.bad."));
        assert!(bad);
    }

    #[test]
    fn open_drops_entries_whose_file_is_missing() {
        let tmp = TempDir::new("missing");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        let b = save(&mut archive, 1, &(20..=26).collect::<Vec<_>>());
        drop(archive);
        fs::remove_file(&a.file).unwrap();
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 1);
        assert_eq!(archive.list(None, 0)[0].id, b.id);
        // The manifest was rewritten without the dead row.
        assert_eq!(manifest_json(dir)["entries"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn lease_blocks_delete_and_gc() {
        let tmp = TempDir::new("leased");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = save(&mut archive, 1, &(1..=8).collect::<Vec<_>>());

        let lease = archive.lease(&entry.id).unwrap();
        assert_eq!(lease.id(), entry.id);
        assert_eq!(archive.leases(&entry.id), 1);
        let err = archive.delete(&entry.id).unwrap_err();
        assert!(err.to_string().contains("leased"), "{err}");
        assert!(entry.file.exists());
        assert_eq!(archive.stats().entries, 1);
        drop(lease);
        assert_eq!(archive.leases(&entry.id), 0);
        assert_eq!(archive.delete(&entry.id).unwrap(), entry.bytes);
        assert!(!entry.file.exists());

        // GC skips a leased entry even when the budget is over the target.
        let entry = save(&mut archive, 2, &(20..=28).collect::<Vec<_>>());
        drop(archive);
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + 1;
        let mut archive = Archive::open(cfg).unwrap();
        let lease = archive.lease(&entry.id).unwrap();
        let report = archive.gc().unwrap();
        assert_eq!(report.evicted, 0);
        assert_eq!(report.freed_bytes, 0);
        assert!(entry.file.exists());
        assert_eq!(archive.leases(&entry.id), 1);

        drop(lease);
        let report = archive.gc().unwrap();
        assert_eq!(report.evicted, 1);
        assert_eq!(report.freed_bytes, entry.bytes);
        assert!(!entry.file.exists());
        assert_eq!(archive.stats().entries, 0);
    }

    #[test]
    fn double_lease_needs_two_drops() {
        let tmp = TempDir::new("double-lease");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = save(&mut archive, 1, &(1..=7).collect::<Vec<_>>());

        let first = archive.lease(&entry.id).unwrap();
        let second = archive.lease(&entry.id).unwrap();
        assert_eq!(archive.leases(&entry.id), 2);
        drop(first);
        assert_eq!(archive.leases(&entry.id), 1);
        assert!(
            archive.delete(&entry.id).is_err(),
            "one lease must still block delete"
        );
        drop(second);
        assert_eq!(archive.leases(&entry.id), 0);
        assert_eq!(archive.delete(&entry.id).unwrap(), entry.bytes);

        // Unknown ids cannot be leased; a stale guard is harmless.
        assert!(archive.lease("0000000000000000").is_none());
        assert_eq!(archive.leases("no-such-entry"), 0);
    }

    #[test]
    fn lookup_leased_matches_lookup_and_counts() {
        let tmp = TempDir::new("lookup-leased");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=8).collect();
        let short = save(&mut archive, 1, &tokens[..4]);
        let long = save(&mut archive, 1, &tokens[..6]);

        let (hit, lease) = archive.lookup_leased(1, &tokens, 8, FP, 100).unwrap().unwrap();
        assert_eq!(hit.id, long.id);
        assert_eq!(hit.n_tokens, 6);
        assert_eq!(archive.leases(&long.id), 1);
        assert_eq!(archive.leases(&short.id), 0);
        assert_eq!(archive.stats().hits, 1);

        // Same exact match as the unleased lookup.
        let unleased = archive.lookup(1, &tokens, 8, FP, 100).unwrap().unwrap();
        assert_eq!(unleased.id, long.id);
        assert_eq!(archive.stats().hits, 2);

        // A miss increments the miss counter and takes no lease.
        assert!(archive.lookup_leased(2, &tokens, 8, FP, 100).unwrap().is_none());
        assert_eq!(archive.stats().misses, 1);
        assert_eq!(archive.leases(&long.id), 1);
        // `keep`/`ctx` behave exactly as in `lookup`.
        let (hit, short_lease) = archive.lookup_leased(1, &tokens, 4, FP, 100).unwrap().unwrap();
        assert_eq!(hit.id, short.id);
        assert_eq!(archive.stats().hits, 3);
        assert_eq!(archive.leases(&short.id), 1);
        drop(short_lease);

        // The lookup lease protects the entry from DELETE.
        assert!(archive.delete(&long.id).is_err());
        drop(lease);
        assert_eq!(archive.leases(&long.id), 0);
        assert_eq!(archive.delete(&long.id).unwrap(), long.bytes);
    }

    #[test]
    fn leases_do_not_leak_after_drop() {
        let tmp = TempDir::new("lease-leak");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = save(&mut archive, 1, &(1..=9).collect::<Vec<_>>());

        for _ in 0..64 {
            let lease = archive.lease(&entry.id).unwrap();
            assert_eq!(archive.leases(&entry.id), 1);
            drop(lease);
        }
        assert_eq!(archive.leases(&entry.id), 0);

        // A guard can outlive the Archive: dropping it must not panic or leak into a reopened store.
        let lease = archive.lease(&entry.id).unwrap();
        drop(archive);
        drop(lease);

        let mut archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.leases(&entry.id), 0);
        assert_eq!(archive.delete(&entry.id).unwrap(), entry.bytes);
        assert!(archive.list(None, 0).is_empty());
    }

    #[test]
    fn policy_is_off_by_default_and_round_trips() {
        let tmp = TempDir::new("policy");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.policy(), (0, 0, 0));
        archive.set_policy(86_400_000, 1 << 30, 1 << 28);
        assert_eq!(archive.policy(), (86_400_000, 1 << 30, 1 << 28));
        archive.set_policy(0, 0, 0);
        assert_eq!(archive.policy(), (0, 0, 0));
    }

    #[test]
    fn ttl_evicts_expired_and_keeps_fresh() {
        let tmp = TempDir::new("ttl");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let old = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        let fresh = save(&mut archive, 1, &(20..=25).collect::<Vec<_>>());
        archive.set_policy(1_000, 0, 0);
        let now = now_ms();
        set_last_used(&mut archive, &old.id, now.saturating_sub(10_000));
        set_last_used(&mut archive, &fresh.id, now.saturating_sub(100));

        let report = archive.gc().unwrap();
        assert_eq!((report.evicted, report.expired), (1, 1), "{report:?}");
        assert!(report.freed_bytes >= old.bytes);
        assert!(!old.file.exists(), "the expired file must be removed");
        assert!(fresh.file.exists(), "the fresh entry must survive");
        assert_eq!(archive.stats().expired, 1);
        assert_eq!(archive.list(None, 0).len(), 1);

        // A `last_used_ms` in the future (clock skew) must not expire or panic.
        set_last_used(&mut archive, &fresh.id, now.saturating_add(1_000_000));
        let report = archive.gc().unwrap();
        assert_eq!((report.evicted, report.expired), (0, 0));
        assert!(fresh.file.exists());
    }

    #[test]
    fn ttl_zero_is_off() {
        let tmp = TempDir::new("ttl-off");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        set_last_used(&mut archive, &entry.id, 0);
        let report = archive.gc().unwrap();
        assert_eq!((report.evicted, report.expired), (0, 0));
        assert_eq!(archive.stats().expired, 0);
        assert!(entry.file.exists());
        assert_eq!(archive.stats().entries, 1);
    }

    #[test]
    fn per_key_quota_evicts_only_that_owner() {
        let tmp = TempDir::new("perkey");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a1 = save(&mut archive, 1, &(1..=10).collect::<Vec<_>>());
        std::thread::sleep(Duration::from_millis(2));
        let a2 = save(&mut archive, 1, &(21..=30).collect::<Vec<_>>());
        // Owner 0 is a normal bucket, not a special case.
        let b1 = save(&mut archive, 0, &(41..=50).collect::<Vec<_>>());
        assert_eq!(a1.bytes, a2.bytes);

        // Room for two entries per owner: the third owner-1 save evicts owner 1's oldest entry only.
        archive.set_policy(0, a1.bytes * 2 + 26, 0);
        std::thread::sleep(Duration::from_millis(2));
        let a3 = save(&mut archive, 1, &(61..=70).collect::<Vec<_>>());

        let ids: Vec<String> = archive.list(None, 0).into_iter().map(|e| e.id).collect();
        assert!(ids.contains(&a2.id), "{ids:?}");
        assert!(ids.contains(&a3.id), "{ids:?}");
        assert!(ids.contains(&b1.id), "owner 0 must be untouched: {ids:?}");
        assert!(!ids.contains(&a1.id), "the oldest owner-1 entry must go: {ids:?}");
        assert!(!a1.file.exists());
        assert!(a2.file.exists() && b1.file.exists());
        assert_eq!(archive.list(Some(0), 0).len(), 1);
    }

    #[test]
    fn per_key_quota_boundary_and_oversize_entry() {
        let tmp = TempDir::new("perkey-boundary");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a1 = save(&mut archive, 1, &(1..=10).collect::<Vec<_>>());
        std::thread::sleep(Duration::from_millis(2));
        let a2 = save(&mut archive, 1, &(21..=30).collect::<Vec<_>>());
        let size = a1.bytes;
        assert_eq!(a2.bytes, size);

        // Exactly at the quota: both entries fit (`>` not `>=`).
        archive.set_policy(0, size * 2, 0);
        assert_eq!(archive.stats().entries, 2);
        assert!(a1.file.exists() && a2.file.exists());

        // The new entry makes the owner exceed the quota by exactly one entry; evicting the oldest brings
        // the bucket back to exactly the quota, which is allowed.
        std::thread::sleep(Duration::from_millis(2));
        let a3 = save(&mut archive, 1, &(41..=50).collect::<Vec<_>>());
        assert!(!a1.file.exists(), "the oldest entry must be evicted");
        assert!(a2.file.exists() && a3.file.exists());

        // One byte under two entries: the owner cannot keep two, so both older entries go.
        archive.set_policy(0, size * 2 - 1, 0);
        std::thread::sleep(Duration::from_millis(2));
        let a4 = save(&mut archive, 1, &(51..=60).collect::<Vec<_>>());
        let owner_entries = archive.list(Some(1), 0);
        assert_eq!(owner_entries.len(), 1, "{owner_entries:?}");
        assert_eq!(owner_entries[0].id, a4.id);
        assert!(!a2.file.exists() && !a3.file.exists());

        // An entry that alone exceeds the quota is rejected and its file deleted.
        archive.set_policy(0, size - 1, 0);
        let tokens: Vec<u32> = (61..=70).collect();
        let (id, path) = match archive.prepare_save(9, &tokens, FP, "test").unwrap() {
            SaveSlot::New { id, path } => (id, path),
            SaveSlot::Existing(_) => panic!("expected New"),
        };
        write_slot(&path, 9, FP, &tokens);
        let err = archive
            .commit_save(&id, &path, 9, &tokens, FP, "test")
            .unwrap_err();
        assert!(err.to_string().contains("per-owner quota"), "{err}");
        assert!(!path.exists(), "the rejected file must be deleted");
        assert!(archive.list(Some(9), 0).is_empty());
    }

    #[test]
    fn per_key_quota_protects_pinned_and_leased() {
        let tmp = TempDir::new("perkey-protected");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a1 = save(&mut archive, 1, &(1..=10).collect::<Vec<_>>());
        std::thread::sleep(Duration::from_millis(2));
        let a2 = save(&mut archive, 1, &(21..=30).collect::<Vec<_>>());
        std::thread::sleep(Duration::from_millis(2));
        let a3 = save(&mut archive, 1, &(41..=50).collect::<Vec<_>>());
        let size = a1.bytes;

        archive.set_pinned(&a1.id, true).unwrap();
        let lease = archive.lease(&a2.id).unwrap();
        // Three entries fill the quota; a fourth must evict only the unpinned, unleased one.
        archive.set_policy(0, size * 3, 0);
        std::thread::sleep(Duration::from_millis(2));
        let a4 = save(&mut archive, 1, &(61..=70).collect::<Vec<_>>());
        assert!(a1.file.exists() && a2.file.exists() && a4.file.exists());
        assert!(!a3.file.exists(), "only the unprotected entry may be evicted");

        // With every remaining entry protected, a new save fails clearly instead of breaking the quota.
        let a4_lease = archive.lease(&a4.id).unwrap();
        archive.set_policy(0, size * 2, 0);
        let tokens: Vec<u32> = (81..=90).collect();
        let (id, path) = match archive.prepare_save(1, &tokens, FP, "test").unwrap() {
            SaveSlot::New { id, path } => (id, path),
            SaveSlot::Existing(_) => panic!("expected New"),
        };
        write_slot(&path, 1, FP, &tokens);
        let err = archive
            .commit_save(&id, &path, 1, &tokens, FP, "test")
            .unwrap_err();
        assert!(err.to_string().contains("pinned or leased"), "{err}");
        assert!(!path.exists());
        assert_eq!(archive.stats().entries, 3);
        drop((lease, a4_lease));
    }

    #[test]
    fn per_key_quota_still_enforces_the_global_budget() {
        let tmp = TempDir::new("perkey-global");
        let dir = tmp.path();
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + 400;
        let mut archive = Archive::open(cfg).unwrap();
        archive.set_policy(0, 1 << 30, 0);
        let mut saved = Vec::new();
        for i in 0..6u32 {
            let tokens: Vec<u32> = (0..10).map(|j| i * 1000 + j + 1).collect();
            std::thread::sleep(Duration::from_millis(2));
            saved.push(save(&mut archive, 1, &tokens));
        }
        assert!(archive.stats().bytes <= 400, "{:?}", archive.stats());
        assert!(archive.stats().evictions > 0);
        assert!(archive.list(None, 0).iter().any(|e| e.id == saved[5].id));
        assert!(archive.list(None, 0).iter().all(|e| e.id != saved[0].id));
    }

    #[test]
    fn pin_persists_across_open() {
        let tmp = TempDir::new("pin-persist");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        let b = save(&mut archive, 1, &(20..=25).collect::<Vec<_>>());
        assert_eq!(archive.pinned_bytes(), 0);
        archive.set_pinned(&a.id, true).unwrap();
        assert_eq!(archive.pinned_bytes(), a.bytes);
        // Idempotent: pinning twice must not double-count.
        archive.set_pinned(&a.id, true).unwrap();
        assert_eq!(archive.pinned_bytes(), a.bytes);
        drop(archive);

        let mut archive = Archive::open(config(dir)).unwrap();
        let entries = archive.list(None, 0);
        assert!(entries.iter().find(|e| e.id == a.id).unwrap().pinned);
        assert!(!entries.iter().find(|e| e.id == b.id).unwrap().pinned);
        assert_eq!(archive.pinned_bytes(), a.bytes);

        archive.set_pinned(&a.id, false).unwrap();
        assert_eq!(archive.pinned_bytes(), 0);
        drop(archive);
        let archive = Archive::open(config(dir)).unwrap();
        assert!(!archive.list(None, 0).iter().any(|e| e.pinned));
        assert_eq!(archive.pinned_bytes(), 0);
    }

    #[test]
    fn pin_cap_is_enforced_and_unpin_always_works() {
        let tmp = TempDir::new("pin-cap");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        let b = save(&mut archive, 1, &(20..=25).collect::<Vec<_>>());
        archive.set_policy(0, 0, a.bytes + 10);
        archive.set_pinned(&a.id, true).unwrap();
        assert_eq!(archive.pinned_bytes(), a.bytes);

        let err = archive.set_pinned(&b.id, true).unwrap_err();
        assert!(err.to_string().contains("pin cap"), "{err}");
        assert!(!archive.list(None, 0).iter().find(|e| e.id == b.id).unwrap().pinned);
        assert_eq!(archive.pinned_bytes(), a.bytes);

        // Unpinning is never blocked, even by a cap smaller than the current pinned total.
        archive.set_policy(0, 0, 1);
        archive.set_pinned(&a.id, false).unwrap();
        assert_eq!(archive.pinned_bytes(), 0);
        assert!(archive.set_pinned(&b.id, true).is_err());

        // Zero means unlimited.
        archive.set_policy(0, 0, 0);
        archive.set_pinned(&b.id, true).unwrap();
        assert_eq!(archive.pinned_bytes(), b.bytes);

        assert!(archive.set_pinned("no-such-entry", true).is_err());
    }

    #[test]
    fn pinned_survives_gc_and_ttl() {
        let tmp = TempDir::new("pin-gc");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        archive.set_pinned(&entry.id, true).unwrap();
        archive.set_policy(1_000, 0, 0);
        set_last_used(&mut archive, &entry.id, 0);

        let report = archive.gc().unwrap();
        assert_eq!((report.evicted, report.expired), (0, 0));
        assert!(entry.file.exists());
        assert_eq!(archive.stats().entries, 1);

        // A pin also protects the entry from budget pressure, even when it is over budget by itself.
        drop(archive);
        let mut cfg = config(dir);
        cfg.budget_bytes = RESERVE_BYTES + 1;
        let mut archive = Archive::open(cfg).unwrap();
        let report = archive.gc().unwrap();
        assert_eq!(report.evicted, 0);
        assert_eq!(archive.stats().entries, 1);
        assert!(entry.file.exists());
    }

    /// Build a v2 entry without a GPU: 4-token segments with a deterministic payload and a `STRSTATE` state file.
    /// The payload is not a real `STRRSEG`; these tests only exercise the archive/format layer.
    fn v2_entry(archive: &mut Archive, owner: u64, tokens: &[u32], reason: &str) -> ArchiveEntry {
        let prefix = prefix_hash(tokens);
        let id = id_for(FP, owner, tokens.len(), prefix);
        let mut parent: Option<String> = None;
        let mut refs = Vec::new();
        let mut new_bytes = 0u64;
        for (index, chunk) in tokens.chunks(4).enumerate() {
            let start = index * 4;
            let end = start + chunk.len();
            let expected_id = slot_segments::segment_id(FP, parent.as_deref(), chunk);
            let existed = archive.segments.contains(&expected_id);
            let payload: Vec<u8> = chunk.iter().flat_map(|t| t.to_le_bytes()).collect();
            let segment = archive
                .segments
                .put(FP, parent.as_deref(), chunk, &payload)
                .unwrap();
            if !existed {
                new_bytes = new_bytes.saturating_add(segment.bytes);
            }
            refs.push(SegmentRef { id: segment.id.clone(), start, end });
            parent = Some(segment.id);
        }
        let path = state_path(&archive.cfg.dir, &id).unwrap();
        let state_bytes = write_state_file_atomic(
            &path,
            &StateHeader { fingerprint: FP, owner, rope_offset: 0, n_tokens: tokens.len() },
            &refs,
            tokens,
            b"test-recurrent-payload",
        )
        .unwrap();
        let now = now_ms();
        let entry = ArchiveEntry {
            id: id.clone(),
            file: path.clone(),
            state_file: Some(path),
            segments: refs,
            bytes: state_bytes.saturating_add(new_bytes),
            n_tokens: tokens.len(),
            owner,
            fingerprint: FP,
            prefix_hash: prefix,
            created_ms: now,
            last_used_ms: now,
            parent: None,
            pinned: false,
            reason: reason.to_string(),
            format: 2,
        };
        archive.entries.insert(id, entry.clone());
        archive.compact().unwrap();
        entry
    }

    /// A synthetic valid 4-token segment chain for state-file tests (no segment files needed).
    fn state_refs(tokens: &[u32]) -> Vec<SegmentRef> {
        let mut refs = Vec::new();
        let mut parent: Option<String> = None;
        for (index, chunk) in tokens.chunks(4).enumerate() {
            let start = index * 4;
            let id = slot_segments::segment_id(FP, parent.as_deref(), chunk);
            refs.push(SegmentRef { id: id.clone(), start, end: start + chunk.len() });
            parent = Some(id);
        }
        refs
    }

    #[test]
    fn state_file_round_trip_and_header_validation() {
        let tmp = TempDir::new("state");
        let dir = tmp.path();
        let path = dir.join("ab").join("entry.state");
        let tokens: Vec<u32> = (1..=17).collect();
        let payload = b"opaque recurrent payload".to_vec();
        let header = StateHeader { fingerprint: FP, owner: 0x1234, rope_offset: -7, n_tokens: tokens.len() };
        let refs = state_refs(&tokens);
        let bytes = write_state_file_atomic(&path, &header, &refs, &tokens, &payload).unwrap();
        assert_eq!(bytes, fs::metadata(&path).unwrap().len());
        assert_eq!(read_state_tokens(&path).unwrap(), tokens);
        let (got, got_refs, got_tokens, got_payload) = read_state_file(&path).unwrap();
        assert_eq!(got, header);
        assert_eq!(got_refs, refs);
        assert_eq!(got_tokens, tokens);
        assert_eq!(got_payload, payload);

        // A wrong token count in the header is refused before anything is written.
        let bad = StateHeader { n_tokens: tokens.len() + 1, ..header };
        assert!(write_state_file_atomic(&dir.join("bad.state"), &bad, &refs, &tokens, &payload).is_err());
        assert!(!dir.join("bad.state").exists());

        let raw = fs::read(&path).unwrap();
        assert_eq!(&raw[..8], &STATE_MAGIC);
    }

    #[test]
    fn state_file_rejects_corrupt_headers_and_lengths() {
        let tmp = TempDir::new("state-corrupt");
        let dir = tmp.path();
        let path = dir.join("entry.state");
        let tokens: Vec<u32> = (1..=9).collect();
        let header = StateHeader { fingerprint: FP, owner: 1, rope_offset: 0, n_tokens: tokens.len() };
        write_state_file_atomic(&path, &header, &state_refs(&tokens), &tokens, b"payload").unwrap();
        let good = fs::read(&path).unwrap();

        let mut bad = good.clone();
        bad[0] ^= 0xff;
        fs::write(&path, &bad).unwrap();
        let err = read_state_tokens(&path).unwrap_err();
        assert!(format!("{err:#}").contains("magic"), "{err:#}");

        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&(STATE_VERSION + 1).to_le_bytes());
        fs::write(&path, &bad).unwrap();
        let err = read_state_tokens(&path).unwrap_err();
        assert!(format!("{err:#}").contains("version"), "{err:#}");

        // n_tokens = 0 and a token count that disagrees with the file length.
        let mut bad = good.clone();
        bad[32..36].copy_from_slice(&0u32.to_le_bytes());
        fs::write(&path, &bad).unwrap();
        assert!(read_state_tokens(&path).is_err());

        let mut bad = good.clone();
        bad[32..36].copy_from_slice(&(tokens.len() as u32 + 1).to_le_bytes());
        fs::write(&path, &bad).unwrap();
        assert!(read_state_tokens(&path).is_err());

        let mut bad = good.clone();
        bad.push(0);
        fs::write(&path, &bad).unwrap();
        let err = read_state_file(&path).unwrap_err();
        assert!(format!("{err:#}").contains("bytes"), "{err:#}");

        fs::write(&path, &good[..good.len() - 1]).unwrap();
        assert!(read_state_file(&path).is_err());
    }

    #[test]
    fn v2_manifest_round_trip_and_v1_backcompat() {
        let tmp = TempDir::new("v2-manifest");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let v1 = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        // A persisted v1 row has neither v2 field.
        let json = manifest_json(dir);
        assert!(json["entries"].as_array().unwrap().iter().all(|row| row.get("segments").is_none()));
        assert!(json["entries"].as_array().unwrap().iter().all(|row| row.get("state_file").is_none()));

        let tokens: Vec<u32> = (1..=13).collect();
        let v2 = v2_entry(&mut archive, 2, &tokens, "segmented");
        assert_eq!(v2.format, 2);
        assert_eq!(v2.segments.len(), 4);
        assert_eq!(v2.segments.iter().map(|s| s.end - s.start).sum::<usize>(), tokens.len());
        let json = manifest_json(dir);
        let row = json["entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == v2.id)
            .unwrap();
        assert_eq!(row["format"], 2);
        assert_eq!(row["segments"].as_array().unwrap().len(), 4);
        assert!(row["state_file"].as_str().unwrap().ends_with(".state"));
        drop(archive);

        let archive = Archive::open(config(dir)).unwrap();
        let entries = archive.list(None, 0);
        assert_eq!(entries.len(), 2);
        let loaded = entries.iter().find(|e| e.id == v2.id).unwrap();
        assert_eq!(loaded.segments, v2.segments);
        assert_eq!(loaded.state_file, v2.state_file);
        assert_eq!(loaded.bytes, v2.bytes);
        assert!(entries.iter().any(|e| e.id == v1.id && e.format == 1));
    }

    #[test]
    fn entry_token_reader_handles_v1_and_v2() {
        let tmp = TempDir::new("tokens");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let v1_tokens: Vec<u32> = (10..=18).collect();
        let v1 = save(&mut archive, 1, &v1_tokens);
        assert_eq!(read_entry_tokens(&v1).unwrap(), v1_tokens);
        let v2_tokens: Vec<u32> = (30..=39).collect();
        let v2 = v2_entry(&mut archive, 1, &v2_tokens, "segmented");
        assert_eq!(read_entry_tokens(&v2).unwrap(), v2_tokens);

        let missing = ArchiveEntry { state_file: None, ..v2.clone() };
        assert!(read_entry_tokens(&missing).is_err(), "a v2 entry without a state file must not silently read file");
    }

    #[test]
    fn lookup_and_longest_prefix_see_v2_entries() {
        let tmp = TempDir::new("v2-lookup");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=20).collect();
        let short = v2_entry(&mut archive, 3, &tokens[..8], "segmented");
        let long = v2_entry(&mut archive, 3, &tokens[..16], "segmented");

        let hit = archive.lookup(3, &tokens, 20, FP, 100).unwrap().unwrap();
        assert_eq!(hit.id, long.id);
        assert_eq!(hit.format, 2);
        // Exact token proof comes from the state file: a change inside the saved prefix rejects the entry.
        let mut changed = tokens.clone();
        changed[3] = 999;
        assert!(archive.lookup(3, &changed, 20, FP, 100).unwrap().is_none());
        let mut after = tokens.clone();
        after[10] = 999;
        assert_eq!(archive.lookup(3, &after, 20, FP, 100).unwrap().unwrap().id, short.id);
        assert_eq!(archive.longest_prefix(3, &tokens, FP).unwrap().id, long.id);
    }

    #[test]
    fn delete_v2_releases_only_its_own_segment_refs() {
        let tmp = TempDir::new("v2-delete");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=20).collect();
        let first = v2_entry(&mut archive, 1, &tokens[..12], "segmented");
        let second = v2_entry(&mut archive, 1, &tokens[..20], "segmented");

        // The first three 4-token segments are shared by both entries.
        let shared: Vec<String> = first.segments.iter().map(|s| s.id.clone()).collect();
        for id in &shared {
            let refs = archive.segments.segment(id).unwrap().refs;
            assert_eq!(refs, 2, "segment {id} must be referenced by both entries");
        }
        let files: Vec<PathBuf> = shared
            .iter()
            .map(|id| archive.segments.segment(id).unwrap().file.clone())
            .collect();
        let freed = archive.delete(&first.id).unwrap();
        assert_eq!(freed, first.bytes);
        assert!(!first.state_file.as_ref().unwrap().exists());
        assert_eq!(archive.stats().entries, 1);
        for (id, file) in shared.iter().zip(&files) {
            assert_eq!(archive.segments.segment(id).unwrap().refs, 1, "{id}");
            assert!(file.exists(), "a shared segment must survive its first entry");
        }
        let tail: Vec<String> = second.segments[3..].iter().map(|s| s.id.clone()).collect();
        assert_eq!(tail.len(), 2);
        assert!(tail.iter().all(|id| archive.segments.contains(id)));

        archive.delete(&second.id).unwrap();
        assert!(archive.segments.list().is_empty(), "all refs released");
        assert!(files.iter().all(|file| !file.exists()));
        drop(archive);
        let archive = Archive::open(config(dir)).unwrap();
        assert!(archive.segments().list().is_empty());
        assert_eq!(archive.stats().entries, 0);
    }

    #[test]
    fn open_reconciles_segment_refs_after_a_manifest_rebuild() {
        let tmp = TempDir::new("v2-reconcile");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=20).collect();
        let first = v2_entry(&mut archive, 1, &tokens[..12], "segmented");
        let second = v2_entry(&mut archive, 1, &tokens[..20], "segmented");
        let shared = second.segments[0].id.clone();
        let shared_file = archive.segments.segment(&shared).unwrap().file.clone();
        assert_eq!(archive.segments.segment(&shared).unwrap().refs, 2);
        drop(archive);

        // A rebuild can only adopt each file with refs = 1; opening the archive must restore the true counts
        // from the entries, or deleting one entry would remove a shared file still used by the other.
        fs::write(dir.join(slot_segments::MANIFEST), b"{ not json").unwrap();
        let mut archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 2);
        assert_eq!(archive.segments.segment(&shared).unwrap().refs, 2);
        archive.delete(&first.id).unwrap();
        assert_eq!(archive.segments.segment(&shared).unwrap().refs, 1);
        assert!(shared_file.exists());
        archive.delete(&second.id).unwrap();
        assert!(archive.segments().list().is_empty());
        assert!(!shared_file.exists());
    }

    #[test]
    fn gc_and_open_drop_invalid_v2_entries() {
        let tmp = TempDir::new("v2-invalid");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = v2_entry(&mut archive, 1, &(1..=12).collect::<Vec<_>>(), "segmented");
        let segment_id = entry.segments[1].id.clone();
        let segment_file = archive.segments.segment(&segment_id).unwrap().file.clone();
        drop(archive);

        // A corrupt segment is dropped by the store at open; the entry then loses a referenced segment.
        let mut raw = fs::read(&segment_file).unwrap();
        raw[0] ^= 0xff;
        fs::write(&segment_file, &raw).unwrap();
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 0, "an entry with a bad segment must be dropped");
        assert!(!segment_file.exists(), "the corrupt segment file is removed");
        assert!(archive.segments().list().is_empty(), "the dropped entry released its refs");

        // A missing state file is dropped at open and releases the segments it referenced.
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = v2_entry(&mut archive, 2, &(100..=112).collect::<Vec<_>>(), "segmented");
        assert_eq!(archive.segments.list().len(), 4);
        fs::remove_file(entry.state_file.as_ref().unwrap()).unwrap();
        drop(archive);
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 0);
        assert!(archive.segments().list().is_empty());

        // GC also evicts a v2 entry whose segment file disappeared while the process was running.
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = v2_entry(&mut archive, 3, &(200..=212).collect::<Vec<_>>(), "segmented");
        let gone = archive.segments.segment(&entry.segments[0].id).unwrap().file.clone();
        fs::remove_file(&gone).unwrap();
        let report = archive.gc().unwrap();
        assert_eq!(report.evicted, 1);
        assert_eq!(archive.stats().entries, 0);
        assert!(!entry.state_file.as_ref().unwrap().exists());
    }

    #[test]
    fn v2_open_rejects_mismatched_state_header_and_segment_cover() {
        let tmp = TempDir::new("v2-header");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = v2_entry(&mut archive, 4, &(1..=9).collect::<Vec<_>>(), "segmented");

        // Manifest n_tokens disagrees with the state file.
        let mut json = manifest_json(dir);
        let row = json["entries"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|row| row["id"] == entry.id)
            .unwrap();
        row["n_tokens"] = serde_json::json!(entry.n_tokens + 1);
        write_manifest_json(dir, &json);
        drop(archive);
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 0);

        // A non-contiguous segment cover is invalid even when every segment exists.
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = v2_entry(&mut archive, 5, &(20..=29).collect::<Vec<_>>(), "segmented");
        assert!(!segments_cover(&[], entry.n_tokens));
        assert!(!segments_cover(&entry.segments[..1], entry.n_tokens));
        let mut broken = entry.segments.clone();
        broken[1].start += 1;
        assert!(!segments_cover(&broken, entry.n_tokens));
        assert!(segments_cover(&entry.segments, entry.n_tokens));
    }

    /// Write a legacy v1 `STRSTATE` file (no segment refs) for compatibility tests.
    fn write_state_v1(path: &Path, header: &StateHeader, tokens: &[u32], payload: &[u8]) -> u64 {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&STATE_MAGIC).unwrap();
        crate::slot_file::write_u32(&mut file, 1).unwrap();
        crate::slot_file::write_u64(&mut file, header.fingerprint).unwrap();
        crate::slot_file::write_u64(&mut file, header.owner).unwrap();
        file.write_all(&header.rope_offset.to_le_bytes()).unwrap();
        crate::slot_file::write_u32(&mut file, tokens.len() as u32).unwrap();
        for &token in tokens {
            crate::slot_file::write_u32(&mut file, token).unwrap();
        }
        crate::slot_file::write_u64(&mut file, payload.len() as u64).unwrap();
        file.write_all(payload).unwrap();
        file.sync_all().unwrap();
        fs::metadata(path).unwrap().len()
    }

    /// Build a v1 entry on disk without adding it to the manifest (for journal `add` tests).
    fn orphan_v1_entry(dir: &Path, owner: u64, tokens: &[u32], reason: &str) -> ArchiveEntry {
        let id = entry_id(FP, owner, tokens);
        let path = dir.join(&id[..2]).join(format!("{id}.slot"));
        let bytes = write_slot(&path, owner, FP, tokens);
        let now = now_ms();
        ArchiveEntry {
            id,
            file: path,
            state_file: None,
            segments: Vec::new(),
            bytes,
            n_tokens: tokens.len(),
            owner,
            fingerprint: FP,
            prefix_hash: prefix_hash(tokens),
            created_ms: now,
            last_used_ms: now,
            parent: None,
            pinned: false,
            reason: reason.to_string(),
            format: 1,
        }
    }

    #[test]
    fn journal_replays_add_touch_pin_and_delete() {
        let tmp = TempDir::new("journal-replay");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        let b = save(&mut archive, 1, &(20..=26).collect::<Vec<_>>());
        let c = save(&mut archive, 1, &(40..=46).collect::<Vec<_>>());
        assert_eq!(archive.stats().entries, 3);

        let added = orphan_v1_entry(dir, 9, &(80..=86).collect::<Vec<_>>(), "journal-add");
        archive
            .append_journal(&JournalOp::Add {
                v: JOURNAL_VERSION,
                entry: archive.manifest_row(&added),
            })
            .unwrap();
        archive
            .append_journal(&JournalOp::Touch {
                v: JOURNAL_VERSION,
                touches: vec![TouchRow { id: a.id.clone(), last_used_ms: a.last_used_ms + 10_000 }],
            })
            .unwrap();
        archive
            .append_journal(&JournalOp::Pin { v: JOURNAL_VERSION, id: b.id.clone(), pinned: true })
            .unwrap();
        archive
            .append_journal(&JournalOp::Delete { v: JOURNAL_VERSION, id: c.id.clone() })
            .unwrap();
        assert!(dir.join(JOURNAL).exists());
        // The compacted snapshot is unchanged until a compaction; the journal is the delta.
        assert_eq!(manifest_json(dir)["entries"].as_array().unwrap().len(), 3);
        drop(archive);

        let archive = Archive::open(config(dir)).unwrap();
        let entries = archive.list(None, 0);
        assert_eq!(entries.len(), 3, "{entries:?}");
        assert!(entries.iter().any(|e| e.id == added.id));
        assert!(!entries.iter().any(|e| e.id == c.id));
        assert!(entries.iter().find(|e| e.id == b.id).unwrap().pinned);
        assert!(
            entries.iter().find(|e| e.id == a.id).unwrap().last_used_ms >= a.last_used_ms + 10_000
        );
    }

    #[test]
    fn truncated_journal_tail_is_ignored_and_compacted() {
        let tmp = TempDir::new("journal-truncated");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        let b = save(&mut archive, 1, &(20..=26).collect::<Vec<_>>());
        let touched = a.last_used_ms + 20_000;
        archive
            .append_journal(&JournalOp::Touch {
                v: JOURNAL_VERSION,
                touches: vec![TouchRow { id: a.id.clone(), last_used_ms: touched }],
            })
            .unwrap();
        drop(archive);

        // A crash in the middle of the last record leaves no trailing newline. It must not be applied.
        let mut file = fs::OpenOptions::new().append(true).open(dir.join(JOURNAL)).unwrap();
        file.write_all(br#"{"op":"touch","v":1,"touches":[{"id":"#).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let archive = Archive::open(config(dir)).unwrap();
        let entries = archive.list(None, 0);
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|e| e.id == b.id));
        assert!(entries.iter().find(|e| e.id == a.id).unwrap().last_used_ms >= touched);
        // A repaired journal is folded into the snapshot and truncated.
        assert_eq!(fs::metadata(dir.join(JOURNAL)).unwrap().len(), 0);
        assert_eq!(manifest_json(dir)["entries"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn batched_touches_do_not_rewrite_the_snapshot_until_flush() {
        let tmp = TempDir::new("touch-batch");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let a = save(&mut archive, 1, &(1..=6).collect::<Vec<_>>());
        let b = save(&mut archive, 1, &(20..=26).collect::<Vec<_>>());
        let c = save(&mut archive, 1, &(40..=46).collect::<Vec<_>>());
        let before = fs::read(dir.join(MANIFEST)).unwrap();

        std::thread::sleep(Duration::from_millis(3));
        archive.touch(&a.id);
        std::thread::sleep(Duration::from_millis(3));
        archive.touch(&b.id);
        std::thread::sleep(Duration::from_millis(3));
        archive.touch(&c.id);
        // Touches are in memory only: no snapshot rewrite and no journal record yet.
        assert_eq!(fs::read(dir.join(MANIFEST)).unwrap(), before);
        assert_eq!(
            fs::metadata(dir.join(JOURNAL)).map(|m| m.len()).unwrap_or(0),
            0,
            "touch must be cheap until a flush is due"
        );
        assert_eq!(archive.list(None, 0)[0].id, c.id);

        archive.flush().unwrap();
        assert_eq!(fs::read(dir.join(MANIFEST)).unwrap(), before, "flush only appends the journal");
        assert!(fs::metadata(dir.join(JOURNAL)).unwrap().len() > 0);
        let touched_at = archive.list(None, 0)[0].last_used_ms;
        drop(archive);

        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.list(None, 0)[0].id, c.id);
        assert!(archive.list(None, 0)[0].last_used_ms >= touched_at);
    }

    #[test]
    fn v2_rebuilds_from_state_refs_and_segment_files() {
        let tmp = TempDir::new("v2-rebuild-state");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=20).collect();
        let entry = v2_entry(&mut archive, 3, &tokens, "segmented");
        let expected_bytes = entry.bytes;
        drop(archive);

        // Both manifests are lost; the state file carries the refs and the `.seg` files are the data.
        fs::remove_file(dir.join(MANIFEST)).unwrap();
        fs::remove_file(dir.join(slot_segments::MANIFEST)).unwrap();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entries = archive.list(None, 0);
        assert_eq!(entries.len(), 1, "{entries:?}");
        let rebuilt = &entries[0];
        assert_eq!(rebuilt.id, entry.id);
        assert_eq!(rebuilt.format, 2);
        assert_eq!(rebuilt.segments, entry.segments);
        assert_eq!(rebuilt.n_tokens, tokens.len());
        assert_eq!(rebuilt.bytes, expected_bytes);
        assert_eq!(read_entry_tokens(rebuilt).unwrap(), tokens);
        for segment in &rebuilt.segments {
            assert_eq!(archive.segments().segment(&segment.id).unwrap().refs, 1);
        }
        // The rebuilt index is compacted, and it is usable for lookups.
        assert_eq!(manifest_json(dir)["entries"].as_array().unwrap().len(), 1);
        assert_eq!(
            archive.lookup(3, &tokens, 20, FP, 100).unwrap().unwrap().id,
            entry.id
        );
    }

    #[test]
    fn v2_rebuild_reconciles_shared_segment_refs() {
        let tmp = TempDir::new("v2-rebuild-shared");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=20).collect();
        let first = v2_entry(&mut archive, 1, &tokens[..12], "segmented");
        let _second = v2_entry(&mut archive, 1, &tokens[..20], "segmented");
        let shared = first.segments[0].id.clone();
        drop(archive);

        fs::remove_file(dir.join(MANIFEST)).unwrap();
        fs::remove_file(dir.join(slot_segments::MANIFEST)).unwrap();
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 2);
        assert_eq!(archive.segments().segment(&shared).unwrap().refs, 2);
        assert!(archive.segments().segment(&shared).unwrap().file.exists());
    }

    #[test]
    fn v2_rebuild_drops_an_entry_with_a_missing_segment_and_orphans_are_cleaned() {
        let tmp = TempDir::new("v2-rebuild-invalid");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let entry = v2_entry(&mut archive, 4, &(1..=12).collect::<Vec<_>>(), "segmented");
        let missing = archive.segments.segment(&entry.segments[1].id).unwrap().file.clone();
        drop(archive);
        fs::remove_file(&missing).unwrap();
        fs::remove_file(dir.join(MANIFEST)).unwrap();

        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 0, "an entry cannot be rebuilt without its segments");
        assert!(
            archive.segments().list().is_empty(),
            "unreferenced segments are removed at startup: {:?}",
            archive.segments().list()
        );
        assert!(manifest_json(dir)["entries"].as_array().unwrap().is_empty());
    }

    #[test]
    fn segments_without_a_state_file_are_orphans_not_entries() {
        let tmp = TempDir::new("segments-orphan");
        let dir = tmp.path();
        let mut archive = Archive::open(config(dir)).unwrap();
        let tokens: Vec<u32> = (1..=6).collect();
        archive
            .segments
            .put(FP, None, &tokens, b"orphan-payload")
            .unwrap();
        drop(archive);
        fs::remove_file(dir.join(MANIFEST)).unwrap();

        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 0);
        assert!(archive.segments().list().is_empty());
    }

    #[test]
    fn v1_state_file_is_readable_but_not_rebuilt_without_refs() {
        let tmp = TempDir::new("v1-state");
        let dir = tmp.path();
        let tokens: Vec<u32> = (1..=9).collect();
        let header = StateHeader { fingerprint: FP, owner: 7, rope_offset: 0, n_tokens: tokens.len() };
        let id = id_for(FP, 7, tokens.len(), prefix_hash(&tokens));
        let path = state_path(dir, &id).unwrap();
        write_state_v1(&path, &header, &tokens, b"legacy-payload");
        let (got, refs, got_tokens) = read_state_manifest(&path).unwrap();
        assert_eq!(got, header);
        assert!(refs.is_empty());
        assert_eq!(got_tokens, tokens);
        let (_, _, full_tokens, payload) = read_state_file(&path).unwrap();
        assert_eq!(full_tokens, tokens);
        assert_eq!(payload, b"legacy-payload");

        // With no manifest, a v1 state file has no refs to reconstruct from and is skipped, not invented.
        let archive = Archive::open(config(dir)).unwrap();
        assert_eq!(archive.stats().entries, 0);
        assert!(path.exists(), "the legacy state file is left for the operator");
    }
}
