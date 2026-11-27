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

use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// File name of the manifest inside [`ArchiveConfig::dir`].
pub const MANIFEST: &str = "manifest.json";
/// Headroom kept free below the logical budget, so a save can always write a temp file and the manifest.
pub const RESERVE_BYTES: u64 = 1 << 30;

/// Archive policy and location. Built from the `--slot-archive-*` flags.
#[derive(Clone, Debug)]
pub struct ArchiveConfig {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    pub min_tokens: usize,
    pub min_delta: usize,
    pub versions: usize,
}

/// One immutable `STRATASLOT` file and its manifest row.
#[derive(Clone, Debug)]
pub struct ArchiveEntry {
    /// Stable content identity; also the dedupe key.
    pub id: String,
    /// Absolute path to the `.slot` file. Never accepted from a client.
    pub file: PathBuf,
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
    /// `STRATASLOT` version (1).
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
    pub failures: u64,
}

/// What [`Archive::gc`] removed.
#[derive(Clone, Debug, Default)]
pub struct GcReport {
    pub evicted: usize,
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
    stats: ArchiveStats,
}

impl Archive {
    /// Open (or create) `cfg.dir`, load `manifest.json`, and rebuild it from the shard files when it is missing or
    /// corrupt. Creates the directory with mode 0700 on Unix when it does not exist.
    pub fn open(cfg: ArchiveConfig) -> Result<Self> {
        let _ = &cfg;
        todo!("implemented by the archive-core agent")
    }

    pub fn config(&self) -> &ArchiveConfig {
        &self.cfg
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
        let _ = (owner, limit);
        todo!("implemented by the archive-core agent")
    }

    /// Reserve an entry id and a final path. Returns [`SaveSlot::Existing`] when the same content is already
    /// archived (the caller should only [`Archive::touch`] it).
    pub fn prepare_save(&mut self, owner: u64, tokens: &[u32], fingerprint: u64, reason: &str) -> Result<SaveSlot> {
        let _ = (owner, tokens, fingerprint, reason);
        todo!("implemented by the archive-core agent")
    }

    /// Register the file written at `path`, enforce the budget/lineage policy, and persist the manifest.
    pub fn commit_save(&mut self, id: &str, path: &Path, owner: u64, tokens: &[u32], fingerprint: u64, reason: &str) -> Result<ArchiveEntry> {
        let _ = (id, path, owner, tokens, fingerprint, reason);
        todo!("implemented by the archive-core agent")
    }

    /// Longest entry whose token history is an exact prefix of `tokens[..keep]`, with the same owner and
    /// fingerprint and `n_tokens <= ctx`. Increments the hit/miss counters.
    pub fn lookup(&mut self, owner: u64, tokens: &[u32], keep: usize, fingerprint: u64, ctx: usize) -> Result<Option<ArchiveEntry>> {
        let _ = (owner, tokens, keep, fingerprint, ctx);
        todo!("implemented by the archive-core agent")
    }

    /// Longest existing entry whose history is a hash-prefix of `tokens` (same owner/fingerprint). Hash only:
    /// used for the `min-delta` heuristic, never to authorize a restore.
    pub fn longest_prefix(&self, owner: u64, tokens: &[u32], fingerprint: u64) -> Option<ArchiveEntry> {
        let _ = (owner, tokens, fingerprint);
        todo!("implemented by the archive-core agent")
    }

    /// Mark an entry as used now (LRU). No-op when it disappeared.
    pub fn touch(&mut self, id: &str) {
        let _ = id;
        todo!("implemented by the archive-core agent")
    }

    /// Delete an entry and its file. Returns the freed bytes.
    pub fn delete(&mut self, id: &str) -> Result<u64> {
        let _ = id;
        todo!("implemented by the archive-core agent")
    }

    /// Evict unusable entries first, then unpinned LRU entries, until `used_bytes <= budget`.
    pub fn gc(&mut self) -> Result<GcReport> {
        todo!("implemented by the archive-core agent")
    }

    pub fn note_restore(&mut self) {
        self.stats.restores += 1;
    }

    pub fn note_failure(&mut self) {
        self.stats.failures += 1;
    }
}

/// Milliseconds since the Unix epoch, saturating to 0 before it.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// Join a manifest-relative file name to the archive directory, rejecting escapes.
pub(crate) fn entry_path(dir: &Path, file: &str) -> Result<PathBuf> {
    let p = Path::new(file);
    if p.is_absolute() || p.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        bail!("archive manifest contains an invalid path: {file}");
    }
    Ok(dir.join(p))
}

/// Read the token list at the start of a slot file without reading the KV. Used by lookup to turn a hash
/// candidate into an exact match.
pub(crate) fn read_tokens(path: &Path) -> Result<Vec<u32>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = std::io::BufReader::new(f);
    let (_header, tokens) = crate::slot_file::read_header(&mut r)?;
    Ok(tokens)
}
