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
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

/// File name of the manifest inside [`ArchiveConfig::dir`].
pub const MANIFEST: &str = "manifest.json";
/// Headroom kept free below the logical budget, so a save can always write a temp file and the manifest.
pub const RESERVE_BYTES: u64 = 1 << 30;
/// Manifest schema version written and understood by this module.
const MANIFEST_VERSION: u32 = 1;

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
    stats: ArchiveStats,
    /// Transient in-flight references, keyed by entry id. Shared with [`Lease`] guards so a guard can release
    /// the lease after the `Archive` borrow has ended.
    leases: Arc<Mutex<BTreeMap<String, usize>>>,
    /// Lifecycle policy; all zero means off. Set with [`Archive::set_policy`], not persisted in the manifest.
    ttl_ms: u64,
    per_key_bytes: u64,
    pin_bytes: u64,
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
    /// Open (or create) `cfg.dir`, load `manifest.json`, and rebuild it from the shard files when it is missing or
    /// corrupt. Creates the directory with mode 0700 on Unix when it does not exist.
    pub fn open(cfg: ArchiveConfig) -> Result<Self> {
        let mut cfg = cfg;
        if !cfg.dir.is_absolute() {
            cfg.dir = std::env::current_dir()
                .context("resolving the relative slot archive directory")?
                .join(&cfg.dir);
        }
        ensure_dir(&cfg.dir)?;
        remove_stale_temps(&cfg.dir);

        let manifest = cfg.dir.join(MANIFEST);
        let (mut entries, rebuilt) = match load_manifest(&cfg.dir) {
            Ok(entries) => (entries, false),
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
                    crate::say!("slot archive: {err:#}; rebuilding the manifest from the shard files");
                }
                (rebuild_manifest(&cfg.dir)?, true)
            }
        };

        // Startup validation: stat every file and compare the header with the manifest row. Missing or
        // inconsistent entries are dropped (the file, if any, stays on disk for a later rebuild).
        let mut changed = rebuilt;
        let ids: Vec<String> = entries.keys().cloned().collect();
        for id in ids {
            let entry = &entries[&id];
            if let Err(err) = validate_entry(entry) {
                crate::say!("slot archive: dropping {id}: {err:#}");
                entries.remove(&id);
                changed = true;
            }
        }

        let archive = Archive {
            cfg,
            entries,
            stats: ArchiveStats::default(),
            leases: Arc::new(Mutex::new(BTreeMap::new())),
            ttl_ms: 0,
            per_key_bytes: 0,
            pin_bytes: 0,
        };
        if changed {
            archive.persist()?;
        }
        Ok(archive)
    }

    pub fn config(&self) -> &ArchiveConfig {
        &self.cfg
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
        self.persist()?;
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
                entry.last_used_ms = now_ms();
                return Ok(SaveSlot::Existing(entry.clone()));
            }
            // A stale row whose file disappeared: forget it and let the caller write a new file.
            self.entries.remove(&id);
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
        let limit = self.cfg.budget_bytes.saturating_sub(RESERVE_BYTES);
        if bytes > limit {
            let _ = fs::remove_file(path);
            bail!(
                "slot archive: a {bytes}-byte entry does not fit in the {limit}-byte budget ({} bytes reserved)",
                RESERVE_BYTES
            );
        }

        // Per-key quota: evict this owner's own LRU entries first so one conversation cannot crowd every
        // other owner out. Pinned and leased entries are protected, and another owner is never touched here.
        if self.per_key_bytes > 0 {
            if bytes > self.per_key_bytes {
                let _ = fs::remove_file(path);
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
                    let _ = fs::remove_file(path);
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
                let _ = fs::remove_file(path);
                bail!(
                    "slot archive is full: a {bytes}-byte entry does not fit under the {limit}-byte limit \
                     even after evicting every unpinned entry"
                );
            }
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
        self.persist()?;
        Ok(entry)
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
            // Hash match: prove it with the token list at the start of the file.
            let exact = match read_tokens(&entry.file) {
                Ok(saved) => saved.len() == n && saved == tokens[..n],
                Err(err) => {
                    crate::say!("slot archive: cannot read {}: {err:#}", entry.file.display());
                    false
                }
            };
            if !exact {
                continue;
            }
            let used = now_ms();
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.last_used_ms = used;
                return Ok(Some(entry.clone()));
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
    pub fn touch(&mut self, id: &str) {
        let Some(entry) = self.entries.get_mut(id) else { return };
        entry.last_used_ms = now_ms();
        if let Err(err) = self.persist() {
            crate::say!("slot archive: manifest write after touch failed: {err:#}");
        }
    }

    /// Delete an entry and its file. Returns the freed bytes.
    ///
    /// An entry with live [`Lease`]s is not deleted: the caller gets an error and should retry after the restore
    /// or save that holds it completes.
    pub fn delete(&mut self, id: &str) -> Result<u64> {
        let Some(entry) = self.entries.get(id) else { return Ok(0) };
        let held = self.leases(id);
        if held > 0 {
            bail!(
                "slot archive: entry {id} is leased by {held} in-flight operation(s); retry after it completes"
            );
        }
        let file = entry.file.clone();
        let bytes = entry.bytes;
        match fs::remove_file(&file) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| format!("removing {}", file.display()));
            }
        }
        self.entries.remove(id);
        self.persist()?;
        Ok(bytes)
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

        // Unusable: the manifest row points at a missing file. (Foreign fingerprints are not identifiable
        // without a current fingerprint argument, so they are only evicted by the LRU pass.)
        let ids: Vec<String> = self.entries.keys().cloned().collect();
        for id in ids {
            let unusable = match fs::metadata(&self.entries[&id].file) {
                Ok(meta) => !meta.is_file(),
                Err(_) => true,
            };
            if unusable {
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

        if changed {
            self.persist()?;
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

    /// Remove an entry and its file. `None` when the file could not be unlinked (the entry is kept) or when the
    /// entry is leased (GC must never pull a file out from under an in-flight restore).
    fn evict_entry(&mut self, id: &str) -> Option<u64> {
        if self.leases(id) > 0 {
            return None;
        }
        let entry = self.entries.get(id)?;
        let file = entry.file.clone();
        let bytes = entry.bytes;
        match fs::remove_file(&file) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                crate::say!("slot archive: cannot remove {}: {err:#}; keeping the entry", file.display());
                return None;
            }
        }
        self.entries.remove(id);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        Some(bytes)
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

    fn persist(&self) -> Result<()> {
        ensure_dir(&self.cfg.dir)?;
        let manifest = ManifestFile {
            version: MANIFEST_VERSION,
            entries: self
                .entries
                .values()
                .map(|e| ManifestEntry {
                    id: e.id.clone(),
                    file: relative_string(&self.cfg.dir, &e.file),
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
                })
                .collect(),
        };
        let data = serde_json::to_vec(&manifest).context("serializing the archive manifest")?;
        write_atomic(&self.cfg.dir, &data)
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

#[derive(Serialize, Deserialize)]
struct ManifestEntry {
    id: String,
    file: String,
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
        let file = entry_path(dir, &row.file)?;
        let entry = ArchiveEntry {
            id: row.id,
            file,
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
        };
        entries.insert(entry.id.clone(), entry);
    }
    Ok(entries)
}

/// Stat the file and compare its `STRATASLOT` header with the manifest row.
fn validate_entry(entry: &ArchiveEntry) -> Result<()> {
    let meta = fs::metadata(&entry.file).with_context(|| format!("stat {}", entry.file.display()))?;
    if !meta.is_file() {
        bail!("{} is not a regular file", entry.file.display());
    }
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
    Ok(())
}

/// Scan `<dir>/*/*.slot`, adopt every valid file as a recovered entry, and return them keyed by id.
fn rebuild_manifest(dir: &Path) -> Result<BTreeMap<String, ArchiveEntry>> {
    let mut entries = BTreeMap::new();
    let root = fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    for shard in root.flatten() {
        let shard_path = shard.path();
        if !shard.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
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
            if !name.ends_with(".slot") || !file.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let meta = match fs::metadata(&path) {
                Ok(meta) => meta,
                Err(err) => {
                    crate::say!("slot archive: cannot stat {}: {err:#}", path.display());
                    continue;
                }
            };
            let (header, tokens) = match read_slot(&path) {
                Ok(parsed) => parsed,
                Err(err) => {
                    crate::say!("slot archive: cannot recover {}: {err:#}", path.display());
                    continue;
                }
            };
            let id = id_for(header.fingerprint, header.owner, tokens.len(), prefix_hash(&tokens));
            if entries.contains_key(&id) {
                crate::say!("slot archive: duplicate entry {id} at {}; keeping the first", path.display());
                continue;
            }
            let used = mtime_ms(&meta);
            entries.insert(
                id.clone(),
                ArchiveEntry {
                    id,
                    file: path,
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
                },
            );
        }
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
}
