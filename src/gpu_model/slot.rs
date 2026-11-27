//! `STRATASLOT` v1 save/restore for one slot (see `docs/SLOT-SAVE.md`).
//!
//! The state is streamed in bounded chunks: a 1M-token q8 slot is about 10.9 GB and must never be held in host
//! RAM. The file framing lives in `crate::slot_file`; this module knows the `GpuModel` buffers and the fingerprint.
//!
//! Wave 1 of the async pipeline lives here too: [`GpuModel::save_slot_to`] and [`GpuModel::restore_slot_from`] split
//! the format from the file handling, and [`GpuModel::save_slot_file_async`] / [`GpuModel::restore_slot_file_async`]
//! move the disk I/O to a writer/reader thread through a bounded channel. The model thread still performs every
//! GPU→host and host→GPU copy; only `write`/`fsync`/`rename` (save) and `read` (restore) leave it.

use super::{Buf, GpuModel};
use crate::slot_file::{self, SlotHeader};
use anyhow::{bail, ensure, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Bytes copied between the GPU and the host in one `get`/`put`, and the payload of one channel chunk.
const CHUNK: usize = 64 << 20;

/// Depth of the async save ring (4 × 64 MB of staged bytes, see `docs/SLOT-ARCHIVE-PHASE2.md` §5.3).
const RING: usize = 4;

/// Restores the request owner when `save_slot_file_as` finishes or unwinds. The owner field is the caller's
/// request identity; only the saved state and header carry the explicit conversation owner.
struct OwnerGuard<'m, 'a> {
    model: &'m mut GpuModel<'a>,
    prev: u64,
}
impl Drop for OwnerGuard<'_, '_> {
    fn drop(&mut self) { self.model.owner = self.prev; }
}

/// A writer that counts the bytes it passes through (the footer stores the payload length).
struct Count<W> { w: W, n: u64 }
impl<W> Count<W> {
    fn new(w: W) -> Self { Self { w, n: 0 } }
    fn count(&self) -> u64 { self.n }
}
impl<W: Write> Write for Count<W> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        let n = self.w.write(b)?;
        self.n += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> { self.w.flush() }
}
impl<R: Read> Read for Count<R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let n = self.w.read(b)?;
        self.n += n as u64;
        Ok(n)
    }
}

/// The first worker error, kept so that both `write()` and `join()` report it instead of a bare "channel closed".
#[derive(Clone, Default)]
struct SharedErr(Arc<Mutex<Option<(io::ErrorKind, String)>>>);
impl SharedErr {
    fn set(&self, e: &io::Error) {
        *self.0.lock().unwrap_or_else(|p| p.into_inner()) = Some((e.kind(), e.to_string()));
    }
    fn get(&self) -> Option<io::Error> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|(k, m)| io::Error::new(*k, m.clone()))
    }
}

/// One message from the model thread to the writer thread.
enum Chunk {
    Data(Vec<u8>),
    /// The stream is complete: flush, fsync and rename the temp file.
    End,
}

/// A `Write` that hands bounded chunks to the writer thread.
///
/// `write` blocks when the ring is full, so a slow disk backpressures the model thread once it is more than
/// `RING` chunks behind. The worker error is kept in [`SharedErr`], so a `write` that loses the channel returns the
/// disk error rather than a generic disconnect, and `AsyncSave::join` returns it too.
struct ChannelWriter {
    tx: Option<SyncSender<Chunk>>,
    limit: usize,
    abort: Arc<AtomicBool>,
    err: SharedErr,
    finished: bool,
}

impl ChannelWriter {
    fn new(tx: SyncSender<Chunk>, limit: usize, abort: Arc<AtomicBool>, err: SharedErr) -> Self {
        Self { tx: Some(tx), limit: limit.max(1), abort, err, finished: false }
    }

    fn send_error(&self) -> io::Error {
        self.err.get().unwrap_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "slot writer thread stopped"))
    }

    fn send(&self, v: Vec<u8>) -> io::Result<()> {
        let Some(tx) = self.tx.as_ref() else {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "slot writer already finished"));
        };
        tx.send(Chunk::Data(v)).map_err(|_| self.send_error())
    }

    /// Send the end marker. Only after this may the writer make the file visible.
    fn finish(&mut self) -> io::Result<()> {
        if self.finished { return Ok(()); }
        self.finished = true;
        let Some(tx) = self.tx.take() else { return Ok(()) };
        tx.send(Chunk::End).map_err(|_| self.send_error())
    }
}

impl Write for ChannelWriter {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if let Some(e) = self.err.get() { return Err(e); }
        if b.is_empty() { return Ok(0); }
        if b.len() <= self.limit {
            self.send(b.to_vec())?;
        } else {
            for c in b.chunks(self.limit) { self.send(c.to_vec())?; }
        }
        Ok(b.len())
    }

    /// The writer thread flushes and syncs before the rename; there is nothing to flush here.
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

impl Drop for ChannelWriter {
    /// Dropping without `finish` aborts the save (error path): the writer removes the temp file and never renames.
    fn drop(&mut self) {
        if !self.finished { self.abort.store(true, Ordering::Relaxed); }
    }
}

/// A `Read` over the chunks produced by the reader thread. Holds at most one chunk plus the caller's buffer.
struct ChannelReader {
    rx: Receiver<io::Result<Vec<u8>>>,
    buf: Vec<u8>,
    pos: usize,
    done: bool,
}

impl ChannelReader {
    fn new(rx: Receiver<io::Result<Vec<u8>>>) -> Self {
        Self { rx, buf: Vec::new(), pos: 0, done: false }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() { return Ok(0); }
        loop {
            if self.pos < self.buf.len() {
                let n = (self.buf.len() - self.pos).min(out.len());
                out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
                self.pos += n;
                return Ok(n);
            }
            if self.done { return Ok(0); }
            match self.rx.recv() {
                Ok(Ok(v)) => { self.buf = v; self.pos = 0; }
                Ok(Err(e)) => { self.done = true; return Err(e); }
                Err(_) => { self.done = true; return Ok(0); }
            }
        }
    }
}

/// Handle for an in-flight async save. `join` waits for the fsync/rename and returns the stream length.
///
/// Dropping the handle before `join` aborts the save: the writer thread removes the temp file, and a previous
/// target file is left untouched.
pub struct AsyncSave {
    handle: Option<JoinHandle<Result<u64>>>,
    abort: Arc<AtomicBool>,
}

impl AsyncSave {
    /// Wait for the writer to fsync and rename the temp file and return the number of bytes written.
    pub fn join(mut self) -> Result<u64> {
        let Some(h) = self.handle.take() else { return Err(anyhow::anyhow!("slot save already joined")); };
        match h.join() {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("slot writer thread panicked")),
        }
    }
}

impl Drop for AsyncSave {
    fn drop(&mut self) {
        if self.handle.is_some() { self.abort.store(true, Ordering::Relaxed); }
    }
}

/// Spawn the writer thread: it owns the temp file and renames it only after `Chunk::End` and an fsync.
fn spawn_writer(f: File, tmp: PathBuf, path: PathBuf) -> (ChannelWriter, AsyncSave) {
    let abort = Arc::new(AtomicBool::new(false));
    let err = SharedErr::default();
    let (tx, rx) = sync_channel::<Chunk>(RING);
    let w = ChannelWriter::new(tx, CHUNK, abort.clone(), err.clone());
    let a = abort.clone();
    let e = err.clone();
    let handle = std::thread::spawn(move || writer_thread(f, &tmp, &path, rx, &a, &e));
    (w, AsyncSave { handle: Some(handle), abort })
}

fn writer_thread(
    mut f: File,
    tmp: &Path,
    path: &Path,
    rx: Receiver<Chunk>,
    abort: &AtomicBool,
    err: &SharedErr,
) -> Result<u64> {
    let mut n = 0u64;
    let r = (|| -> io::Result<u64> {
        loop {
            if abort.load(Ordering::Relaxed) {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "slot save aborted"));
            }
            match rx.recv() {
                Ok(Chunk::Data(v)) => {
                    if abort.load(Ordering::Relaxed) {
                        return Err(io::Error::new(io::ErrorKind::Interrupted, "slot save aborted"));
                    }
                    f.write_all(&v)?;
                    n = n.saturating_add(v.len() as u64);
                }
                Ok(Chunk::End) => break,
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "slot save aborted before the stream was complete",
                    ));
                }
            }
        }
        f.flush()?;
        f.sync_all()?;
        Ok(n)
    })();
    match r {
        Ok(n) => match std::fs::rename(tmp, path) {
            Ok(()) => Ok(n),
            Err(e) => {
                let _ = std::fs::remove_file(tmp);
                err.set(&e);
                Err(anyhow::Error::new(e).context(format!("renaming {} to {}", tmp.display(), path.display())))
            }
        },
        Err(e) => {
            let _ = std::fs::remove_file(tmp);
            err.set(&e);
            Err(anyhow::Error::new(e).context(format!("writing {}", tmp.display())))
        }
    }
}

/// Spawn the reader thread. Chunks are `chunk` bytes (the async restore uses 64 MB); a read error is sent before the
/// thread ends so the model-side `read_exact` fails instead of seeing a silent EOF.
fn spawn_reader<R: Read + Send + 'static>(
    mut r: R,
    tx: SyncSender<io::Result<Vec<u8>>>,
    chunk: usize,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = vec![0u8; chunk.max(1)];
        loop {
            match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(Ok(buf[..n].to_vec())).is_err() { break; }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }
    })
}

impl GpuModel<'_> {
    /// Seconds of a slot's state file. The file contains plaintext KV: prompt, system prompt, tools and reply.
    pub fn save_slot_file(&mut self, slot: usize, path: &Path) -> Result<u64> {
        self.save_slot_file_as(slot, path, self.owner)
    }

    /// `save_slot_file` with an explicit conversation owner. The archive uses `slot_owner` here, not the owner of
    /// the request that triggered the checkpoint: an eviction caused by another key must not re-label the state. The
    /// caller's request owner is restored on success, error and unwind.
    pub fn save_slot_file_as(&mut self, slot: usize, path: &Path, owner: u64) -> Result<u64> {
        let prev = std::mem::replace(&mut self.owner, owner);
        let guard = OwnerGuard { model: self, prev };
        guard.model.save_slot_file_owned(slot, path)
    }

    /// Write the complete `STRATASLOT` v1 stream (header, token ids, every part's recurrent and KV bytes, footer)
    /// for `slot` to `out`, and return the number of bytes written. No file is created or renamed.
    ///
    /// This calls `save_state` (the recurrent/DeltaNet snapshot is the one streamed). It does **not** set
    /// `saved_ids`/`saved_owner`: those are the "reusable at this history" marker and only a caller that owns the
    /// commit may set them (the file path after the rename, the async path once every byte is captured).
    /// `save_state` itself clears `saved_ids`, as in the synchronous save.
    pub fn save_slot_to<W: Write>(&mut self, slot: usize, out: &mut W) -> Result<u64> {
        let (tokens, header) = self.slot_snapshot(slot)?;
        self.write_slot_stream(&tokens, &header, out)
    }

    /// Restore a `STRATASLOT` v1 stream from `inp` into `slot`. Same validation, unified-KV growth, install and
    /// error handling (a partially installed slot is cleared) as [`Self::restore_slot_file`], without opening a
    /// path. `allow_evict` is as in `restore_slot_file`.
    pub fn restore_slot_from<R: Read>(&mut self, slot: usize, inp: &mut R, allow_evict: bool) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        let mut inp = Count::new(BufReader::with_capacity(8 << 20, inp));
        let (header, tokens) = slot_file::read_header(&mut inp)?;
        ensure!(header.fingerprint == self.slot_fingerprint(), "slot file model/config fingerprint mismatch");
        ensure!(header.n_tokens as usize == tokens.len(), "slot file token count mismatch");
        ensure!(!tokens.is_empty(), "slot file has no tokens");
        ensure!(tokens.len() <= self.ctx, "slot file has {} tokens, --ctx is {}", tokens.len(), self.ctx);
        ensure!(header.n_parts as usize == self.parts().count(), "slot file has {} parts, this model has {}", header.n_parts, self.parts().count());
        self.use_slot(slot);
        let n = tokens.len();
        let cap = self.kv_cap.get(slot).copied().unwrap_or(self.ctx);
        if self.kv_budget > 0 && n > cap {
            let max = self.kv_budget.saturating_sub(self.kv_cap.len().saturating_sub(1));
            ensure!(n <= max, "restored state has {n} tokens; with the other slots reserved the budget holds {max}");
            let new = n.next_multiple_of(1024).min(self.ctx).min(max);
            let victims = self.restore_victims(slot, new);
            if !victims.is_empty() && !allow_evict {
                bail!("restore needs to evict non-empty slot(s) {:?}; pass force=1 to allow it", victims);
            }
            // The target is not a victim of its own growth.
            let mut active = self.active.clone();
            active.resize(self.kv_cap.len(), false);
            if slot < active.len() { active[slot] = true; }
            self.set_active(&active);
        }
        self.kv_need(slot, n)?;
        self.save_state()?;
        let r = (|| -> Result<u64> {
            let mut q = Some(&mut *self);
            while let Some(p) = q {
                let ple0 = slot_file::read_u64(&mut inp)? as i64;
                let ple1 = slot_file::read_u64(&mut inp)? as i64;
                let n_slot = slot_file::read_u32(&mut inp)? as usize;
                let bufs = p.slot();
                ensure!(n_slot == bufs.len(), "slot file has {n_slot} recurrent buffers, this model has {}", bufs.len());
                for b in bufs {
                    let len = slot_file::read_u64(&mut inp)? as usize;
                    ensure!(len == b.len(), "slot file recurrent buffer {len} bytes, this model has {}", b.len());
                    read_buf(&*p.gpu, b, len, &mut inp)?;
                }
                let n_kv = slot_file::read_u32(&mut inp)? as usize;
                let kv = p.kv_bufs();
                ensure!(n_kv == kv.len(), "slot file has {n_kv} KV buffers, this model has {}", kv.len());
                for (b, rows) in kv {
                    let row = slot_file::read_u64(&mut inp)? as usize;
                    ensure!(row == b.len() / rows.max(1), "slot file KV row is {row} bytes, this model has {}", b.len() / rows.max(1));
                    let bytes = row.checked_mul(n).context("slot file KV size overflow")?;
                    ensure!(bytes <= b.len(), "slot file KV rows do not fit the slot buffer");
                    read_buf(&*p.gpu, b, bytes, &mut inp)?;
                }
                p.saved_ple_prev = [ple0, ple1];
                p.saved_pos = n;
                p.saved_ids = tokens.clone();
                p.history = tokens.clone();
                p.saved_owner = header.owner;
                p.rope_offset = header.rope_offset;
                q = p.next.as_deref_mut();
            }
            let payload = inp.count();
            slot_file::read_footer(&mut inp, payload)?;
            Ok(inp.count())
        })();
        let n_read = match r {
            Ok(n) => n,
            Err(e) => {
                // A partially written slot must not be used: clear it and report the error.
                self.use_slot(slot);
                let _ = self.reset();
                return Err(e);
            }
        };
        self.restore_state()?;
        Ok(n_read)
    }

    /// Save `slot` to `path` on a writer thread. The temp file is created (and freed) as in the synchronous path;
    /// the model thread streams the complete format through a bounded ring, then returns.
    ///
    /// The returned [`AsyncSave`] must be joined to learn whether the `fsync`/`rename` succeeded. On success,
    /// `saved_ids` and `saved_owner` are set before returning: once every byte has been copied out of the model, the
    /// snapshot is immutable, so the in-memory saved history is valid even if the file write later fails (the
    /// failure means no archive entry, not a bad in-memory prefix). If the stream cannot be handed off, they are
    /// left as `save_state` cleared them and the temp file is removed.
    ///
    /// Backpressure: `write` blocks when the writer is more than `RING` chunks behind, so a disk slower than the
    /// GPU→host copies still stalls the model thread (after ~256 MB of staging), and `join` waits for the disk.
    pub fn save_slot_file_async(&mut self, slot: usize, path: &Path) -> Result<AsyncSave> {
        let (tokens, header) = self.slot_snapshot(slot)?;
        let (tmp, f) = create_tmp(path)?;
        let (mut out, save) = spawn_writer(f, tmp, path.to_path_buf());
        match self.write_slot_stream(&tokens, &header, &mut out) {
            Ok(_) => {
                if let Err(e) = out.finish() {
                    drop(out);
                    let _ = save.join();
                    return Err(anyhow::Error::new(e).context(format!("finishing {}", path.display())));
                }
                // All bytes are in the channel; the snapshot can no longer change.
                let owner = self.owner;
                let mut q = Some(&mut *self);
                while let Some(p) = q {
                    p.saved_ids = tokens.clone();
                    p.saved_owner = owner;
                    q = p.next.as_deref_mut();
                }
                Ok(save)
            }
            Err(e) => {
                drop(out);
                let _ = save.join();
                Err(e)
            }
        }
    }

    fn save_slot_file_owned(&mut self, slot: usize, path: &Path) -> Result<u64> {
        let (tokens, header) = self.slot_snapshot(slot)?;
        let (tmp, f) = create_tmp(path)?;
        let r = (|| -> Result<u64> {
            let mut out = BufWriter::with_capacity(8 << 20, f);
            let n = self.write_slot_stream(&tokens, &header, &mut out)?;
            let f = out.into_inner().map_err(|e| anyhow::anyhow!("flushing {}: {}", tmp.display(), e.error()))?;
            f.sync_all().with_context(|| format!("syncing {}", tmp.display()))?;
            Ok(n)
        })();
        let n = match r {
            Ok(n) => n,
            Err(e) => { let _ = std::fs::remove_file(&tmp); return Err(e); }
        };
        std::fs::rename(&tmp, path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
        // The slot is now reusable at the full history, not only at the prompt prefix.
        let owner = self.owner;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            p.saved_ids = tokens.clone();
            p.saved_owner = owner;
            q = p.next.as_deref_mut();
        }
        Ok(n)
    }

    /// Restore a slot file into `slot`. `allow_evict`: under `--kv-unified`, allow reclaiming non-empty slots when
    /// the state does not fit; otherwise such a restore is refused. The caller is responsible for pinning `slot`
    /// (it is marked active here while the restore runs).
    pub fn restore_slot_file(&mut self, slot: usize, path: &Path, allow_evict: bool) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        self.restore_slot_from(slot, &mut f, allow_evict)
    }

    /// Read `path` on a reader thread and install it on the model thread. Same semantics as
    /// [`Self::restore_slot_file`]; the disk reads are off the model thread but the host→GPU copies are not.
    pub fn restore_slot_file_async(&mut self, slot: usize, path: &Path, allow_evict: bool) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let (tx, rx) = sync_channel::<io::Result<Vec<u8>>>(RING);
        let handle = spawn_reader(f, tx, CHUNK);
        let mut inp = ChannelReader::new(rx);
        let r = self.restore_slot_from(slot, &mut inp, allow_evict);
        // Stop a reader that is blocked on a full ring before joining it.
        drop(inp);
        match handle.join() {
            Ok(()) => r,
            Err(_) => match r {
                Err(e) => Err(e),
                Ok(_) => Err(anyhow::anyhow!("slot reader thread panicked")),
            },
        }
    }

    /// Validate `slot` and capture the token list and header a save would write. Shared by the sync, streaming and
    /// async paths so they reject the same states with the same messages and before any file is created.
    fn slot_snapshot(&mut self, slot: usize) -> Result<(Vec<u32>, SlotHeader)> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        self.use_slot(slot);
        let tokens = self.history.clone();
        ensure!(!tokens.is_empty(), "slot {slot} has no token history to save");
        ensure!(self.pos == tokens.len(), "slot {slot}: pos {} but {} tokens in its history", self.pos, tokens.len());
        let fp = self.slot_fingerprint();
        let header = SlotHeader {
            fingerprint: fp,
            n_tokens: tokens.len() as u64,
            rope_offset: self.rope_offset,
            owner: self.owner,
            flags: (self.rope_offset != 0).then_some(slot_file::FLAG_IMAGE).unwrap_or(0)
                | (self.kv_budget > 0).then_some(slot_file::FLAG_UNIFIED).unwrap_or(0)
                | (self.slot().len() > 0 || self.ple_ring.is_some() || self.qsa.is_some()).then_some(slot_file::FLAG_AUX).unwrap_or(0),
            n_parts: self.parts().count() as u32,
        };
        Ok((tokens, header))
    }

    /// The framing and buffer copies of a save: `save_state`, header, every part, footer.
    fn write_slot_stream<W: Write>(&mut self, tokens: &[u32], header: &SlotHeader, out: &mut W) -> Result<u64> {
        // Allocate the recurrent copies and make the current recurrent state the saved one.
        self.save_state()?;
        let mut out = Count::new(out);
        slot_file::write_header(&mut out, header, tokens)?;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            for v in p.saved_ple_prev { slot_file::write_u64(&mut out, v as u64)?; }
            let slot_bufs = p.slot();
            slot_file::write_u32(&mut out, slot_bufs.len() as u32)?;
            for b in slot_bufs {
                slot_file::write_u64(&mut out, b.len() as u64)?;
                write_buf(&*p.gpu, b, b.len(), &mut out)?;
            }
            let kv = p.kv_bufs();
            slot_file::write_u32(&mut out, kv.len() as u32)?;
            for (b, rows) in kv {
                let row = b.len() / rows.max(1);
                slot_file::write_u64(&mut out, row as u64)?;
                write_buf(&*p.gpu, b, row * tokens.len(), &mut out)?;
            }
            q = p.next.as_deref_mut();
        }
        let payload = out.count();
        slot_file::write_footer(&mut out, payload)?;
        out.flush()?;
        Ok(out.count())
    }

    /// Clear `slot` and return how many tokens it held.
    pub fn erase_slot(&mut self, slot: usize) -> Result<usize> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        let n = self.slot_tokens(slot).len();
        self.use_slot(slot);
        self.reset()?;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            p.saved.clear();
            p.saved_pos = 0;
            p.saved_owner = 0;
            p.saved_ple_prev = [-1, -1];
            q = p.next.as_deref_mut();
        }
        Ok(n)
    }

    /// The caps a unified restore of `new` tokens would have to reclaim, only the ones that are not empty.
    fn restore_victims(&self, slot: usize, new: usize) -> Vec<usize> {
        super::reclaim_plan(&self.kv_cap, &self.active, self.kv_budget, slot, new)
            .into_iter()
            .filter(|&i| !self.slot_empty(i))
            .collect()
    }

    fn slot_empty(&self, s: usize) -> bool {
        let q = &self.seqs[s];
        q.pos <= 1 && q.history.is_empty() && q.saved_ids.is_empty()
    }

    /// The model/layout identity a slot file is tied to. It must not depend on a slot's current KV capacity: under
    /// `--kv-unified` the capacities change as prompts grow, while the bytes per token do not.
    pub fn slot_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let c = &self.m.cfg;
        c.arch.hash(&mut h);
        for v in [c.n_layer, c.n_embd, c.n_vocab, c.n_head, c.n_head_kv, c.head_dim, c.n_rot, c.n_expert, c.n_expert_used,
                  c.n_ff, c.n_ff_exp, c.n_ff_shexp, c.ssm_d_inner, c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank, c.ssm_conv,
                  c.hc_n, c.mhc_streams] {
            v.hash(&mut h);
        }
        c.mrope_sections.hash(&mut h);
        c.dflash.is_some().hash(&mut h);
        c.rope.base.to_bits().hash(&mut h);
        c.rope.n_rot.hash(&mut h);
        c.rope.freq_scale.to_bits().hash(&mut h);
        c.rope.ext_factor.to_bits().hash(&mut h);
        c.rope.corr.map(f32::to_bits).hash(&mut h);
        for (name, t) in &self.m.w.g.tensors {
            name.hash(&mut h);
            t.dims.hash(&mut h);
            t.ty.hash(&mut h);
            t.offset.hash(&mut h);
            t.file.hash(&mut h);
        }
        let mut p = Some(self);
        while let Some(q) = p {
            q.first.hash(&mut h);
            q.layers.len().hash(&mut h);
            for l in &q.layers {
                if let super::Mixer::Delta { conv, s, .. } = &l.mixer {
                    conv.first().map_or(0, |b| b.len()).hash(&mut h);
                    s.first().map_or(0, |b| b.len()).hash(&mut h);
                }
            }
            if let Some(m) = &q.mtp { m.saved_x.len().hash(&mut h); }
            if let Some((_, ring)) = &q.ple_ring { ring.len().hash(&mut h); }
            if let Some(qsa) = &q.qsa { qsa.saved.len().hash(&mut h); }
            q.kv_bufs().len().hash(&mut h);
            for (b, rows) in q.kv_bufs() { (b.len() / rows.max(1)).hash(&mut h); }
            q.gpu.kv_words(q.ctx).hash(&mut h);
            q.gpu.kv_words_v(q.ctx).hash(&mut h);
            p = q.next.as_deref();
        }
        h.finish()
    }
}

/// Create the temp file next to `path` (`create_new`, `0600` on unix) and return its name.
fn create_tmp(path: &Path) -> Result<(PathBuf, File)> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let name = path.file_name().map_or_else(|| "slot".into(), |s| s.to_string_lossy().into_owned());
    let tmp = dir.join(format!(".{name}.{}.{stamp}.tmp", std::process::id()));
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let f = match opts.open(&tmp) {
        Ok(f) => f,
        Err(e) => return Err(e).with_context(|| format!("creating {}", tmp.display())),
    };
    Ok((tmp, f))
}

fn write_buf(g: &dyn crate::backend::Backend, b: &Buf, bytes: usize, out: &mut impl Write) -> Result<()> {
    let mut off = 0;
    while off < bytes {
        let n = (bytes - off).min(CHUNK);
        let v = g.get::<u8>(b, off, n)?;
        out.write_all(&v)?;
        off += n;
    }
    Ok(())
}

fn read_buf(g: &dyn crate::backend::Backend, b: &Buf, bytes: usize, inp: &mut impl Read) -> Result<()> {
    let mut off = 0;
    let mut v = vec![0u8; CHUNK.min(bytes)];
    while off < bytes {
        let n = (bytes - off).min(v.len());
        inp.read_exact(&mut v[..n])?;
        g.put(&v[..n], b, off)?;
        off += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn test_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("strata-slot-unit-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect()
    }

    /// The handle crosses threads, so the model thread can return it to the scheduler.
    #[test]
    fn async_save_handle_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AsyncSave>();
    }

    /// Arbitrary bytes through several small chunks, with a capacity of 2, reassemble exactly.
    #[test]
    fn channel_writer_streams_whole_chunks_to_the_consumer() {
        let data = pattern(10_000);
        let (tx, rx) = sync_channel::<Chunk>(2);
        let mut w = ChannelWriter::new(tx, 64, Arc::new(AtomicBool::new(false)), SharedErr::default());
        let consumer = std::thread::spawn(move || {
            let mut out = Vec::new();
            while let Ok(msg) = rx.recv() {
                match msg {
                    Chunk::Data(v) => out.extend_from_slice(&v),
                    Chunk::End => break,
                }
            }
            out
        });
        w.write_all(&data).unwrap();
        w.finish().unwrap();
        assert_eq!(consumer.join().unwrap(), data);
    }

    /// A worker error is returned by `write`, not hidden behind the channel disconnect.
    #[test]
    fn channel_writer_propagates_the_worker_error() {
        let err = SharedErr::default();
        let (tx, rx) = sync_channel::<Chunk>(1);
        let e2 = err.clone();
        let worker = std::thread::spawn(move || {
            let _ = rx.recv();
            e2.set(&io::Error::new(io::ErrorKind::Other, "disk full"));
            drop(rx);
        });
        let mut w = ChannelWriter::new(tx, 8, Arc::new(AtomicBool::new(false)), err);
        let e = w.write_all(&[7u8; 1024]).unwrap_err();
        assert!(e.to_string().contains("disk full"), "{e}");
        worker.join().unwrap();
    }

    /// The reader thread produces many small chunks and the adapter returns them short-read by short-read.
    #[test]
    fn channel_reader_streams_chunks_to_eof() {
        let data = pattern(300);
        let (tx, rx) = sync_channel(1);
        let worker = spawn_reader(Cursor::new(data.clone()), tx, 7);
        let mut r = ChannelReader::new(rx);
        let mut out = Vec::new();
        let mut small = [0u8; 5];
        loop {
            let n = r.read(&mut small).unwrap();
            if n == 0 { break; }
            out.extend_from_slice(&small[..n]);
        }
        worker.join().unwrap();
        assert_eq!(out, data);
    }

    struct FailAfter { left: usize }
    impl Read for FailAfter {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            if self.left == 0 { return Err(io::Error::new(io::ErrorKind::Other, "read failed")); }
            let n = self.left.min(b.len());
            b[..n].fill(b'x');
            self.left -= n;
            Ok(n)
        }
    }

    /// A read error reaches the model side after the bytes that were read.
    #[test]
    fn channel_reader_propagates_a_read_error() {
        let (tx, rx) = sync_channel(2);
        let worker = spawn_reader(FailAfter { left: 3 }, tx, 2);
        let mut r = ChannelReader::new(rx);
        let mut out = Vec::new();
        let e = r.read_to_end(&mut out).unwrap_err();
        assert!(e.to_string().contains("read failed"), "{e}");
        assert_eq!(out, b"xxx");
        worker.join().unwrap();
    }

    /// The writer thread renames the temp file only after `End`, and returns the byte count.
    #[test]
    fn writer_thread_renames_only_after_end() {
        let d = test_dir("rename");
        let tmp = d.join(".s.tmp");
        let dst = d.join("s.bin");
        let f = File::create(&tmp).unwrap();
        let (mut w, save) = spawn_writer(f, tmp.clone(), dst.clone());
        w.write_all(b"abc").unwrap();
        w.write_all(b"def").unwrap();
        assert!(!dst.exists(), "renamed before the stream ended");
        w.finish().unwrap();
        assert_eq!(save.join().unwrap(), 6);
        assert_eq!(std::fs::read(&dst).unwrap(), b"abcdef");
        assert!(!tmp.exists(), "temp left after a successful save");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Dropping an unfinished save aborts it: temp removed, previous target untouched.
    #[test]
    fn writer_thread_abort_keeps_the_previous_target() {
        let d = test_dir("abort");
        let tmp = d.join(".s.tmp");
        let dst = d.join("s.bin");
        std::fs::write(&dst, b"old").unwrap();
        let f = File::create(&tmp).unwrap();
        let (mut w, save) = spawn_writer(f, tmp.clone(), dst.clone());
        w.write_all(b"partial").unwrap();
        drop(w);
        assert!(save.join().is_err(), "aborted save reported success");
        assert_eq!(std::fs::read(&dst).unwrap(), b"old");
        assert!(!tmp.exists(), "temp left after an aborted save");
        let _ = std::fs::remove_dir_all(&d);
    }
}
