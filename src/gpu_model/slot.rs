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

/// One bounded GPU↔host copy: part index, buffer, byte offset, byte length. `len` is at most [`CHUNK`].
type CopyPlan = (usize, Buf, usize, usize);

/// Split `len` bytes of `buf` into at most-[`CHUNK`] copy steps, attributed to `part` and starting at `offset`.
///
/// This is the unit the resumable `_with` variants yield on: after every entry of the resulting plan the callback
/// runs, so the longest stretch without a scheduler turn is one 64 MiB copy (plus the caller's own framing).
fn push_copy_chunks(plan: &mut Vec<CopyPlan>, part: usize, buf: &Buf, offset: usize, len: usize) {
    let mut done = 0;
    while done < len {
        let n = (len - done).min(CHUNK);
        plan.push((part, Buf::clone(buf), offset + done, n));
        done += n;
    }
}

/// The no-op interleaving point used by the existing public API: `save_slot_to`, `restore_slot_from` and the
/// segmented variants keep their old behaviour by passing this for `between`.
fn no_yield(_: &mut GpuModel<'_>) -> Result<()> { Ok(()) }

/// One step of a `STRATASLOT` save: a framing write or a bounded GPU→host copy. The frame and its data must stay
/// interleaved exactly as the v1 format defines (`len`, bytes, `len`, bytes, ...).
enum SlotWriteStep {
    Frame(Vec<u8>),
    Copy(CopyPlan),
}

/// Magic of a segmented K/V payload (`STRKVSEG`), see [`GpuModel::save_kv_segment`].
const KV_SEGMENT_MAGIC: [u8; 8] = *b"STRKVSEG";
/// Magic of a segmented recurrent/DeltaNet/MTP/PLE/QSA payload (`STRRSEG`), see
/// [`GpuModel::save_recurrent_to`].
const RECURRENT_SEGMENT_MAGIC: [u8; 7] = *b"STRRSEG";
/// Version of both segmented payloads understood by this module.
const SEGMENT_VERSION: u32 = 1;
/// Sanity bound on the number of model parts in a segmented payload. A split model has a handful; the bound is
/// only there so a corrupt header cannot make the reader allocate an arbitrary vector.
const MAX_SEGMENT_PARTS: usize = 256;
/// Sanity bound on the number of buffers in one part (KV pairs, DeltaNet pairs, MTP/PLE/QSA copies).
const MAX_SEGMENT_BUFFERS: usize = 4096;
/// Sanity bound on one buffer or KV row. 16 GiB is far above any per-layer buffer but keeps length arithmetic from
/// accepting an impossible value before it is compared with the model layout.
const MAX_SEGMENT_BUFFER_LEN: u64 = 1 << 34;

/// Restores the request owner when `save_slot_file_as` finishes or unwinds. The owner field is the caller's
/// request identity; only the saved state and header carry the explicit conversation owner.
struct OwnerGuard<'m, 'a> {
    model: &'m mut GpuModel<'a>,
    prev: u64,
}
impl Drop for OwnerGuard<'_, '_> {
    fn drop(&mut self) { self.model.owner = self.prev; }
}

/// Restores the caller's current slot when a segmented save/restore finishes or unwinds. Selecting another slot is
/// safe: `use_slot` swaps the whole per-slot state, so the caller finds its cursor where it left it.
struct SlotGuard<'m, 'a> {
    model: &'m mut GpuModel<'a>,
    prev: usize,
}
impl Drop for SlotGuard<'_, '_> {
    fn drop(&mut self) { self.model.use_slot(self.prev); }
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

// ---------------------------------------------------------------------------------------------------------
// Segmented payloads
// ---------------------------------------------------------------------------------------------------------
//
// The archive v2 dedup stores one opaque payload per KV segment and one per entry's fixed recurrent state. They
// are deliberately small, self-delimiting streams (no file path, temp file, footer or fingerprint: the entry
// header and the segment id carry those). All integers are little-endian, like the `STRATASLOT` format.
//
// `STRKVSEG` v1 for the K/V rows of `[start, end)`:
//
// ```text
// magic[8] = "STRKVSEG"
// version: u32 = 1
// start: u64
// end: u64                       (start < end)
// n_parts: u32
// per part:
//     n_buffers: u32
//     per buffer: row_bytes: u64
// then the bytes of every part/buffer in the same order, each buffer `row_bytes * (end - start)` long.
// ```
//
// `STRRSEG` v1 for the recurrent state of one slot:
//
// ```text
// magic[7] = "STRRSEG"
// version: u32 = 1
// ple_prev[0]: u64 (i64 bits), ple_prev[1]: u64
// n_parts: u32
// per part:
//     n_slot: u32
//     per saved buffer: len: u64, then `len` bytes
// ```
//
// `n_parts` and `n_buffers` are validated against the model before any GPU copy; a corrupt count, an empty range
// or a payload length overflow is rejected up front. The model/layout fingerprint is not in these payloads: the
// caller (the archive entry) owns that check, because a segment may be shared by several entries of one model.

/// Parsed `STRKVSEG` header: the range, one row-size vector per part, and the computed payload length.
#[derive(Debug, PartialEq, Eq)]
struct KvSegmentHeader {
    start: u64,
    end: u64,
    parts: Vec<Vec<u64>>,
    payload_len: u64,
}

/// The bytes of all KV rows of a segment, checked for overflow.
fn kv_segment_payload_len(start: u64, end: u64, parts: &[Vec<u64>]) -> Result<u64> {
    let rows = end.checked_sub(start).context("KV segment range underflow")?;
    ensure!(rows > 0, "KV segment range [{start}, {end}) is empty or reversed");
    let mut total = 0u64;
    for &row in parts.iter().flatten() {
        let bytes = row
            .checked_mul(rows)
            .context("KV segment payload length overflow")?;
        total = total
            .checked_add(bytes)
            .context("KV segment payload length overflow")?;
    }
    Ok(total)
}

/// Write a `STRKVSEG` v1 header and return the payload length it describes.
fn write_kv_segment_header<W: Write>(out: &mut W, start: u64, end: u64, parts: &[Vec<u64>]) -> Result<u64> {
    ensure!(
        (1..=MAX_SEGMENT_PARTS).contains(&parts.len()),
        "KV segment has {} parts, the range is 1..={MAX_SEGMENT_PARTS}",
        parts.len()
    );
    let payload_len = kv_segment_payload_len(start, end, parts)?;
    out.write_all(&KV_SEGMENT_MAGIC).context("writing the KV segment magic")?;
    slot_file::write_u32(out, SEGMENT_VERSION)?;
    slot_file::write_u64(out, start)?;
    slot_file::write_u64(out, end)?;
    slot_file::write_u32(out, parts.len() as u32)?;
    for part in parts {
        ensure!(
            part.len() <= MAX_SEGMENT_BUFFERS,
            "KV segment part has {} buffers, the limit is {MAX_SEGMENT_BUFFERS}",
            part.len()
        );
        slot_file::write_u32(out, part.len() as u32)?;
        for &row in part {
            ensure!(row > 0, "KV segment row is empty");
            ensure!(
                row <= MAX_SEGMENT_BUFFER_LEN,
                "KV segment row of {row} bytes exceeds the {MAX_SEGMENT_BUFFER_LEN}-byte limit"
            );
            slot_file::write_u64(out, row)?;
        }
    }
    Ok(payload_len)
}

/// Read and validate a `STRKVSEG` v1 header, including its counts and its computed payload length.
fn read_kv_segment_header<R: Read>(inp: &mut R) -> Result<KvSegmentHeader> {
    let mut magic = [0u8; 8];
    inp.read_exact(&mut magic).context("reading the KV segment magic")?;
    ensure!(
        magic == KV_SEGMENT_MAGIC,
        "bad KV segment magic {magic:02x?}, expected {KV_SEGMENT_MAGIC:02x?}"
    );
    let version = slot_file::read_u32(inp).context("reading the KV segment version")?;
    ensure!(
        version == SEGMENT_VERSION,
        "unsupported KV segment version {version}, expected {SEGMENT_VERSION}"
    );
    let start = slot_file::read_u64(inp).context("reading the KV segment start")?;
    let end = slot_file::read_u64(inp).context("reading the KV segment end")?;
    ensure!(
        start < end,
        "KV segment range [{start}, {end}) is empty or reversed"
    );
    let n_parts = slot_file::read_u32(inp).context("reading the KV segment part count")? as usize;
    ensure!(
        (1..=MAX_SEGMENT_PARTS).contains(&n_parts),
        "KV segment has {n_parts} parts, the range is 1..={MAX_SEGMENT_PARTS}"
    );
    let mut parts = Vec::with_capacity(n_parts);
    for part_index in 0..n_parts {
        let n_buffers = slot_file::read_u32(inp)
            .with_context(|| format!("reading the KV buffer count of part {part_index}"))?
            as usize;
        ensure!(
            n_buffers <= MAX_SEGMENT_BUFFERS,
            "KV segment part {part_index} has {n_buffers} buffers, the limit is {MAX_SEGMENT_BUFFERS}"
        );
        let mut rows = Vec::with_capacity(n_buffers);
        for buffer_index in 0..n_buffers {
            let row = slot_file::read_u64(inp).with_context(|| {
                format!("reading the row size of part {part_index} buffer {buffer_index}")
            })?;
            ensure!(
                row > 0,
                "KV segment part {part_index} buffer {buffer_index} has a zero row size"
            );
            ensure!(
                row <= MAX_SEGMENT_BUFFER_LEN,
                "KV segment row of {row} bytes exceeds the {MAX_SEGMENT_BUFFER_LEN}-byte limit"
            );
            rows.push(row);
        }
        parts.push(rows);
    }
    let payload_len = kv_segment_payload_len(start, end, &parts)?;
    Ok(KvSegmentHeader { start, end, parts, payload_len })
}

/// Parsed `STRRSEG` header: the PLE token pair and one saved-buffer length vector per part.
#[derive(Debug, PartialEq, Eq)]
struct RecurrentSegmentHeader {
    ple_prev: [i64; 2],
    parts: Vec<Vec<u64>>,
}

/// Write a `STRRSEG` v1 header.
fn write_recurrent_segment_header<W: Write>(out: &mut W, header: &RecurrentSegmentHeader) -> Result<()> {
    ensure!(
        (1..=MAX_SEGMENT_PARTS).contains(&header.parts.len()),
        "recurrent segment has {} parts, the range is 1..={MAX_SEGMENT_PARTS}",
        header.parts.len()
    );
    out.write_all(&RECURRENT_SEGMENT_MAGIC)
        .context("writing the recurrent segment magic")?;
    slot_file::write_u32(out, SEGMENT_VERSION)?;
    for value in header.ple_prev {
        slot_file::write_u64(out, value as u64)?;
    }
    slot_file::write_u32(out, header.parts.len() as u32)?;
    for (part_index, part) in header.parts.iter().enumerate() {
        ensure!(
            part.len() <= MAX_SEGMENT_BUFFERS,
            "recurrent segment part {part_index} has {} buffers, the limit is {MAX_SEGMENT_BUFFERS}",
            part.len()
        );
        slot_file::write_u32(out, part.len() as u32)?;
        for &len in part {
            ensure!(
                len <= MAX_SEGMENT_BUFFER_LEN,
                "recurrent segment buffer of {len} bytes exceeds the {MAX_SEGMENT_BUFFER_LEN}-byte limit"
            );
            slot_file::write_u64(out, len)?;
        }
    }
    Ok(())
}

/// Read and validate a `STRRSEG` v1 header, including its counts and buffer lengths.
fn read_recurrent_segment_header<R: Read>(inp: &mut R) -> Result<RecurrentSegmentHeader> {
    let mut magic = [0u8; 7];
    inp.read_exact(&mut magic)
        .context("reading the recurrent segment magic")?;
    ensure!(
        magic == RECURRENT_SEGMENT_MAGIC,
        "bad recurrent segment magic {magic:02x?}, expected {RECURRENT_SEGMENT_MAGIC:02x?}"
    );
    let version = slot_file::read_u32(inp).context("reading the recurrent segment version")?;
    ensure!(
        version == SEGMENT_VERSION,
        "unsupported recurrent segment version {version}, expected {SEGMENT_VERSION}"
    );
    let ple0 = slot_file::read_u64(inp).context("reading the recurrent PLE position")? as i64;
    let ple1 = slot_file::read_u64(inp).context("reading the recurrent PLE position")? as i64;
    let n_parts = slot_file::read_u32(inp).context("reading the recurrent part count")? as usize;
    ensure!(
        (1..=MAX_SEGMENT_PARTS).contains(&n_parts),
        "recurrent segment has {n_parts} parts, the range is 1..={MAX_SEGMENT_PARTS}"
    );
    let mut parts = Vec::with_capacity(n_parts);
    for part_index in 0..n_parts {
        let n_slot = slot_file::read_u32(inp)
            .with_context(|| format!("reading the recurrent buffer count of part {part_index}"))?
            as usize;
        ensure!(
            n_slot <= MAX_SEGMENT_BUFFERS,
            "recurrent segment part {part_index} has {n_slot} buffers, the limit is {MAX_SEGMENT_BUFFERS}"
        );
        let mut lens = Vec::with_capacity(n_slot);
        for buffer_index in 0..n_slot {
            let len = slot_file::read_u64(inp).with_context(|| {
                format!("reading the length of part {part_index} buffer {buffer_index}")
            })?;
            ensure!(
                len <= MAX_SEGMENT_BUFFER_LEN,
                "recurrent segment buffer of {len} bytes exceeds the {MAX_SEGMENT_BUFFER_LEN}-byte limit"
            );
            lens.push(len);
        }
        parts.push(lens);
    }
    Ok(RecurrentSegmentHeader { ple_prev: [ple0, ple1], parts })
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
        self.save_slot_to_with(slot, out, &mut no_yield)
    }

    /// Resumable [`Self::save_slot_to`]: `between(self)` runs after every 64 MiB GPU→host chunk, so a scheduler can
    /// service other slots between copies. The callback must not change `slot`'s state (buffers, history or saved
    /// marker); the caller marks the target active/saving for the duration so no request can use it. The target slot
    /// is selected again before each copy because the callback may have switched the current slot.
    pub fn save_slot_to_with<W: Write, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        out: &mut W,
        between: &mut F,
    ) -> Result<u64> {
        let (tokens, header) = self.slot_snapshot(slot)?;
        self.write_slot_stream_with(slot, &tokens, &header, out, between)
    }

    /// Restore a `STRATASLOT` v1 stream from `inp` into `slot`. Same validation, unified-KV growth, install and
    /// error handling (a partially installed slot is cleared) as [`Self::restore_slot_file`], without opening a
    /// path. `allow_evict` is as in `restore_slot_file`.
    pub fn restore_slot_from<R: Read>(&mut self, slot: usize, inp: &mut R, allow_evict: bool) -> Result<u64> {
        self.restore_slot_from_with(slot, inp, allow_evict, &mut no_yield)
    }

    /// Resumable [`Self::restore_slot_from`]: `between(self)` runs after every 64 MiB host→GPU chunk. The callback
    /// must not change `slot`'s state (buffers, history or saved marker); the caller marks the target active/saving
    /// for the duration so no request can use it. The target slot is selected again before each copy because the
    /// callback may have switched the current slot.
    pub fn restore_slot_from_with<R: Read, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        inp: &mut R,
        allow_evict: bool,
        between: &mut F,
    ) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        let mut inp = Count::new(BufReader::with_capacity(8 << 20, inp));
        let (header, tokens) = slot_file::read_header(&mut inp)?;
        ensure!(header.fingerprint == self.slot_fingerprint(), "slot file model/config fingerprint mismatch");
        ensure!(header.n_tokens as usize == tokens.len(), "slot file token count mismatch");
        ensure!(!tokens.is_empty(), "slot file has no tokens");
        ensure!(tokens.len() <= self.ctx, "slot file has {} tokens, --ctx is {}", tokens.len(), self.ctx);
        let n_parts = self.parts().count();
        ensure!(header.n_parts as usize == n_parts, "slot file has {} parts, this model has {n_parts}", header.n_parts);
        let n = tokens.len();
        self.prepare_segmented_restore(slot, n, allow_evict)?;
        self.save_state()?;
        let r = (|| -> Result<u64> {
            for part in 0..n_parts {
                // The v1 format interleaves each length with its bytes: PLE pair, n_slot, then
                // (len, bytes) per recurrent buffer, n_kv, then (row, bytes) per KV buffer. Read and install each
                // buffer immediately; the resumable callback runs between its bounded chunks.
                self.use_slot(slot);
                let (ple0, ple1) = {
                    let p = self.part_mut(part)?;
                    let _ = p;
                    (slot_file::read_u64(&mut inp)? as i64, slot_file::read_u64(&mut inp)? as i64)
                };
                let n_slot = {
                    self.use_slot(slot);
                    let p = self.part_mut(part)?;
                    let n_slot = slot_file::read_u32(&mut inp)? as usize;
                    ensure!(n_slot == p.slot().len(), "slot file has {n_slot} recurrent buffers, this model has {}", p.slot().len());
                    n_slot
                };
                for i in 0..n_slot {
                    self.use_slot(slot);
                    let (b, len) = {
                        let p = self.part_mut(part)?;
                        let b = p.slot()[i];
                        let len = slot_file::read_u64(&mut inp)? as usize;
                        ensure!(len == b.len(), "slot file recurrent buffer {len} bytes, this model has {}", b.len());
                        (Buf::clone(b), len)
                    };
                    self.read_copies(slot, &[(part, b, 0, len)], &mut inp, between)?;
                }
                let n_kv = {
                    self.use_slot(slot);
                    let p = self.part_mut(part)?;
                    let n_kv = slot_file::read_u32(&mut inp)? as usize;
                    ensure!(n_kv == p.kv_bufs().len(), "slot file has {n_kv} KV buffers, this model has {}", p.kv_bufs().len());
                    n_kv
                };
                for i in 0..n_kv {
                    self.use_slot(slot);
                    let (b, bytes) = {
                        let p = self.part_mut(part)?;
                        let (b, rows) = p.kv_bufs()[i];
                        let row = slot_file::read_u64(&mut inp)? as usize;
                        ensure!(row == b.len() / rows.max(1), "slot file KV row is {row} bytes, this model has {}", b.len() / rows.max(1));
                        let bytes = row.checked_mul(n).context("slot file KV size overflow")?;
                        ensure!(bytes <= b.len(), "slot file KV rows do not fit the slot buffer");
                        (Buf::clone(b), bytes)
                    };
                    self.read_copies(slot, &[(part, b, 0, bytes)], &mut inp, between)?;
                }
                self.use_slot(slot);
                let p = self.part_mut(part)?;
                p.saved_ple_prev = [ple0, ple1];
                p.saved_pos = n;
                p.saved_ids = tokens.clone();
                p.history = tokens.clone();
                p.saved_owner = header.owner;
                p.rope_offset = header.rope_offset;
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
        self.save_slot_file_async_as(slot, path, self.owner)
    }

    /// `save_slot_file_async` with an explicit conversation owner. The archive uses `slot_owner` here, not the
    /// owner of the request that triggered the checkpoint: an eviction caused by another key must not re-label the
    /// state. The owner is put in the header and in `saved_owner`; the caller's request owner is restored on
    /// success, error and unwind, as in [`Self::save_slot_file_as`].
    pub fn save_slot_file_async_as(&mut self, slot: usize, path: &Path, owner: u64) -> Result<AsyncSave> {
        let prev = std::mem::replace(&mut self.owner, owner);
        let guard = OwnerGuard { model: self, prev };
        guard.model.save_slot_file_async_owned(slot, path)
    }

    fn save_slot_file_async_owned(&mut self, slot: usize, path: &Path) -> Result<AsyncSave> {
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
                // Informational: the model has MTP/PLE/QSA buffers, not whether `saved` happens to be allocated
                // yet. Computing this from `slot()` made the flags depend on whether an earlier save had run.
                | (self.mtp.is_some() || self.ple_ring.is_some() || self.qsa.is_some()).then_some(slot_file::FLAG_AUX).unwrap_or(0),
            n_parts: self.parts().count() as u32,
        };
        Ok((tokens, header))
    }

    /// The framing and buffer copies of a save: `save_state`, header, every part, footer.
    fn write_slot_stream<W: Write>(&mut self, tokens: &[u32], header: &SlotHeader, out: &mut W) -> Result<u64> {
        let slot = self.current_slot();
        self.write_slot_stream_with(slot, tokens, header, out, &mut no_yield)
    }

    /// [`Self::write_slot_stream`] with resumable copies: `between(self)` runs after every 64 MiB GPU→host chunk.
    fn write_slot_stream_with<W: Write, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        tokens: &[u32],
        header: &SlotHeader,
        out: &mut W,
        between: &mut F,
    ) -> Result<u64> {
        // Allocate the recurrent copies and make the current recurrent state the saved one.
        self.save_state()?;

        // Capture the framing and the buffer handles once, preserving the v1 interleaving: each buffer's length is
        // written immediately before its bytes. The clones keep the target slot's device memory alive if the
        // callback selects another slot; `use_slot(slot)` at each copy makes the model state target it too.
        let mut steps = Vec::new();
        let mut part = 0usize;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            let mut head = Vec::new();
            for v in p.saved_ple_prev { slot_file::write_u64(&mut head, v as u64)?; }
            let slot_bufs = p.slot();
            slot_file::write_u32(&mut head, slot_bufs.len() as u32)?;
            steps.push(SlotWriteStep::Frame(head));
            for b in slot_bufs {
                let mut len = Vec::new();
                slot_file::write_u64(&mut len, b.len() as u64)?;
                steps.push(SlotWriteStep::Frame(len));
                let mut plan = Vec::new();
                push_copy_chunks(&mut plan, part, b, 0, b.len());
                steps.extend(plan.into_iter().map(SlotWriteStep::Copy));
            }
            let kv = p.kv_bufs();
            let mut kv_head = Vec::new();
            slot_file::write_u32(&mut kv_head, kv.len() as u32)?;
            steps.push(SlotWriteStep::Frame(kv_head));
            for (b, rows) in kv {
                let row = b.len() / rows.max(1);
                let mut frame = Vec::new();
                slot_file::write_u64(&mut frame, row as u64)?;
                steps.push(SlotWriteStep::Frame(frame));
                let bytes = row.checked_mul(tokens.len()).context("slot file KV size overflow")?;
                let mut plan = Vec::new();
                push_copy_chunks(&mut plan, part, b, 0, bytes);
                steps.extend(plan.into_iter().map(SlotWriteStep::Copy));
            }
            part += 1;
            q = p.next.as_deref_mut();
        }

        let mut out = Count::new(out);
        slot_file::write_header(&mut out, header, tokens)?;
        for step in steps {
            match step {
                SlotWriteStep::Frame(frame) => out.write_all(&frame)?,
                SlotWriteStep::Copy(copy) => self.write_copies(slot, &[copy], &mut out, between)?,
            }
        }
        let payload = out.count();
        slot_file::write_footer(&mut out, payload)?;
        out.flush()?;
        Ok(out.count())
    }

    /// Copy the bytes of a save plan to `out`, calling `between(self)` after each bounded chunk. The target slot is
    /// selected again before every copy because the callback may have left a different slot current.
    fn write_copies<W: Write, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        plan: &[CopyPlan],
        out: &mut W,
        between: &mut F,
    ) -> Result<()> {
        for (part, b, offset, len) in plan {
            self.use_slot(slot);
            let v = {
                let g = self.part_ref(*part)?;
                g.get::<u8>(b, *offset, *len)?
            };
            out.write_all(&v)?;
            if let Err(e) = between(self) {
                self.use_slot(slot);
                return Err(e);
            }
        }
        self.use_slot(slot);
        Ok(())
    }

    /// Read a restore plan from `inp` into the target slot, calling `between(self)` after each bounded chunk. The
    /// target slot is selected again before every copy because the callback may have left a different slot current.
    fn read_copies<R: Read, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        plan: &[CopyPlan],
        inp: &mut R,
        between: &mut F,
    ) -> Result<()> {
        for (part, b, offset, len) in plan {
            self.use_slot(slot);
            let mut v = vec![0u8; *len];
            inp.read_exact(&mut v)?;
            {
                let g = self.part_ref(*part)?;
                g.put(&v, b, *offset)?;
            }
            if let Err(e) = between(self) {
                self.use_slot(slot);
                return Err(e);
            }
        }
        self.use_slot(slot);
        Ok(())
    }

    /// The backend of model part `index` (0 is `self`, then the `next` chain).
    fn part_ref(&self, index: usize) -> Result<&dyn crate::backend::Backend> {
        let mut p = self;
        for _ in 0..index {
            p = p.next.as_deref().with_context(|| format!("model part {index} is missing"))?;
        }
        Ok(p.gpu.as_ref())
    }

    /// Model part `index` (0 is `self`, then the `next` chain).
    fn part_mut(&mut self, index: usize) -> Result<&mut Self> {
        let mut p = self;
        for _ in 0..index {
            p = p.next.as_deref_mut().with_context(|| format!("model part {index} is missing"))?;
        }
        Ok(p)
    }

    /// Save the K/V rows of the token range `[start, end)` of `slot` as an opaque `STRKVSEG` v1 payload and return
    /// the number of bytes written.
    ///
    /// The K/V representation is exactly the running model's (q8/f16/q4/q2), so this is a raw row copy: the caller
    /// checks the model fingerprint through the archive entry. `use_slot` selects the slot; `pos`, `history`,
    /// `saved_*` and the caller's current slot are left as they were.
    pub fn save_kv_segment<W: Write>(
        &mut self,
        slot: usize,
        start: usize,
        end: usize,
        out: &mut W,
    ) -> Result<u64> {
        self.save_kv_segment_with(slot, start, end, out, &mut no_yield)
    }

    /// Resumable [`Self::save_kv_segment`]: `between(self)` runs after each at-most-64-MiB GPU→host chunk of the
    /// segment (at least once for a non-empty segment). The callback must not change `slot`'s KV rows; the caller
    /// marks the target active/saving for the duration. The target slot is selected again before each copy.
    pub fn save_kv_segment_with<W: Write, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        start: usize,
        end: usize,
        out: &mut W,
        between: &mut F,
    ) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        ensure!(
            start < end,
            "KV segment range [{start}, {end}) is empty or reversed"
        );
        let prev = self.current_slot();
        self.use_slot(slot);
        let guard = SlotGuard { model: self, prev };
        guard.model.write_kv_segment_with(slot, start, end, out, between)
    }

    /// Restore a `STRKVSEG` v1 payload into the K/V rows of `[start, end)` of `slot` and return the bytes consumed.
    ///
    /// Every header count, the range, the row sizes and the model layout are validated before the first host→GPU
    /// copy, so a malformed header does not partially change the slot. The recurrent state, `pos`, `history`, the
    /// saved history and the caller's current slot are untouched.
    pub fn restore_kv_segment<R: Read>(
        &mut self,
        slot: usize,
        start: usize,
        end: usize,
        inp: &mut R,
    ) -> Result<u64> {
        self.restore_kv_segment_with(slot, start, end, inp, &mut no_yield)
    }

    /// Resumable [`Self::restore_kv_segment`]: `between(self)` runs after each at-most-64-MiB host→GPU chunk of the
    /// segment (at least once for a non-empty segment). The callback must not change `slot`'s KV rows; the caller
    /// marks the target active/saving for the duration. The target slot is selected again before each copy.
    pub fn restore_kv_segment_with<R: Read, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        start: usize,
        end: usize,
        inp: &mut R,
        between: &mut F,
    ) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        ensure!(
            start < end,
            "KV segment range [{start}, {end}) is empty or reversed"
        );
        let prev = self.current_slot();
        self.use_slot(slot);
        let guard = SlotGuard { model: self, prev };
        guard.model.read_kv_segment_with(slot, start, end, inp, between)
    }

    fn write_kv_segment_with<W: Write, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        start: usize,
        end: usize,
        out: &mut W,
        between: &mut F,
    ) -> Result<u64> {
        // Capture and validate one row size per buffer first, so a state that cannot be addressed fails before the
        // header is written. The same pass records the bounded copies the callback can yield between.
        let mut parts: Vec<Vec<u64>> = Vec::new();
        let mut plan: Vec<CopyPlan> = Vec::new();
        let mut part = 0usize;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            let kv = p.kv_bufs();
            let mut rows = Vec::with_capacity(kv.len());
            for &(b, n_rows) in &kv {
                ensure!(n_rows > 0, "slot {slot}: a KV buffer has no rows");
                let row = b.len() / n_rows;
                ensure!(
                    row > 0,
                    "slot {slot}: a KV buffer is {}-1 rows, shorter than {n_rows} rows",
                    b.len()
                );
                ensure!(
                    end <= n_rows,
                    "KV segment ends at {end}, but slot {slot} has {n_rows} rows"
                );
                let offset = row.checked_mul(start).context("KV segment offset overflow")?;
                let bytes = row
                    .checked_mul(end - start)
                    .context("KV segment size overflow")?;
                push_copy_chunks(&mut plan, part, b, offset, bytes);
                rows.push(row as u64);
            }
            parts.push(rows);
            part += 1;
            q = p.next.as_deref_mut();
        }

        let mut out = Count::new(out);
        let payload_len = write_kv_segment_header(&mut out, start as u64, end as u64, &parts)?;
        self.write_copies(slot, &plan, &mut out, between)?;
        let written: u64 = plan.iter().map(|&(_, _, _, len)| len as u64).sum();
        ensure!(
            written == payload_len,
            "KV segment wrote {written} payload bytes, the header describes {payload_len}"
        );
        out.flush()?;
        Ok(out.count())
    }

    fn read_kv_segment_with<R: Read, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        start: usize,
        end: usize,
        inp: &mut R,
        between: &mut F,
    ) -> Result<u64> {
        let mut inp = Count::new(inp);
        let header = read_kv_segment_header(&mut inp)?;
        ensure!(
            header.start == start as u64 && header.end == end as u64,
            "KV segment range [{}, {}) does not match the requested [{start}, {end})",
            header.start,
            header.end
        );
        let n_parts = self.parts().count();
        ensure!(
            header.parts.len() == n_parts,
            "KV segment has {} parts, this model has {n_parts}",
            header.parts.len()
        );

        // Validation pass: compare every descriptor with the model before touching the slot. The same pass records
        // the bounded host→GPU copies the callback can yield between.
        let rows = end - start;
        let mut plan: Vec<CopyPlan> = Vec::new();
        let mut part_index = 0;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            let kv = p.kv_bufs();
            let described = &header.parts[part_index];
            ensure!(
                described.len() == kv.len(),
                "KV segment part {part_index} has {} buffers, this model has {}",
                described.len(),
                kv.len()
            );
            for (buffer_index, &(b, n_rows)) in kv.iter().enumerate() {
                ensure!(
                    n_rows > 0,
                    "slot {slot}: part {part_index} buffer {buffer_index} has no rows"
                );
                let row = b.len() / n_rows;
                ensure!(
                    row > 0,
                    "slot {slot}: part {part_index} buffer {buffer_index} has an empty row"
                );
                ensure!(
                    end <= n_rows,
                    "KV segment ends at {end}, but slot {slot} part {part_index} has {n_rows} rows"
                );
                ensure!(
                    described[buffer_index] == row as u64,
                    "KV segment part {part_index} buffer {buffer_index} has {} bytes a row, this model has {row}",
                    described[buffer_index]
                );
                let offset = row
                    .checked_mul(start)
                    .context("KV segment offset overflow")?;
                let bytes = row
                    .checked_mul(rows)
                    .context("KV segment size overflow")?;
                ensure!(
                    offset.checked_add(bytes).is_some_and(|n| n <= b.len()),
                    "KV segment part {part_index} buffer {buffer_index} does not fit its slot buffer"
                );
                push_copy_chunks(&mut plan, part_index, b, offset, bytes);
            }
            part_index += 1;
            q = p.next.as_deref_mut();
        }

        // Install pass: the header and the layout are known good.
        let header_bytes = inp.count();
        self.read_copies(slot, &plan, &mut inp, between)?;
        let payload_read = inp.count() - header_bytes;
        ensure!(
            payload_read == header.payload_len,
            "KV segment payload is {payload_read} bytes, the header describes {}",
            header.payload_len
        );
        Ok(inp.count())
    }

    /// Save `slot`'s recurrent/DeltaNet/MTP/PLE/QSA state as an opaque `STRRSEG` v1 payload and return the number
    /// of bytes written.
    ///
    /// This calls `save_state` (which clears `saved_ids`, as the full-slot save does), writes that snapshot and
    /// leaves `pos`/`history` and the caller's current slot alone. `saved_owner` is not part of the payload: the
    /// archive entry stores it and passes it to [`Self::finish_segmented_restore`].
    pub fn save_recurrent_to<W: Write>(&mut self, slot: usize, out: &mut W) -> Result<u64> {
        self.save_recurrent_to_with(slot, out, &mut no_yield)
    }

    /// Resumable [`Self::save_recurrent_to`]: `between(self)` runs after each at-most-64-MiB GPU→host chunk of the
    /// snapshot (at least once when the snapshot has buffers). The callback must not change `slot`'s saved buffers;
    /// the caller marks the target active/saving for the duration. The target slot is selected again before each
    /// copy.
    pub fn save_recurrent_to_with<W: Write, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        out: &mut W,
        between: &mut F,
    ) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        let prev = self.current_slot();
        self.use_slot(slot);
        let guard = SlotGuard { model: self, prev };
        guard.model.write_recurrent_segment_with(slot, out, between)
    }

    /// Restore a `STRRSEG` v1 payload into `slot`'s saved buffers and return the bytes consumed.
    ///
    /// The saved buffers are filled and `saved_ple_prev` is set, but `saved_pos`, `saved_ids`, `history` and
    /// `saved_owner` are left for [`Self::finish_segmented_restore`], after every KV segment has also been installed.
    /// A corrupt header/count/length is rejected before any buffer is changed.
    pub fn restore_recurrent_from<R: Read>(&mut self, slot: usize, inp: &mut R) -> Result<u64> {
        self.restore_recurrent_from_with(slot, inp, &mut no_yield)
    }

    /// Resumable [`Self::restore_recurrent_from`]: `between(self)` runs after each at-most-64-MiB host→GPU chunk of
    /// the snapshot (at least once when the snapshot has buffers). The callback must not change `slot`'s saved
    /// buffers; the caller marks the target active/saving for the duration. The target slot is selected again before
    /// each copy.
    pub fn restore_recurrent_from_with<R: Read, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        inp: &mut R,
        between: &mut F,
    ) -> Result<u64> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        let prev = self.current_slot();
        self.use_slot(slot);
        let guard = SlotGuard { model: self, prev };
        guard.model.read_recurrent_segment_with(slot, inp, between)
    }

    fn write_recurrent_segment_with<W: Write, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        out: &mut W,
        between: &mut F,
    ) -> Result<u64> {
        // The payload is the `saved` snapshot, not the live state.
        self.save_state()?;
        let mut parts: Vec<Vec<u64>> = Vec::new();
        let mut plan: Vec<CopyPlan> = Vec::new();
        let mut part = 0usize;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            let bufs = p.slot();
            parts.push(bufs.iter().map(|b| b.len() as u64).collect());
            for b in bufs {
                push_copy_chunks(&mut plan, part, b, 0, b.len());
            }
            part += 1;
            q = p.next.as_deref_mut();
        }
        let header = RecurrentSegmentHeader {
            // PLE lives on one part of a split model; the single header field takes that part's pair.
            ple_prev: self
                .parts()
                .find(|p| p.ple_ring.is_some())
                .map_or(self.saved_ple_prev, |p| p.saved_ple_prev),
            parts,
        };
        let mut out = Count::new(out);
        write_recurrent_segment_header(&mut out, &header)?;
        self.write_copies(slot, &plan, &mut out, between)?;
        out.flush()?;
        Ok(out.count())
    }

    fn read_recurrent_segment_with<R: Read, F: FnMut(&mut GpuModel<'_>) -> Result<()>>(
        &mut self,
        slot: usize,
        inp: &mut R,
        between: &mut F,
    ) -> Result<u64> {
        let mut inp = Count::new(inp);
        let header = read_recurrent_segment_header(&mut inp)?;
        let n_parts = self.parts().count();
        ensure!(
            header.parts.len() == n_parts,
            "recurrent segment has {} parts, this model has {n_parts}",
            header.parts.len()
        );
        // A fresh model that never saved a state has no DeltaNet copies yet. Allocate the same set `save_state`
        // would, without overwriting the target's saved position/history/owner.
        self.ensure_saved_buffers()?;

        // Validate every count and length before the first copy, and record the bounded copies.
        let mut plan: Vec<CopyPlan> = Vec::new();
        let mut part_index = 0;
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            let bufs = p.slot();
            let described = &header.parts[part_index];
            ensure!(
                described.len() == bufs.len(),
                "recurrent segment part {part_index} has {} buffers, this model has {}",
                described.len(),
                bufs.len()
            );
            for (buffer_index, b) in bufs.iter().enumerate() {
                ensure!(
                    described[buffer_index] == b.len() as u64,
                    "recurrent segment part {part_index} buffer {buffer_index} has {} bytes, this model has {}",
                    described[buffer_index],
                    b.len()
                );
                push_copy_chunks(&mut plan, part_index, b, 0, b.len());
            }
            part_index += 1;
            q = p.next.as_deref_mut();
        }

        self.read_copies(slot, &plan, &mut inp, between)?;
        for part_index in 0..n_parts {
            self.use_slot(slot);
            self.part_mut(part_index)?.saved_ple_prev = header.ple_prev;
        }
        Ok(inp.count())
    }

    /// Complete a segmented restore: adopt `tokens` as the slot's history at position `tokens.len()`, set its owner
    /// and rope offset on every part, and install the recurrent snapshot loaded by [`Self::restore_recurrent_from`].
    ///
    /// A segmented restore has no full-slot header, so this is where the history marker (`saved_ids`), `saved_pos`,
    /// `history` and `saved_owner` are committed. The K/V rows must already be installed. After it returns, `pos`
    /// and `history` are consistent for the next prefill/decode.
    pub fn finish_segmented_restore(
        &mut self,
        slot: usize,
        tokens: &[u32],
        owner: u64,
        rope_offset: i32,
    ) -> Result<()> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        ensure!(!tokens.is_empty(), "segmented restore has no tokens");
        ensure!(
            tokens.len() <= self.ctx,
            "segmented restore has {} tokens, --ctx is {}",
            tokens.len(),
            self.ctx
        );
        let prev = self.current_slot();
        self.use_slot(slot);
        let guard = SlotGuard { model: self, prev };
        guard.model.finish_segmented(slot, tokens, owner, rope_offset)
    }

    fn finish_segmented(
        &mut self,
        slot: usize,
        tokens: &[u32],
        owner: u64,
        rope_offset: i32,
    ) -> Result<()> {
        // Reject a state that was never installed rather than letting `restore_state` find an empty snapshot.
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            let has_delta = p
                .layers
                .iter()
                .any(|l| matches!(l.mixer, super::Mixer::Delta { .. }));
            ensure!(
                !has_delta || !p.saved.is_empty(),
                "slot {slot}: no recurrent state restored (call restore_recurrent_from first)"
            );
            q = p.next.as_deref_mut();
        }

        let mut q = Some(&mut *self);
        while let Some(p) = q {
            p.saved_pos = tokens.len();
            p.saved_ids = tokens.to_vec();
            p.history = tokens.to_vec();
            p.saved_owner = owner;
            p.rope_offset = rope_offset;
            q = p.next.as_deref_mut();
        }
        self.restore_state()
    }

    /// Allocate the per-part saved buffers if they were not allocated yet, without touching the saved marker.
    fn ensure_saved_buffers(&mut self) -> Result<()> {
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            if p.saved.is_empty() {
                p.saved = super::alloc_saved(&*p.gpu, &p.layers, p.seqs.len() > 1)?;
            }
            q = p.next.as_deref_mut();
        }
        Ok(())
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

    /// Make `slot` able to hold `n` tokens: validate the context/budget, grow a unified slot when needed, and refuse
    /// (or reclaim empty victims, when `allow_evict`) exactly like [`Self::restore_slot_from`].
    ///
    /// This is the capacity half of a restore without reading a file: a segmented restore calls it before
    /// [`Self::restore_kv_segment`], which needs the rows `[0, n)` to exist. An automatic restore passes
    /// `allow_evict = false` and never touches a non-empty victim.
    pub fn prepare_segmented_restore(&mut self, slot: usize, n: usize, allow_evict: bool) -> Result<()> {
        ensure!(slot < self.slots(), "slot {slot} of {}", self.slots());
        ensure!(n > 0, "restored state has no tokens");
        ensure!(n <= self.ctx, "restored state has {n} tokens, --ctx is {}", self.ctx);
        self.use_slot(slot);
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
        Ok(())
    }

    /// The M-RoPE offset of slot `s` (0 for a plain text slot), read without switching slots.
    pub fn slot_rope_offset(&self, s: usize) -> i32 {
        if s == self.cur { self.rope_offset } else { self.seqs[s].rope }
    }

    /// Mark `slot` reusable at `tokens` for `owner` without installing a snapshot. A v1 synchronous save does this
    /// after the rename; the segmented save calls it after the state file and the manifest are committed.
    pub fn mark_slot_saved(&mut self, slot: usize, tokens: &[u32], owner: u64) {
        if slot >= self.slots() { return; }
        let prev = self.current_slot();
        self.use_slot(slot);
        let mut q = Some(&mut *self);
        while let Some(p) = q {
            p.saved_pos = tokens.len();
            p.saved_ids = tokens.to_vec();
            p.history = tokens.to_vec();
            p.saved_owner = owner;
            q = p.next.as_deref_mut();
        }
        self.use_slot(prev);
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

    // Segmented payload headers are pure CPU framing; these tests do not need a GPU.

    fn put_u32(buf: &mut [u8], off: usize, v: u32) {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn put_u64(buf: &mut [u8], off: usize, v: u64) {
        buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// A `STRKVSEG` stream assembled without validation, for corrupt-header tests.
    fn kv_header_bytes(start: u64, end: u64, parts: &[Vec<u64>]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&KV_SEGMENT_MAGIC);
        v.extend_from_slice(&SEGMENT_VERSION.to_le_bytes());
        v.extend_from_slice(&start.to_le_bytes());
        v.extend_from_slice(&end.to_le_bytes());
        v.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for part in parts {
            v.extend_from_slice(&(part.len() as u32).to_le_bytes());
            for &row in part {
                v.extend_from_slice(&row.to_le_bytes());
            }
        }
        v
    }

    /// A `STRRSEG` stream assembled without validation, for corrupt-header tests.
    fn recurrent_header_bytes(ple_prev: [i64; 2], parts: &[Vec<u64>]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&RECURRENT_SEGMENT_MAGIC);
        v.extend_from_slice(&SEGMENT_VERSION.to_le_bytes());
        for value in ple_prev {
            v.extend_from_slice(&(value as u64).to_le_bytes());
        }
        v.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for part in parts {
            v.extend_from_slice(&(part.len() as u32).to_le_bytes());
            for &len in part {
                v.extend_from_slice(&len.to_le_bytes());
            }
        }
        v
    }

    #[test]
    fn kv_segment_header_round_trip() {
        let parts = vec![vec![128, 64], vec![256]];
        let mut buf = Vec::new();
        let payload_len = write_kv_segment_header(&mut buf, 10, 20, &parts).unwrap();
        assert_eq!(payload_len, (128 + 64 + 256) * 10);
        let header = read_kv_segment_header(&mut &buf[..]).unwrap();
        assert_eq!(
            header,
            KvSegmentHeader { start: 10, end: 20, parts: parts.clone(), payload_len }
        );

        // A model part with no attention layer describes zero buffers; the payload is then just the header.
        let mut buf = Vec::new();
        assert_eq!(write_kv_segment_header(&mut buf, 0, 8, &[vec![]]).unwrap(), 0);
        let header = read_kv_segment_header(&mut &buf[..]).unwrap();
        assert!(header.parts[0].is_empty());
        assert_eq!(header.payload_len, 0);
    }

    #[test]
    fn kv_segment_header_rejects_bad_magic_version_and_range() {
        let good = kv_header_bytes(0, 4, &[vec![128]]);

        let mut bad = good.clone();
        bad[0] ^= 0xff;
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("bad KV segment magic"), "{err}");

        let mut bad = good.clone();
        put_u32(&mut bad, 8, SEGMENT_VERSION + 1);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("unsupported KV segment version"), "{err}");

        let mut bad = good.clone();
        put_u64(&mut bad, 20, 0);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("empty or reversed"), "{err}");

        let mut bad = good.clone();
        bad.truncate(bad.len() - 2);
        assert!(read_kv_segment_header(&mut &bad[..]).is_err());
    }

    #[test]
    fn kv_segment_header_rejects_bad_counts_and_lengths() {
        let good = kv_header_bytes(0, 4, &[vec![128, 256]]);

        let mut bad = good.clone();
        put_u32(&mut bad, 28, 0);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("parts"), "{err}");

        let mut bad = good.clone();
        put_u32(&mut bad, 28, (MAX_SEGMENT_PARTS + 1) as u32);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("parts"), "{err}");

        let mut bad = good.clone();
        put_u32(&mut bad, 32, (MAX_SEGMENT_BUFFERS + 1) as u32);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("buffers"), "{err}");

        let mut bad = good.clone();
        put_u64(&mut bad, 36, 0);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("zero row size"), "{err}");

        let mut bad = good.clone();
        put_u64(&mut bad, 36, MAX_SEGMENT_BUFFER_LEN + 1);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");

        // A payload length that does not fit u64 is rejected while reading the header.
        let bad = kv_header_bytes(0, 1 << 40, &[vec![MAX_SEGMENT_BUFFER_LEN]]);
        let err = read_kv_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    #[test]
    fn recurrent_segment_header_round_trip() {
        let header = RecurrentSegmentHeader {
            ple_prev: [-1, 0x1_0000_0002],
            parts: vec![vec![8, 16], vec![32]],
        };
        let mut buf = Vec::new();
        write_recurrent_segment_header(&mut buf, &header).unwrap();
        let got = read_recurrent_segment_header(&mut &buf[..]).unwrap();
        assert_eq!(got, header);
    }

    #[test]
    fn recurrent_segment_header_rejects_bad_magic_version_and_counts() {
        let good = recurrent_header_bytes([3, 4], &[vec![8]]);

        let mut bad = good.clone();
        bad[1] ^= 0xff;
        let err = read_recurrent_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("bad recurrent segment magic"), "{err}");

        let mut bad = good.clone();
        put_u32(&mut bad, 7, SEGMENT_VERSION + 1);
        let err = read_recurrent_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("unsupported recurrent segment version"), "{err}");

        let mut bad = good.clone();
        put_u32(&mut bad, 27, 0);
        let err = read_recurrent_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("parts"), "{err}");

        let mut bad = good.clone();
        put_u32(&mut bad, 27, (MAX_SEGMENT_PARTS + 1) as u32);
        assert!(read_recurrent_segment_header(&mut &bad[..]).is_err());

        let mut bad = good.clone();
        put_u32(&mut bad, 31, (MAX_SEGMENT_BUFFERS + 1) as u32);
        let err = read_recurrent_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("buffers"), "{err}");

        let mut bad = good.clone();
        put_u64(&mut bad, 35, MAX_SEGMENT_BUFFER_LEN + 1);
        let err = read_recurrent_segment_header(&mut &bad[..]).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");

        let mut bad = good.clone();
        bad.truncate(bad.len() - 1);
        assert!(read_recurrent_segment_header(&mut &bad[..]).is_err());
    }

    /// A buffer larger than one `CHUNK` becomes one copy per chunk, which is one `between` call per entry: the
    /// resumable variants cannot spend more than 64 MiB in a single uninterruptible copy.
    #[test]
    fn copy_plan_splits_a_multi_chunk_stream() {
        let len = 2 * CHUNK + 17;
        let buf = Buf::new(0, len, ());
        let mut plan = Vec::new();
        push_copy_chunks(&mut plan, 2, &buf, 3, len);

        assert_eq!(plan.len(), 3, "a {len}-byte copy is not split into one entry per chunk");
        assert!(plan.iter().all(|&(part, _, _, n)| part == 2 && n > 0 && n <= CHUNK));
        assert_eq!(plan.iter().map(|entry| entry.3).sum::<usize>(), len);
        assert_eq!(plan.iter().map(|entry| entry.2).collect::<Vec<_>>(), vec![3, 3 + CHUNK, 3 + 2 * CHUNK]);

        // An empty copy (a part with no buffers) yields no callback and no bytes.
        let mut empty = Vec::new();
        push_copy_chunks(&mut empty, 0, &buf, 0, 0);
        assert!(empty.is_empty());
    }
}
