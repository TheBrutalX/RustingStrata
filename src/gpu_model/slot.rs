//! `STRATASLOT` v1 save/restore for one slot (see `docs/SLOT-SAVE.md`).
//!
//! The state is streamed in bounded chunks: a 1M-token q8 slot is about 10.9 GB and must never be held in host
//! RAM. The file framing lives in `crate::slot_file`; this module knows the `GpuModel` buffers and the fingerprint.

use super::{Buf, GpuModel};
use crate::slot_file::{self, SlotHeader};
use anyhow::{bail, ensure, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Bytes copied between the GPU and the host in one `get`/`put`.
const CHUNK: usize = 64 << 20;

/// A writer that counts the bytes it passes through (the footer stores the payload length).
struct Count<W> { w: W, n: u64 }
impl<W> Count<W> {
    fn new(w: W) -> Self { Self { w, n: 0 } }
    fn count(&self) -> u64 { self.n }
    fn into_inner(self) -> W { self.w }
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

impl GpuModel<'_> {
    /// Seconds of a slot's state file. The file contains plaintext KV: prompt, system prompt, tools and reply.
    pub fn save_slot_file(&mut self, slot: usize, path: &Path) -> Result<u64> {
        self.save_slot_file_as(slot, path, self.owner)
    }

    /// `save_slot_file` with an explicit conversation owner. The archive uses `slot_owner` here, not the owner of
    /// the request that triggered the checkpoint: an eviction caused by another key must not re-label the state.
    pub fn save_slot_file_as(&mut self, slot: usize, path: &Path, owner: u64) -> Result<u64> {
        let prev = self.owner;
        self.owner = owner;
        let r = self.save_slot_file_owned(slot, path);
        self.owner = prev;
        r
    }

    fn save_slot_file_owned(&mut self, slot: usize, path: &Path) -> Result<u64> {
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
        // Allocate the recurrent copies and make the current recurrent state the saved one.
        self.save_state()?;

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
        let r = (|| -> Result<u64> {
            let mut out = Count::new(BufWriter::with_capacity(8 << 20, f));
            slot_file::write_header(&mut out, &header, &tokens)?;
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
            let n = out.count();
            let f = out.into_inner().into_inner().map_err(|e| anyhow::anyhow!("flushing {}: {}", tmp.display(), e.error()))?;
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
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut inp = Count::new(BufReader::with_capacity(8 << 20, f));
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
        let mut caps = self.kv_cap.clone();
        let mut order: Vec<usize> = (0..caps.len())
            .filter(|&i| i != slot && !self.active.get(i).copied().unwrap_or(false) && caps[i] > 1)
            .collect();
        order.sort_by_key(|&i| std::cmp::Reverse(caps[i]));
        let fits = |caps: &[usize]| caps.iter().sum::<usize>() - caps[slot] + new <= self.kv_budget;
        let mut victims = Vec::new();
        for i in order {
            if fits(&caps) { break; }
            if !self.slot_empty(i) { victims.push(i); }
            caps[i] = 1;
        }
        victims
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
