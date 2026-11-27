//! Low-level `STRATASLOT` v1 file format (see `docs/SLOT-SAVE.md`, section 3).
//!
//! This module only deals with the byte framing: header, token list and footer. The
//! per-part buffers are written and read by the caller, which knows the model layout.
//! All integers are little-endian.

use anyhow::{Context, Result, ensure};
use std::io::{Read, Write};

/// Header magic: first 8 bytes of every slot file.
pub const MAGIC: [u8; 8] = *b"STRATASL";
/// Footer magic: written after the payload.
pub const FOOTER: [u8; 8] = *b"STRATEND";
/// Format version understood by this module.
pub const VERSION: u32 = 1;

/// Flag bit 0: image/M-RoPE slot (`rope_offset != 0`).
pub const FLAG_IMAGE: u32 = 1 << 0;
/// Flag bit 1: saved with `--kv-unified` (informational).
pub const FLAG_UNIFIED: u32 = 1 << 1;
/// Flag bit 2: MTP/PLE/QSA buffers present (informational).
pub const FLAG_AUX: u32 = 1 << 2;

/// Bits 3..31 are reserved and must be zero in v1.
const FLAG_KNOWN_MASK: u32 = FLAG_IMAGE | FLAG_UNIFIED | FLAG_AUX;

/// Fixed portion of the header, before the token ids.
pub const HEADER_LEN: u64 = 56;

/// Parsed slot header. The token list is returned separately by [`read_header`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotHeader {
    pub fingerprint: u64,
    pub n_tokens: u64,
    pub rope_offset: i32,
    pub owner: u64,
    pub flags: u32,
    pub n_parts: u32,
}

fn read_exact<const N: usize, R: Read>(r: &mut R, what: &str) -> Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)
        .with_context(|| format!("reading {what}"))?;
    Ok(buf)
}

/// Write a little-endian `u32`.
pub fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes()).context("writing u32")?;
    Ok(())
}

/// Read a little-endian `u32`.
pub fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4, _>(r, "u32")?))
}

/// Write a little-endian `u64`.
pub fn write_u64<W: Write>(w: &mut W, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes()).context("writing u64")?;
    Ok(())
}

/// Read a little-endian `u64`.
pub fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact::<8, _>(r, "u64")?))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    Ok(i32::from_le_bytes(read_exact::<4, _>(r, "i32")?))
}

/// Write the header and the token ids.
///
/// `h.n_tokens` must equal `tokens.len()`; that is the token count stored in the
/// header and the number of ids written after it.
pub fn write_header<W: Write>(w: &mut W, h: &SlotHeader, tokens: &[u32]) -> Result<()> {
    let n_tokens =
        u64::try_from(tokens.len()).context("token list length does not fit in u64")?;
    ensure!(
        h.n_tokens == n_tokens,
        "header n_tokens {} does not match token list length {}",
        h.n_tokens,
        n_tokens
    );

    w.write_all(&MAGIC).context("writing header magic")?;
    write_u32(w, VERSION)?;
    write_u32(w, h.flags)?;
    write_u64(w, h.fingerprint)?;
    write_u64(w, n_tokens)?;
    w.write_all(&h.rope_offset.to_le_bytes())
        .context("writing rope_offset")?;
    write_u32(w, 0)?; // reserved
    write_u64(w, h.owner)?;
    write_u32(w, h.n_parts)?;
    write_u32(w, 0)?; // reserved

    for &t in tokens {
        write_u32(w, t)?;
    }
    Ok(())
}

/// Read and validate a header, then read exactly `n_tokens` token ids.
pub fn read_header<R: Read>(r: &mut R) -> Result<(SlotHeader, Vec<u32>)> {
    let magic = read_exact::<8, _>(r, "header magic")?;
    ensure!(magic == MAGIC, "bad slot magic {magic:02x?}, expected {MAGIC:02x?}");

    let version = read_u32(r)?;
    ensure!(
        version == VERSION,
        "unsupported slot version {version}, expected {VERSION}"
    );

    let flags = read_u32(r)?;
    let unknown = flags & !FLAG_KNOWN_MASK;
    ensure!(
        unknown == 0,
        "unknown slot flags {unknown:#x} (allowed mask {FLAG_KNOWN_MASK:#x})"
    );

    let fingerprint = read_u64(r)?;
    let n_tokens = read_u64(r)?;
    let rope_offset = read_i32(r)?;
    let _reserved = read_u32(r)?;
    let owner = read_u64(r)?;
    let n_parts = read_u32(r)?;
    let _reserved = read_u32(r)?;

    // Reject impossible counts before reserving anything. `n_tokens` must fit
    // usize and the byte length of the token list must fit `isize::MAX`.
    let n = usize::try_from(n_tokens)
        .with_context(|| format!("n_tokens {n_tokens} does not fit in usize"))?;
    ensure!(
        n <= (isize::MAX as usize) / std::mem::size_of::<u32>(),
        "n_tokens {n_tokens} is too large (would overflow the token buffer)"
    );

    let mut tokens = Vec::new();
    for i in 0..n {
        tokens.push(read_u32(r).with_context(|| format!("reading token {i} of {n}"))?);
    }

    let header = SlotHeader {
        fingerprint,
        n_tokens,
        rope_offset,
        owner,
        flags,
        n_parts,
    };
    Ok((header, tokens))
}

/// Write the footer: `FOOTER` followed by the payload length.
///
/// `payload_len` is the number of bytes written before the footer.
pub fn write_footer<W: Write>(w: &mut W, payload_len: u64) -> Result<()> {
    w.write_all(&FOOTER).context("writing footer magic")?;
    write_u64(w, payload_len)?;
    Ok(())
}

/// Read and validate the footer.
///
/// `payload_len` must be the byte offset at which the footer starts (bytes
/// consumed before it). The stored length must match it, which catches both a
/// truncated file and trailing garbage.
pub fn read_footer<R: Read>(r: &mut R, payload_len: u64) -> Result<()> {
    let magic = read_exact::<8, _>(r, "footer magic")?;
    ensure!(
        magic == FOOTER,
        "bad slot footer magic {magic:02x?}, expected {FOOTER:02x?}"
    );

    let stored = read_u64(r).context("reading footer payload length")?;
    ensure!(
        stored == payload_len,
        "slot footer payload length {stored} does not match consumed bytes {payload_len}"
    );
    Ok(())
}

/// Validate a slot filename: a single path component, at most 255 bytes.
///
/// Rejects empty names, `.`/`..`, absolute paths, path separators (`/`, `\`),
/// NUL and control characters.
pub fn validate_filename(name: &str) -> Result<()> {
    ensure!(!name.is_empty(), "invalid filename: empty");
    ensure!(
        name.len() <= 255,
        "invalid filename: {} bytes exceeds the 255-byte limit",
        name.len()
    );
    ensure!(
        name != "." && name != "..",
        "invalid filename: {name:?} is a directory reference"
    );

    for c in name.chars() {
        ensure!(
            !c.is_control(),
            "invalid filename: {name:?} contains a control character"
        );
        ensure!(
            c != '/' && c != '\\' && c != '\0',
            "invalid filename: {name:?} contains a path separator or NUL"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header(flags: u32, n_tokens: u64) -> SlotHeader {
        SlotHeader {
            fingerprint: 0x0123_4567_89ab_cdef,
            n_tokens,
            rope_offset: -42,
            owner: 7,
            flags,
            n_parts: 3,
        }
    }

    /// Header bytes with an arbitrary (possibly huge) `n_tokens`, for rejection tests.
    fn raw_header(version: u32, flags: u32, n_tokens: u64) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&version.to_le_bytes());
        v.extend_from_slice(&flags.to_le_bytes());
        v.extend_from_slice(&0xaabb_ccdd_eeff_0011u64.to_le_bytes());
        v.extend_from_slice(&n_tokens.to_le_bytes());
        v.extend_from_slice(&7i32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&9u64.to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v
    }

    #[test]
    fn header_round_trip() {
        let tokens: Vec<u32> = (0..17).map(|i| 1000 + i as u32).collect();
        let h = sample_header(FLAG_IMAGE | FLAG_UNIFIED | FLAG_AUX, tokens.len() as u64);

        let mut buf = Vec::new();
        write_header(&mut buf, &h, &tokens).unwrap();
        assert_eq!(buf.len() as u64, HEADER_LEN + 4 * tokens.len() as u64);

        let (got, got_tokens) = read_header(&mut &buf[..]).unwrap();
        assert_eq!(got, h);
        assert_eq!(got_tokens, tokens);
    }

    #[test]
    fn empty_token_list_round_trip() {
        let h = sample_header(0, 0);
        let mut buf = Vec::new();
        write_header(&mut buf, &h, &[]).unwrap();
        let (got, tokens) = read_header(&mut &buf[..]).unwrap();
        assert_eq!(got, h);
        assert!(tokens.is_empty());
    }

    #[test]
    fn header_token_count_mismatch_is_rejected_on_write() {
        let h = sample_header(0, 3);
        let mut buf = Vec::new();
        let err = write_header(&mut buf, &h, &[1, 2]).unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
    }

    #[test]
    fn bad_magic_is_rejected() {
        let tokens = [1u32, 2, 3];
        let h = sample_header(0, tokens.len() as u64);
        let mut buf = Vec::new();
        write_header(&mut buf, &h, &tokens).unwrap();
        buf[0] ^= 0xff;

        let err = read_header(&mut &buf[..]).unwrap_err();
        assert!(err.to_string().contains("bad slot magic"), "{err}");
    }

    #[test]
    fn bad_version_is_rejected() {
        let buf = raw_header(2, 0, 0);
        let err = read_header(&mut &buf[..]).unwrap_err();
        assert!(err.to_string().contains("unsupported slot version"), "{err}");
    }

    #[test]
    fn unknown_flag_is_rejected() {
        let buf = raw_header(VERSION, 1 << 3, 0);
        let err = read_header(&mut &buf[..]).unwrap_err();
        assert!(err.to_string().contains("unknown slot flags"), "{err}");

        // All reserved bits together are also rejected.
        let buf = raw_header(VERSION, u32::MAX & !FLAG_KNOWN_MASK, 0);
        assert!(read_header(&mut &buf[..]).is_err());
    }

    #[test]
    fn truncated_token_list_is_rejected() {
        let tokens = [10u32, 20, 30];
        let h = sample_header(0, tokens.len() as u64);
        let mut buf = Vec::new();
        write_header(&mut buf, &h, &tokens).unwrap();
        buf.truncate(buf.len() - 4);

        let err = read_header(&mut &buf[..]).unwrap_err();
        assert!(format!("{err:#}").contains("reading token"), "{err:#}");
    }

    #[test]
    fn huge_n_tokens_is_rejected_before_allocating() {
        let buf = raw_header(VERSION, 0, u64::MAX);
        let err = read_header(&mut &buf[..]).unwrap_err();
        assert!(err.to_string().contains("too large"), "{err}");
    }

    #[test]
    fn footer_round_trip() {
        let payload_len = 123_456_789u64;
        let mut buf = Vec::new();
        write_footer(&mut buf, payload_len).unwrap();
        assert_eq!(buf.len(), 16);
        read_footer(&mut &buf[..], payload_len).unwrap();
    }

    #[test]
    fn footer_length_mismatch_is_rejected() {
        let mut buf = Vec::new();
        write_footer(&mut buf, 100).unwrap();

        let err = read_footer(&mut &buf[..], 99).unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
    }

    #[test]
    fn bad_footer_magic_is_rejected() {
        let mut buf = Vec::new();
        write_footer(&mut buf, 100).unwrap();
        buf[0] ^= 0xff;

        let err = read_footer(&mut &buf[..], 100).unwrap_err();
        assert!(err.to_string().contains("bad slot footer magic"), "{err}");
    }

    #[test]
    fn truncated_footer_is_rejected() {
        let mut buf = Vec::new();
        write_footer(&mut buf, 100).unwrap();
        buf.truncate(10);
        assert!(read_footer(&mut &buf[..], 100).is_err());
    }

    #[test]
    fn filename_matrix() {
        for ok in [
            "slot.bin",
            "a",
            "slot_1-2026.01.slot",
            &"a".repeat(255),
        ] {
            validate_filename(ok).unwrap_or_else(|e| panic!("{ok:?} should be valid: {e}"));
        }

        for bad in [
            "",
            ".",
            "..",
            "../x",
            "/abs",
            "a/b",
            "a\\b",
            "a\0b",
            "a\nb",
            "a\rb",
            "a\u{1}b",
            &"a".repeat(256),
        ] {
            assert!(
                validate_filename(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }
}
