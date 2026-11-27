//! RoPE frequency plans: plain, linear and YaRN.
//!
//! YaRN (Peng et al., <https://github.com/jquesnelle/yarn>) is frequency-dependent: low-frequency dimensions are
//! interpolated by `1/factor`, high-frequency ones keep extrapolating, and a ramp between `beta_fast` and `beta_slow`
//! mixes the two. The angle stays linear in the position, so the whole plan is one effective frequency per pair:
//!
//! ```text
//! f_i      = base^(-2i/n_rot)
//! mix_i    = (1 - clamp((i - corr0) / max(0.001, corr1 - corr0), 0, 1)) * ext
//! freq_i   = f_i * (freq_scale * (1 - mix_i) + mix_i)
//! theta_i  = pos * freq_i
//! ```
//!
//! This mirrors `rope_yarn`/`rope_yarn_ramp` in llama.cpp's ggml. The attention magnitude factor `mscale` is 1: ggml
//! applies `1 + 0.1*ln(factor)` inside the rotation and then divides it out in `llama-context` (`yarn_attn_factor`),
//! so the net scale is 1 for a standard YaRN checkpoint, as in `deepseek::ops::Rope`.

use anyhow::{ensure, Context};

/// One RoPE plan for a model: the base, the YaRN/linear parameters and the effective frequency of every pair.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeParams {
    pub base: f32,
    pub n_rot: usize,
    /// `1/factor` (1 = no interpolation).
    pub freq_scale: f32,
    /// 0 = no YaRN ramp (plain or linear); 1 = full YaRN.
    pub ext_factor: f32,
    /// Correction dims `[start, end]` in pair units, from `beta_fast` and `beta_slow`.
    pub corr: [f32; 2],
    /// Effective frequency of pair `i`, `n_rot/2` entries.
    freqs: Vec<f64>,
}

impl RopeParams {
    /// No scaling: `theta = pos * base^(-2i/n_rot)`.
    pub fn plain(base: f32, n_rot: usize) -> Self {
        let n = n_rot.max(2);
        let freqs = (0..n / 2).map(|i| (base as f64).powf(-2.0 * i as f64 / n as f64)).collect();
        Self { base, n_rot: n, freq_scale: 1.0, ext_factor: 0.0, corr: [0.0, 0.0], freqs }
    }

    /// Linear position interpolation: every frequency is multiplied by `1/factor`.
    pub fn linear(base: f32, n_rot: usize, factor: f32) -> anyhow::Result<Self> {
        ensure!(factor.is_finite() && factor > 0.0, "rope: linear factor {factor} must be > 0");
        let mut r = Self::plain(base, n_rot);
        r.freq_scale = 1.0 / factor;
        for f in &mut r.freqs {
            *f *= r.freq_scale as f64;
        }
        Ok(r)
    }

    /// YaRN: `n_ctx_orig` is the trained context, `factor` the context extension (2 for 256k -> 512k).
    /// `beta_fast`/`beta_slow` are llama.cpp's `--yarn-beta-fast`/`--yarn-beta-slow` (32 and 1 by default).
    pub fn yarn(base: f32, n_rot: usize, n_ctx_orig: usize, factor: f32, beta_fast: f32, beta_slow: f32) -> anyhow::Result<Self> {
        ensure!(factor.is_finite() && factor > 0.0, "rope: yarn factor {factor} must be > 0");
        ensure!(n_ctx_orig > 0, "rope: yarn needs the trained context (--yarn-orig-ctx), got 0");
        ensure!(beta_fast.is_finite() && beta_fast > 0.0, "rope: yarn beta_fast {beta_fast} must be > 0");
        ensure!(beta_slow.is_finite() && beta_slow > 0.0, "rope: yarn beta_slow {beta_slow} must be > 0");
        let n = n_rot.max(2);
        let freq_scale = 1.0 / factor;
        // ggml_rope_yarn_corr_dim: n_dims * log(n_ctx_orig / (n_rot * 2*pi)) / (2 * log(base))
        let cd = |n_rot_beta: f32| -> f32 {
            (n as f32) * (n_ctx_orig as f32 / (n_rot_beta * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let corr = [cd(beta_fast).floor().max(0.0), cd(beta_slow).ceil().min((n - 1) as f32)];
        let mut freqs = Vec::with_capacity(n / 2);
        for i in 0..n / 2 {
            let f = (base as f64).powf(-2.0 * i as f64 / n as f64);
            let y = (i as f32 - corr[0]) / (corr[1] - corr[0]).max(0.001);
            let mix = (1.0 - y.clamp(0.0, 1.0)) * 1.0; // ext_factor = 1 for YaRN
            let eff = f * (freq_scale as f64 * (1.0 - mix as f64) + mix as f64);
            freqs.push(eff);
        }
        Ok(Self { base, n_rot: n, freq_scale, ext_factor: 1.0, corr, freqs })
    }

    /// The effective frequencies of the rotated pairs (`n_rot/2` entries).
    pub fn freqs(&self) -> &[f64] { &self.freqs }

    /// The `(sin, cos)` of pair `i` at `pos`.
    pub fn cs(&self, pos: usize, i: usize) -> (f64, f64) {
        let theta = pos as f64 * self.freqs.get(i).copied().unwrap_or(0.0);
        (theta.sin(), theta.cos())
    }

    /// Whether the plan differs from plain RoPE.
    pub fn is_scaled(&self) -> bool { self.freq_scale != 1.0 || self.ext_factor != 0.0 }

    /// NEOX-style rotation of the first `n_rot` values: pairs `(i, i + n_rot/2)`.
    pub fn apply_neox(&self, x: &mut [f32], pos: usize) {
        let half = self.n_rot / 2;
        for i in 0..half {
            let (s, c) = self.cs(pos, i);
            let (s, c) = (s as f32, c as f32);
            let (a, b) = (x[i], x[i + half]);
            x[i] = a * c - b * s;
            x[i + half] = a * s + b * c;
        }
    }
}

/// Read a model's RoPE scaling from GGUF (`{arch}.rope.scaling.*`), the keys llama.cpp writes for a YaRN checkpoint.
/// No key = plain RoPE.
pub fn from_gguf(g: &crate::gguf::Gguf, arch: &str, base: f32, n_rot: usize) -> anyhow::Result<RopeParams> {
    let key = |s: &str| format!("{arch}.rope.scaling.{s}");
    let get = |s: &str| g.meta.get(&key(s));
    let kind = get("type").and_then(|v| v.as_str()).unwrap_or("none");
    let factor = get("factor").and_then(|v| v.as_f32()).unwrap_or(1.0);
    match kind {
        "none" | "" => Ok(RopeParams::plain(base, n_rot)),
        "linear" => RopeParams::linear(base, n_rot, factor).with_context(|| key("factor")),
        "yarn" => {
            let orig = get("original_context_length").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let fast = get("yarn_beta_fast").and_then(|v| v.as_f32()).unwrap_or(32.0);
            let slow = get("yarn_beta_slow").and_then(|v| v.as_f32()).unwrap_or(1.0);
            RopeParams::yarn(base, n_rot, orig, factor, fast, slow).with_context(|| key("type"))
        }
        other => anyhow::bail!("unsupported RoPE scaling {other:?} in {arch}.rope.scaling.type (expected none, linear, yarn)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_matches_pow() {
        let r = RopeParams::plain(1e7, 64);
        for i in 0..32 {
            let want = (1e7f64).powf(-2.0 * i as f64 / 64.0);
            assert!((r.freqs()[i] - want).abs() < 1e-18);
        }
        // pos 0 is the identity.
        let r4 = RopeParams::plain(1e7, 4);
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let before = x;
        r4.apply_neox(&mut x, 0);
        assert_eq!(x, before);
    }

    #[test]
    fn yarn_corr_dims_match_ggml() {
        // ggml_rope_yarn_corr_dims(64, 262144, 1e7, beta_fast=32, beta_slow=1) -> [14, 22]
        let r = RopeParams::yarn(1e7, 64, 262144, 2.0, 32.0, 1.0).unwrap();
        assert_eq!(r.corr, [14.0, 22.0]);
        assert!((r.freq_scale - 0.5).abs() < 1e-7);
    }

    #[test]
    fn yarn_extrapolates_low_dims_and_interpolates_high_dims() {
        let r = RopeParams::yarn(1e7, 64, 262144, 2.0, 32.0, 1.0).unwrap();
        let plain = RopeParams::plain(1e7, 64);
        // i below corr0: mix = 1, theta is extrapolated (same as plain).
        for i in 0..14 {
            assert!((r.freqs()[i] - plain.freqs()[i]).abs() < 1e-18, "i={i}");
        }
        // i above corr1: mix = 0, theta is interpolated by 1/factor.
        for i in 22..32 {
            assert!((r.freqs()[i] - plain.freqs()[i] * 0.5).abs() < 1e-18, "i={i}");
        }
        // The ramp is monotone between the two.
        for i in 14..22 {
            assert!(r.freqs()[i] <= plain.freqs()[i] + 1e-18 && r.freqs()[i] >= plain.freqs()[i] * 0.5 - 1e-18);
        }
    }

    #[test]
    fn yarn_rejects_bad_parameters() {
        assert!(RopeParams::yarn(1e7, 64, 0, 2.0, 32.0, 1.0).is_err());
        assert!(RopeParams::yarn(1e7, 64, 262144, 0.0, 32.0, 1.0).is_err());
        assert!(RopeParams::linear(1e7, 64, -1.0).is_err());
    }
}
