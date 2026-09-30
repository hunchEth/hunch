//! Fixed-point MLP with online SGD. Q16.16 arithmetic throughout.
//! Shared by the Stylus contract and the native learning-curve sim.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

pub const IN: usize = 16;
pub const Q: i64 = 16;
pub const ONE: i64 = 1 << Q;

#[inline]
pub fn qmul(a: i64, b: i64) -> i64 {
    ((a as i128 * b as i128) >> Q) as i64
}

/// xorshift64* PRNG for deterministic weight init.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// uniform in [-limit, limit] (Q16)
    pub fn sym(&mut self, limit: i64) -> i64 {
        let r = (self.next() >> 33) as i64; // 31 bits
        (r % (2 * limit + 1)) - limit
    }
}

/// Layout: w1[IN*H] | b1[H] | w2[H] | b2[1]  (all Q16)
pub fn param_count(h: usize) -> usize {
    IN * h + h + h + 1
}

pub fn init_weights(h: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng(seed | 1);
    let limit = ONE / 20; // +-0.05
    (0..param_count(h)).map(|_| rng.sym(limit)).collect()
}

/// Forward pass. Returns (prediction, hidden activations).
pub fn forward(w: &[i64], h: usize, x: &[i64; IN]) -> (i64, Vec<i64>) {
    let (w1, rest) = w.split_at(IN * h);
    let (b1, rest) = rest.split_at(h);
    let (w2, b2) = rest.split_at(h);
    let mut z = vec![0i64; h];
    for j in 0..h {
        let mut acc = b1[j];
        for i in 0..IN {
            acc += qmul(w1[j * IN + i], x[i]);
        }
        z[j] = acc.max(0); // ReLU
    }
    let mut y = b2[0];
    for j in 0..h {
        y += qmul(w2[j], z[j]);
    }
    (y, z)
}

/// One SGD step on one sample. Updates weights in place.
/// Returns squared error (Q16). lr_shift: update = grad >> lr_shift.
pub fn lesson(w: &mut [i64], h: usize, x: &[i64; IN], target: i64, lr_shift: u32) -> i64 {
    let (y, z) = forward(w, h, x);
    let e = (y - target).clamp(-(ONE * 64), ONE * 64);
    let (w1, rest) = w.split_at_mut(IN * h);
    let (b1, rest) = rest.split_at_mut(h);
    let (w2, b2) = rest.split_at_mut(h);
    // output layer
    for j in 0..h {
        let g = qmul(e, z[j]);
        w2[j] -= g >> lr_shift;
    }
    b2[0] -= e >> lr_shift;
    // hidden layer
    for j in 0..h {
        if z[j] > 0 {
            let d = qmul(e, w2[j]);
            for i in 0..IN {
                w1[j * IN + i] -= qmul(d, x[i]) >> lr_shift;
            }
            b1[j] -= d >> lr_shift;
        }
    }
    qmul(e, e)
}

/// Q16 feature from a beat gap in seconds: 65536 * log2(gap/60), clamped
/// to [-4, 8] like the off-chain twin.
pub fn feat_gap(gap: u64) -> i64 {
    let x = gap.max(1);
    let msb = 63 - x.leading_zeros() as i64;
    let mut y: u128 = ((x as u128) << 32) >> msb; // [1,2) in Q32
    let mut frac: i64 = 0;
    for _ in 0..16 {
        y = (y * y) >> 32;
        frac <<= 1;
        if y >= 2u128 << 32 {
            y >>= 1;
            frac |= 1;
        }
    }
    const LOG2_60_Q16: i64 = 387113;
    ((msb << 16) | frac).wrapping_sub(LOG2_60_Q16).clamp(-4 * 65536, 8 * 65536)
}

/// Feature window from the 16 most recent gaps (oldest first).
pub fn features(gaps: &[u32; 16]) -> [i64; IN] {
    let mut x = [0i64; IN];
    for i in 0..IN {
        x[i] = feat_gap(gaps[i] as u64);
    }
    x
}
