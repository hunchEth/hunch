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
