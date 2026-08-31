//! An MLP that lives in chain storage: forward pass, backprop and SGD all
//! run inside the contract. Weights are packed 4 x i64 per storage slot.
//! `write_cap` bounds how many slots persist per lesson (round-robin), so
//! the write-gas cost is tunable independently of brain size.
#![cfg_attr(not(any(test, feature = "export-abi")), no_main)]
extern crate alloc;

pub mod net;

use alloc::vec::Vec;
use net::{forward, init_weights, lesson as sgd_lesson, param_count, IN};
use stylus_sdk::{alloy_primitives::{aliases::U64, U256}, prelude::*};

sol_storage! {
    #[entrypoint]
    pub struct Brain {
        uint256[] packed;
        uint64 hidden;
        uint64 lr_shift;
        uint64 cursor;
        uint64 lessons;
    }
}

fn pack(vals: &[i64]) -> U256 {
    let mut limbs = [0u64; 4];
    for (k, v) in vals.iter().enumerate() {
        limbs[k] = *v as u64;
    }
    U256::from_limbs(limbs)
}

fn unpack(word: U256, out: &mut Vec<i64>, remaining: usize) {
    for k in 0..4.min(remaining) {
        out.push(word.as_limbs()[k] as i64);
    }
}

impl Brain {
    /// Read every weight slot into memory.
    fn read_weights(&self) -> Vec<i64> {
        let h = self.hidden.get().to::<u64>() as usize;
        let n = param_count(h);
        let slots = n.div_ceil(4);
        let mut w = Vec::with_capacity(n);
        for s in 0..slots {
            unpack(self.packed.get(s).unwrap(), &mut w, n - s * 4);
        }
        w
    }

}

#[public]
impl Brain {
    /// One-time setup: hidden size, PRNG seed for weights, learning-rate shift.
    pub fn init(&mut self, hidden: u64, seed: u64, lr_shift: u64) {
        if self.hidden.get().to::<u64>() != 0 {
            return;
        }
        let h = hidden as usize;
        let w = init_weights(h, seed);
        let mut i = 0;
        while i < w.len() {
            let end = (i + 4).min(w.len());
            self.packed.push(pack(&w[i..end]));
            i = end;
        }
        self.hidden.set(U64::from(hidden));
        self.lr_shift.set(U64::from(lr_shift));
    }

    /// Pure inference, free via eth_call.
    pub fn predict(&self, x: Vec<i64>) -> i64 {
        let h = self.hidden.get().to::<u64>() as usize;
        let mut xin = [0i64; IN];
        xin.copy_from_slice(&x[..IN]);
        forward(&self.read_weights(), h, &xin).0
    }

    /// One on-chain SGD step. Persists at most `write_cap` slots (round-robin),
    /// so storage gas is bounded; unpersisted deltas are dropped by design.
    /// Returns squared error (Q16).
    pub fn lesson(&mut self, x: Vec<i64>, target: i64, write_cap: u64) -> u64 {
        let h = self.hidden.get().to::<u64>() as usize;
        let lr = self.lr_shift.get().to::<u64>() as u32;
        let mut xin = [0i64; IN];
        xin.copy_from_slice(&x[..IN]);

        let mut w = self.read_weights();
        let err2 = sgd_lesson(&mut w, h, &xin, target, lr);

        let slots = w.len().div_ceil(4);
        let cap = (write_cap as usize).min(slots);
        let start = self.cursor.get().to::<u64>() as usize % slots;
        for k in 0..cap {
            let s = (start + k) % slots;
            let end = (s * 4 + 4).min(w.len());
            self.packed.setter(s).unwrap().set(pack(&w[s * 4..end]));
        }
        self.cursor.set(U64::from(((start + cap) % slots) as u64));
        self.lessons.set(U64::from(self.lessons.get().to::<u64>() + 1));
        err2 as u64
    }

    pub fn lessons_done(&self) -> u64 {
        self.lessons.get().to::<u64>()
    }
}

#[cfg(test)]
mod sim {
    use super::net::*;
    use alloc::vec::Vec;

    /// GARCH(1,1) return generator (f64), quantized to Q16 with x100 scale.
    struct Garch {
        rng: Rng,
        sigma2: f64,
    }
    impl Garch {
        fn new(seed: u64) -> Self {
            Garch { rng: Rng(seed), sigma2: 1e-4 }
        }
        fn next_ret(&mut self) -> f64 {
            // approx normal: sum of 12 uniforms - 6
            let mut n = -6.0;
            for _ in 0..12 {
                n += (self.rng.next() >> 11) as f64 / (1u64 << 53) as f64;
            }
            let r = self.sigma2.sqrt() * n;
            self.sigma2 = 3e-6 + 0.12 * r * r + 0.85 * self.sigma2;
            r
        }
    }

    fn q16(v: f64) -> i64 {
        (v * ONE as f64) as i64
    }

    /// Online learning on GARCH data; returns (mse_net, mse_baseline) over the
    /// final quarter. `cap_frac_4` = persisted slots per lesson in quarters
    /// (4 = full persistence, 1 = 25% of slots round-robin).
    fn run(h: usize, lr_shift: u32, cap_frac_4: usize, steps: usize) -> (f64, f64) {
        const W: usize = 16;
        let mut g = Garch::new(0xC0FFEE);
        let rets: Vec<f64> = (0..steps + 2 * W).map(|_| g.next_ret().abs() * 100.0).collect();

        let mut w = init_weights(h, 7);
        let slots = w.len().div_ceil(4);
        let cap = (slots * cap_frac_4 / 4).max(1);
        let mut cursor = 0usize;

        let mut base = q16(1.0); // running-mean baseline (EWMA)
        let (mut se_net, mut se_base, mut cnt) = (0f64, 0f64, 0u64);

        for t in W..steps {
            let mut x = [0i64; IN];
            for i in 0..IN {
                x[i] = q16(rets[t - W + i]);
            }
            let target = q16(rets[t..t + W].iter().sum::<f64>() / W as f64);

            let (y, _) = forward(&w, h, &x);
            if t > steps * 3 / 4 {
                let ef = (y - target) as f64 / ONE as f64;
                let eb = (base - target) as f64 / ONE as f64;
                se_net += ef * ef;
                se_base += eb * eb;
                cnt += 1;
            }

            // partial persistence: train a copy, keep only `cap` slots
            let mut trained = w.clone();
            lesson(&mut trained, h, &x, target, lr_shift);
            for k in 0..cap {
                let s = (cursor + k) % slots;
                let end = (s * 4 + 4).min(w.len());
                w[s * 4..end].copy_from_slice(&trained[s * 4..end]);
            }
            cursor = (cursor + cap) % slots;

            base = base + (target - base) / 64; // EWMA baseline
        }
        (se_net / cnt as f64, se_base / cnt as f64)
    }

    #[test]
    fn learns_vol_full_persistence() {
        let (net, base) = run(32, 11, 4, 40_000);
        std::println!("full persistence: net mse {net:.5} vs baseline {base:.5}");
        assert!(net < base * 0.9, "net {net} not < 0.9x baseline {base}");
    }

    #[test]
    fn learns_vol_quarter_persistence() {
        let (net, base) = run(32, 11, 1, 40_000);
        std::println!("quarter persistence: net mse {net:.5} vs baseline {base:.5}");
        assert!(net < base, "net {net} not < baseline {base}");
    }

    /// Scheme B: buffer samples, apply 4 SGD steps + one full weight write
    /// every 4th crank. Same write gas as 25% round-robin, full learning.
    #[test]
    fn learns_vol_batched_writes() {
        const W: usize = 16;
        let steps = 40_000;
        let (h, lr_shift) = (32usize, 11u32);
        let mut g = Garch::new(0xC0FFEE);
        let rets: Vec<f64> = (0..steps + 2 * W).map(|_| g.next_ret().abs() * 100.0).collect();
        let mut w = init_weights(h, 7);
        let mut batch: Vec<([i64; IN], i64)> = Vec::new();
        let mut base = q16(1.0);
        let (mut se_net, mut se_base, mut cnt) = (0f64, 0f64, 0u64);
        for t in W..steps {
