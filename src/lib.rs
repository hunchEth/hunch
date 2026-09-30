//! HUNCH: an MLP that lives in chain storage and feeds itself.
//! `lesson()` takes no arguments: the brain reads its Chainlink feed, measures
//! the gap since the last beat, grades its previous hunch, runs backprop and
//! SGD in place, and forms the next hunch. Nobody can teach it a lie, because
//! nobody gets to hand it inputs. Weights are packed 4 x i64 per slot.
#![cfg_attr(not(any(test, feature = "export-abi")), no_main)]
extern crate alloc;

pub mod net;

use alloc::vec::Vec;
use net::{feat_gap, features, forward, init_weights, lesson as sgd_lesson, param_count, IN};
use stylus_sdk::{
    alloy_primitives::{
        aliases::{I64, U64, U80},
        Address, U256,
    },
    prelude::*,
};

sol_interface! {
    interface IAggregator {
        function latestRoundData() external view returns (uint80, int256, uint256, uint256, uint80);
        function getRoundData(uint80 round) external view returns (uint80, int256, uint256, uint256, uint80);
    }
}

sol_storage! {
    #[entrypoint]
    pub struct Brain {
        uint256[] packed;
        uint64 hidden;
        uint64 lr_shift;
        address feed;
        uint256 last_round;
        uint64 last_t;
        uint64 beats_seen;
        uint256 gaps_lo;   // gaps[0..8], oldest lanes first, u32 each
        uint256 gaps_hi;   // gaps[8..16]
        int64 pending;     // Q16 hunch for the next gap
        uint64 has_pending;
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

fn gaps_pack(g: &[u32; 16]) -> (U256, U256) {
    let mut lo = [0u64; 4];
    let mut hi = [0u64; 4];
    for i in 0..8 {
        lo[i / 2] |= (g[i] as u64) << (32 * (i % 2));
        hi[i / 2] |= (g[8 + i] as u64) << (32 * (i % 2));
    }
    (U256::from_limbs(lo), U256::from_limbs(hi))
}

fn gaps_unpack(lo: U256, hi: U256) -> [u32; 16] {
    let mut g = [0u32; 16];
    for i in 0..8 {
        g[i] = (lo.as_limbs()[i / 2] >> (32 * (i % 2))) as u32;
        g[8 + i] = (hi.as_limbs()[i / 2] >> (32 * (i % 2))) as u32;
    }
    g
}

impl Brain {
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

    fn write_weights(&mut self, w: &[i64]) {
        let slots = w.len().div_ceil(4);
        for s in 0..slots {
            let end = (s * 4 + 4).min(w.len());
            self.packed.setter(s).unwrap().set(pack(&w[s * 4..end]));
        }
    }

    fn push_gap(&mut self, gap: u32) {
        let mut g = gaps_unpack(self.gaps_lo.get(), self.gaps_hi.get());
        g.rotate_left(1);
        g[15] = gap;
        let (lo, hi) = gaps_pack(&g);
        self.gaps_lo.set(lo);
        self.gaps_hi.set(hi);
        self.beats_seen.set(U64::from(self.beats_seen.get().to::<u64>() + 1));
    }
}

#[public]
impl Brain {
    /// One-time setup: feed to listen to, hidden size, weight seed, learning
    /// rate shift. Walks 16 rounds back so the brain wakes up with a full
    /// window and can learn from the very next beat.
    pub fn init(&mut self, feed: Address, hidden: u64, seed: u64, lr_shift: u64) -> Result<(), Vec<u8>> {
        if self.hidden.get().to::<u64>() != 0 {
            return Err(b"already alive".to_vec());
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
        self.feed.set(feed);

        let agg = IAggregator::new(feed);
        let (round, _, _, updated, _) = agg.latest_round_data(self.vm(), Call::new())?;
        self.last_round.set(U256::from(round));
        self.last_t.set(U64::from(updated.to::<u64>()));

        // walk back: collect up to 17 timestamps, oldest last in the walk
        let mut ts = Vec::with_capacity(17);
        ts.push(updated.to::<u64>());
        for k in 1u64..17 {
            let r = round - U80::from(k);
            let agg = IAggregator::new(feed);
            match agg.get_round_data(self.vm(), Call::new(), r) {
                Ok((_, _, _, u, _)) => {
                    let t = u.to::<u64>();
                    if t == 0 { break; }
                    ts.push(t);
                }
                Err(_) => break,
            }
        }
        ts.reverse(); // oldest first
        let mut g = [0u32; 16];
        let have = ts.len().saturating_sub(1);
        for i in 0..have {
            g[16 - have + i] = (ts[i + 1] - ts[i]).max(1) as u32;
        }
        let (lo, hi) = gaps_pack(&g);
        self.gaps_lo.set(lo);
        self.gaps_hi.set(hi);
        self.beats_seen.set(U64::from(have as u64));

        if have >= 16 {
            let x = features(&g);
            let w = self.read_weights();
            let p = forward(&w, h, &x).0;
            self.pending.set(I64::try_from(p).unwrap_or_default());
            self.has_pending.set(U64::from(1u64));
        }
        Ok(())
    }

    /// The bell. Anyone may ring it once a new beat exists. The brain reads
    /// the feed itself, grades its old hunch, learns, and forms a new one.
    /// Returns the squared error of the graded hunch (0 while warming up).
    pub fn lesson(&mut self) -> Result<u64, Vec<u8>> {
        let feed = self.feed.get();
        if feed == Address::ZERO {
            return Err(b"not born yet".to_vec());
        }
        let agg = IAggregator::new(feed);
        let (round, _, _, updated, _) = agg.latest_round_data(self.vm(), Call::new())?;
        if U256::from(round) <= self.last_round.get() {
            return Err(b"no new beat".to_vec());
        }
        let h = self.hidden.get().to::<u64>() as usize;
        let lr = self.lr_shift.get().to::<u64>() as u32;
        let gap = updated.to::<u64>().saturating_sub(self.last_t.get().to::<u64>()).max(1);

        let mut err2 = 0u64;
        let mut w: Option<Vec<i64>> = None;
        if self.beats_seen.get().to::<u64>() >= 16 && self.has_pending.get().to::<u64>() == 1 {
            let g = gaps_unpack(self.gaps_lo.get(), self.gaps_hi.get());
            let x = features(&g);
            let target = feat_gap(gap);
            let mut ww = self.read_weights();
            err2 = sgd_lesson(&mut ww, h, &x, target, lr) as u64;
            self.write_weights(&ww);
            self.lessons.set(U64::from(self.lessons.get().to::<u64>() + 1));
            w = Some(ww);
        }

        self.push_gap(gap as u32);
        self.last_round.set(U256::from(round));
        self.last_t.set(U64::from(updated.to::<u64>()));

        if self.beats_seen.get().to::<u64>() >= 16 {
            let g = gaps_unpack(self.gaps_lo.get(), self.gaps_hi.get());
            let x = features(&g);
            let ww = w.unwrap_or_else(|| self.read_weights());
            let p = forward(&ww, h, &x).0;
            self.pending.set(I64::try_from(p).unwrap_or_default());
            self.has_pending.set(U64::from(1u64));
        }
        Ok(err2)
    }

    /// Poke the brain with any window you like, free via eth_call.
    pub fn predict(&self, x: Vec<i64>) -> i64 {
        let h = self.hidden.get().to::<u64>() as usize;
        let mut xin = [0i64; IN];
        xin.copy_from_slice(&x[..IN]);
        forward(&self.read_weights(), h, &xin).0
    }

    /// Everything a page needs in one call.
    pub fn state(&self) -> (u64, u64, i64, u64, u64, U256) {
        (
            self.lessons.get().to::<u64>(),
            self.beats_seen.get().to::<u64>(),
            self.pending.get().try_into().unwrap_or(0i64),
            self.has_pending.get().to::<u64>(),
            self.last_t.get().to::<u64>(),
            self.last_round.get(),
        )
    }

    /// The raw synapses, one call.
    pub fn synapses(&self) -> Vec<U256> {
        let h = self.hidden.get().to::<u64>() as usize;
        let slots = param_count(h).div_ceil(4);
        (0..slots).map(|s| self.packed.get(s).unwrap()).collect()
    }

    pub fn gaps(&self) -> Vec<u64> {
        gaps_unpack(self.gaps_lo.get(), self.gaps_hi.get()).iter().map(|g| *g as u64).collect()
    }

    pub fn feed_address(&self) -> Address {
        self.feed.get()
    }
}

#[cfg(test)]
mod pure {
    use super::net::*;

    #[test]
    fn feat_gap_matches_the_twin() {
        // JS twin: Math.round(clamp(log2(gap/60), -4, 8) * 65536)
        for (gap, want) in [(60u64, 0i64), (120, 65536), (30, -65536), (1, -4 * 65536), (100_000, 8 * 65536)] {
            let got = feat_gap(gap);
            assert!((got - want).abs() <= 2, "gap {gap}: got {got} want {want}");
        }
    }

    #[test]
    fn gap_window_learns_clustered_vol() {
        // alternating calm/storm regimes must beat a constant guess
        let mut rng = Rng(9);
        let mut gaps: Vec<u64> = Vec::new();
        for block in 0..600 {
            let calm = (block / 30) % 2 == 0;
            for _ in 0..8 {
                let base = if calm { 1800 } else { 120 };
                gaps.push(base + (rng.next() % (base as u64)) );
            }
        }
        let mut w = init_weights(32, 7);
        let (mut se_net, mut se_c, mut cnt) = (0i128, 0i128, 0);
        let mean = {
            let s: u64 = gaps.iter().sum();
            feat_gap(s / gaps.len() as u64)
        };
        for t in 16..gaps.len() {
            let mut win = [0u32; 16];
            for i in 0..16 { win[i] = gaps[t - 16 + i] as u32; }
            let x = features(&win);
            let target = feat_gap(gaps[t]);
            if t > gaps.len() / 2 {
                let p = forward(&w, 32, &x).0;
                se_net += ((p - target) as i128).pow(2);
                se_c += ((mean - target) as i128).pow(2);
                cnt += 1;
            }
            lesson(&mut w, 32, &x, target, 11);
        }
        assert!(cnt > 0 && se_net * 10 < se_c * 9, "net {se_net} not < 0.9x const {se_c}");
    }
}
