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
