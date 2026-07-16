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
