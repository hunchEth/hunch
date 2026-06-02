//! Fixed-point MLP with online SGD. Q16.16 arithmetic throughout.
//! Shared by the Stylus contract and the native learning-curve sim.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

pub const IN: usize = 16;
pub const Q: i64 = 16;
pub const ONE: i64 = 1 << Q;
