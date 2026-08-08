#![cfg_attr(not(feature = "export-abi"), no_main)]

#[cfg(not(feature = "export-abi"))]
#[unsafe(no_mangle)]
pub extern "C" fn main() {}
