# hunch

An MLP that lives in Robinhood Chain storage. It reads the rhythm of
Chainlink price feeds, forms a hunch about when the next beat lands, and
being wrong rewires it, on chain, in one call.

## Layout

- `src/net.rs`  Q16.16 fixed-point MLP, forward + one-sample SGD.
- `src/lib.rs`  Stylus contract, weight packing, learning-curve tests.
- `web/`        Static site + Hood feed pulse collector.

## Tests

`cargo test --release --lib` runs the learning-curve simulation on GARCH(1,1)
returns and asserts the net beats an EWMA baseline both with full-slot writes
and with a 4-sample batched-write schedule (same write gas as 25% round-robin,
full learning).

## Gas (Nitro devnet, Hood twin)

    h=32 (145 slots)   lesson full-write    572,585
    h=32               lesson min-write     383,682
    h=32               predict (view)       379,255
    h=128              lesson full-write  2,097,259

SLOAD dominates (2,100 gas per cold slot).
