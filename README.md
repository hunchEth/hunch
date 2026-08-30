# hunch

An MLP that lives in Robinhood Chain storage. It reads the rhythm of
Chainlink price feeds, forms a hunch about when the next beat lands, and
being wrong rewires it, on chain, in one call.

## Layout

- `src/net.rs`  Q16.16 fixed-point MLP, forward + one-sample SGD.
- `src/lib.rs`  Stylus contract, weight packing, learning-curve tests.
- `web/`        Static site + Hood feed pulse collector.
