<p align="center">
  <img src=".github/banner.webp" alt="HUNCH">
</p>

<h1 align="center">HUNCH</h1>

<p align="center">
  A small neural net that lives in Robinhood Chain storage.<br>
  It listens to the market's pulse and forms a hunch about<br>
  when the next beat lands. Being wrong rewires it, on chain,<br>
  in one call.
</p>

CA: 0x82236d023e3f2c31e67f9945714688fd6cd8ecdf

<p align="center">
  <a href="https://hunch.guru"><strong>hunch.guru</strong></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://x.com/hunchdotguru"><strong>@hunchdotguru</strong></a>
</p>

<br>

## What it eats

Chainlink beats on Robinhood Chain. Each time a stock moves 0.5%, a beat
lands in the feed. That rhythm is the input.

## Where the synapses live

Contract storage. 577 numbers, readable by anyone, packed four to a slot.
A lesson costs about 570,000 gas. A prediction is free.

## Who teaches it

Nobody. `lesson()` takes no arguments. The brain reads its own feed,
measures the gap since the last beat, grades its previous hunch, and
runs backprop and SGD in place. Nobody gets to hand it inputs.

## Layout

    src/net.rs    Q16.16 fixed-point MLP, forward and one-sample SGD
    src/lib.rs    Stylus contract, self-feeding lesson, weight packing
    evm/          mock oracle for local rehearsals

## Tests

    cargo test --release --lib

runs the learning-curve simulation on generated gap data and checks that
the net beats the constant mean baseline.

## Gas (Nitro devnet, Hood twin)

    h=32 (145 slots)   lesson full-write    572,585
    h=32               lesson min-write     383,682
    h=32               predict (view)       379,255
    h=128              lesson full-write  2,097,259

SLOAD dominates, 2,100 gas per cold slot.
