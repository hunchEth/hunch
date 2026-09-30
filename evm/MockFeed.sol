// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Programmable heartbeat for rehearsals: beats spaced 300s apart in the
/// past, latest round and timestamp settable.
contract MockFeed {
    uint80 public round = 100;
    uint256 public ts = 1_000_000;

    function set(uint80 r, uint256 t) external { round = r; ts = t; }

    function latestRoundData() external view returns (uint80, int256, uint256, uint256, uint80) {
        return (round, 100e8, ts, ts, round);
    }

    function getRoundData(uint80 r) external view returns (uint80, int256, uint256, uint256, uint80) {
        require(r <= round, "future");
        uint256 t = ts - uint256(round - r) * 300;
        return (r, 100e8, t, t, r);
    }
}
