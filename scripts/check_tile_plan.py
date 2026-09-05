#!/usr/bin/env python3
"""Validate MiniNPU tile attributes against the v3 UB model."""

import argparse
import math
import pathlib
import re


def read_int_attr(text: str, name: str) -> int:
    match = re.search(rf"{re.escape(name)}\s*=\s*([0-9]+)", text)
    if not match:
        raise ValueError(f"missing integer attribute: {name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlir_file", type=pathlib.Path)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--element-bytes", type=int, required=True)
    parser.add_argument("--ub-bytes", type=int, required=True)
    parser.add_argument("--granularity", type=int, default=16)
    args = parser.parse_args()

    text = args.mlir_file.read_text(encoding="utf-8")
    tile_m = read_int_attr(text, "mininpu.tile_m")
    tile_n = read_int_attr(text, "mininpu.tile_n")
    tile_k = read_int_attr(text, "mininpu.tile_k")
    working_set = read_int_attr(
        text, "mininpu.estimated_working_set_bytes"
    )
    tile_count = read_int_attr(text, "mininpu.estimated_tile_count")
    planned_ub = read_int_attr(text, "mininpu.ub_bytes")

    for tile, dimension, label in (
        (tile_m, args.m, "M"),
        (tile_n, args.n, "N"),
        (tile_k, args.k, "K"),
    ):
        if not 0 < tile <= dimension:
            raise ValueError(f"tile {label}={tile} is outside (0, {dimension}]")
        if dimension >= args.granularity and tile % args.granularity != 0:
            raise ValueError(
                f"tile {label}={tile} is not aligned to {args.granularity}"
            )

    expected_working_set = (
        2 * (tile_m * tile_k + tile_k * tile_n) * args.element_bytes
        + tile_m * tile_n * 4
        + tile_m * tile_n * args.element_bytes
        + tile_n * args.element_bytes
    )
    expected_tile_count = (
        math.ceil(args.m / tile_m)
        * math.ceil(args.n / tile_n)
        * math.ceil(args.k / tile_k)
    )

    if working_set != expected_working_set:
        raise ValueError(
            f"working set mismatch: IR={working_set}, expected={expected_working_set}"
        )
    if working_set > args.ub_bytes:
        raise ValueError(
            f"working set {working_set} exceeds UB {args.ub_bytes}"
        )
    if tile_count != expected_tile_count:
        raise ValueError(
            f"tile count mismatch: IR={tile_count}, expected={expected_tile_count}"
        )
    if planned_ub != args.ub_bytes:
        raise ValueError(
            f"UB attribute mismatch: IR={planned_ub}, expected={args.ub_bytes}"
        )
    if "mininpu.double_buffered = true" not in text:
        raise ValueError("double-buffered plan attribute is missing")

    print(
        f"[PASS] {args.mlir_file.name}: "
        f"tile=({tile_m},{tile_n},{tile_k}) "
        f"working_set={working_set}/{args.ub_bytes} "
        f"tile_count={tile_count}"
    )


if __name__ == "__main__":
    main()
