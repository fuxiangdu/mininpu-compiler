# MiniNPU v3 tile model

## Symbols

- `tm`, `tn`, `tk`: candidate tile sizes for M, N and K.
- `b`: bytes per input/output element.
- `U`: configured UB capacity in bytes.

## Working-set estimate

```text
W = 2b(tm*tk + tk*tn) + 4(tm*tn) + b(tm*tn) + b*tn
```

The terms model:

1. double-buffered lhs and rhs input tiles;
2. one fp32 accumulator tile;
3. one output tile;
4. one bias vector tile.

A candidate is legal only when `W <= U`.

## Objective

The primary score is estimated arithmetic intensity:

```text
score = (tm * tn * tk) / W
```

If two candidates have the same score, the planner chooses the one with fewer
total tiles:

```text
tile_count = ceil(M/tm) * ceil(N/tn) * ceil(K/tk)
```

The final tie-breaker prefers more MACs per tile, making selection stable and
deterministic.

## Reference case

For M=128, N=512, K=256 and fp32 elements, the included tests expect both 64
KiB and 256 KiB configurations to produce legal but different plans. The test
recomputes `W` and `tile_count` independently in Python instead of trusting the
attributes emitted by the C++ Pass.

## Scope

This compact model is intentionally inspectable. A production cost model would
also account for bank conflicts, alignment padding, instruction selection,
pipeline overlap, core count, DMA startup cost and measured target latency.
