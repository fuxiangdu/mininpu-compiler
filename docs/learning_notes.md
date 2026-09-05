# MiniNPU v3 learning notes

## Why tiling exists

Large tensor operands do not fit in a small on-chip buffer at once. Tiling
partitions M, N and K so each kernel instance works on a bounded subset. Tile
sizes affect data reuse, DMA traffic, tail handling, parallelism and memory
capacity simultaneously.

## Compile-time planning sequence

1. Read UB capacity and tile granularity from module target attributes.
2. Read static M, N and K from the fused operation's tensor types.
3. Enumerate aligned candidate tile triples.
4. Reject candidates whose estimated working set exceeds UB.
5. Score legal candidates by MACs per resident byte.
6. Break ties using fewer total tiles and then more MACs per tile.
7. Attach the selected plan to the fused operation.

## Capacity is a hard constraint

A high-score candidate is irrelevant when it does not fit. The planner filters
by capacity before comparing performance scores. The 128-byte negative test
therefore fails compilation instead of silently emitting an illegal plan.

## Static and dynamic shapes

Static shapes permit full enumeration at compile time. A dynamic dimension is
not guessed: v3 emits a remark and leaves the operation unplanned so a future
runtime specialization or fallback can handle it.

## Interview checkpoint

Be ready to explain:

1. Why larger tiles often improve reuse but consume more on-chip memory.
2. Which resident buffers appear in the working-set equation.
3. Why the accumulator is modeled as fp32 even when input precision is lower.
4. How `ceilDiv` accounts for tail tiles.
5. Why a cost-model result is a compiler estimate rather than benchmark proof.
