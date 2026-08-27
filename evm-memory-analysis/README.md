# EVM memory limits — reproducible evidence

This directory backs the report *"What can we do about EVM memory"* with
runnable evidence for every number it cites. Nothing in the report is a
hand-wave: each figure is either **filled under the reference EVM** or **derived
from an explicit formula** you can re-run.

Branch: `evm-memory-limit-tests` (off `forks/amsterdam`).

## Two classes of evidence

| Class | What | How to verify |
|-------|------|---------------|
| **Fill-verified** | Today's max memory under the *current* (quadratic) rules — single-frame, nested-peak, cumulative. | Actual state/blockchain fills executed by the in-repo EELS EVM (the benchmarks below). |
| **Analytical** | Projections for **EIP-7923**, **EIP-7686**, linear-only, and the cap trade-offs. Those EIPs are **not implemented** in the spec. | Numbers follow from the EIPs' *stated formulas*, encoded in [`models.py`](models.py) and printed by [`reproduce.py`](reproduce.py). |

The two classes agree on the current-rules numbers: the analytical model in
`reproduce.py` predicts the same single/nested/cumulative maxima that the fills
produce. That cross-check is the point — the model is calibrated against real
fills, then extended to the unimplemented pricing models.

## 1. Analytical figures — one command

```bash
python3 evm-memory-analysis/reproduce.py
```

No dependencies (bare Python 3). It prints every projected figure next to its
target report value, and finishes with the report's comparison matrix. Figure →
function map:

| Report figure | Value | Source |
|---|---|---|
| Single frame, today | 2.80 MiB (offset 2,939,456; 16,756,193 gas; 14 under cap) | `current_single()` |
| Total / nested peak, today | 18.3 MiB (600,914 words, 148 frames, 6.5×) | `current_nested()` |
| Cumulative touched, today | ~126 MiB (278 words/call, 14,776 calls, 8.7 KiB live) | `current_cumulative()` |
| EIP-7923, cap off | ~654 MiB single = total = cumulative | `eip7923_single()` |
| EIP-7923, per-frame cap 2.8 MiB | 275 MiB across 99 frames | `eip7923_nested_capped()` |
| EIP-7923, cap to bind ≤18 MiB | 14 pages (56 KiB) | `eip7923_nested_capped()` sweep |
| EIP-7686 ceiling (k=1) | 16.0 MiB, ~9% of gas paid | `eip7686(1)` |
| EIP-7686 ceiling (k=6) | 2.7 MiB | `eip7686(6)` |
| Linear-only, today's 3/word, no cap | 170 MiB (single = nested = cumulative) | `linear_only()` |
| Gas-per-byte spectrum (the knob) + 32/3 | table | `spectrum()` |

Key relation (`spectrum()`): once the quadratic is removed, **max memory =
GAS / gas-per-byte**, so every model is one choice of that rate — linear 3/word
= 170 MiB, EIP-7923 100/page = 654 MiB, EIP-7686 reservation 1 gas/byte = 16 MiB.
EIP-7686's reservation is 32/3 ≈ 10.7× the linear cost: it pays the cheap rate
but reserves at the steep one, decoupling price from ceiling.

## 2. Fill-verified figures — run the benchmarks

The three current-rules maxima are also encoded as fill-able benchmarks:
[`tests/benchmark/compute/scenario/test_max_memory_allocation.py`](../tests/benchmark/compute/scenario/test_max_memory_allocation.py).

```bash
# quick smoke fill (all three, small budget) — proves the bytecode is valid
uv run fill tests/benchmark/compute/scenario/test_max_memory_allocation.py \
    --fork Osaka --gas-benchmark-values 1

# fill at (near) the tx gas cap — exercises the real ~2.8 / ~18 MiB cases
uv run fill tests/benchmark/compute/scenario/test_max_memory_allocation.py \
    -k "blockchain_test-" --fork Osaka --gas-benchmark-values 16
```

The three benchmarks:

- `test_single_frame_memory_expansion` — grows one frame's memory (`MSTORE(MSIZE)`
  loop) until out of gas; reaches the single-frame maximum. No hardcoded offset,
  so it tracks the fork's memory pricing.
- `test_nested_frame_memory_expansion` — nested self-calls maximizing
  *simultaneously live* memory; its per-frame schedule is derived from the
  fork's own memory pricing (`nested_frame_plan`), mirrored exactly by
  `current_nested()` in `reproduce.py`.
- `test_cumulative_memory_expansion` — sequential returning calls maximizing
  memory *touched over the whole tx*; per-call size from `cumulative_worker_words`,
  mirrored by `current_cumulative()`.

To confirm the single-frame boundary is exact rather than approximate, the
plain arithmetic (`current_single()`) shows the maximal `MSTORE` lands 14 gas
under the cap and the next byte overflows it.

## 3. What is *not* directly fillable, and why

EIP-7923 and EIP-7686 are not implemented in this spec, so there is no EVM to
fill against for their numbers. The evidence for them is the model in
[`models.py`](models.py), which uses only each EIP's published constants and
formulas:

- **EIP-7923**: `ALLOCATE_PAGE_COST = 100`, `PAGE_SIZE = 4096`, page 0 free per
  message call, quadratic and linear word terms removed.
- **EIP-7686**: linear `3 * words`; forwarding `max_call_gas = gas - max(gas//64,
  memory_byte_size)`; per-call cap `memory <= initial gas`.

Anyone can audit those formulas against the EIP text and re-run `reproduce.py`.
If either EIP is implemented later, the same three benchmarks can be filled on
that fork to replace the analytical numbers with fills.

## Assumptions & sensitivity

- All figures use the EIP-7825 cap of `2**24` gas and a plain call transaction
  (intrinsic 21,000).
- Single-frame figures are **exact**. Nested/cumulative use a modest per-frame
  overhead (nested ≈200 gas, cumulative ≈150 gas for the CALL + argument pushes);
  the results move <0.5% across the 130–300 gas range, so the MiB figures are
  robust to that estimate.
- The nested optimum from a finer value-iteration DP is ~603k words (18.4 MiB);
  the pure-Python geometric sweep here reports 600,914 words (18.3 MiB). The
  report quotes the conservative on-chain-realizable figure.
