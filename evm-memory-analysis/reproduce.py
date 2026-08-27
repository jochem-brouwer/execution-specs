#!/usr/bin/env python3
"""
Reproduce every projected number in the "What can we do about EVM memory"
report. Run with a bare interpreter:

    python3 evm-memory-analysis/reproduce.py

Each block prints the figure, the model it comes from, and the report value it
should match. The current-pricing (quadratic) figures are *also* verified end
to end by filling the benchmarks under the reference EVM; see README.md for the
exact ``fill`` commands. The EIP-7923 / EIP-7686 figures are analytical: those
EIPs are not implemented in the spec, so their numbers follow from the EIPs'
stated formulas, encoded in models.py.
"""

from models import (
    GAS,
    INTRINSIC,
    PAGE,
    TX_GAS_CAP,
    WORD,
    eip7923_cost,
    eip7923_maxpages,
    mib,
    quad_cost,
    quad_maxwords,
)


def hr(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---------------------------------------------------------------------------
# 1. TODAY (quadratic) -- single frame
# ---------------------------------------------------------------------------
def current_single() -> int:
    """Largest single-frame memory under the cap. Report: 2.80 MiB."""
    # bytecode is MSTORE(offset, 1) + STOP:
    #   intrinsic + PUSH value(3) + PUSH offset(3) + MSTORE base(3)
    base = INTRINSIC + 3 + 3 + 3
    words = quad_maxwords(TX_GAS_CAP - base)
    size = words * WORD
    offset = size - WORD
    cost = quad_cost(words)
    total = base + cost
    print(f"  MSTORE offset          : {offset:,}")
    print(f"  memory size            : {size:,} bytes = {mib(size):.2f} MiB")
    print(f"  words                  : {words:,}")
    print(f"  memory gas cost        : {cost:,}")
    print(f"  base + memory          : {total:,}  (cap {TX_GAS_CAP:,})")
    print(f"  headroom under cap     : {TX_GAS_CAP - total} gas")
    print(f"  -> single-frame max    : {mib(size):.2f} MiB   [report: 2.80 MiB]")
    return words


# ---------------------------------------------------------------------------
# 2. TODAY (quadratic) -- nested peak (simultaneously live)
# ---------------------------------------------------------------------------
def current_nested(overhead: int = 200):
    """
    Max simultaneously-live memory via nested self-calls. Mirrors the on-chain
    contract in test_max_memory_allocation.py::nested_frame_plan: each frame
    expands to T words, forwards 63/64 of the rest, next target = T - T//50
    (~0.98). Report: ~18.3 MiB, ~150 frames, 6.5x a single frame.
    """
    best = (0, 0, 0)  # (peak_words, T0, frames)
    for t0 in range(2000, 30_001, 200):
        g, target, peak, frames = GAS, t0, 0, 0
        for _ in range(1024):                 # EVM call-depth limit
            if target < 32:
                break
            cost = quad_cost(target) + overhead
            if cost > g:
                break
            peak += target
            frames += 1
            g = (g - cost) * 63 // 64          # 63/64 forwarding
            target -= target // 50             # ~0.98 geometric decay
        if peak > best[0]:
            best = (peak, t0, frames)
    peak_words, t0, frames = best
    print(f"  best initial target T0 : {t0:,} words")
    print(f"  peak live memory       : {peak_words:,} words "
          f"= {mib(peak_words * WORD):.2f} MiB")
    print(f"  nesting depth reached  : {frames} frames")
    print(f"  -> nested peak         : {mib(peak_words * WORD):.1f} MiB   "
          f"[report: 18.3 MiB]")
    return peak_words


# ---------------------------------------------------------------------------
# 3. TODAY (quadratic) -- cumulative touched (freed + re-touched)
# ---------------------------------------------------------------------------
def current_cumulative(overhead: int = 150):
    """
    Max memory *touched* over the whole tx via sequential returning calls.
    Mirrors test_max_memory_allocation.py::cumulative_worker_words: choose the
    per-call words that minimize gas-per-word. Report: ~126 MiB, ~8.7 KiB live.
    """
    best_w, best_rate = 1, float("inf")
    for w in range(1, 4001):
        rate = (quad_cost(w) + overhead) / w
        if rate < best_rate:
            best_w, best_rate = w, rate
    per_call = quad_cost(best_w) + overhead
    calls = GAS // per_call
    touched = calls * best_w
    print(f"  best words per call    : {best_w}")
    print(f"  gas per word           : {best_rate:.2f}")
    print(f"  calls per tx           : {calls:,}")
    print(f"  cumulative touched     : {touched:,} words "
          f"= {mib(touched * WORD):.1f} MiB")
    print(f"  live at any instant    : {best_w * WORD:,} bytes "
          f"= {best_w * WORD / 1024:.1f} KiB")
    print(f"  -> cumulative          : {mib(touched * WORD):.0f} MiB   "
          f"[report: ~126 MiB]")
    return touched


# ---------------------------------------------------------------------------
# 4. EIP-7923 -- page-based linear, cap OFF
# ---------------------------------------------------------------------------
def eip7923_single() -> int:
    """Single frame under EIP-7923 with no cap. Report: ~654 MiB."""
    pages = eip7923_maxpages(GAS)
    size = pages * PAGE
    print(f"  affordable pages       : {pages:,}  ({eip7923_cost(pages):,} gas)")
    print(f"  -> single-frame max    : {mib(size):.1f} MiB   [report: ~654 MiB]")
    return pages


def eip7923_nested_capped(cap_pages: int, overhead: int = 200):
    """
    Worst-case total under EIP-7923 with a PER-FRAME cap of ``cap_pages``:
    attacker nests, each frame at the cap. Returns (total_pages, frames).
    """
    g, total, frames = GAS, 0, 0
    while True:
        cost = eip7923_cost(cap_pages)
        if cost + overhead > g:
            total += eip7923_maxpages(g)
            frames += 1
            break
        total += cap_pages
        frames += 1
        g = (g - cost - overhead) * 63 // 64
    return total, frames


# ---------------------------------------------------------------------------
# 5. EIP-7686 -- linear price + gas-reservation forwarding
# ---------------------------------------------------------------------------
def eip7686(k: int = 1):
    """
    EIP-7686 invariant: reserve k gas per live byte on forwarding, so total
    live memory <= gas // k bytes, purely from a per-frame rule. Report (k=1):
    16.0 MiB ceiling, ~9% of gas actually paid.
    """
    ceiling = GAS // k
    fill_cost = 3 * (ceiling // WORD)        # linear price 3 gas / word
    print(f"  reservation multiplier : k = {k}")
    print(f"  memory ceiling         : {ceiling:,} bytes "
          f"= {mib(ceiling):.1f} MiB")
    print(f"  gas actually paid      : {fill_cost:,} "
          f"({100 * fill_cost / GAS:.0f}% of budget; the rest is reserved)")
    return ceiling


# ---------------------------------------------------------------------------
def main() -> None:
    """Run every reproduction and print the report's comparison matrix."""
    print(f"Transaction gas cap (EIP-7825): {TX_GAS_CAP:,}")
    print(f"Gas entering top frame        : {GAS:,}  (cap - {INTRINSIC:,})")

    hr("1. TODAY (quadratic)  -  single frame")
    single_w = current_single()

    hr("2. TODAY (quadratic)  -  nested peak (live at once)")
    nested_w = current_nested()
    print(f"  nested / single ratio  : {nested_w / single_w:.1f}x   "
          f"[report: 6.5x]")

    hr("3. TODAY (quadratic)  -  cumulative touched")
    current_cumulative()

    hr("4. EIP-7923 (100 gas/page)  -  cap OFF")
    eip7923_single()
    total_28, fr_28 = eip7923_nested_capped(2_939_488 // PAGE)  # 2.8 MiB/frame
    print(f"  per-frame cap 2.8 MiB  : nested total {mib(total_28 * PAGE):.0f} "
          f"MiB across {fr_28} frames   [report: 275 MiB / 99 frames]")
    # smallest per-frame cap holding total <= today's nested 18.3 MiB
    target = 19_229_248 // PAGE
    bind = max(L for L in range(1, 20000)
               if eip7923_nested_capped(L)[0] <= target)
    print(f"  cap to bind <=18.3 MiB : {bind} pages "
          f"= {bind * PAGE / 1024:.0f} KiB   [report: 56 KiB]")

    hr("5. EIP-7686 (linear + gas-reservation forwarding)")
    eip7686(k=1)
    print("  ---")
    eip7686(k=6)
    print("  -> k=6 pins the ceiling near today's single frame "
          "[report: ~2.7 MiB]")

    hr("COMPARISON MATRIX (report Section 06)")
    single_mib = mib(single_w * WORD)
    nested_mib = mib(nested_w * WORD)
    rows = [
        ("today (quadratic)", f"{single_mib:.1f}", f"{nested_mib:.1f}", "126"),
        ("EIP-7923 no cap", "~654", "~654", "~654"),
        ("EIP-7923 +64MiB global cap", "<=64", "<=64", "<=64"),
        ("EIP-7923 +2.8MiB/frame cap", "2.8",
         f"{mib(total_28 * PAGE):.0f}", "~654"),
        ("EIP-7686 (k=1)", f"{mib(GAS):.1f}", f"{mib(GAS):.1f}",
         f"{mib(GAS):.1f}"),
    ]
    print(f"  {'model':<28}{'single':>9}{'total':>9}{'cumul':>9}  (MiB)")
    for name, s, t, c in rows:
        print(f"  {name:<28}{s:>9}{t:>9}{c:>9}")


if __name__ == "__main__":
    main()
