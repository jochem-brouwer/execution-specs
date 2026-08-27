"""
Cost models and helpers for the EVM memory analysis.

Every projected figure in the report is derived from the functions here, so
each number can be traced back to an explicit formula. All figures assume the
EIP-7825 transaction gas limit cap of 2**24 gas.

Three pricing models:

* ``quad_*``     - today's rule: memory_cost(w) = 3*w + w**2 // 512
* ``eip7923_*``  - page-based linear: 100 gas per 4096-byte page, page 0 free
                   per message call (EIP-7923).
* EIP-7686 keeps the linear word price (3*w) but bounds the aggregate through a
  gas *reservation* in call-forwarding; see ``eip7686.py``.

Nothing here depends on numpy or the execution-specs package, so the scripts
run with a bare Python 3 interpreter.
"""

import math

# --- transaction envelope -------------------------------------------------
TX_GAS_CAP = 2**24          # EIP-7825 per-transaction gas limit cap
INTRINSIC = 21_000          # base cost of a plain value-less call transaction
GAS = TX_GAS_CAP - INTRINSIC  # gas that reaches the top execution frame

WORD = 32                   # bytes per EVM word
PAGE = 4096                 # EIP-7923 PAGE_SIZE
PAGE_COST = 100             # EIP-7923 ALLOCATE_PAGE_COST


def mib(num_bytes: float) -> float:
    """Bytes -> mebibytes."""
    return num_bytes / 1024 / 1024


# --- today: quadratic memory price ----------------------------------------
def quad_cost(words: int) -> int:
    """Current memory expansion cost for ``words`` 32-byte words."""
    return 3 * words + (words * words) // 512


def quad_maxwords(budget: int) -> int:
    """Largest word count whose quadratic expansion fits within ``budget``."""
    if budget <= 0:
        return 0
    # invert 3w + w^2/512 = budget, then correct for integer rounding
    w = int(256 * (math.sqrt(9 + budget / 128.0) - 3))
    while quad_cost(w + 1) <= budget:
        w += 1
    while w > 0 and quad_cost(w) > budget:
        w -= 1
    return w


# --- EIP-7923: page-based linear price -------------------------------------
def pages_for_bytes(num_bytes: int) -> int:
    """Number of 4096-byte pages spanned by ``num_bytes`` of memory."""
    return (num_bytes + PAGE - 1) // PAGE


def eip7923_cost(pages: int) -> int:
    """Gas to hold ``pages`` pages in one frame (page 0 is free per frame)."""
    return PAGE_COST * max(0, pages - 1)


def eip7923_maxpages(budget: int) -> int:
    """Largest page count affordable within ``budget`` (page 0 free)."""
    if budget < 0:
        return 0
    return 1 + budget // PAGE_COST
