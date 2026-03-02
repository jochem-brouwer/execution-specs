"""
Benchmark memory expansion attack scenarios in nested calls.

Memory expansion triggers Array.Clear on the new region. While well-priced
at large sizes due to the quadratic gas term, at moderate sizes (~32KB) the
gas cost is relatively low while generating ArrayPool churn and GC pressure.
In nested call patterns, each call frame allocates its own memory, amplifying
the total memory allocation and clearing work.

Attack vector from the analysis:
- Section 4.7: Memory Expansion Array.Clear
"""

import pytest
from execution_testing import (
    Alloc,
    BenchmarkTestFiller,
    JumpLoopGenerator,
    Op,
)


@pytest.mark.parametrize(
    "expansion_size",
    [
        pytest.param(4096, id="4KiB"),
        pytest.param(16384, id="16KiB"),
        pytest.param(32768, id="32KiB"),
        pytest.param(65536, id="64KiB"),
    ],
)
def test_memory_expansion_in_subcalls(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    expansion_size: int,
) -> None:
    """
    Benchmark memory expansion overhead across repeated subcalls.

    A child contract expands its memory to expansion_size bytes, triggering
    Array.Clear on the new region, then returns. Each call frame has its own
    memory, so the caller can repeat this in a tight loop. The memory
    expansion gas is paid per call, but the actual allocation, clearing, and
    eventual GC pressure are implementation-level costs that may exceed the
    gas-implied work.
    """
    # Child: expand memory then return immediately
    child_code = Op.MSTORE8(expansion_size - 1, 0xFF) + Op.RETURN(0, 0)
    child_addr = pre.deploy_contract(code=child_code)

    setup = Op.PUSH20(child_addr)

    # Call child repeatedly; each call allocates + clears expansion_size bytes
    attack_block = Op.POP(
        Op.STATICCALL(
            gas=Op.GAS,
            address=Op.DUP1,
            args_offset=Op.PUSH0,
            args_size=Op.PUSH0,
            ret_offset=Op.PUSH0,
            ret_size=Op.PUSH0,
        )
    )

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )
