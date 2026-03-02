"""
Benchmark return data copy attack scenarios.

After a CALL returns, return data is copied to the caller's memory at the
specified output region. If the output region was already expanded (memory
expansion gas already paid), the actual memcpy is free — no per-word copy
gas is charged. This allows crafting scenarios where significant memory
copy work happens without corresponding gas cost.

Attack vector from the analysis:
- Section 4.6: Return Data Copy — Free Memcpy
"""

import pytest
from execution_testing import (
    Alloc,
    BenchmarkTestFiller,
    JumpLoopGenerator,
    Op,
)


@pytest.mark.parametrize(
    "return_size",
    [
        pytest.param(1024, id="1KiB"),
        pytest.param(4096, id="4KiB"),
        pytest.param(16384, id="16KiB"),
        pytest.param(32768, id="32KiB"),
    ],
)
def test_return_data_free_memcpy(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    return_size: int,
) -> None:
    """
    Benchmark free return data memcpy to pre-expanded memory.

    A child contract returns return_size bytes of data. The caller
    pre-expands its memory to cover the output region so that the CALL's
    return data copy is a pure memcpy with no additional memory expansion
    gas. The memcpy work scales with return_size but is not gas-metered.
    """
    # Child: fill memory with non-zero data via CODECOPY, then RETURN it
    child_code = Op.CODECOPY(0, 0, return_size) + Op.RETURN(0, return_size)
    # Pad to at least return_size so CODECOPY has data to copy
    if len(child_code) < return_size:
        child_code += bytes([0xFE] * (return_size - len(child_code)))

    child_addr = pre.deploy_contract(code=child_code)

    # Caller: expand memory once in setup, then call child in a loop
    # Memory expansion gas is paid once; subsequent calls do free memcpy
    setup = (
        Op.MSTORE8(return_size - 1, 0xFF)  # Expand memory
        + Op.PUSH20(child_addr)
    )

    # STATICCALL with output region in already-expanded memory
    attack_block = Op.POP(
        Op.STATICCALL(
            gas=Op.GAS,
            address=Op.DUP1,
            args_offset=Op.PUSH0,
            args_size=Op.PUSH0,
            ret_offset=Op.PUSH0,
            ret_size=return_size,
        )
    )

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )
