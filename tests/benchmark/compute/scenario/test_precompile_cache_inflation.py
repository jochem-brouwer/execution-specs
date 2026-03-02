"""
Benchmark precompile cache inflation attack scenarios.

Some implementations cache precompile results in unbounded data structures
(e.g. ConcurrentDictionary) that are cleared between blocks. By calling
precompiles with many unique inputs, an attacker can inflate the cache to
significant memory sizes (~100+ MB per block) while paying only the gas
cost of each precompile call.

Attack vector from the analysis:
- Section 3.2: Precompile Result Cache (unbounded ConcurrentDictionary)
"""

import pytest
from execution_testing import (
    BenchmarkTestFiller,
    JumpLoopGenerator,
    Op,
)

SHA256_ADDRESS = 0x02


@pytest.mark.parametrize(
    "input_size",
    [
        pytest.param(32, id="32B"),
        pytest.param(64, id="64B"),
        pytest.param(128, id="128B"),
    ],
)
def test_precompile_cache_unique_inputs(
    benchmark_test: BenchmarkTestFiller,
    input_size: int,
) -> None:
    """
    Inflate the precompile result cache with unique inputs.

    Call SHA256 with a different input each iteration by writing the current
    GAS value into memory before each call. Each unique (address, input) pair
    creates a new cache entry. With minimal input sizes, this maximizes the
    number of cache entries per unit of gas.
    """
    # Expand memory to input_size
    setup = Op.MSTORE8(input_size - 1, 0xFF)

    # Modify memory to create unique input, then call precompile
    attack_block = Op.MSTORE(Op.PUSH0, Op.GAS) + Op.POP(
        Op.STATICCALL(
            gas=Op.GAS,
            address=SHA256_ADDRESS,
            args_offset=Op.PUSH0,
            args_size=input_size,
            ret_offset=Op.PUSH0,
            ret_size=32,
        )
    )

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )


def test_precompile_cache_same_input(
    benchmark_test: BenchmarkTestFiller,
) -> None:
    """
    Baseline: call SHA256 with the same input each iteration.

    Compare against test_precompile_cache_unique_inputs to measure the
    benefit of precompile caching. With identical inputs, implementations
    that cache precompile results will return immediately on cache hit.
    """
    input_size = 32
    setup = Op.MSTORE(Op.PUSH0, 0x42)

    attack_block = Op.POP(
        Op.STATICCALL(
            gas=Op.GAS,
            address=SHA256_ADDRESS,
            args_offset=Op.PUSH0,
            args_size=input_size,
            ret_offset=Op.PUSH0,
            ret_size=32,
        )
    )

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )
