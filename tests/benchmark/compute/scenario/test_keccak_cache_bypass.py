"""
Benchmark keccak cache bypass attack scenarios.

Some implementations cache keccak hash results for inputs up to a certain
size threshold (e.g. 92 bytes). Inputs just above the threshold bypass the
cache entirely, forcing full SHA3 computation each time. This creates up to
a 30x slowdown compared to cached inputs of similar size.

Attack vector from the analysis:
- Section 4.3: KeccakCache bypass (inputs > 92 bytes)
"""

import pytest
from execution_testing import (
    BenchmarkTestFiller,
    JumpLoopGenerator,
    Op,
)


@pytest.mark.parametrize(
    "input_size",
    [
        pytest.param(32, id="32B-cached"),
        pytest.param(64, id="64B-cached"),
        pytest.param(92, id="92B-cached-boundary"),
        pytest.param(93, id="93B-uncached-boundary"),
        pytest.param(96, id="96B-uncached"),
        pytest.param(128, id="128B-uncached"),
        pytest.param(256, id="256B-uncached"),
    ],
)
def test_keccak_cache_boundary(
    benchmark_test: BenchmarkTestFiller,
    input_size: int,
) -> None:
    """
    Benchmark SHA3 at the keccak cache size boundary.

    Implementations may cache keccak results for inputs up to 92 bytes.
    Inputs of 93+ bytes bypass the cache, forcing full computation each time.
    This test parametrizes input sizes around the boundary to quantify the
    cache bypass amplification factor.
    """
    # Expand memory to input_size bytes
    setup = Op.MSTORE8(input_size - 1, 0xFF)

    # Hash the memory region repeatedly. Store result back to memory offset 0
    # to change input data each iteration and prevent hash elision.
    attack_block = Op.MSTORE(Op.PUSH0, Op.SHA3(Op.PUSH0, input_size))

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )
