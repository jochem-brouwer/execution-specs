"""
Benchmark journal amplification attack scenarios.

These tests target the storage journal infrastructure by maximizing the number
of journal entries (writes and rollbacks) per unit of gas. Storage journals
track SSTORE and TSTORE changes for commit and revert operations, and cleanup
cost scales linearly with journal size but is not fully gas-metered.

Attack vectors from the analysis:
- Section 4.1: Warm SSTORE same-slot churn (~360K journal entries per tx)
- Section 4.2: TSTORE + REVERT amplification (~714K journal operations)
- Section 4.4: CommitCore reverse iteration (HashSet deduplication overhead)
"""

import pytest
from execution_testing import (
    Alloc,
    BenchmarkTestFiller,
    JumpLoopGenerator,
    Op,
    While,
)


def test_warm_sstore_same_slot_churn(
    benchmark_test: BenchmarkTestFiller,
) -> None:
    """
    Maximize journal entries via repeated warm SSTORE to the same slot.

    After the initial cold write (22,100 gas), each subsequent write to the
    same dirty slot costs only 100 gas but appends a new entry to the storage
    journal. The eventual commit iterates all entries with deduplication,
    performing a hash and equality check per StorageCell.
    """
    # Initial cold write makes slot 0 dirty
    setup = Op.SSTORE(0, 1)
    # Each subsequent write costs 100 gas (warm, dirty slot)
    # GAS provides a changing value each iteration
    attack_block = Op.SSTORE(Op.PUSH0, Op.GAS)

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )


@pytest.mark.parametrize(
    "writes_per_call",
    [100, 500, 1000],
)
def test_tstore_revert_amplification(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    writes_per_call: int,
) -> None:
    """
    Amplify journal operations via TSTORE in subcalls that revert.

    A child contract performs K transient storage writes to different keys
    (100 gas each) then REVERTs. Each REVERT triggers O(K) journal rollback
    operations that are not gas-metered. The caller repeats this in a loop,
    maximizing the ratio of free rollback work to paid gas.
    """
    K = writes_per_call

    # Child: TSTORE to K different keys, then REVERT
    # Stack: [limit=K, counter=0]
    child_code = (
        Op.PUSH2(K)
        + Op.PUSH0
        + While(
            # TSTORE(counter, 1): push value first, then dup counter as key
            body=Op.PUSH1(1) + Op.DUP2 + Op.TSTORE,
            # Increment counter, check limit > counter
            condition=(Op.PUSH1(1) + Op.ADD + Op.DUP1 + Op.DUP3 + Op.GT),
        )
        + Op.REVERT(0, 0)
    )
    child_addr = pre.deploy_contract(code=child_code)

    setup = Op.PUSH20(child_addr)
    attack_block = Op.POP(Op.CALL(gas=Op.GAS, address=Op.DUP1))

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )


@pytest.mark.parametrize(
    "writes_per_call",
    [100, 500, 1000],
)
def test_sstore_revert_amplification(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    writes_per_call: int,
) -> None:
    """
    Amplify journal operations via warm SSTORE in subcalls that revert.

    A child contract performs an initial cold SSTORE (making the slot dirty),
    then writes K-1 more times to the same warm dirty slot (100 gas each),
    then REVERTs. The revert triggers O(K) journal rollback for both the
    storage journal and the access tracker.
    """
    K = writes_per_call

    # Child: cold SSTORE to slot 0, then K-1 warm SSTOREs, then REVERT
    # Stack after cold write: [limit=K, counter=1]
    child_code = (
        Op.SSTORE(0, 1)  # Cold write: 22,100 gas
        + Op.PUSH2(K)
        + Op.PUSH1(1)  # Counter starts at 1 (cold write counted)
        + While(
            # SSTORE(0, counter): push counter as value, push 0 as key
            body=Op.DUP1 + Op.PUSH0 + Op.SSTORE,
            # Increment counter, check limit > counter
            condition=(Op.PUSH1(1) + Op.ADD + Op.DUP1 + Op.DUP3 + Op.GT),
        )
        + Op.REVERT(0, 0)
    )
    child_addr = pre.deploy_contract(code=child_code)

    setup = Op.PUSH20(child_addr)
    attack_block = Op.POP(Op.CALL(gas=Op.GAS, address=Op.DUP1))

    benchmark_test(
        code_generator=JumpLoopGenerator(
            setup=setup,
            attack_block=attack_block,
        ),
    )
