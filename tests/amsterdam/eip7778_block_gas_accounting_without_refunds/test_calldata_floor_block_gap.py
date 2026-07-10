"""
Demonstrate that the EIP-7976 calldata floor is invisible to the block gas
limit.

[EIP-7976](https://eips.ethereum.org/EIPS/eip-7976) raises the calldata floor
so that a data-heavy transaction must *occupy* at least the floor's worth of
block gas, bounding the worst-case block size. The sender-side gas formula
enforces this for payment
(``tx_gas_used = max(tx_gas_used_after_refund, calldata_floor_gas_cost)``), but
the block-level accounting sums the *pre-floor* regular gas
(``tx_gas_used_before_refund`` minus state gas). The floor is therefore charged
to senders yet never counted against ``block_gas_used``.

The consequence: a block can pack far more calldata than the floor-based
worst-case bound intends. Every transaction is individually valid and pays the
full floor, but the block-level sum -- and hence the header ``gas_used`` and
the base-fee update -- sees only the cheaper execution-metered cost.

Rates for non-zero calldata bytes: standard calldata = 16 gas/byte (the
execution-metered schedule, ``TX_DATA_TOKEN_STANDARD``); floor = 64 gas/byte
(``TX_DATA_TOKEN_FLOOR``, EIP-7976). Zero bytes are metered at 4 gas/byte on
the standard schedule but still 64 gas/byte at the floor, widening the gap.

This test constructs a block of identical pure-data-post transactions and
asserts that it is accepted as valid even though the sum of the floors the
senders pay exceeds the block gas limit several times over. If the floor were
correctly enforced at the block level, only ``block_gas_limit // floor``
transactions would fit.
"""

import pytest
from execution_testing import (
    Alloc,
    Block,
    BlockchainTestFiller,
    Environment,
    Fork,
    Transaction,
)
from execution_testing.vm import Op

from .spec import ref_spec_7778

REFERENCE_SPEC_GIT_PATH = ref_spec_7778.git_path
REFERENCE_SPEC_VERSION = ref_spec_7778.version


@pytest.mark.parametrize(
    "calldata_byte,calldata_len",
    [
        # The scenario from the EIP-8037 accounting write-up: a 260,000-byte
        # non-zero data post. floor = 16,655,000, execution cost = 4,175,000.
        pytest.param(0xFF, 260_000, id="nonzero_260k_bytes"),
        # Zero bytes are metered even cheaper on the standard schedule
        # (4 gas/byte) while paying the same floor, so the gap is far larger.
        pytest.param(0x00, 260_000, id="zero_260k_bytes"),
    ],
)
@pytest.mark.valid_from("Amsterdam")
def test_calldata_floor_invisible_to_block_limit(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    calldata_byte: int,
    calldata_len: int,
) -> None:
    """
    Pack more floor-bound data posts into a block than the floor should allow.

    The block is expected to be *valid*: its ``block_gas_used`` counts only the
    pre-floor execution cost, which stays under the limit, even though the sum
    of the calldata floors the senders pay is several times the block gas
    limit.
    """
    tx_gas_limit_cap = fork.transaction_gas_limit_cap()
    assert tx_gas_limit_cap is not None, "fork must cap per-tx gas (EIP-7825)"

    # A block that holds two maximum-size transactions (EIP-7825 cap x2).
    block_gas_limit = 2 * tx_gas_limit_cap

    intrinsic_calc = fork.transaction_intrinsic_cost_calculator()
    floor_calc = fork.transaction_data_floor_cost_calculator()

    data = bytes([calldata_byte]) * calldata_len

    # F: the calldata floor. The sender pays this and must set a gas limit of
    # at least this much for the transaction to be valid.
    floor = floor_calc(data=data)

    # c: what the block-level accounting actually counts. For a value-less call
    # to a STOP target there is no execution or state gas, so the block
    # contribution is exactly the regular gas deducted before execution.
    block_contribution = intrinsic_calc(
        calldata=data,
        return_cost_deducted_prior_execution=True,
    )
    assert floor > block_contribution, (
        "the calldata floor must bind above the execution-metered cost"
    )

    # If the floor were enforced at the block level, at most this many such
    # transactions could be included.
    intended_max_txs = block_gas_limit // floor

    # What the current accounting actually admits: the per-transaction
    # inclusion check reserves min(TX_MAX_GAS_LIMIT, tx.gas) == floor of block
    # gas, but only `block_contribution` is added to `block_gas_used` after
    # the transaction runs. Mirror that loop to size the block exactly.
    reserve = min(tx_gas_limit_cap, floor)
    block_gas_used = 0
    num_txs = 0
    while reserve <= block_gas_limit - block_gas_used:
        block_gas_used += block_contribution
        num_txs += 1

    assert num_txs > intended_max_txs, (
        f"scenario must exceed the intended worst case "
        f"({num_txs} vs {intended_max_txs})"
    )
    # Sanity: the senders collectively pay more floor than a whole block of
    # gas, yet the block is under its limit -- the floor is invisible to it.
    assert num_txs * floor > block_gas_limit
    assert block_gas_used <= block_gas_limit

    target = pre.deterministic_deploy_contract(deploy_code=Op.STOP)

    # Each transaction is a value-less data post that pays the full floor. A
    # separate funded sender per transaction keeps the nonces trivial.
    txs = [
        Transaction(
            to=target,
            data=data,
            gas_limit=floor,
            sender=pre.fund_eoa(10**20),
        )
        for _ in range(num_txs)
    ]

    blockchain_test(
        pre=pre,
        blocks=[
            Block(
                txs=txs,
                gas_limit=block_gas_limit,
                expected_gas_used=block_gas_used,
            )
        ],
        post={},
        genesis_environment=Environment(gas_limit=block_gas_limit),
    )
