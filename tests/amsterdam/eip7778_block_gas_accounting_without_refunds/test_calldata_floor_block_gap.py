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

To fill the block, transactions are packed largest-first: the per-transaction
inclusion check reserves each transaction's floor against the block, but only
its cheaper pre-floor cost is added to ``block_gas_used``. Progressively
smaller data posts mop up the reservation the large ones could not use, so the
block's counted gas is driven to ~95% of the limit while it carries several
times the calldata a correctly floored block limit would allow
(``block_gas_limit // 64`` bytes).

Measured results (Amsterdam, ``block_gas_limit = 2 * TX_MAX_GAS_LIMIT =
33,554,432``; floor budget ``33,554,432 // 64 = 524,288`` bytes = 512 KiB).
Each block below is accepted as valid:

| variant  | txs | block gas_used / limit    | calldata in block   | oversize |
|----------|-----|---------------------------|---------------------|----------|
| nonzero  | 12  | 31,900,896 / 33,554,432   | 1,982,556 B (~1.9M) | 3.8x     |
| zero     | 50  | 31,896,068 / 33,554,432   | 7,786,517 B (~7.4M) | 14.9x    |

Both blocks report ``gas_used`` at 95.1% of the limit, yet the calldata floors
the senders collectively pay -- 127,063,584 gas (nonzero) and 499,087,088 gas
(zero) -- are 3.8x and 14.9x the block gas limit, none of which the block-level
accounting counts. Re-run to verify:

    uv run fill tests/amsterdam -k
    test_calldata_floor_invisible_to_block_limit --fork Amsterdam
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

# Fill the block's counted gas to at least this fraction of the limit.
FILL_TARGET_PERCENT = 95
# Cap individual transactions to keep the fixture size bounded.
MAX_CALLDATA_LEN = 260_000


@pytest.mark.parametrize(
    "calldata_byte",
    [
        # 16 gas/byte on the standard schedule, 64 at the floor: a 4x gap.
        pytest.param(0xFF, id="nonzero"),
        # 4 gas/byte standard vs the same 64 at the floor: a 16x gap.
        pytest.param(0x00, id="zero"),
    ],
)
@pytest.mark.valid_from("Amsterdam")
def test_calldata_floor_invisible_to_block_limit(
    blockchain_test: BlockchainTestFiller,
    pre: Alloc,
    fork: Fork,
    calldata_byte: int,
) -> None:
    """
    Fill a block with floor-bound data posts beyond the floor's intended cap.

    The block is expected to be *valid*: its ``block_gas_used`` counts only the
    pre-floor execution cost and is driven to ~95% of the limit, while the sum
    of the calldata floors the senders pay is several times the block gas
    limit and the calldata carried far exceeds ``block_gas_limit // 64`` bytes.
    """
    tx_gas_limit_cap = fork.transaction_gas_limit_cap()
    assert tx_gas_limit_cap is not None, "fork must cap per-tx gas (EIP-7825)"

    # A block that holds two maximum-size transactions (EIP-7825 cap x2).
    block_gas_limit = 2 * tx_gas_limit_cap

    intrinsic_calc = fork.transaction_intrinsic_cost_calculator()
    floor_calc = fork.transaction_data_floor_cost_calculator()

    # The floor is linear in calldata length: base + floor_per_byte * length.
    base_floor = floor_calc(data=b"")
    floor_per_byte = floor_calc(data=b"\x00") - base_floor

    # Greedily pack data posts largest-first. Each transaction reserves its
    # floor (min(TX_MAX_GAS_LIMIT, gas_limit)) against the remaining block gas,
    # but only its pre-floor execution cost is added to ``block_gas_used``. As
    # the block fills, the reservation shrinks, so later transactions carry
    # less calldata -- exactly the surplus the accounting gap allows.
    fill_target = block_gas_limit * FILL_TARGET_PERCENT // 100

    calldata_lengths = []
    block_gas_used = 0
    while block_gas_used < fill_target:
        remaining_reserve = block_gas_limit - block_gas_used
        max_floor = min(tx_gas_limit_cap, remaining_reserve)
        length = min(
            (max_floor - base_floor) // floor_per_byte, MAX_CALLDATA_LEN
        )
        if length <= 0:
            break
        data = bytes([calldata_byte]) * length
        assert floor_calc(data=data) <= max_floor
        block_gas_used += intrinsic_calc(
            calldata=data,
            return_cost_deducted_prior_execution=True,
        )
        calldata_lengths.append(length)

    assert block_gas_used >= fill_target, "block should be nearly full"
    assert block_gas_used <= block_gas_limit

    # The block carries more calldata than the floor rate would let it pay for.
    total_calldata = sum(calldata_lengths)
    floor_budget = block_gas_limit // floor_per_byte
    assert total_calldata > floor_budget

    # The floors the senders collectively pay dwarf a whole block of gas, yet
    # the block stays under its limit -- the floor is invisible to it.
    total_floor_paid = sum(
        base_floor + floor_per_byte * length for length in calldata_lengths
    )
    assert total_floor_paid > block_gas_limit

    target = pre.deterministic_deploy_contract(deploy_code=Op.STOP)

    # Each transaction is a value-less data post that pays the full floor. A
    # separate funded sender per transaction keeps the nonces trivial.
    txs = [
        Transaction(
            to=target,
            data=bytes([calldata_byte]) * length,
            gas_limit=base_floor + floor_per_byte * length,
            sender=pre.fund_eoa(10**20),
        )
        for length in calldata_lengths
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
