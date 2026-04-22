"""
abstract: BloatNet single-opcode benchmark cases for state-related operations.

   These tests focus on individual EVM opcodes (SLOAD, SSTORE) to measure
   their performance when accessing many storage slots across pre-deployed
   contracts. Unlike multi-opcode tests, these isolate single operations
   to benchmark specific state-handling bottlenecks.
"""

from enum import Enum, auto
from functools import partial
from math import ceil, floor
from typing import Callable, Generator, List

import pytest
from execution_testing import (
    EOA,
    AccessList,
    Address,
    Alloc,
    AuthorizationTuple,
    BalAccountExpectation,
    BalNonceChange,
    BalStorageSlot,
    BenchmarkTestFiller,
    Block,
    BlockAccessListExpectation,
    Bytecode,
    CreatePreimageLayout,
    Fork,
    Hash,
    IteratingBytecode,
    JumpLoopGenerator,
    Op,
    SequentialAddressLayout,
    Storage,
    TestPhaseManager,
    Transaction,
    While,
    keccak256,
)
from execution_testing.base_types.base_types import Number

from tests.benchmark.stateful.helpers import (
    APPROVE_SELECTOR,
    BALANCEOF_SELECTOR,
    CacheStrategy,
    build_cache_strategy_blocks,
)

REFERENCE_SPEC_GIT_PATH = "DUMMY/bloatnet.md"
REFERENCE_SPEC_VERSION = "1.0"

# keccak256("random") for non-existing slots, masked as address,
# Solidity does input checks on the size and throws if we input
# something different than an address
START_SLOT = (
    0xA4896A3F93BF4BF58378E579F3CF193BB4AF1022AF7D2089F37D8BAE7157B85F
    % (2**160)
)


def _max_sloads_per_tx(tx_gas_limit: int, fork: Fork) -> int:
    """
    Conservative upper bound on cold SLOADs that fit in a max-gas tx.

    Derived from the cold SLOAD cost (EIP-2929: 2100 gas) and used by
    the bloated SLOAD benchmarks both as the inter-tx offset stride
    (to keep consecutive txs' SLOAD ranges disjoint) and as the
    per-target storage pre-load count.
    """
    cold_sload_cost = Op.SLOAD(key_warm=False).gas_cost(fork)
    return tx_gas_limit // cold_sload_cost


@pytest.mark.repricing
def test_mainnet_prestate(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    gas_benchmark_value: int,
    tx_gas_limit: int,
) -> None:
    sender = pre.fund_eoa()
    blocks = []
    with TestPhaseManager.setup():
        # account bloater range 0x1000 -> 0x1000 + 150_000
        accounts = 150_000
        current_addr = 0x1000
        account_bloat_txs_per_block = floor(60_000_000 / 21000)
        blocks_needed = ceil(accounts / account_bloat_txs_per_block)
        for _ in range(blocks_needed):
            txs = []
            for _ in range(account_bloat_txs_per_block):
                txs.append(Transaction(
                    to=current_addr,
                    sender=sender,
                    value=1
                ))
                current_addr = current_addr + 1
            blocks.append(Block(txs=txs))

        addr = pre.deploy_contract(code=Bytecode(bytes.fromhex("5f541515600b5760015f555b5f54805b818155600101906001019061ffff5a11600f575f55"), popped_stack_items=0, pushed_stack_items=0))
        auth = pre.stub_eoa("bloated_eoa_10GB")

        blocks.append(Block(txs=[Transaction(
            gas_limit=100_000,
            to=auth,
            value=1,
            sender=sender,
            authorization_list=[
                AuthorizationTuple(
                    chain_id=0,
                    address=addr,
                    nonce=auth.nonce,
                    signer=auth,
                ),
            ],
        )]))
        auth.nonce = Number(auth.nonce + 1)

        txs = 890020
        txs_per_block = 4
        blocks_needed = ceil(txs / txs_per_block)

        # gas limit necessary 67108864 ( 4 * (2 **24))
        for _ in range(blocks_needed):
            txs = []
            for _ in range(txs_per_block):
                txs.append(Transaction(
                    gas_limit=tx_gas_limit,
                    to=auth,
                    sender=sender,
                    value=1
                ))
            blocks.append(Block(txs=txs))
    benchmark_test(
        pre=pre,
        blocks=blocks,
        skip_gas_used_validation=True,
        expected_receipt_status=True,
    )