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


def _sender_generator(
    pre: Alloc, distinct_senders: bool
) -> Generator[EOA, None, None]:
    """
    Yield one sender per tx.

    In distinct mode, yields a fresh EOA per call. Otherwise, yields
    the same shared sender for every call. Used by the bloated SLOAD
    benchmarks so the BAL builder can group nonce changes by sender
    uniformly regardless of mode.
    """
    sender = pre.fund_eoa()
    while True:
        yield sender if not distinct_senders else pre.fund_eoa()


def delegate_with_calldata(
    pre: Alloc,
    authority: EOA,
    address: Address,
    calldata: Hash,
) -> Transaction:
    """
    Create a tx that delegates the authority and calls it with calldata.

    The delegated code determines what happens with the calldata.
    The authority nonce is incremented in-place.
    """
    tx = Transaction(
        gas_limit=100_000,
        to=authority,
        value=0,
        data=calldata,
        sender=pre.fund_eoa(),
        authorization_list=[
            AuthorizationTuple(
                chain_id=0,
                address=address,
                nonce=authority.nonce,
                signer=authority,
            ),
        ],
    )
    authority.nonce = Number(authority.nonce + 1)
    return tx


def run_bloated_eoa_benchmark(
    *,
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
    tx_gas_limit: int,
    authority: EOA,
    existing_slots: bool,
    runtime_code: Bytecode,
    cache_strategy: CacheStrategy,
) -> None:
    """
    Run a bloated-EOA benchmark with the given runtime delegation code.

    Handles authority setup, slot 0 initialization, delegation to
    runtime code, benchmark tx generation, and test invocation.
    """
    slot_0_value = Hash(1) if existing_slots else Hash(START_SLOT)

    setter_address = pre.deploy_contract(code=Op.SSTORE(0, Op.CALLDATALOAD(0)))
    runtime_address = pre.deploy_contract(code=runtime_code)

    init_tx = delegate_with_calldata(
        pre, authority, setter_address, slot_0_value
    )
    runtime_tx = delegate_with_calldata(
        pre, authority, runtime_address, Hash(0)
    )

    blocks: list[Block] = [Block(txs=[init_tx, runtime_tx])]

    gas_available = gas_benchmark_value
    intrinsic_gas = fork.transaction_intrinsic_cost_calculator()()
    sender = pre.fund_eoa()

    txs: list[Transaction] = []
    with TestPhaseManager.execution():
        while gas_available >= intrinsic_gas:
            tx_gas = min(gas_available, tx_gas_limit)
            txs.append(
                Transaction(
                    gas_limit=tx_gas,
                    to=authority,
                    sender=sender,
                )
            )
            gas_available -= tx_gas

    cache_txs: list[Transaction] = []
    if cache_strategy == CacheStrategy.CACHE_PREVIOUS_BLOCK:
        with TestPhaseManager.setup():
            cache_sender = pre.fund_eoa()
            for tx in txs:
                cache_txs.append(
                    Transaction(
                        gas_limit=tx.gas_limit,
                        to=authority,
                        sender=cache_sender,
                    )
                )

    blocks += build_cache_strategy_blocks(cache_strategy, txs, cache_txs)

    benchmark_test(
        pre=pre,
        blocks=blocks,
        skip_gas_used_validation=True,
        expected_receipt_status=True,
    )


@pytest.mark.repricing
def test_mainnet_prestate(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
    tx_gas_limit: int,
    token_name: str,
    existing_slots: bool,
    cache_strategy: CacheStrategy,
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

        addr = pre.deploy_contract(code="0x5f541515600b5760015f555b5f54805b818155600101906001019061ffff5a11600f575f55")
        auth = pre.stub_eoa("bloated_eoa_10GB")
        delegate_with_calldata(pre, auth, addr, Hash(0))
        txs = 890020
        txs_per_block = 4
        blocks_needed = ceil(txs / txs_per_block)

        sender = pre.fund_eoa()
        # gas limit necessary 67108864 ( 4 * (2 **24))
        for _ in range(blocks_needed):
            txs = []
            for _ in range(txs_per_block):
                txs.append(Transaction(
                    gas_limit=tx_gas_limit,
                    to=auth,
                    sender=sender,
                ))
            blocks.append(Block(txs=txs))
    benchmark_test(
        pre=pre,
        blocks=blocks,
        skip_gas_used_validation=True,
        expected_receipt_status=True,
    )