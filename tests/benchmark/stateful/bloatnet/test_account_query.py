"""
Benchmark operations that require querying the account state, either on the
current executing account or on a target account.

Supported Opcodes:
- SELFBALANCE
- CODESIZE
- CODECOPY
- EXTCODESIZE
- EXTCODEHASH
- EXTCODECOPY
- BALANCE
"""

from enum import Enum, auto
from typing import Any

import pytest
from execution_testing import (
    DETERMINISTIC_FACTORY_ADDRESS,
    Account,
    Alloc,
    BenchmarkTestFiller,
    Block,
    Bytecode,
    Create2PreimageLayout,
    Fork,
    Hash,
    IteratingBytecode,
    JumpLoopGenerator,
    Op,
    SequentialAddressLayout,
    TestPhaseManager,
    Transaction,
    While,
    keccak256,
)

from tests.benchmark.stateful.helpers import CacheStrategy


@pytest.mark.repricing(
    empty_code=True,
    initial_balance=True,
    initial_storage=True,
)
@pytest.mark.parametrize(
    "opcode",
    [
        Op.BALANCE,
        Op.EXTCODESIZE,
        Op.EXTCODEHASH,
        Op.CALL,
        Op.CALLCODE,
        Op.DELEGATECALL,
        Op.STATICCALL,
    ],
)
@pytest.mark.parametrize(
    "empty_code",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "initial_balance",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "initial_storage",
    [
        True,
        False,
    ],
)
def test_ext_account_query_warm(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    opcode: Op,
    empty_code: bool,
    initial_balance: bool,
    initial_storage: bool,
) -> None:
    """
    Test running a block with as many stateful opcodes doing warm access
    for an account.
    """
    # Setup
    post = {}

    # Case 1: Completely empty account (no balance, no storage, no code)
    if not initial_balance and not initial_storage and empty_code:
        target_addr = pre.nonexistent_account()
    # Case 2: EOA with optional balance and storage
    elif empty_code:
        eoa_kwargs: dict[str, Any] = {}
        if initial_balance:
            eoa_kwargs["amount"] = 100
        if initial_storage:
            eoa_kwargs["storage"] = {0: 0x1337}
        target_addr = pre.fund_eoa(**eoa_kwargs)
    # Case 3: Contract with optional balance and storage
    else:
        contract_kwargs: dict[str, Any] = {"code": Op.STOP + Op.JUMPDEST * 100}
        if initial_balance:
            contract_kwargs["balance"] = 100
        if initial_storage:
            contract_kwargs["storage"] = {0: 0x1337}
        target_addr = pre.deploy_contract(**contract_kwargs)
        post[target_addr] = Account(**contract_kwargs)

    benchmark_test(
        target_opcode=opcode,
        post=post,
        code_generator=JumpLoopGenerator(
            setup=Op.MSTORE(0, target_addr),
            attack_block=Op.POP(opcode(address=Op.MLOAD(0))),
        ),
    )


class AccountMode(Enum):
    """Target Account Mode."""

    EXISTING_CONTRACT_MINIMAL = auto()
    EXISTING_CONTRACT_SAME = auto()
    EXISTING_CONTRACT_DIFF = auto()
    EXISTING_EOA = auto()
    NON_EXISTING_ACCOUNT = auto()


def account_access_params() -> list:
    """Generate (opcode, value_sent, account_mode) triples."""
    params = []

    for mode in AccountMode:
        for op in [Op.CALL, Op.CALLCODE]:
            params.append(pytest.param(op, 0, mode))
            params.append(pytest.param(op, 1, mode))

        for op in [Op.BALANCE, Op.STATICCALL, Op.DELEGATECALL]:
            params.append(pytest.param(op, 0, mode))

    for op in [Op.EXTCODECOPY, Op.EXTCODESIZE, Op.EXTCODEHASH]:
        for mode in [
            AccountMode.EXISTING_CONTRACT_MINIMAL,
            AccountMode.EXISTING_CONTRACT_SAME,
            AccountMode.EXISTING_CONTRACT_DIFF,
            AccountMode.NON_EXISTING_ACCOUNT,
        ]:
            params.append(pytest.param(op, 0, mode))

    return params


def build_existing_contract_initcode(
    fork: Fork, account_mode: AccountMode
) -> Bytecode:
    """
    Build the initcode for an existing contract.
    """
    max_code_size = fork.max_code_size()

    # MCOPY fills MEM[0:0x8000] with JUMPDEST.
    # Runtime only uses MEM[0:0x6000].
    code = Op.MSTORE(0, bytes(Op.JUMPDEST * 32))
    for size in (1 << s for s in range(5, 15)):
        code += Op.MCOPY(size, 0, size)

    if account_mode == AccountMode.EXISTING_CONTRACT_DIFF:
        code += Op.MSTORE(0, Op.ADDRESS)
    else:
        code += Op.MSTORE8(0, 0)
    code += Op.RETURN(0, max_code_size)

    return code


@pytest.mark.repricing
@pytest.mark.parametrize("cache_strategy", list(CacheStrategy))
@pytest.mark.parametrize(
    "opcode,value_sent,account_mode", account_access_params()
)
def test_account_access(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    opcode: Op,
    value_sent: int,
    gas_benchmark_value: int,
    fixed_opcode_count: int | None,
    account_mode: AccountMode,
    cache_strategy: CacheStrategy,
) -> None:
    """Benchmark account access with caching strategies."""
    address_retriever: Bytecode
    # Read start_iteration from calldata so that when transactions are
    # split across gas limits, each transaction continues from where
    # the previous one left off instead of re-targeting the same accounts.
    calldataload_start = Op.CALLDATALOAD(0)
    if account_mode == AccountMode.EXISTING_CONTRACT_MINIMAL:
        # initcode returns a single zero byte (STOP) as the runtime.
        init_code = Op.PUSH1(1) + Op.PUSH1(0) + Op.RETURN
        address_retriever = Create2PreimageLayout(
            factory_address=DETERMINISTIC_FACTORY_ADDRESS,
            salt=calldataload_start,
            init_code_hash=keccak256(bytes(init_code)),
        )
        increment_op = address_retriever.increment_salt_op()
    elif account_mode in (
        AccountMode.EXISTING_CONTRACT_SAME,
        AccountMode.EXISTING_CONTRACT_DIFF,
    ):
        init_code = build_existing_contract_initcode(fork, account_mode)
        address_retriever = Create2PreimageLayout(
            factory_address=DETERMINISTIC_FACTORY_ADDRESS,
            salt=calldataload_start,
            init_code_hash=keccak256(bytes(init_code)),
        )
        increment_op = address_retriever.increment_salt_op()
    elif account_mode == AccountMode.EXISTING_EOA:
        # Spamoor EOA creator (https://github.com/CPerezz/spamoor/pull/12)
        # created these accounts on bloatnet with these values (are also the
        # defaults of SequentialAddressLayout)
        address_retriever = SequentialAddressLayout(
            starting_address=Op.ADD(0x1000, calldataload_start),
            increment=1,
        )
        increment_op = address_retriever.increment_address_op()
    else:
        address_retriever = SequentialAddressLayout(
            starting_address=Op.ADD(keccak256(b"random"), calldataload_start),
            increment=1,
        )
        increment_op = address_retriever.increment_address_op()

    setup_code: Bytecode = address_retriever

    cache_op = (
        Op.POP(
            Op.BALANCE(
                address=address_retriever.address_op(),
                # Gas accounting
                address_warm=False,
            )
        )
        if cache_strategy == CacheStrategy.CACHE_TX
        else Bytecode()
    )

    access_warm = cache_strategy == CacheStrategy.CACHE_TX

    if opcode == Op.EXTCODECOPY:
        attack_call = opcode(
            address=address_retriever.address_op(),
            size=1024,
            # Gas accounting
            address_warm=access_warm,
        )
    elif opcode in (Op.CALL, Op.CALLCODE):
        attack_call = Op.POP(
            opcode(
                address=address_retriever.address_op(),
                value=value_sent,
                # Gas accounting
                address_warm=access_warm,
                value_transfer=value_sent > 0,
                account_new=value_sent > 0
                and account_mode == AccountMode.NON_EXISTING_ACCOUNT,
            )
        )
    elif opcode in (Op.STATICCALL, Op.DELEGATECALL):
        attack_call = Op.POP(
            opcode(
                address=address_retriever.address_op(),
                # Gas accounting
                address_warm=access_warm,
            )
        )
    else:
        # BALANCE, EXTCODESIZE, EXTCODEHASH
        attack_call = Op.POP(
            opcode(
                address=address_retriever.address_op(),
                # Gas accounting
                address_warm=access_warm,
            )
        )

    loop_code = While(
        body=cache_op + attack_call + increment_op,
        condition=Op.GT(Op.GAS, 0x9000) if value_sent > 0 else None,
    )

    attack_code = IteratingBytecode(
        setup=setup_code,
        iterating=loop_code,
        # Since the target contract is guaranteed to have a STOP as the first
        # instruction, we can use a STOP as the iterating subcall code.
        iterating_subcall=Op.STOP,
    )

    # Calldata generator for each transaction of the iterating bytecode.
    def calldata(iteration_count: int, start_iteration: int) -> bytes:
        del iteration_count
        return Hash(start_iteration)

    attack_address = pre.deploy_contract(code=attack_code, balance=10**21)

    post: dict = {}
    cache_txs = []

    with TestPhaseManager.execution():
        attack_sender = pre.fund_eoa()
        if fixed_opcode_count is not None:
            attack_txs = list(
                attack_code.transactions_by_total_iteration_count(
                    fork=fork,
                    total_iterations=int(fixed_opcode_count * 1000),
                    sender=attack_sender,
                    to=attack_address,
                    calldata=calldata,
                )
            )
        else:
            attack_txs = list(
                attack_code.transactions_by_gas_limit(
                    fork=fork,
                    gas_limit=gas_benchmark_value,
                    sender=attack_sender,
                    to=attack_address,
                    calldata=calldata,
                )
            )

    if cache_strategy == CacheStrategy.CACHE_PREVIOUS_BLOCK:
        with TestPhaseManager.setup():
            cache_sender = pre.fund_eoa()
            for tx in attack_txs:
                cache_txs.append(
                    Transaction(
                        gas_limit=tx.gas_limit,
                        data=tx.data,
                        to=attack_address,
                        sender=cache_sender,
                    )
                )

    blocks = (
        [Block(txs=attack_txs)]
        if cache_strategy != CacheStrategy.CACHE_PREVIOUS_BLOCK
        else [Block(txs=cache_txs), Block(txs=attack_txs)]
    )

    benchmark_test(
        pre=pre,
        post=post,
        blocks=blocks,
        target_opcode=opcode,
        skip_gas_used_validation=True,
        expected_receipt_status=1,
    )
