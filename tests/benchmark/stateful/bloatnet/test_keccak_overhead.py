"""Keccak overhead baseline for the account-access benchmark."""

from enum import Enum, auto

import pytest
from execution_testing import (
    DETERMINISTIC_FACTORY_ADDRESS,
    Alloc,
    BenchmarkTestFiller,
    Block,
    Bytecode,
    Create2PreimageLayout,
    Fork,
    Hash,
    IteratingBytecode,
    Op,
    TestPhaseManager,
    While,
    keccak256,
)

from tests.benchmark.stateful.helpers import CacheStrategy

REFERENCE_SPEC_GIT_PATH = "DUMMY/bloatnet.md"
REFERENCE_SPEC_VERSION = "1.0"


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


class AccountMode(Enum):
    """Target Account Mode."""

    EXISTING_CONTRACT_MINIMAL = auto()
    EXISTING_CONTRACT_SAME = auto()
    EXISTING_CONTRACT_DIFF = auto()
    EXISTING_EOA = auto()
    NON_EXISTING_ACCOUNT = auto()


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


def _account_access_loop(
    *,
    opcode: Op,
    value_sent: int,
    cache_strategy: CacheStrategy,
    address_retriever: Create2PreimageLayout,
    increment_op: Bytecode,
) -> Bytecode:
    """Rebuild test_account_access's EXISTING_CONTRACT loop (count only)."""
    cache_op = (
        Op.POP(
            Op.BALANCE(
                address=address_retriever.address_op(),
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
            address_warm=access_warm,
        )
    elif opcode in (Op.CALL, Op.CALLCODE):
        attack_call = Op.POP(
            opcode(
                address=address_retriever.address_op(),
                value=value_sent,
                # gas accounting
                address_warm=access_warm,
                value_transfer=value_sent > 0,
                account_new=False,
            )
        )
    else:
        attack_call = Op.POP(
            opcode(
                address=address_retriever.address_op(),
                address_warm=access_warm,
            )
        )

    return While(
        body=cache_op + attack_call + increment_op,
        condition=Op.GT(Op.GAS, 0x9000) if value_sent > 0 else None,
    )


@pytest.mark.repricing
@pytest.mark.parametrize("cache_strategy", list(CacheStrategy))
@pytest.mark.parametrize(
    "opcode,value_sent,account_mode", account_access_params()
)
def test_account_access_keccak(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    opcode: Op,
    value_sent: int,
    gas_benchmark_value: int,
    account_mode: AccountMode,
    cache_strategy: CacheStrategy,
) -> None:
    """
    Benchmark the keccak256 work of the account-access address derivation.

    Compute the iteration count test_account_access[EXISTING_CONTRACT] runs
    for this gas budget, then execute only the CREATE2 address derivation
    (SHA3 over 0xff ++ factory ++ salt ++ keccak(initcode)) for that count.
    Same factory, initcode and salt sequence as the full benchmark, so the
    keccak count and input match: one per iteration, two for cache_tx.
    """
    calldataload_start = Op.CALLDATALOAD(0)

    initcode = Bytecode()
    if account_mode == AccountMode.EXISTING_CONTRACT_MINIMAL:
        initcode += Op.PUSH1(1) + Op.PUSH1(0) + Op.RETURN
    elif account_mode in (
        AccountMode.EXISTING_CONTRACT_SAME,
        AccountMode.EXISTING_CONTRACT_DIFF,
    ):
        initcode += build_existing_contract_initcode(fork, account_mode)

    address_retriever = Create2PreimageLayout(
        factory_address=DETERMINISTIC_FACTORY_ADDRESS,
        salt=calldataload_start,
        init_code_hash=keccak256(bytes(initcode)),
    )
    increment_op = address_retriever.increment_salt_op()

    def calldata(iteration_count: int, start_iteration: int) -> bytes:
        del iteration_count
        return Hash(start_iteration)

    account_access_code = IteratingBytecode(
        setup=address_retriever,
        iterating=_account_access_loop(
            opcode=opcode,
            value_sent=value_sent,
            cache_strategy=cache_strategy,
            address_retriever=address_retriever,
            increment_op=increment_op,
        ),
        iterating_subcall=Op.STOP,
    )
    total_iterations = sum(
        account_access_code.tx_iterations_by_gas_limit(
            fork=fork,
            gas_limit=gas_benchmark_value,
            calldata=calldata,
        )
    )

    keccak_op = Op.POP(address_retriever.address_op())
    if cache_strategy == CacheStrategy.CACHE_TX:
        keccak_op = keccak_op * 2
    keccak_code = IteratingBytecode(
        setup=address_retriever,
        iterating=While(body=keccak_op + increment_op),
    )

    attack_address = pre.deploy_contract(code=keccak_code)

    with TestPhaseManager.execution():
        attack_sender = pre.fund_eoa()
        attack_txs = list(
            keccak_code.transactions_by_total_iteration_count(
                fork=fork,
                total_iterations=total_iterations,
                sender=attack_sender,
                to=attack_address,
                calldata=calldata,
            )
        )

    benchmark_test(
        pre=pre,
        blocks=[Block(txs=attack_txs)],
        target_opcode=Op.SHA3,
        skip_gas_used_validation=True,
        expected_receipt_status=1,
    )
