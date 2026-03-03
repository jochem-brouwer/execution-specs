"""
abstract: BloatNet bench cases extracted from https://hackmd.io/9icZeLN7R0Sk5mIjKlZAHQ.

   The idea of all these tests is to stress client implementations to find out
   where the limits of processing are focusing specifically on state-related
   operations.
"""

from typing import Dict, List

import pytest
from execution_testing import (
    AccessList,
    Account,
    Alloc,
    BenchmarkTestFiller,
    Block,
    Bytecode,
    Conditional,
    Create2PreimageLayout,
    CreatePreimageLayout,
    Fork,
    Hash,
    IteratingBytecode,
    Op,
    ParameterSet,
    SequentialAddressLayout,
    TestPhaseManager,
    Transaction,
    While,
)

from tests.benchmark.compute.helpers import (
    AccountQueryMode,
    ContractDeploymentTransaction,
    CustomSizedContractFactory,
)
from tests.benchmark.stateful.helpers import (
    APPROVE_SELECTOR,
    BALANCEOF_SELECTOR,
    MIXED_TOKENS,
)

REFERENCE_SPEC_GIT_PATH = "DUMMY/bloatnet.md"
REFERENCE_SPEC_VERSION = "1.0"


# BLOATNET ARCHITECTURE:
#
#   [Initcode Contract]        [Factory Contract]              [24KB Contracts]
#         (9.5KB)                    (116B)                     (N x 24KB each)
#           │                          │                              │
#           │  EXTCODECOPY             │   CREATE2(salt++)            │
#           └──────────────►           ├──────────────────►     Contract_0
#                                      ├──────────────────►     Contract_1
#                                      ├──────────────────►     Contract_2
#                                      └──────────────────►     Contract_N
#
#   [Attack Contract] ──STATICCALL──► [Factory.getConfig()]
#           │                              returns: (N, hash)
#           └─► Loop(i=0 to N):
#                 1. Generate CREATE2 addr: keccak256(0xFF|factory|i|hash)[12:]
#                 2. BALANCE(addr)    → 2600 gas (cold access)
#                 3. EXTCODESIZE(addr) → 100 gas (warm access)
#
# HOW IT WORKS:
#   1. Factory uses EXTCODECOPY to load initcode, avoiding PC-relative jumps
#   2. Each CREATE2 deployment produces unique 24KB bytecode (via ADDRESS)
#   3. All contracts share same initcode hash for deterministic addresses
#   4. Attack rapidly accesses all contracts, stressing client's state handling


@pytest.mark.parametrize(
    "balance_first",
    [True, False],
    ids=["balance_extcodesize", "extcodesize_balance"],
)
def test_bloatnet_balance_extcodesize(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
    tx_gas_limit: int,
    balance_first: bool,
) -> None:
    """Benchmark BALANACE and EXTCODESIZE combination on bloatnet."""
    # Stub Account
    factory_address = pre.deploy_contract(
        code=Bytecode(),  # Required parameter, but will be ignored for stubs
        stub="bloatnet_factory",
    )

    # Contract Construction
    setup = Bytecode()

    setup += Conditional(
        condition=Op.STATICCALL(
            gas=Op.GAS,
            address=factory_address,
            args_offset=0,
            args_size=0,
            ret_offset=96,
            ret_size=64,
            # gas accounting
            address_warm=False,
            old_memory_size=0,
            new_memory_size=160,
        ),
        if_false=Op.INVALID,
    )

    create2_preimage = Create2PreimageLayout(
        factory_address=factory_address,
        salt=Op.CALLDATALOAD(32),
        init_code_hash=Op.MLOAD(128),
        old_memory_size=160,
    )

    setup += create2_preimage
    setup += Op.CALLDATALOAD(0)  # [num_contract]

    balance_op = Op.POP(Op.BALANCE)
    extcodesize_op = Op.POP(Op.EXTCODESIZE)
    benchmark_ops = (
        (balance_op + extcodesize_op)
        if balance_first
        else (extcodesize_op + balance_op)
    )

    loop = While(
        body=(
            create2_preimage.address_op()
            + Op.DUP1
            + benchmark_ops
            + create2_preimage.increment_salt_op()
        ),
        condition=Op.PUSH1(1)  # [1, num_contract]
        + Op.SWAP1  # [num_contract, 1]
        + Op.SUB  # [num_contract-1]
        + Op.DUP1  # [num_contract-1, num_contract-1]
        + Op.ISZERO  # [num_contract-1==0, num_contract-1]
        + Op.ISZERO,  # [num_contract-1!=0, num_contract-1]
    )

    # Contract Deployment
    code = setup + loop
    attack_contract_address = pre.deploy_contract(code=code)

    # Gas Accounting
    setup_cost = setup.gas_cost(fork)
    loop_cost = loop.gas_cost(fork)
    intrinsic_gas = fork.transaction_intrinsic_cost_calculator()(
        calldata=b"\xff" * 64
    )

    # Attack Loop
    gas_remaining = gas_benchmark_value
    txs = []
    salt_offset = 0

    while gas_remaining > intrinsic_gas + setup_cost + loop_cost:
        gas_available = min(gas_remaining, tx_gas_limit)

        if gas_available < intrinsic_gas + setup_cost:
            break

        num_contract = (
            gas_available - intrinsic_gas - setup_cost
        ) // loop_cost

        if num_contract == 0:
            break

        calldata = Hash(num_contract) + Hash(salt_offset)

        txs.append(
            Transaction(
                gas_limit=gas_available,
                data=calldata,
                to=attack_contract_address,
                sender=pre.fund_eoa(),
            )
        )

        gas_remaining -= gas_available
        salt_offset += num_contract

    benchmark_test(
        pre=pre,
        blocks=[Block(txs=txs)],
    )


@pytest.mark.parametrize(
    "balance_first",
    [True, False],
    ids=["balance_extcodecopy", "extcodecopy_balance"],
)
def test_bloatnet_balance_extcodecopy(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
    tx_gas_limit: int,
    balance_first: bool,
) -> None:
    """Benchmark BALANACE and EXTCODECOPY combination on bloatnet."""
    # Stub Account
    factory_address = pre.deploy_contract(
        code=Bytecode(),  # Required parameter, but will be ignored for stubs
        stub="bloatnet_factory",
    )

    # Contract Construction
    setup = Bytecode()

    setup += Conditional(
        condition=Op.STATICCALL(
            gas=Op.GAS,
            address=factory_address,
            args_offset=0,
            args_size=0,
            ret_offset=96,
            ret_size=64,
            # gas accounting
            address_warm=False,
            old_memory_size=0,
            new_memory_size=160,
        ),
        if_false=Op.INVALID,
    )

    create2_preimage = Create2PreimageLayout(
        factory_address=factory_address,
        salt=Op.CALLDATALOAD(32),
        init_code_hash=Op.MLOAD(128),
        old_memory_size=160,
    )

    setup += create2_preimage
    setup += Op.CALLDATALOAD(0)  # [num_contract]

    max_contract_size = fork.max_code_size()

    balance_op = Op.POP(Op.BALANCE)
    extcodecopy_op = Op.POP(
        Op.EXTCODECOPY(
            address=Op.DUP4,
            destOffset=Op.ADD(Op.MLOAD(32), 96),
            offset=max_contract_size - 1,
            size=1,
        )
    )
    benchmark_ops = (
        (balance_op + extcodecopy_op)
        if balance_first
        else (extcodecopy_op + balance_op)
    )

    loop = While(
        body=(
            create2_preimage.address_op()
            + Op.DUP1
            + benchmark_ops
            + create2_preimage.increment_salt_op()
        ),
        condition=Op.PUSH1(1)  # [1, num_contract]
        + Op.SWAP1  # [num_contract, 1]
        + Op.SUB  # [num_contract-1]
        + Op.DUP1  # [num_contract-1, num_contract-1]
        + Op.ISZERO  # [num_contract-1==0, num_contract-1]
        + Op.ISZERO,  # [num_contract-1!=0, num_contract-1]
    )

    # Contract Deployment
    code = setup + loop
    attack_contract_address = pre.deploy_contract(code=code)

    # Gas Accounting
    setup_cost = setup.gas_cost(fork)
    loop_cost = loop.gas_cost(fork)
    intrinsic_gas = fork.transaction_intrinsic_cost_calculator()(
        calldata=b"\xff" * 64
    )

    # Attack Loop
    gas_remaining = gas_benchmark_value
    txs = []
    salt_offset = 0

    while gas_remaining > intrinsic_gas + setup_cost + loop_cost:
        gas_available = min(gas_remaining, tx_gas_limit)

        if gas_available < intrinsic_gas + setup_cost:
            break

        num_contract = (
            gas_available - intrinsic_gas - setup_cost
        ) // loop_cost

        if num_contract == 0:
            break

        calldata = Hash(num_contract) + Hash(salt_offset)

        txs.append(
            Transaction(
                gas_limit=gas_available,
                data=calldata,
                to=attack_contract_address,
                sender=pre.fund_eoa(),
            )
        )

        gas_remaining -= gas_available
        salt_offset += num_contract

    benchmark_test(
        pre=pre,
        blocks=[Block(txs=txs)],
    )


@pytest.mark.parametrize(
    "balance_first",
    [True, False],
    ids=["balance_extcodehash", "extcodehash_balance"],
)
def test_bloatnet_balance_extcodehash(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
    tx_gas_limit: int,
    balance_first: bool,
) -> None:
    """Benchmark BALANACE and EXTCODEHASH combination on bloatnet."""
    # Stub Account
    factory_address = pre.deploy_contract(
        code=Bytecode(),  # Required parameter, but will be ignored for stubs
        stub="bloatnet_factory",
    )

    # Contract Construction
    setup = Bytecode()

    setup += Conditional(
        condition=Op.STATICCALL(
            gas=Op.GAS,
            address=factory_address,
            args_offset=0,
            args_size=0,
            ret_offset=96,
            ret_size=64,
            # gas accounting
            address_warm=False,
            old_memory_size=0,
            new_memory_size=160,
        ),
        if_false=Op.INVALID,
    )

    create2_preimage = Create2PreimageLayout(
        factory_address=factory_address,
        salt=Op.CALLDATALOAD(32),
        init_code_hash=Op.MLOAD(128),
        old_memory_size=160,
    )

    setup += create2_preimage
    setup += Op.CALLDATALOAD(0)  # [num_contract]

    balance_op = Op.POP(Op.BALANCE)
    extcodehash_op = Op.POP(Op.EXTCODEHASH)
    benchmark_ops = (
        (balance_op + extcodehash_op)
        if balance_first
        else (extcodehash_op + balance_op)
    )

    loop = While(
        body=(
            create2_preimage.address_op()
            + Op.DUP1
            + benchmark_ops
            + create2_preimage.increment_salt_op()
        ),
        condition=Op.PUSH1(1)  # [1, num_contract]
        + Op.SWAP1  # [num_contract, 1]
        + Op.SUB  # [num_contract-1]
        + Op.DUP1  # [num_contract-1, num_contract-1]
        + Op.ISZERO  # [num_contract-1==0, num_contract-1]
        + Op.ISZERO,  # [num_contract-1!=0, num_contract-1]
    )

    # Contract Deployment
    code = setup + loop
    attack_contract_address = pre.deploy_contract(code=code)

    # Gas Accounting
    setup_cost = setup.gas_cost(fork)
    loop_cost = loop.gas_cost(fork)
    intrinsic_gas = fork.transaction_intrinsic_cost_calculator()(
        calldata=b"\xff" * 64
    )

    # Attack Loop
    gas_remaining = gas_benchmark_value
    txs = []
    salt_offset = 0

    while gas_remaining > intrinsic_gas + setup_cost + loop_cost:
        gas_available = min(gas_remaining, tx_gas_limit)

        if gas_available < intrinsic_gas + setup_cost:
            break

        num_contract = (
            gas_available - intrinsic_gas - setup_cost
        ) // loop_cost

        if num_contract == 0:
            break

        calldata = Hash(num_contract) + Hash(salt_offset)

        txs.append(
            Transaction(
                gas_limit=gas_available,
                data=calldata,
                to=attack_contract_address,
                sender=pre.fund_eoa(),
            )
        )

        gas_remaining -= gas_available
        salt_offset += num_contract

    benchmark_test(
        pre=pre,
        blocks=[Block(txs=txs)],
    )


@pytest.mark.parametrize("token_name", MIXED_TOKENS)
@pytest.mark.parametrize(
    "sload_percent,sstore_percent",
    [
        pytest.param(10, 90, id="10-90"),
        pytest.param(30, 70, id="30-70"),
        pytest.param(50, 50, id="50-50"),
        pytest.param(70, 30, id="70-30"),
        pytest.param(90, 10, id="90-10"),
    ],
)
def test_mixed_sload_sstore(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
    tx_gas_limit: int,
    token_name: str,
    sload_percent: int,
    sstore_percent: int,
) -> None:
    """Benchmark mixed SLOAD/SSTORE on bloatnet."""
    # Stub Account
    erc20_address = pre.deploy_contract(
        code=Bytecode(),
        stub=f"test_mixed_sload_sstore_{token_name}",
    )

    # Contract Construction
    # MEM[0] = function selector
    # MEM[32] = address/slot offset (incremented each iteration)
    # MEM[64] = spender/amount for approve (copied from MEM[32])
    setup = (
        Op.MSTORE(
            0,
            BALANCEOF_SELECTOR,
            # gas accounting
            old_memory_size=0,
            new_memory_size=32,
        )
        + Op.MSTORE(
            32,
            Op.CALLDATALOAD(64),  # Slot Offset
            # gas accounting
            old_memory_size=32,
            new_memory_size=64,
        )
        + Op.CALLDATALOAD(0)  # [num_sload_calls]
    )

    sload_loop = While(
        body=Op.POP(
            Op.CALL(
                address=erc20_address,
                value=0,
                args_offset=28,
                args_size=36,
                ret_offset=0,
                ret_size=0,
                # gas accounting
                address_warm=True,
            )
        )
        + Op.MSTORE(32, Op.ADD(Op.MLOAD(32), 1)),
        condition=Op.PUSH1(1)  # [1, num_sload]
        + Op.SWAP1  # [num_sload, 1]
        + Op.SUB  # [num_sload-1]
        + Op.DUP1  # [num_sload-1, num_sload-1]
        + Op.ISZERO  # [num_sload-1==0, num_sload-1]
        + Op.ISZERO,  # [num_sload-1!=0, num_sload-1]
    )

    transition = (
        Op.POP  # remove 0 counter from sload loop
        + Op.MSTORE(0, APPROVE_SELECTOR)
        + Op.CALLDATALOAD(32)  # [num_sstore_calls]
    )

    sstore_loop = While(
        body=(
            Op.MSTORE(64, Op.MLOAD(32))
            + Op.POP(
                Op.CALL(
                    address=erc20_address,
                    value=0,
                    args_offset=28,
                    args_size=68,
                    ret_offset=0,
                    ret_size=0,
                    # gas accounting
                    address_warm=True,
                )
            )
            + Op.MSTORE(32, Op.ADD(Op.MLOAD(32), 1))
        ),
        condition=Op.PUSH1(1)  # [1, num_sstore]
        + Op.SWAP1  # [num_sstore, 1]
        + Op.SUB  # [num_sstore-1]
        + Op.DUP1  # [num_sstore-1, num_sstore-1]
        + Op.ISZERO  # [num_sstore-1==0, num_sstore-1]
        + Op.ISZERO,  # [num_sstore-1!=0, num_sstore-1]
    )

    # Contract Deployment
    code = setup + sload_loop + transition + sstore_loop
    attack_contract_address = pre.deploy_contract(code=code)

    # Gas Accounting
    setup_cost = setup.gas_cost(fork)
    sload_loop_cost = sload_loop.gas_cost(fork)
    transition_cost = transition.gas_cost(fork)
    sstore_loop_cost = sstore_loop.gas_cost(fork)

    access_list = [AccessList(address=erc20_address, storage_keys=[])]
    intrinsic_gas = fork.transaction_intrinsic_cost_calculator()(
        access_list=access_list,
        calldata=b"\xff" * 96,
    )

    # ERC20 balanceOf bytecode structure:
    sload_dispatch = (
        # Selector dispatch
        Op.PUSH4(BALANCEOF_SELECTOR)
        + Op.EQ
        + Op.JUMPI
        # Function body
        + Op.JUMPDEST
        + Op.CALLDATALOAD(4)
        + Op.MSTORE(0)
        + Op.MSTORE(32, 0)
        + Op.SHA3(
            0,
            64,
            # gas accounting
            data_size=64,
            old_memory_size=0,
            new_memory_size=64,
        )
        + Op.SLOAD
        # Return value
        + Op.MSTORE(0)
        + Op.RETURN(0, 32)
    )

    sload_dispatch_cost = sload_dispatch.gas_cost(fork)

    # ERC20 approve bytecode structure:
    sstore_dispatch = (
        # Selector dispatch
        Op.PUSH4(APPROVE_SELECTOR)
        + Op.EQ
        + Op.JUMPI
        # Function body
        + Op.JUMPDEST
        + Op.CALLDATALOAD(4)
        + Op.CALLDATALOAD(36)
        + Op.MSTORE(0, Op.CALLER)
        + Op.MSTORE(32, 1)
        + Op.SHA3(
            0,
            64,
            # gas accounting
            data_size=64,
            old_memory_size=0,
            new_memory_size=64,
        )
        + Op.MSTORE(32)
        + Op.MSTORE(0, Op.CALLDATALOAD(4))
        + Op.SHA3(
            0,
            64,
            # gas accounting
            data_size=64,
        )
        + Op.DUP1
        + Op.SLOAD.with_metadata(access_warm=False)
        + Op.POP
        + Op.SSTORE
        # Return true
        + Op.PUSH1(1)
        + Op.MSTORE(0)
        + Op.PUSH1(32)
        + Op.PUSH1(0)
        + Op.RETURN(0, 32)
    )

    sstore_dispatch_cost = sstore_dispatch.gas_cost(fork)

    sload_iter_cost = sload_loop_cost + sload_dispatch_cost
    sstore_iter_cost = sstore_loop_cost + sstore_dispatch_cost
    fixed_overhead = intrinsic_gas + setup_cost + transition_cost

    # Attack Loop
    gas_remaining = gas_benchmark_value
    txs = []
    slot_offset = 0

    while gas_remaining > fixed_overhead + sload_iter_cost + sstore_iter_cost:
        gas_available = min(gas_remaining, tx_gas_limit)

        if gas_available < fixed_overhead + sload_iter_cost + sstore_iter_cost:
            break

        available = gas_available - fixed_overhead
        sload_gas = (available * sload_percent) // 100
        sstore_gas = (available * sstore_percent) // 100

        num_sload = sload_gas // sload_iter_cost
        num_sstore = sstore_gas // sstore_iter_cost

        if num_sload == 0 or num_sstore == 0:
            break

        calldata = Hash(num_sload) + Hash(num_sstore) + Hash(slot_offset)

        txs.append(
            Transaction(
                gas_limit=gas_available,
                data=calldata,
                to=attack_contract_address,
                sender=pre.fund_eoa(),
                access_list=access_list,
            )
        )

        gas_remaining -= gas_available
        slot_offset += num_sload + num_sstore

    benchmark_test(
        pre=pre,
        blocks=[Block(txs=txs)],
    )


def generate_account_query_params() -> List[ParameterSet]:
    """
    Generate valid parameter combinations for test_account_query.

    Returns tuples of: (opcode, access_warm, mem_size, code_size, value_sent)
    """
    all_mem_sizes = [0, 32, 256, 1024]
    all_code_sizes = [0, 32, 256, 1024]
    all_access_warm = [True, False]
    all_value_sent = [0, 1]

    params = []

    # BALANCE, EXTCODESIZE, EXTCODEHASH:
    # only mem_size=0, code_size=0, value_sent=0
    for opcode in [Op.BALANCE, Op.EXTCODESIZE, Op.EXTCODEHASH]:
        for access_warm in all_access_warm:
            params.append(pytest.param(opcode, access_warm, 0, 0, 0))

    # EXTCODECOPY: all mem_size, all code_size, value_sent=0
    for access_warm in all_access_warm:
        for mem_size in all_mem_sizes:
            for code_size in all_code_sizes:
                params.append(
                    pytest.param(
                        Op.EXTCODECOPY, access_warm, mem_size, code_size, 0
                    )
                )
            # Add None (max_code_size) separately with custom ID
            params.append(
                pytest.param(
                    Op.EXTCODECOPY,
                    access_warm,
                    mem_size,
                    None,
                    0,
                    id=f"EXTCODECOPY-{access_warm}-{mem_size}-max_code_size-0",
                )
            )

    # CALL, CALLCODE: all mem_size, code_size=0, all value_sent
    for opcode in [Op.CALL, Op.CALLCODE]:
        for access_warm in all_access_warm:
            for mem_size in all_mem_sizes:
                for value_sent in all_value_sent:
                    params.append(
                        pytest.param(
                            opcode, access_warm, mem_size, 0, value_sent
                        )
                    )

    # STATICCALL, DELEGATECALL: all mem_size, code_size=0, value_sent=0
    for opcode in [Op.STATICCALL, Op.DELEGATECALL]:
        for access_warm in all_access_warm:
            for mem_size in all_mem_sizes:
                params.append(
                    pytest.param(opcode, access_warm, mem_size, 0, 0)
                )

    return params


@pytest.mark.repricing
@pytest.mark.parametrize(
    "account_query_mode",
    [
        AccountQueryMode.CREATE2_FACTORY,
        AccountQueryMode.CREATE_FACTORY,
        AccountQueryMode.SEQUENTIAL,
    ],
)
@pytest.mark.parametrize(
    "opcode,access_warm,mem_size,code_size,value_sent",
    generate_account_query_params(),
)
def test_account_query(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    opcode: Op,
    access_warm: bool,
    mem_size: int,
    code_size: int,
    value_sent: int,
    gas_benchmark_value: int,
    fixed_opcode_count: int | None,
    account_query_mode: AccountQueryMode,
) -> None:
    """Benchmark scenario of accessing max-code size bytecode."""
    attack_gas_limit = gas_benchmark_value

    # Create the max-sized fork-dependent contract factory.
    custom_sized_contract_factory = CustomSizedContractFactory(
        pre=pre, fork=fork, contract_size=code_size
    )
    factory_address = custom_sized_contract_factory.address()
    initcode = custom_sized_contract_factory.initcode

    # Prepare the attack iterating bytecode.
    # Setup is just placing the CREATE2 Preimage in memory.
    address_retriever: Bytecode
    if account_query_mode == AccountQueryMode.CREATE2_FACTORY:
        address_retriever = Create2PreimageLayout(
            factory_address=factory_address,
            salt=Op.CALLDATALOAD(0),
            init_code_hash=initcode.keccak256(),
        )
    elif account_query_mode == AccountQueryMode.CREATE_FACTORY:
        address_retriever = CreatePreimageLayout(
            sender_address=factory_address,
            nonce=1,
        )
    else:
        address_retriever = SequentialAddressLayout()

    setup_code: Bytecode = address_retriever

    if mem_size > 96:
        setup_code += Op.MSTORE8(
            mem_size - 1,
            0,
            # Gas accounting
            old_memory_size=96,
            new_memory_size=mem_size,
        )

    if opcode == Op.EXTCODECOPY:
        attack_call = Op.EXTCODECOPY(
            address=address_retriever.address_op(),
            dest_offset=0,
            size=mem_size,
            # Gas accounting
            data_size=mem_size,
            address_warm=access_warm,
        )
    elif opcode in (Op.CALL, Op.CALLCODE):
        # CALL and CALLCODE accept value parameter
        attack_call = Op.POP(
            opcode(
                address=address_retriever.address_op(),
                value=value_sent,
                args_size=mem_size,
                # Gas accounting
                address_warm=access_warm,
                new_memory_size=max(mem_size, 96),
            )
        )
    elif opcode in (Op.STATICCALL, Op.DELEGATECALL):
        # STATICCALL and DELEGATECALL don't have value parameter
        attack_call = Op.POP(
            opcode(
                address=address_retriever.address_op(),
                args_size=mem_size,
                # Gas accounting
                address_warm=access_warm,
                new_memory_size=max(mem_size, 96),
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

    if account_query_mode == AccountQueryMode.CREATE2_FACTORY:
        assert isinstance(address_retriever, Create2PreimageLayout)
        increment_op = address_retriever.increment_salt_op()
    elif account_query_mode == AccountQueryMode.CREATE_FACTORY:
        assert isinstance(address_retriever, CreatePreimageLayout)
        increment_op = address_retriever.increment_nonce_op()
    else:
        assert isinstance(address_retriever, SequentialAddressLayout)
        increment_op = address_retriever.increment_address_op()

    loop_code = While(
        body=attack_call + increment_op,
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
        # We only pass the start iteration index as calldata for this bytecode
        return Hash(start_iteration)

    # Access list generator for warm access tests.
    # When access_warm=True, include all contract addresses that will be
    # accessed in each transaction to warm them up via access list.
    # Note: This access list generation is very expensive due to the binary
    # search, which builds different access lists using the same elements
    # over and over. Caching the elements helps a bit.
    access_list_cache: Dict[int, AccessList] = {}

    def access_list_generator(
        iteration_count: int, start_iteration: int
    ) -> list[AccessList] | None:
        if not access_warm:
            return None
        return [
            access_list_cache.setdefault(
                i,
                AccessList(
                    address=custom_sized_contract_factory.created_contract_address(
                        salt=i
                    ),
                    storage_keys=[],
                ),
            )
            for i in range(start_iteration, start_iteration + iteration_count)
        ]

    attack_address = pre.deploy_contract(code=attack_code, balance=10**21)

    # Calculate the number of contracts to be targeted.
    if fixed_opcode_count is not None:
        # Fixed opcode count mode
        num_contracts = int(fixed_opcode_count * 1000)
    else:
        # Gas limit mode
        num_contracts = sum(
            attack_code.tx_iterations_by_gas_limit(
                fork=fork,
                gas_limit=attack_gas_limit,
                calldata=calldata,
                access_list=access_list_generator,
            )
        )

    # Deploy num_contracts via multiple txs (each capped by tx gas limit).
    post = {}
    with TestPhaseManager.setup():
        setup_sender = pre.fund_eoa()
        contracts_deployment_txs: List[ContractDeploymentTransaction] = []
        for contract_creating_tx in (
            custom_sized_contract_factory.transactions_by_total_contract_count(
                fork=fork,
                sender=setup_sender,
                contract_count=num_contracts,
            )
        ):
            contracts_deployment_txs.append(contract_creating_tx)
            if custom_sized_contract_factory.contract_size > 0:
                post[contract_creating_tx.deployed_contracts[-1]] = Account(
                    nonce=1
                )

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
                    access_list=access_list_generator,
                )
            )
        else:
            attack_txs = list(
                attack_code.transactions_by_gas_limit(
                    fork=fork,
                    gas_limit=attack_gas_limit,
                    sender=attack_sender,
                    to=attack_address,
                    calldata=calldata,
                    access_list=access_list_generator,
                )
            )
        total_gas_cost = sum(tx.gas_cost for tx in attack_txs)

    benchmark_test(
        pre=pre,
        post=post,
        blocks=[
            Block(txs=contracts_deployment_txs),
            Block(txs=attack_txs),
        ],
        target_opcode=opcode,
        expected_benchmark_gas_used=total_gas_cost,
    )
