"""Deploy the CREATE2 contracts assumed to exist by bloatnet benchmarks."""

from execution_testing import (
    DETERMINISTIC_FACTORY_ADDRESS,
    Account,
    Alloc,
    BenchmarkTestFiller,
    Block,
    Fork,
    Hash,
    Op,
    Transaction,
    compute_create2_address,
)

from tests.benchmark.stateful.helpers import (
    AccountMode,
    account_mode_initcode,
)

RECEIVER_CONTRACT_COUNT = 50_000

CONTRACT_MODES = [
    AccountMode.EXISTING_CONTRACT_MINIMAL,
    AccountMode.EXISTING_CONTRACT_SAME,
    AccountMode.EXISTING_CONTRACT_DIFF,
    AccountMode.EXISTING_CONTRACT_JUMPDEST,
]


def deployment_gas_limit(
    fork: Fork, initcode: bytes, deployed_code_size: int
) -> int:
    """
    Return an upper bound on the gas for one factory deployment.

    Sums the modeled costs and adds headroom for the factory dispatch,
    the initcode's memory writes, and the 63/64 rule.
    """
    intrinsic = fork.transaction_intrinsic_cost_calculator()(
        calldata=Hash(0) + initcode
    )
    create_cost = Op.CREATE2(
        value=0,
        offset=0,
        size=len(initcode),
        salt=0,
        init_code_size=len(initcode),
    ).gas_cost(fork)
    deposit_cost = Op.RETURN(
        0,
        deployed_code_size,
        code_deposit_size=deployed_code_size,
    ).gas_cost(fork)
    base = intrinsic + create_cost + deposit_cost
    return base + base // 16 + 100_000


def test_deploy_existing_contracts(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Deploy the contracts behind the `AccountMode.EXISTING_CONTRACT_*`
    receivers via the deterministic CREATE2 factory.

    Runs once as a global setup before the bloatnet benchmarks; the
    transactions intentionally carry no test phase so that the payload
    pipeline replays them ahead of every scenario.
    """
    txs = []
    post = {}
    for account_mode in CONTRACT_MODES:
        initcode = account_mode_initcode(fork, account_mode)
        deployed_code_size = (
            1
            if account_mode == AccountMode.EXISTING_CONTRACT_MINIMAL
            else fork.max_code_size()
        )
        gas_limit = deployment_gas_limit(fork, initcode, deployed_code_size)
        sender = pre.fund_eoa()
        for salt in range(RECEIVER_CONTRACT_COUNT):
            txs.append(
                Transaction(
                    to=DETERMINISTIC_FACTORY_ADDRESS,
                    data=Hash(salt) + initcode,
                    gas_limit=gas_limit,
                    sender=sender,
                )
            )
        for salt in (0, RECEIVER_CONTRACT_COUNT - 1):
            created_address = compute_create2_address(
                address=DETERMINISTIC_FACTORY_ADDRESS,
                salt=salt,
                initcode=initcode,
            )
            post[created_address] = Account(nonce=1)

    benchmark_test(
        post=post,
        blocks=[Block(txs=txs)],
        skip_gas_used_validation=True,
        expected_receipt_status=1,
    )
