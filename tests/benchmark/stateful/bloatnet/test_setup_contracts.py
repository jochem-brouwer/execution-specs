"""Deploy the CREATE2 contracts assumed to exist by bloatnet benchmarks."""

from execution_testing import (
    DETERMINISTIC_FACTORY_ADDRESS,
    Alloc,
    BenchmarkTestFiller,
    Fork,
    Hash,
    Op,
    Transaction,
)

from tests.benchmark.stateful.helpers import (
    AccountMode,
    account_mode_initcode,
    account_mode_runtime_size,
    pack_transactions_into_blocks,
)

# Distinct contracts deployed per account mode. Sized for the most
# demanding consumer, test_account_access, whose cheapest iteration is
# ~2,691 gas: a 300M block reaches ~111,482 distinct salts, rounded up
# with margin.
RECEIVER_CONTRACT_COUNT = 120_000

CONTRACT_MODES = [
    AccountMode.EXISTING_CONTRACT_MINIMAL,
    AccountMode.EXISTING_CONTRACT_SAME,
    AccountMode.EXISTING_CONTRACT_DIFF,
    AccountMode.EXISTING_CONTRACT_JUMPDEST,
]


def deployment_gas_limit(
    fork: Fork, initcode: bytes, runtime_size: int
) -> int:
    """
    Return the gas limit for one CREATE2 deployment, derived from the
    intrinsic, CREATE2, and code-deposit costs with a small margin.
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
        runtime_size,
        code_deposit_size=runtime_size,
    ).gas_cost(fork)
    base = intrinsic + create_cost + deposit_cost
    return base + base // 16


def test_deploy_existing_contracts(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
) -> None:
    """
    Deploy the contracts behind the `AccountMode.EXISTING_CONTRACT_*`
    receivers via the deterministic CREATE2 factory.

    Runs once as a global setup before the bloatnet benchmarks; the
    transactions intentionally carry no test phase so that the payload
    pipeline replays them ahead of every scenario. Each deployment's
    gas limit is computed from its cost, then transactions are packed
    into blocks up to the block gas budget.
    """
    txs = []
    for account_mode in CONTRACT_MODES:
        initcode = account_mode_initcode(fork, account_mode)
        gas_limit = deployment_gas_limit(
            fork, initcode, account_mode_runtime_size(fork, account_mode)
        )
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

    blocks = pack_transactions_into_blocks(txs, gas_benchmark_value)

    benchmark_test(
        post={},
        blocks=blocks,
        skip_gas_used_validation=True,
        expected_receipt_status=1,
    )
