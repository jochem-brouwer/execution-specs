"""Deploy the CREATE2 contracts assumed to exist by bloatnet benchmarks."""

from execution_testing import (
    DETERMINISTIC_FACTORY_ADDRESS,
    Alloc,
    BenchmarkTestFiller,
    Fork,
    Hash,
    Transaction,
)

from tests.benchmark.stateful.helpers import (
    AccountMode,
    account_mode_initcode,
    pack_transactions_into_blocks,
)

RECEIVER_CONTRACT_COUNT = 5

CONTRACT_MODES = [
    AccountMode.EXISTING_CONTRACT_MINIMAL,
    AccountMode.EXISTING_CONTRACT_SAME,
    AccountMode.EXISTING_CONTRACT_DIFF,
    AccountMode.EXISTING_CONTRACT_JUMPDEST,
]

# Generous per-deployment gas limit. A max-size contract deposit costs
# ~54M under EIP-8037; the single-byte runtime is far cheaper but the
# same limit is harmless. Kept well below the block budget so many
# deployments pack into one block.
DEPLOYMENT_TX_GAS_LIMIT = 80_000_000


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
    pipeline replays them ahead of every scenario. Deployments are
    packed into blocks up to the block gas budget so each block carries
    many transactions instead of one.
    """
    txs = []
    for account_mode in CONTRACT_MODES:
        initcode = account_mode_initcode(fork, account_mode)
        sender = pre.fund_eoa()
        for salt in range(RECEIVER_CONTRACT_COUNT):
            txs.append(
                Transaction(
                    to=DETERMINISTIC_FACTORY_ADDRESS,
                    data=Hash(salt) + initcode,
                    gas_limit=DEPLOYMENT_TX_GAS_LIMIT,
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
