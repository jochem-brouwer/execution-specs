"""Deploy the CREATE2 contracts assumed to exist by bloatnet benchmarks."""

from execution_testing import (
    DETERMINISTIC_FACTORY_ADDRESS,
    Account,
    Alloc,
    BenchmarkTestFiller,
    Block,
    Fork,
    Hash,
    Transaction,
    compute_create2_address,
)

from tests.benchmark.stateful.helpers import (
    AccountMode,
    account_mode_initcode,
)

RECEIVER_CONTRACT_COUNT = 5

CONTRACT_MODES = [
    AccountMode.EXISTING_CONTRACT_MINIMAL,
    AccountMode.EXISTING_CONTRACT_SAME,
    AccountMode.EXISTING_CONTRACT_DIFF,
    AccountMode.EXISTING_CONTRACT_JUMPDEST,
]


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
    pipeline replays them ahead of every scenario. Each deployment gets
    the full block gas budget and its own block, so no transaction is
    ever dropped at block building.
    """
    blocks = []
    post = {}
    for account_mode in CONTRACT_MODES:
        initcode = account_mode_initcode(fork, account_mode)
        sender = pre.fund_eoa()
        for salt in range(RECEIVER_CONTRACT_COUNT):
            blocks.append(
                Block(
                    txs=[
                        Transaction(
                            to=DETERMINISTIC_FACTORY_ADDRESS,
                            data=Hash(salt) + initcode,
                            gas_limit=gas_benchmark_value,
                            sender=sender,
                        )
                    ]
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
        blocks=blocks,
        skip_gas_used_validation=True,
        expected_receipt_status=1,
    )
