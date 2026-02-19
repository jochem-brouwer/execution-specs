"""Benchmark scenario for maximizing call frame creation."""

import pytest
from execution_testing import (
    Alloc,
    BenchmarkTestFiller,
    Environment,
    JumpLoopGenerator,
    Op,
)

# Default coinbase address used in test environments.
# Coinbase is implicitly warm in every transaction (EIP-2929).
COINBASE = Environment().fee_recipient


@pytest.mark.parametrize(
    "call_opcode",
    [
        Op.CALL,
        Op.CALLCODE,
        Op.DELEGATECALL,
        Op.STATICCALL,
    ],
)
def test_max_call_frames(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    call_opcode: Op,
) -> None:
    """Benchmark creating the maximum number of call frames.

    Deploy a minimal contract (STOP) at the coinbase address and
    repeatedly call it using each call opcode. The coinbase address is
    implicitly warm, so each call pays only the warm account access cost,
    maximizing the number of frames that fit within the gas limit.
    """
    # Deploy a contract with only STOP (0x00) at the coinbase address.
    pre.deploy_contract(code=Op.STOP, address=COINBASE)

    # CALL and CALLCODE require a value argument.
    if call_opcode in (Op.CALL, Op.CALLCODE):
        attack_block = Op.POP(
            call_opcode(address=COINBASE, value=0)
        )
    else:
        attack_block = Op.POP(call_opcode(address=COINBASE))

    benchmark_test(
        target_opcode=call_opcode,
        code_generator=JumpLoopGenerator(
            attack_block=attack_block,
        ),
    )
