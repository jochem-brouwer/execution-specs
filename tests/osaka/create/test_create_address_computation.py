"""
Test CreatePreimageLayout dynamic nonce encoding against actual CREATE.

Deploy a contract that loops calling CREATE (empty initcode), computes
the expected address on-chain via CreatePreimageLayout, and reverts on
any mismatch. The post-state check confirms all iterations succeeded.
"""

import pytest

from execution_testing import (
    Account,
    Alloc,
    Conditional,
    CreatePreimageLayout,
    StateTestFiller,
    Transaction,
    While,
)
from execution_testing.vm import Op


@pytest.mark.valid_from("Osaka")
def test_create_address_dynamic_nonce(
    pre: Alloc,
    state_test: StateTestFiller,
) -> None:
    """
    Verify CreatePreimageLayout dynamic nonce encoding matches CREATE.

    A contract calls CREATE(value=0, offset=0, size=0) in a loop,
    computes the expected address using the dynamic nonce RLP encoder,
    and reverts if any computed address differs from the actual one.

    The loop runs from nonce 1 to 260, crossing the RLP encoding
    boundary at nonce 128 (1-byte to 2-byte encoding) and at
    256 where it has to change the 0x80 prefix to 0x81.
    """
    iterations = 260

    # Memory[0:32] is used as the loop counter.
    # Layout starts at offset 32 to avoid conflict.
    layout = CreatePreimageLayout(
        sender_address=Op.ADDRESS,
        nonce=Op.PUSH1(1),
        offset=32,
    )

    # Build the loop body: check address, revert on mismatch,
    # increment nonce, decrement counter.
    body = (
        Conditional(
            condition=Op.EQ(
                layout.address_op(),
                Op.CREATE(value=0, offset=0, size=0),
            ),
            if_false=Op.REVERT(0, 0),
        )
        + layout.increment_nonce_op()
        + Op.MSTORE(0, Op.SUB(Op.MLOAD(0), 1))
    )

    code = layout
    code += Op.MSTORE(0, iterations)
    code += While(body=body, condition=Op.MLOAD(0))
    code += Op.SSTORE(0, 1)
    code += Op.STOP

    contract = pre.deploy_contract(code=code)
    sender = pre.fund_eoa()

    tx = Transaction(
        to=contract,
        gas_limit=15_000_000,
        sender=sender,
    )

    post = {contract: Account(storage={0: 1})}

    state_test(pre=pre, tx=tx, post=post)
