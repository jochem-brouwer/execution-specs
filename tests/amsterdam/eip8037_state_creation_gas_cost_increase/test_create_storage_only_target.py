"""
Contract creation onto an account that holds only storage.

An account with zero nonce, no code and a non-empty storage trie is not a
creation collision in Amsterdam: EIP-684 only rejects a target with code
or a non-zero nonce, and the EIP-7610 storage check is not part of the
fork. The creation therefore succeeds and the pre-existing storage stays
readable by the init code.
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Fork,
    Initcode,
    Op,
    StateTestFiller,
    Transaction,
    compute_create_address,
)

from .spec import ref_spec_8037

REFERENCE_SPEC_GIT_PATH = ref_spec_8037.git_path
REFERENCE_SPEC_VERSION = ref_spec_8037.version

# Sentinel written into the target's storage before the transaction.
PRE_EXISTING_SLOT = 1
PRE_EXISTING_VALUE = 0x42
# Slot the init code writes the observed value of PRE_EXISTING_SLOT to.
WITNESS_SLOT = 2


@pytest.mark.pre_alloc_mutable
@pytest.mark.valid_from("Amsterdam")
def test_create_tx_onto_storage_only_account(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """
    Deploy with a creation transaction onto a storage-only account.

    The target has neither code nor nonce, so the creation must succeed;
    the init code copies the pre-existing slot into a witness slot to
    show the old storage is visible to the new contract.
    """
    deploy_code = Op.STOP
    init_code = Initcode(
        deploy_code=deploy_code,
        initcode_prefix=Op.SSTORE(WITNESS_SLOT, Op.SLOAD(PRE_EXISTING_SLOT)),
    )

    sender = pre.fund_eoa()
    target = compute_create_address(address=sender, nonce=0)
    # Only reachable through pre-state: post-Spurious-Dragon execution
    # cannot leave an account with storage but no nonce and no code.
    pre[target] = Account(
        nonce=0,
        balance=0,
        code=b"",
        storage={PRE_EXISTING_SLOT: PRE_EXISTING_VALUE},
    )

    tx = Transaction(
        to=None,
        data=init_code,
        sender=sender,
    )

    state_test(
        pre=pre,
        post={
            target: Account(
                nonce=1,
                code=deploy_code,
                storage={
                    PRE_EXISTING_SLOT: PRE_EXISTING_VALUE,
                    WITNESS_SLOT: PRE_EXISTING_VALUE,
                },
            ),
            sender: Account(nonce=1),
        },
        tx=tx,
    )
