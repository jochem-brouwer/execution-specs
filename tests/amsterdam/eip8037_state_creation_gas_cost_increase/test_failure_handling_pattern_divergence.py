"""
Show that failure-handling contract patterns diverge across the EIP-8037
boundary, with transaction gas (the state gas reservoir) and the immediate
clear refill as the levers.

Background (ChainSecurity, "Glamsterdam and Failure Handling Based on Gas
Checks", https://www.chainsecurity.com/blog/glamsterdam-gas/): a large class
of contracts inspects the *outcome* of an external call and then makes a
decision from it ("insufficient-gas griefing" defenses). The blog groups the
defenses into four patterns:

- A.  Pre-call budget check: forward a fixed `gasReq`, trusting it is enough
      for the callee's honest work.
- B1. Post-call check against a fixed requirement (`gasleft() < gasReq / 63`).
- B2. Post-call check against the pre-call gas reading.
- C.  Relying on reverts: assume the failure-handling write (e.g. marking a
      message permanently failed) will itself run out of gas, and unwind the
      whole transaction, if the callee was starved.

Two tests live here, one per group of patterns:

- `test_forwarded_budget_outcome_divergence` (patterns A / B1 / C): the
  forwarded budget's *meaning* changed. Lever: the reservoir (tx gas limit).
- `test_post_call_gas_check_divergence` (patterns B1 / B2): the post-call
  `gasleft()` check is defeated by the immediate refill when a slot that was
  zero at transaction start is cleared (`0 -> 1 -> 0`). Before EIP-8037 the
  clear is a refund settled at transaction end and invisible to `GAS`, so the
  check correctly rejects a starved call; after it the refill lands in
  `gas_left` at once, so the same reading passes and the griefing succeeds.
  Lever: the clear refill.

The first test exercises the headline break shared by patterns A, B1 and C:
the
*meaning of a forwarded gas budget changed*. Under EIP-8037 a storage-slot
creation costs the same ~22,100 execution gas only when the reservoir funds
the 97,920 state charge; with an empty reservoir that charge spills into
`gas_left`, so the identical write needs ~110,000. Because the reservoir is
sized by the transaction gas limit (`cap + reservoir`), which the submitter
chooses and the callee cannot observe, the *same* contracts and the *same*
forwarded budget can make a valid message succeed or be permanently marked
failed.

Scenario: an executor forwards a fixed, honestly-signed budget to a target
whose success path creates one storage slot, and marks the message failed if
the call reports failure. `FORWARDED_GAS` is sized to sit above the Osaka cost
of that success path but below the Amsterdam empty-reservoir cost, so:

- pre-EIP-8037:            target succeeds, message not marked failed.
- EIP-8037, reservoir funded:   target succeeds (reservoir pays the state
                                charge, forwarded in full to the child).
- EIP-8037, reservoir empty:    target runs out of gas on the same budget, and
                                the executor permanently marks a valid message
                                failed.

The failure-handling write (pattern C) is expressed branch-free as
`SSTORE(SLOT_FAILED, ISZERO(ok))`: writing zero to a never-touched slot creates
nothing, so the 97,920 state charge only materializes when the call failed,
exactly as a conditional `if (!ok) failed = true` would.

Tests for [EIP-8037: State Creation Gas Cost Increase]
(https://eips.ethereum.org/EIPS/eip-8037).
"""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Bytecode,
    Fork,
    Op,
    StateTestFiller,
    Transaction,
)
from execution_testing import (
    Macros as Om,
)
from execution_testing.forks import Amsterdam, Osaka

from .spec import ref_spec_8037

REFERENCE_SPEC_GIT_PATH = ref_spec_8037.git_path
REFERENCE_SPEC_VERSION = ref_spec_8037.version

DONE_VALUE = 0x11  # marker a completing executor writes to its done slot


# ---------------------------------------------------------------------------
# Variant A - patterns A / B1 / C: a fixed forwarded budget changes meaning.
# Lever: the state gas reservoir (the transaction gas limit).
# ---------------------------------------------------------------------------

# Executor witness slots (pre-set to canary sentinels so an untouched slot is
# distinguishable from a deliberate write, and so the writes are slot
# modifications rather than state creations).
SLOT_SUCCESS = 0x10  # stores `ok + 1`: 2 if call succeeded, 1 if it failed
SLOT_DONE = 0x12  # stores DONE_VALUE unconditionally: proof executor ran
SUCCESS_CANARY = 0xFF
DONE_CANARY = 0xAA

# The permanent "message failed" flag. Deliberately NOT pre-set: it must be
# zero at transaction start so that marking it is a fresh state creation (the
# 97,920 state charge that is the subject of the break).
SLOT_FAILED = 0x11

# The target's success-path slot, also a fresh creation.
SLOT_TARGET = 0x20


@pytest.mark.parametrize(
    "reservoir_funded",
    [
        pytest.param(False, id="reservoir_empty"),
        pytest.param(True, id="reservoir_funded"),
    ],
)
@pytest.mark.valid_from("Osaka")
def test_forwarded_budget_outcome_divergence(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    reservoir_funded: bool,
) -> None:
    """
    A fixed forwarded budget keeps a valid message alive before EIP-8037 but,
    after it, lets an empty reservoir starve the callee and permanently mark
    the message failed; the transaction gas limit is the only lever.
    """
    # Target success path: create one storage slot. Priced on both schedules
    # to bracket the forwarded budget. The metadata marks a cold, fresh
    # zero-to-nonzero write.
    fresh_write = Op.SSTORE.with_metadata(
        key_warm=False, original_value=0, current_value=0, new_value=1
    )
    target_code = fresh_write(SLOT_TARGET, 1) + Op.STOP

    # osaka_total: execution only (no state dimension yet).
    # amsterdam_exec: state charge paid from the reservoir.
    # amsterdam_full: state charge spilled into gas_left (empty reservoir).
    osaka_total = target_code.gas_cost(Osaka)
    amsterdam_exec = target_code.execution_cost(Amsterdam)
    amsterdam_full = target_code.gas_cost(Amsterdam)

    # The app-signed budget. Fixed across forks (signed once), chosen inside
    # the window where it covers the success path on Osaka and on Amsterdam
    # when the reservoir funds the state charge, yet falls short when it does
    # not.
    forwarded_gas = (osaka_total + amsterdam_full) // 2
    assert amsterdam_exec <= osaka_total <= forwarded_gas < amsterdam_full, (
        "forwarded budget must suffice on Osaka and on a funded reservoir, "
        "but starve the target when the reservoir is empty"
    )

    target = pre.deploy_contract(code=target_code)

    # Executor: forward the budget, record the outcome, and mark the message
    # failed. `SSTORE(SLOT_FAILED, ISZERO(ok))` writes 1 (a fresh creation)
    # only when the call failed; on success it writes 0 to the untouched slot,
    # which creates nothing. This is pattern C expressed without a branch.
    executor_code = (
        Op.MSTORE(0, Op.CALL(forwarded_gas, target, 0, 0, 0, 0, 0))
        + Op.SSTORE(SLOT_SUCCESS, Op.ADD(Op.MLOAD(0), 1))
        + Op.SSTORE(SLOT_FAILED, Op.ISZERO(Op.MLOAD(0)))
        + Op.SSTORE(SLOT_DONE, DONE_VALUE)
        + Op.STOP
    )
    executor = pre.deploy_contract(
        code=executor_code,
        storage={SLOT_SUCCESS: SUCCESS_CANARY, SLOT_DONE: DONE_CANARY},
    )

    # tx.gas is the lever. Before EIP-8037 the reservoir does not exist, so
    # the field is omitted and the implicit gas limit (the cap) runs both
    # parameter cases identically. After it, `state_gas_reservoir=0` pins the
    # gas limit to the cap (empty reservoir), and a positive value funds the
    # reservoir enough to cover the target's state charge, forwarded in full
    # to the child.
    tx_kwargs: dict = {}
    if fork.is_eip_enabled(8037):
        tx_kwargs["state_gas_reservoir"] = (
            target_code.state_cost(fork) * 2 if reservoir_funded else 0
        )
    tx = Transaction(to=executor, sender=pre.fund_eoa(), **tx_kwargs)

    # The divergence. Pre-8037 the reservoir is irrelevant and the honest
    # budget always suffices; post-8037 the empty reservoir flips the outcome.
    target_succeeds = (not fork.is_eip_enabled(8037)) or reservoir_funded

    post = {
        executor: Account(
            storage={
                SLOT_SUCCESS: 2 if target_succeeds else 1,
                SLOT_FAILED: 0 if target_succeeds else 1,
                SLOT_DONE: DONE_VALUE,
            }
        ),
        target: Account(storage={SLOT_TARGET: 1 if target_succeeds else 0}),
    }
    state_test(pre=pre, post=post, tx=tx)


# ---------------------------------------------------------------------------
# Variants B1 & B2 - the post-call gasleft() check defeated by the clear
# refill. Lever: clearing a transaction-fresh slot (0 -> 1 -> 0).
# ---------------------------------------------------------------------------

# Executor is given a tight gas budget, so the forwarded call is clipped to
# the EIP-150 63/64 rule and the executor retains only ~1/64. The B1/B2 witness
# slots are canary-preset (cheap modifications, distinct from untouched),
# except the pending slot, which must be zero at transaction start so that
# clearing it triggers the refill.
SLOT_PENDING = 0x30  # fresh: set to 1 before the call, cleared to 0 after
SLOT_DECISION = 0x31  # stores `pass + 1`: 2 if the check passed, 1 if rejected
SLOT_CALL_OK = 0x32  # stores `ok + 1`: must be 1 (call failed) on every fork
SLOT_DONE_B = 0x33  # stores DONE_VALUE: proof the executor ran to completion
B_CANARY = 0xEE

# Executor start gas. Large enough that the forwarded call clips to the 63/64
# rule (leaving ~1/64 retained) and the executor can still run the clear after
# a starved call, yet small enough that the retained ~1/64 stays below the
# check threshold before any refill.
EXECUTOR_START_GAS = 2_000_000


@pytest.mark.parametrize(
    "check_style",
    [
        # B1: compare the post-call reading against a fixed requirement (here
        # the nominal 1/64 the executor expects to retain).
        pytest.param("b1_fixed_requirement", id="b1_fixed_requirement"),
        # B2: compare the post-call reading against a fraction of the gas read
        # before the call.
        pytest.param("b2_pre_call_ratio", id="b2_pre_call_ratio"),
    ],
)
@pytest.mark.valid_from("Osaka")
def test_post_call_gas_check_divergence(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
    check_style: str,
) -> None:
    """
    A post-call `gasleft()` check that correctly rejects a starved call before
    EIP-8037 is defeated after it, because clearing a transaction-fresh slot
    refills `gas_left` immediately and inflates the reading past the threshold.
    """
    # The refill restored on the clear dominates the check threshold on every
    # fork that has the state dimension.
    fresh_write = Op.SSTORE.with_metadata(
        key_warm=False, original_value=0, current_value=0, new_value=1
    )
    refill = fresh_write.state_cost(Amsterdam)
    threshold_fixed = EXECUTOR_START_GAS // 64  # the nominal 1/64 retention
    assert refill > threshold_fixed, (
        "the clear refill must dominate the check threshold"
    )

    # Target consumes all forwarded gas and fails, identically on every fork,
    # so the call outcome is constant and only the check verdict can diverge.
    target = pre.deploy_contract(code=Om.OOG)

    # Build the check. B1 compares against a fixed requirement; B2 against the
    # gas read before the call (stashed in memory word 1).
    if check_style == "b1_fixed_requirement":
        capture_pre_call = Bytecode()
        threshold: int | Bytecode = threshold_fixed
    else:
        capture_pre_call = Op.MSTORE(32, Op.GAS)
        threshold = Op.DIV(Op.MLOAD(32), 64)
    check_passed = Op.ISZERO(Op.LT(Op.GAS, threshold))

    # Executor: set a fresh pending slot, forward all gas to the starved
    # target, clear the pending slot (the refill), then run the post-call check
    # and record whether it passed. `ok` is stashed in memory word 0.
    executor_code = (
        Op.SSTORE(SLOT_PENDING, 1)
        + capture_pre_call
        + Op.MSTORE(0, Op.CALL(Op.GAS, target, 0, 0, 0, 0, 0))
        + Op.SSTORE(SLOT_PENDING, 0)
        + Op.SSTORE(SLOT_DECISION, Op.ADD(check_passed, 1))
        + Op.SSTORE(SLOT_CALL_OK, Op.ADD(Op.MLOAD(0), 1))
        + Op.SSTORE(SLOT_DONE_B, DONE_VALUE)
        + Op.STOP
    )
    executor = pre.deploy_contract(
        code=executor_code,
        storage={
            SLOT_DECISION: B_CANARY,
            SLOT_CALL_OK: B_CANARY,
            SLOT_DONE_B: B_CANARY,
        },
    )

    # Pin the executor's start gas so the forwarded call is clipped to 63/64.
    # The limit stays below the cap on every fork, so the reservoir is empty
    # and the pending set spills into gas_left, where the clear can refill it.
    intrinsic = fork.transaction_intrinsic_cost_calculator()()
    tx = Transaction(
        to=executor,
        gas_limit=EXECUTOR_START_GAS + intrinsic,
        sender=pre.fund_eoa(),
    )

    # The divergence. Pre-8037 the clear is a deferred refund invisible to GAS,
    # so the post-call reading stays below the threshold and the check rejects
    # the starved call (decision 1). Post-8037 the clear refills gas_left at
    # once, so the reading passes and the griefing succeeds (decision 2).
    check_passes = fork.is_eip_enabled(8037)

    post = {
        executor: Account(
            storage={
                SLOT_PENDING: 0,  # set then cleared
                SLOT_DECISION: 2 if check_passes else 1,
                SLOT_CALL_OK: 1,  # ok == 0 (call ran out of gas) everywhere
                SLOT_DONE_B: DONE_VALUE,
            }
        ),
    }
    state_test(pre=pre, post=post, tx=tx)
