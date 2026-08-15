"""
Benchmark the largest memory footprint a single transaction can force, bounded
by the [EIP-7825 transaction gas limit
cap](https://eips.ethereum.org/EIPS/eip-7825).

- ``test_single_frame_memory_expansion`` grows one frame's memory until the
  gas runs out. The quadratic memory-expansion term makes a single frame the
  worst way to buy memory, capping it at a few MiB.
- ``test_nested_frame_memory_expansion`` spends the same gas across nested
  self-calls, maximizing memory that is *simultaneously live*. Every fresh
  frame resets memory to zero and pays the expansion term from scratch, so
  splitting the allocation trades the quadratic term for the linear one and
  multiplies the live footprint several-fold. The depth is bounded by the
  63/64 gas-forwarding rule long before the call-stack limit.
- ``test_cumulative_memory_expansion`` maximizes memory *touched over the whole
  transaction* (freed and re-touched). Sequential calls each expand a small,
  near-linear-cost region and return, so the returned gas is reused and the
  63/64 rule is not a loss; the live footprint at any instant stays tiny while
  the cumulative bytes expanded reach into the hundreds of MiB.
"""

import pytest
from execution_testing import (
    Alloc,
    BenchmarkTestFiller,
    Conditional,
    Fork,
    Hash,
    JumpLoopGenerator,
    Op,
    Transaction,
)

# Each frame hands its child a target that is a fixed fraction smaller, so the
# per-frame allocation decays geometrically (``next = T - T // 50`` ~= 0.98 T).
# This matches the optimal decreasing schedule: the shallow, gas-rich frames
# allocate the most, deep frames the least.
FRAME_WORD_DECAY_DIVISOR = 50

# A frame stops recursing once its target drops below this many words. Reaching
# it means gas, not this floor, has bounded the recursion.
MIN_FRAME_WORDS = 32


@pytest.mark.valid_from("Osaka")
def test_single_frame_memory_expansion(
    benchmark_test: BenchmarkTestFiller,
) -> None:
    """Grow a single frame's memory until it runs out of gas."""
    # MSTORE at the current memory size grows memory by one word each store,
    # so the loop walks memory upward until the next expansion is unaffordable.
    # No offset is hardcoded, so the reachable size follows the fork's pricing.
    benchmark_test(
        target_opcode=Op.MSTORE,
        code_generator=JumpLoopGenerator(
            attack_block=Op.MSTORE(Op.MSIZE, 0),
        ),
    )


def largest_affordable_words(fork: Fork, gas: int) -> int:
    """Return the most words a single frame can expand to within ``gas``."""
    mem_gas = fork.memory_expansion_gas_calculator()
    low, high = 0, 1
    while mem_gas(new_bytes=high * 32) <= gas:
        low, high = high, high * 2
    while low < high:
        mid = (low + high + 1) // 2
        if mem_gas(new_bytes=mid * 32) <= gas:
            low = mid
        else:
            high = mid - 1
    return low


def nested_frame_plan(fork: Fork) -> tuple[int, int]:
    """
    Return ``(initial_words, predicted_peak_words)`` for the nested-frame
    attack.

    The top-frame target is picked from the fork's own memory pricing so the
    plan tracks any future repricing rather than a hardcoded offset. The peak
    is flat around the optimum, so the exact target is not sensitive.
    """
    cap = fork.transaction_gas_limit_cap()
    assert cap is not None, "Fork does not have a transaction gas limit cap"
    mem_gas = fork.memory_expansion_gas_calculator()
    intrinsic = fork.transaction_intrinsic_cost_calculator()

    # Gas the top frame receives; its 32-byte calldata carries the word target.
    frame_gas = cap - intrinsic(calldata=Hash(0))
    # Per-frame opcode overhead: the target arithmetic plus a warm 7-arg CALL.
    overhead = 200

    def peak_for(initial_words: int) -> int:
        remaining, target, peak = frame_gas, initial_words, 0
        for _ in range(1024):  # bounded by the EVM call-depth limit
            if target < MIN_FRAME_WORDS:
                break
            frame_cost = mem_gas(new_bytes=target * 32) + overhead
            if frame_cost > remaining:
                break
            peak += target
            remaining = (remaining - frame_cost) * 63 // 64
            target -= target // FRAME_WORD_DECAY_DIVISOR
        return peak

    return max(
        ((words, peak_for(words)) for words in range(2000, 30_001, 200)),
        key=lambda plan: plan[1],
    )


@pytest.mark.valid_from("Osaka")
def test_nested_frame_memory_expansion(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """Allocate memory across nested self-calls to maximize live memory."""
    cap = fork.transaction_gas_limit_cap()
    assert cap is not None, "Fork does not have a transaction gas limit cap"

    initial_words, predicted_peak = nested_frame_plan(fork)
    single_frame = largest_affordable_words(fork, cap)
    # Nesting must beat a single frame by a wide margin (empirically ~6x);
    # a loose bound keeps the guard robust to repricing.
    assert predicted_peak > 2 * single_frame

    # calldata[0:32] is this frame's word target T. When it drops below the
    # floor (only reachable if the depth limit, not gas, bounds the recursion)
    # the frame stops descending and falls through to the burn loop.
    target = Op.CALLDATALOAD(0)
    frame = Conditional(
        condition=Op.LT(target, MIN_FRAME_WORDS),
        if_false=(
            # Expand this frame to T words with a single store at (T-1)*32.
            Op.MSTORE(Op.MUL(Op.SUB(target, 1), 32), 0)
            # Write the child's smaller target into the call's args region.
            + Op.MSTORE(
                0,
                Op.SUB(target, Op.DIV(target, FRAME_WORD_DECAY_DIVISOR)),
            )
            # Recurse into self, forwarding all (63/64) of the remaining gas.
            + Op.POP(
                Op.CALL(
                    gas=Op.GAS,
                    address=Op.ADDRESS,
                    value=0,
                    args_offset=0,
                    args_size=32,
                    ret_offset=0,
                    ret_size=0,
                )
            )
        ),
    )

    # A frame cannot forward the 1/64 of gas it retains. Once the recursive
    # call returns, keep growing this frame's own memory until out of gas, so
    # the whole benchmark budget is spent on memory work rather than spinning.
    # This does not raise the peak (deeper frames are already freed) but keeps
    # every leftover gas unit attributable to memory expansion.
    grow_until_oog = Op.JUMPDEST + Op.MSTORE(Op.MSIZE, 0) + Op.JUMP(len(frame))
    contract = pre.deploy_contract(code=frame + grow_until_oog)
    tx = Transaction(
        to=contract,
        sender=pre.fund_eoa(),
        data=Hash(initial_words),
    )
    benchmark_test(pre=pre, tx=tx)


# Rough per-call gas overhead of the driver loop: a warm CALL plus the
# argument pushes and the worker's own prologue. Used only to pick the
# gas-optimal expansion size; the loop itself is gas-adaptive.
SEQUENTIAL_CALL_OVERHEAD = 150


def cumulative_worker_words(fork: Fork) -> int:
    """
    Return the per-call expansion size (words) that maximizes total memory
    touched across many sequential calls.

    Each call pays ``memory_expansion + overhead`` for ``words`` words, so the
    best size minimizes gas-per-word: too small wastes the fixed call overhead,
    too large pays the quadratic term. The size is read off the fork's own
    memory pricing so it tracks any future repricing.
    """
    mem_gas = fork.memory_expansion_gas_calculator()
    best_words, best_rate = 1, float("inf")
    for words in range(1, 4001):
        call_gas = mem_gas(new_bytes=words * 32) + SEQUENTIAL_CALL_OVERHEAD
        rate = call_gas / words
        if rate < best_rate:
            best_words, best_rate = words, rate
    return best_words


@pytest.mark.valid_from("Osaka")
def test_cumulative_memory_expansion(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """Touch the most cumulative memory across many freed call frames."""
    words = cumulative_worker_words(fork)

    # The worker expands to `words` in a single store, then returns so its
    # memory is freed. Because it returns rather than recursing, the gas the
    # driver forwarded but the worker did not spend flows back to the driver,
    # so sequential calls are not taxed by the 63/64 forwarding rule.
    worker = pre.deploy_contract(
        code=Op.MSTORE((words - 1) * 32, 0) + Op.STOP,
    )

    # The driver calls the worker in a loop until it runs out of gas.
    benchmark_test(
        pre=pre,
        target_opcode=Op.MSTORE,
        code_generator=JumpLoopGenerator(
            attack_block=Op.POP(
                Op.CALL(
                    gas=Op.GAS,
                    address=worker,
                    value=0,
                    args_offset=0,
                    args_size=0,
                    ret_offset=0,
                    ret_size=0,
                )
            ),
        ),
    )
