# Nethermind EVM Optimization Analysis

Research into Nethermind's optimistic execution, non-standard EVM optimizations, and attack surfaces where crafted transactions can cause disproportionately slow runtimes.

---

## 1. Parallel Execution Architecture

### Key Finding: No True Parallel Transaction Execution

Nethermind does NOT execute transactions in parallel. Transactions within a block are processed strictly sequentially in a simple for-loop:

`src/Nethermind/Nethermind.Consensus/Processing/BlockProcessor.BlockValidationTransactionsExecutor.cs:31-41`:
```csharp
for (int i = 0; i < block.Transactions.Length; i++)
{
    Transaction currentTx = block.Transactions[i];
    ProcessTransaction(block, currentTx, i, receiptsTracer, processingOptions);
}
```

What Nethermind does have is a sophisticated **cache pre-warming** system that speculatively executes transactions in parallel on background threads before the real sequential processing begins, purely to populate lock-free concurrent caches. The real block processor then benefits from cache hits, avoiding expensive trie traversals.

### The Pipeline Architecture

```
                      +-------------------+
                      | BlockchainProcessor |
                      |  (recovery queue)   |
                      |  (processing queue) |
                      +--------+-----------+
                               |
                      +--------v-----------+
                      |  BranchProcessor    |
                      |  - starts prewarming|  <-- OVERLAP: prewarm starts before ProcessOne
                      |  - starts prefetch  |  <-- OVERLAP: blockhash prefetch in background
                      +--------+-----------+
                               |
          +--------------------+--------------------+
          |                                         |
  BACKGROUND (concurrent)                  MAIN THREAD (sequential)
  +-------------------------+       +-------------------------------+
  | BlockCachePreWarmer     |       | BlockProcessor.ProcessOne()   |
  | - Address warming       |       |   1. PrepareBlock             |
  | - Tx warming by sender  |       |   2. ProcessBlock()           |
  |   (parallel senders,    |       |     a. Beacon root / blockhash|
  |    sequential per sender)|      |     b. Commit                 |
  | - Withdrawal warming    |       |     c. ProcessTransactions    |
  +-------------------------+       |        (sequential for loop)  |
          |                         |     d. Commit                 |
          | populates               |     e. Blooms (parallel)      |
          v                         |     f. Receipts root          |
  +-------------------------+       |     g. Miner rewards          |
  | PreBlockCaches          |       |     h. Withdrawals            |
  | - SeqlockCache<Account> |------>|     i. Commit (with roots)    |
  | - SeqlockCache<Storage> | reads |     j. State root computation |
  | - SeqlockCache<NodeRlp> |       +-------------------------------+
  | - Precompile cache      |
  +-------------------------+
```

### Pre-Warmer: Sender-Group Parallelism

`src/Nethermind/Nethermind.Consensus/Processing/BlockCachePreWarmer.cs:157-223` groups transactions by sender address:
- **Different senders** execute in parallel (on up to `min(ProcessorCount-1, 16)` threads)
- **Same sender** transactions execute sequentially within their group (preserving nonce/balance ordering)

Each parallel thread gets its own `IReadOnlyTxProcessingScope` (a private WorldState copy). Execution runs with `ExecutionOptions.Warmup | ExecutionOptions.SkipValidation` — full EVM execution but all state changes are discarded. Only cache-warming side effects persist.

### Same-Source Transactions Cannot Run in Parallel

The prewarmer explicitly groups same-sender transactions sequentially (`BlockCachePreWarmer.cs:225-243`) because they have nonce dependencies and sequential balance deductions.

### No Conflict Detection or Re-Execution

There is no read-write set tracking, no conflict detection, and no re-execution mechanism. The prewarmer is purely optimistic with graceful degradation:
- Cache hit (common case): main processor avoids trie traversal
- Cache miss (cross-sender dependency): main processor reads from authoritative trie at baseline cost

---

## 2. Coinbase Fee Handling

Since execution is sequential, coinbase creates no serialization bottleneck.

**Fee flow per transaction** (`src/Nethermind/Nethermind.Evm/TransactionProcessing/TransactionProcessor.cs:867-895`):
1. **BuyGas** (line 531): Sender balance deducted by `gasLimit * effectiveGasPrice`
2. **EVM Execution**
3. **Refund** (line 762): Unused gas refunded to sender
4. **PayFees** (line 867-895):
   - `premiumPerGas * spentGas` credited to `header.GasBeneficiary` (tip)
   - Base fee burned on mainnet (FeeCollector is null)
   - On Gnosis/Chiado, base fee goes to configured FeeCollector address

In the prewarmer, each parallel thread has its own WorldState copy — coinbase writes are independent and discarded.

---

## 3. Non-Standard Optimizations & Cache Attack Surfaces

### 3.1 SeqlockCache — The Fundamental Primitive (CRITICAL)

`src/Nethermind/Nethermind.Core/Collections/SeqlockCache.cs:40-400`

A custom 2-way skew-associative lock-free cache with **32,768 entries** (16,384 sets x 2 ways). Used for account state, storage values, and trie node RLP.

- Lock-free reads via seqlock pattern (readers never take locks)
- CAS-based writes that **silently drop on contention** (line 353-356)
- O(1) logical Clear() via epoch counter

**Attack: Cache Thrashing.** A block accessing >32K unique storage cells causes constant eviction, making the entire prewarming infrastructure wasted CPU.

**Attack: Hash Collision.** Set indices use bits[0:13] for way 0, bits[42:55] for way 1, from a non-cryptographic hash. Crafted keys with colliding set indices can force targeted evictions.

**Attack: Write Contention.** Under high parallelism, CAS failures cause prewarmed data to be silently dropped.

### 3.2 Precompile Result Cache (Unbounded ConcurrentDictionary)

`src/Nethermind/Nethermind.Blockchain/CachedCodeInfoRepository.cs:71-93`

All precompiles except Identity (0x04) are cached in `ConcurrentDictionary<PrecompileCacheKey, Result<byte[]>>` with no size limit and no eviction.

**Which precompiles are cached:** All except Identity (`SupportsCaching => false`). The cheapest cacheable precompile is BLAKE2F (0x09) at 0 gas with rounds=0, though the CALL overhead is 100 gas (warm access). Maximum ~262K unique cache entries per 36M gas block, consuming ~116 MB.

**Gas-bounded:** Yes. The cache is implicitly bounded by gas. With the cheapest path (BLAKE2F rounds=0 at ~137 gas per unique call), maximum is ~262K entries ≈ 116 MB per block. Cleared between blocks via `NoResizeClear()`.

### 3.3 Code Cache (5,120 entries)

`src/Nethermind/Nethermind.Evm/CodeInfoRepository.cs:21,153-188`

Static `ClockCache` with 5,120 entries across 16 shards (~320 per shard). Cache index is last byte of code hash.

**Attack:** A block calling >5,120 unique contracts causes every call to miss, triggering DB reads + CodeInfo construction + background jump destination analysis.

### 3.4 Jump Destination Analysis Blocking

`src/Nethermind/Nethermind.Evm/CodeAnalysis/JumpDestinationAnalyzer.cs:32-68`

Analysis is lazy and runs on background ThreadPool. If the EVM hits a JUMP before analysis completes, the **main processing thread blocks** on `ManualResetEventSlim.Wait()` and priority is demoted to normal.

**Attack:** Deploy large contracts and immediately JUMP within them. Background analysis uses SIMD (Vector512/Vector128) but must complete before the main thread can proceed.

**Attack: ThreadPool Saturation.** Many unique new contracts queue unbounded `ThreadPool.UnsafeQueueUserWorkItem` calls for analysis.

### 3.5 Cross-Sender Independence Assumption

`src/Nethermind/Nethermind.Consensus/Processing/BlockCachePreWarmer.cs:157-223`

The prewarmer assumes transactions from different senders don't interact.

**Attack:** Craft a block where Tx_A (sender X) writes a storage slot that Tx_B (sender Y) reads. The prewarmer processes them in parallel with wrong results, making the cache entries invalid. Main processor misses where it expected hits.

---

## 4. Main-Thread Sequential Execution Attack Vectors

### 4.1 The CALL-SSTORE-REVERT Journal Amplification (HIGH)

**How the journal works:**

Every SSTORE appends to two data structures (`src/Nethermind/Nethermind.State/PartialStorageProviderBase.cs:214-218`):
```csharp
private void PushUpdate(in StorageCell cell, byte[] value)
{
    StackList<int> stack = SetupRegistry(cell);  // dictionary lookup
    stack.Push(_changes.Count);                   // list append
    _changes.Add(new Change(in cell, value, ChangeType.Update));
}
```

**How Restore works:**

On REVERT, `Restore(snapshot)` (`PartialStorageProviderBase.cs:77-141`) iterates every change since the snapshot — O(k) dictionary lookups + stack pops. StateProvider.Restore (`StateProvider.cs:352-418`) does the same for accounts. JournalSet.Restore does the same for the access tracker.

**Warm SSTORE churn — maximum amplification:**

Writing to the same slot N times within a single call frame:
```
sstore(slot, 1)  // 100 gas (warm, net metered)
sstore(slot, 2)  // 100 gas
... N times
```

Each costs only 100 gas (`GetNetMeteredSStoreCost`) but appends to the journal. The eventual `CommitCore` (`PersistentStorageProvider.cs:125-235`) iterates ALL N entries, deduplicating via HashSet:

- 36M gas / 100 gas per warm SSTORE = **360,000 journal entries** per transaction
- `CommitCore` iterates all 360K entries with HashSet lookups
- `StorageCell` hashing involves 20-byte address + UInt256 index — non-trivial per lookup

### 4.2 TSTORE + REVERT Amplification (HIGH)

TSTORE costs flat 100 gas (`GasCostOf.cs:67`) with no cold/warm distinction. Uses the same journal infrastructure as persistent storage.

**Combination attack:**
```
for N iterations:
    call {
        for K = 1000:
            tstore(j, 1)   // 100 gas each
        revert()            // O(K) Restore
    }
```
Per iteration: K * 100 + ~700 gas. With K=1000: ~100,700 gas → ~357 iterations.
Total: 357,000 writes + 357,000 rollbacks = **714,000 journal operations**.

Each rollback does O(K) dictionary lookups + stack pops — **unpaid by gas**.

### 4.3 KeccakCache Bypass (MEDIUM-HIGH)

`src/Nethermind/Nethermind.Core/Crypto/KeccakCache.cs:25-244`

128K-entry cache (16MB native memory). **Inputs > 92 bytes bypass the cache entirely** (line 65):
```csharp
if (input.Length is 0 or > Entry.MaxPayloadLength)  // MaxPayloadLength = 92
    goto Uncommon;  // full Keccak computation, no caching
```

SHA3 opcode on 93-byte input: 48 gas, ~200ns uncached Keccak.
SHA3 opcode on 32-byte input: 36 gas, ~5ns cached.

**Attack:** Repeatedly hash 93-byte inputs:
- 36M / 48 = 750,000 uncached Keccak computations
- ~150ms of pure Keccak work vs ~5ms if cached
- **30x slowdown for inputs just above the 92-byte threshold**

### 4.4 CommitCore Reverse Iteration (MEDIUM)

After each transaction, `PersistentStorageProvider.CommitCore` (`PersistentStorageProvider.cs:125-235`) iterates ALL changes in reverse:
```csharp
for (int i = 0; i <= currentPosition; i++)
{
    Change change = _changes[currentPosition - i];
    if (_committedThisRound.Contains(change!.StorageCell))
        continue;
    // ...
}
```

With 360K journal entries from warm SSTOREs, this iterates 360K times. 359,999 are HashSet.Contains hits (no-op) but each is a real hash + equality check on StorageCell.

### 4.5 State Root Computation Amplification (MEDIUM)

`RecalculateStateRoot` (`StateProvider.cs:61-65`) hashes every dirty trie node from leaf to root:

`(modified accounts) * O(64 nibbles depth) * O(Keccak per node)`

Plus for each modified account with storage changes:
`(modified storage slots) * O(64 nibbles depth) * O(Keccak per node)`

A block touching 1,000 unique storage slots across 100 contracts forces ~1,100+ Keccak computations on trie node RLP. This work is implicit in block finalization, not gas-metered per-slot.

### 4.6 Return Data Copy — Free Memcpy (LOW-MEDIUM)

After a CALL returns, return data is copied back to the caller's memory (`VirtualMachine.cs:1148-1151`). Memory expansion is charged during the CALL, but if the output region was already expanded, the actual memcpy is **free** — no per-word copy gas.

### 4.7 Memory Expansion Array.Clear (LOW)

`src/Nethermind/Nethermind.Evm/EvmPooledMemory.cs:308-349`: Memory expansion does `Array.Clear` on the new region. Well-priced at large sizes due to quadratic gas term, but at 32KB expansions the gas cost is moderate while generating ArrayPool churn and GC pressure in nested call patterns.

---

## 5. Object Pool & Memory Concerns

### 5.1 Unbounded Object Pools

- **VmState pool** (`VmState.cs:160-161`): `ConcurrentQueue` with no size limit
- **ExecutionEnvironment pool** (`ExecutionEnvironment.cs:16-123`): Same pattern
- **StackAccessTracker pool** (`StackAccessTracker.cs:97-124`): Collections retain capacity after `Clear()`

Deep call chains (1024 depth) with parallel prewarming allocate thousands of pooled objects that are never released.

### 5.2 Pinned Stack Memory

`src/Nethermind/Nethermind.Evm/StackPool.cs:13-66`: EVM stacks are 33KB+ each, allocated as pinned arrays. Pool holds up to 2048 stacks (~66MB pinned). GC cannot compact pinned memory, causing heap fragmentation over time.

### 5.3 PerContractState Pool with Capacity Guard

`PersistentStorageProvider.cs:616-651`: The PerContractState pool caps at 2048 entries and rejects items with `BlockChange.Capacity > 512`. Contracts with >512 storage changes per block cause PerContractState objects to be GC'd rather than pooled.

### 5.4 StackList Capacity Retention

`src/Nethermind/Nethermind.Core/Collections/StackList.cs:82-96`: StackList.Return() rejects items with `Capacity > 128`. Large StackLists from high-churn slots are not pooled.

### 5.5 JournalSet/TrackingState Capacity Retention

`StackAccessTracker.cs:115-123`: `Clear()` on `List<T>` and `HashSet<T>` does not release internal arrays. Pooled `TrackingState` objects with large internal buffers stay large forever.

---

## 6. EVM Opcode-Level Details

### 6.1 SSTORE Net Metering (EIP-2200)

`src/Nethermind/Nethermind.Evm/Instructions/EvmInstructions.Storage.cs:442-586`

The net-metered SSTORE calls `GetOriginal` (`PersistentStorageProvider.cs:90-109`) which does `StackList.TryGetSearchedItem` — a binary search O(log N) where N is the number of changes for that cell. With 360K writes to the same slot, the StackList grows large.

### 6.2 KeccakCache Design

- 128K entries, 16MB native memory, one-way set-associative
- Seqlock pattern for lock-free reads
- Fast paths for 32-byte (Hash256) and 20-byte (Address) inputs using SIMD
- **Critical: MaxPayloadLength = 92 bytes** — anything larger bypasses the cache

### 6.3 Function Pointer Dispatch

`VirtualMachine.cs:860-888`: Opcodes dispatch via function pointer table (`delegate*<...>[]`). Table is refreshed every 10,000 transactions (up to 500K total) to pick up JIT PGO recompilations.

### 6.4 EOA Fast-Path

`EvmInstructions.Call.cs:249-303`: Calls to EOAs (no code) skip full call frame setup. But calls to contracts with minimal code (single STOP byte) do NOT hit this fast path, incurring full frame allocation overhead.

### 6.5 EXTCODESIZE Peephole Optimizer

`EvmInstructions.CodeCopy.cs:247-293`: Detects `EXTCODESIZE ISZERO`, `EXTCODESIZE GT` (with 0), `EXTCODESIZE EQ` (with 0) patterns and short-circuits by checking `IsContract()` (code hash only) instead of fetching the full code.

---

## 7. Summary: Attack Severity Rankings

| Attack | Gas Cost/Op | Real Work | Max Ops/Block | Severity |
|---|---|---|---|---|
| Warm SSTORE churn (same slot) | 100 | Journal append + commit scan | 360K | **High** |
| TSTORE + REVERT | 100 | Write + O(K) revert | 714K ops | **High** |
| KeccakCache bypass (93+ bytes) | 48 | Full Keccak computation | 750K | **Medium-High** |
| SeqlockCache thrashing (>32K cells) | 2,100+ | Cache eviction, wasted prewarming | N/A | **Medium-High** |
| CALL-SSTORE-REVERT (cold) | 22,100 | Write + revert + access tracker | Limited | **Medium** |
| State root amplification | Implicit | O(N*D) Keccak for trie nodes | N/A | **Medium** |
| Code cache thrashing (>5K contracts) | 2,600+ | DB reads + analysis | 5K+ | **Medium** |
| JUMPDEST analysis blocking | 32,000+ | Main thread stalls | Limited | **Medium** |
| Cross-sender prewarmer defeat | N/A | Wasted CPU on wrong cache entries | N/A | **Medium** |
| Precompile cache inflation | 137+ | Memory allocation | ~262K | **Low-Medium** |
| Return data free copy | 0 | memcpy | Bounded | **Low-Medium** |
| Unbounded object pools | N/A | Memory growth | N/A | **Low** |
| Pinned stack memory fragmentation | N/A | GC degradation | 2048 stacks | **Low** |

---

## 8. Key File References

| Area | File | Lines |
|---|---|---|
| Block processing loop | `Nethermind.Consensus/Processing/BlockProcessor.BlockValidationTransactionsExecutor.cs` | 31-41 |
| Pre-warmer | `Nethermind.Consensus/Processing/BlockCachePreWarmer.cs` | 28-431 |
| Branch processor pipeline | `Nethermind.Consensus/Processing/BranchProcessor.cs` | 28-169 |
| SeqlockCache | `Nethermind.Core/Collections/SeqlockCache.cs` | 40-400 |
| PreBlockCaches | `Nethermind.State/PreBlockCaches.cs` | 15-72 |
| PrewarmerScopeProvider | `Nethermind.State/PrewarmerScopeProvider.cs` | 30-267 |
| Transaction fee payment | `Nethermind.Evm/TransactionProcessing/TransactionProcessor.cs` | 867-895 |
| Storage journal | `Nethermind.State/PartialStorageProviderBase.cs` | 19-275 |
| Storage commit | `Nethermind.State/PersistentStorageProvider.cs` | 125-325 |
| State provider journal | `Nethermind.State/StateProvider.cs` | 29-993 |
| KeccakCache | `Nethermind.Core/Crypto/KeccakCache.cs` | 25-244 |
| SSTORE (net metered) | `Nethermind.Evm/Instructions/EvmInstructions.Storage.cs` | 442-586 |
| TSTORE/TLOAD | `Nethermind.Evm/Instructions/EvmInstructions.Storage.cs` | 35-127 |
| Code cache | `Nethermind.Evm/CodeInfoRepository.cs` | 21-188 |
| Jump dest analysis | `Nethermind.Evm/CodeAnalysis/JumpDestinationAnalyzer.cs` | 19-399 |
| Precompile caching | `Nethermind.Blockchain/CachedCodeInfoRepository.cs` | 19-94 |
| CALL fast-path | `Nethermind.Evm/Instructions/EvmInstructions.Call.cs` | 249-303 |
| EVM memory | `Nethermind.Evm/EvmPooledMemory.cs` | 16-357 |
| Gas constants | `Nethermind.Evm/GasCostOf.cs` | 6-95 |
| VmState pool | `Nethermind.Evm/VmState.cs` | 160-161 |
| Stack pool (pinned) | `Nethermind.Evm/StackPool.cs` | 13-66 |
| Access tracker | `Nethermind.Evm/StackAccessTracker.cs` | 14-125 |
| JournalSet | `Nethermind.Core/Collections/JournalSet.cs` | 21-64 |
| StackList | `Nethermind.Core/Collections/StackList.cs` | 12-98 |
| Trie dirty nodes cache | `Nethermind.Trie/Pruning/TrieStoreDirtyNodesCache.cs` | 20-100 |
| Blockhash cache | `Nethermind.Blockchain/BlockhashCache.cs` | 20-271 |
