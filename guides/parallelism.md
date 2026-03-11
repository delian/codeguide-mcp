# Parallel and Concurrent Programming Guidelines
Mandatory principles and best practices for parallel, concurrent, and asynchronous programming with emphasis on safety, correctness, performance, and maintainability. Language-agnostic principles for multiprocess, multithreaded, and async programming.

---

**Agent Profile**: The Concurrency Architect
**Role**: Senior Parallel Computing Engineer & Concurrency Specialist
**Objective**: Generate safe, correct, efficient concurrent and parallel code following best practices for race condition prevention, deadlock avoidance, and optimal resource utilization. Prioritize async/await over threads, threads over multiprocessing.
**Tools**: Language-agnostic concurrency patterns, async/await, threads, locks, lock-free structures, atomic operations, memory models.

---

## 1. Core Philosophies: ASYNC-FIRST

The agent must adhere to the **ASYNC-FIRST** principles for every concurrent/parallel implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY concurrency bug MUST receive a test BEFORE fixing to prevent regression.

**CRITICAL CONCURRENCY PRINCIPLES**:
🔴 **Concurrent code is harder to test, debug, and reason about - keep it simple**
🔴 **Prefer async/await over threads, threads over processes**
🔴 **Make race conditions impossible by design, not by careful coding**
🔴 **Immutability and message passing prevent most concurrency bugs**

- **A**sync First: Prefer async/await and event loops over OS threads
- **S**afe by Design: Make race conditions impossible, not just unlikely
- **Y**ield Control: Cooperative scheduling over preemptive when possible
- **N**o Shared Mutable State: Immutability and message passing by default
- **C**lear Ownership: Well-defined data ownership prevents races

- **F**unctional Core: Pure functions, immutable data, no side effects
- **I**solation: Isolate concurrent operations from sequential logic
- **R**ace-Free: Design to prevent races, not just detect them
- **S**ynchronization Minimal: Minimize critical sections, prefer lock-free
- **T**estable: Concurrent code MUST be thoroughly tested

**Additional Principles:**

- **Correctness Over Performance**: Safe code is more important than fast code
- **Simplicity Over Cleverness**: Simple concurrent code is maintainable
- **Message Passing Over Shared Memory**: Communicate by sharing channels, not memory
- **Backpressure Handling**: Always handle flow control and resource limits

**Verified Code**: Agent-generated concurrent code MUST be race-free, deadlock-free, and tested before delivery.

---

## 1A. The Concurrency Hierarchy (MANDATORY)

🔴 **CRITICAL: Follow the concurrency hierarchy - always prefer higher levels**

### The Golden Rule of Concurrency

**Always choose the highest level of abstraction that meets your needs. Lower levels are more dangerous and harder to get right.**

### Concurrency Hierarchy (Prefer Higher → Lower)

```
┌─────────────────────────────────────────────────────────────┐
│ Level 1: Sequential Code (SAFEST)                           │
│   ↓ Only parallelize if proven necessary                    │
│ Level 2: Async/Await (Cooperative Concurrency)              │
│   ↓ Use when I/O-bound or need many concurrent operations   │
│ Level 3: Thread Pools with Immutable Data                   │
│   ↓ Use when CPU-bound with independent work units          │
│ Level 4: Threads with Message Passing                       │
│   ↓ Use when threads need to communicate                    │
│ Level 5: Threads with Locks (DANGEROUS)                     │
│   ↓ Use only when absolutely necessary                      │
│ Level 6: Lock-Free Algorithms (EXPERT ONLY)                 │
│   ↓ Use only when proven bottleneck and expert available    │
│ Level 7: Multiprocessing (HIGHEST OVERHEAD)                 │
│   ↓ Use when need memory isolation or true parallelism      │
└─────────────────────────────────────────────────────────────┘
```

### When to Use Each Level

#### Level 1: Sequential Code (Default)
```
WHEN:
- Performance is adequate
- Problem is inherently sequential
- Code simplicity is critical

WHY:
- No race conditions possible
- Easy to test and debug
- Predictable behavior

EXAMPLE USE CASE:
- Single-threaded processing
- Command-line tools
- Simple scripts
```

#### Level 2: Async/Await (Preferred for I/O)
```
WHEN:
- I/O-bound operations (network, disk, database)
- Need many concurrent operations (thousands)
- Operations spend most time waiting

WHY:
- No race conditions with proper design
- Very low overhead per operation
- Cooperative scheduling prevents many bugs
- Easy to reason about execution order

EXAMPLE USE CASE:
- Web servers handling many requests
- Concurrent API calls
- Database connection pools
- WebSocket servers

ADVANTAGES:
✅ Single-threaded execution (mostly)
✅ No data races on single-threaded runtime
✅ Low memory overhead
✅ Explicit control flow

DISADVANTAGES:
❌ Doesn't use multiple CPU cores
❌ Requires async ecosystem/libraries
❌ One blocking operation blocks entire runtime
```

#### Level 3: Thread Pools with Immutable Data (Preferred for CPU)
```
WHEN:
- CPU-bound operations
- Independent work units
- All data is immutable or copied

WHY:
- Uses multiple CPU cores
- No synchronization needed
- Predictable performance
- Limited number of threads (pool)

EXAMPLE USE CASE:
- Parallel map/reduce operations
- Image processing (per-pixel operations)
- Mathematical computations
- Batch processing

ADVANTAGES:
✅ Uses multiple cores
✅ No data races (immutable data)
✅ Controlled resource usage (pool size)

DISADVANTAGES:
❌ Memory overhead for copying data
❌ Thread creation/destruction cost (mitigated by pool)
```

#### Level 4: Threads with Message Passing (When Threads Must Communicate)
```
WHEN:
- Threads need to communicate
- Can use channels/queues for communication
- State can be isolated

WHY:
- No shared mutable state
- Clear data ownership
- Easier to reason about than locks

EXAMPLE USE CASE:
- Producer-consumer patterns
- Pipeline processing
- Actor model systems

ADVANTAGES:
✅ No explicit locking needed
✅ Clear data flow
✅ Easier to reason about

DISADVANTAGES:
❌ Message passing overhead
❌ Potential deadlocks (channel-based)
❌ Complex error handling
```

#### Level 5: Threads with Locks (Use Sparingly)
```
WHEN:
- Must share mutable state
- Message passing is impractical
- No higher-level alternative exists

WHY:
- Sometimes unavoidable
- Can be efficient when done right
- Legacy systems may require it

EXAMPLE USE CASE:
- Shared caches
- Counter increments
- Complex shared data structures

ADVANTAGES:
✅ Direct memory access (fast)
✅ Fine-grained control

DISADVANTAGES:
❌ Race conditions likely
❌ Deadlocks possible
❌ Hard to test
❌ Hard to debug
❌ Performance bottlenecks (lock contention)
```

#### Level 6: Lock-Free Algorithms (Expert Only)
```
WHEN:
- Proven performance bottleneck
- Lock contention is demonstrated issue
- Expert programmer available
- Extensive testing possible

WHY:
- Highest performance
- No blocking
- Progress guarantees

EXAMPLE USE CASE:
- High-performance queues
- Memory allocators
- Performance-critical libraries

ADVANTAGES:
✅ No blocking
✅ Very high performance
✅ Good scalability

DISADVANTAGES:
❌ Extremely complex
❌ Easy to get wrong
❌ Hard to debug
❌ Platform-specific
❌ Requires deep expertise
```

#### Level 7: Multiprocessing (Highest Isolation)
```
WHEN:
- Need memory isolation
- Running untrusted code
- Must bypass GIL (Python, Ruby)
- Fault isolation required

WHY:
- Complete isolation
- No shared memory races
- Fault tolerance (process crash isolation)

EXAMPLE USE CASE:
- Parallel Python processing (GIL bypass)
- Sandboxed execution
- Distributed computing

ADVANTAGES:
✅ Complete memory isolation
✅ No race conditions across processes
✅ Fault isolation

DISADVANTAGES:
❌ Highest overhead
❌ Slow inter-process communication
❌ Large memory footprint
❌ Process creation cost
```

### Decision Flowchart

```
Start: Need concurrency?
  ↓
NO → Use sequential code
  ↓
YES → Is it I/O-bound?
  ↓                    ↓
YES                   NO
  ↓                    ↓
Use async/await    Is it CPU-bound?
                       ↓
                     YES
                       ↓
                Can data be immutable?
                  ↓              ↓
                YES             NO
                  ↓              ↓
          Thread pool      Message passing?
                                 ↓
                               YES  → Use channels/queues
                                 ↓
                               NO   → Must use locks
                                      (minimize scope!)
```

---

## 2. Async/Await Patterns (PREFERRED)

### A. Async/Await Best Practices

**CRITICAL: Async/await is the preferred concurrency model for I/O-bound operations.**

#### ✅ CORRECT - Async/Await Usage

```pseudocode
// Conceptual async/await pattern (language-agnostic)

// 1. Use async functions for I/O operations
async function fetch_user_data(user_id):
    // Await suspends this function, allowing other work
    user = await database.query("SELECT * FROM users WHERE id = ?", user_id)
    posts = await database.query("SELECT * FROM posts WHERE user_id = ?", user_id)
    return {user: user, posts: posts}

// 2. Run independent operations concurrently
async function fetch_multiple_users(user_ids):
    // Launch all requests concurrently
    promises = user_ids.map(id => fetch_user_data(id))

    // Wait for all to complete
    results = await Promise.all(promises)
    return results

// 3. Handle errors properly
async function safe_fetch(url):
    try:
        response = await http.get(url)
        return {success: true, data: response}
    catch error:
        log_error(error)
        return {success: false, error: error}

// 4. Set timeouts to prevent indefinite waiting
async function fetch_with_timeout(url, timeout_ms):
    timeout_promise = sleep(timeout_ms).then(() => {
        throw new TimeoutError("Request timed out")
    })

    fetch_promise = http.get(url)

    // Race between fetch and timeout
    return await Promise.race([fetch_promise, timeout_promise])

// 5. Implement backpressure
async function process_stream(stream, max_concurrent):
    semaphore = new Semaphore(max_concurrent)

    for item in stream:
        await semaphore.acquire()

        // Process in background, release semaphore when done
        spawn async {
            try:
                await process_item(item)
            finally:
                semaphore.release()
        }
```

#### ❌ WRONG - Async/Await Anti-Patterns

```pseudocode
// ❌ WRONG - Sequential async calls (not concurrent)
async function slow_fetch():
    user = await fetch_user()      // Wait
    posts = await fetch_posts()    // Wait
    comments = await fetch_comments()  // Wait
    // These could run concurrently!

// ✅ CORRECT - Concurrent async calls
async function fast_fetch():
    // Launch all concurrently
    [user, posts, comments] = await Promise.all([
        fetch_user(),
        fetch_posts(),
        fetch_comments()
    ])

// ❌ WRONG - Blocking the event loop
async function bad_async():
    await some_async_operation()
    // Don't do CPU-intensive work here!
    for i in range(1_000_000_000):
        heavy_computation()  // Blocks event loop!

// ✅ CORRECT - Move CPU work to thread pool
async function good_async():
    await some_async_operation()
    // Offload to thread pool
    result = await run_in_thread_pool(heavy_computation)

// ❌ WRONG - Forgetting to await
async function forgot_await():
    fetch_data()  // Returns promise, doesn't wait!
    // Data won't be ready

// ✅ CORRECT - Always await async operations
async function remembered_await():
    await fetch_data()  // Actually waits
```

### B. Async Error Handling

```pseudocode
// Error handling in async code

// 1. Always use try/catch with await
async function safe_operation():
    try:
        result = await risky_operation()
        return result
    catch error:
        // Handle or log error
        log_error(error)
        return default_value

// 2. Handle multiple concurrent errors
async function handle_multiple():
    results = await Promise.allSettled([
        operation1(),
        operation2(),
        operation3()
    ])

    // Check each result
    for result in results:
        if result.status == "rejected":
            log_error(result.reason)

// 3. Cleanup with finally
async function with_cleanup():
    resource = await acquire_resource()
    try:
        await use_resource(resource)
    finally:
        await release_resource(resource)  // Always executes
```

---

## 3. Thread Safety Principles (MANDATORY)

### A. Shared Mutable State (Avoid When Possible)

**CRITICAL: Shared mutable state is the root cause of most concurrency bugs.**

#### The Four Rules of Shared State

1. **No Shared State** (Best): Each thread owns its data
2. **Shared Immutable State** (Good): Read-only data can be freely shared
3. **Shared Mutable State with Message Passing** (OK): Change ownership through channels
4. **Shared Mutable State with Locks** (Last Resort): Protect with synchronization

#### ✅ CORRECT - Avoiding Shared Mutable State

```pseudocode
// Pattern 1: No shared state (BEST)
function process_items_parallel(items):
    // Each thread gets its own copy
    thread_pool.map(items, item => {
        result = process_item(item)  // No sharing
        return result
    })

// Pattern 2: Shared immutable state (GOOD)
immutable config = {
    max_connections: 100,
    timeout: 5000
}

function worker_thread(config):
    // Can read config safely, can't modify
    connect(config.max_connections)

// Pattern 3: Message passing (OK)
channel = new Channel()

// Producer thread
function producer():
    for item in data:
        channel.send(item)  // Transfer ownership
    channel.close()

// Consumer thread
function consumer():
    while item = channel.receive():
        process(item)  // Now owns the item

// Pattern 4: Thread-local storage (GOOD)
thread_local cache = new Map()

function worker():
    // Each thread has its own cache
    cache.put(key, value)  // No synchronization needed
```

#### ❌ WRONG - Dangerous Shared Mutable State

```pseudocode
// ❌ WRONG - Unsynchronized shared mutable state
global counter = 0  // Shared

function worker_thread():
    for i in range(1000):
        counter = counter + 1  // RACE CONDITION!
        // Multiple threads read-modify-write concurrently

// ❌ WRONG - Partial synchronization
global data = {}
lock = new Lock()

function dangerous_update():
    lock.acquire()
    value = data.get(key)  // Protected
    lock.release()

    // DANGER: Data could change here!
    new_value = compute(value)

    lock.acquire()
    data.set(key, new_value)  // Protected
    lock.release()
    // RACE CONDITION between reads!

// ✅ CORRECT - Complete critical section
function safe_update():
    lock.acquire()
    try:
        value = data.get(key)
        new_value = compute(value)
        data.set(key, new_value)  // All protected together
    finally:
        lock.release()
```

### B. Lock Ordering and Deadlock Prevention

**CRITICAL: Deadlocks occur when threads wait for each other in a cycle.**

#### Four Conditions for Deadlock (Break at Least One)

1. **Mutual Exclusion**: Resources cannot be shared
2. **Hold and Wait**: Thread holds resources while waiting for others
3. **No Preemption**: Resources cannot be forcibly taken
4. **Circular Wait**: Circular chain of threads waiting for resources

#### ✅ CORRECT - Deadlock Prevention

```pseudocode
// Strategy 1: Lock ordering (prevent circular wait)
// Always acquire locks in the same global order

global lock_A = new Lock()
global lock_B = new Lock()

function thread_1():
    lock_A.acquire()  // Always A before B
    lock_B.acquire()
    // Do work
    lock_B.release()
    lock_A.release()

function thread_2():
    lock_A.acquire()  // Same order: A before B
    lock_B.acquire()
    // Do work
    lock_B.release()
    lock_A.release()

// Strategy 2: Try-lock with timeout (detect and recover)
function try_acquire_both(lock_A, lock_B, timeout):
    start_time = now()

    while now() - start_time < timeout:
        if lock_A.try_acquire(timeout=100):
            if lock_B.try_acquire(timeout=100):
                return true  // Got both locks
            else:
                lock_A.release()  // Release A, try again

        sleep(random(1, 10))  // Random backoff

    return false  // Timeout

// Strategy 3: Acquire all locks atomically
function atomic_acquire(locks):
    // Either acquire all or none
    all_locks = new Lock()
    all_locks.acquire()

    for lock in locks:
        lock.acquire()

    all_locks.release()
    return locks

// Strategy 4: Use higher-level abstractions
// Message passing avoids explicit locks
channel = new Channel()

function producer():
    channel.send(data)  // No locks

function consumer():
    data = channel.receive()  // No locks
```

#### ❌ WRONG - Deadlock Prone Code

```pseudocode
// ❌ WRONG - Inconsistent lock ordering
lock_A = new Lock()
lock_B = new Lock()

function thread_1():
    lock_A.acquire()
    lock_B.acquire()  // Order: A then B
    // ...
    lock_B.release()
    lock_A.release()

function thread_2():
    lock_B.acquire()  // Order: B then A (OPPOSITE!)
    lock_A.acquire()  // DEADLOCK POSSIBLE
    // ...
    lock_A.release()
    lock_B.release()

// ❌ WRONG - Nested lock acquisition without ordering
function transfer(from_account, to_account, amount):
    from_account.lock.acquire()
    to_account.lock.acquire()  // Deadlock if two threads transfer in opposite directions
    // ...
    to_account.lock.release()
    from_account.lock.release()

// ✅ CORRECT - Use consistent ordering
function safe_transfer(from_account, to_account, amount):
    // Always lock accounts in ID order
    first, second = sort_by_id(from_account, to_account)

    first.lock.acquire()
    second.lock.acquire()
    // Do transfer
    second.lock.release()
    first.lock.release()
```

---

## 4. Memory Models and Synchronization (MANDATORY)

### A. Memory Visibility

**CRITICAL: Without synchronization, changes made by one thread may not be visible to others.**

#### Memory Visibility Rules

```pseudocode
// Modern processors and compilers reorder operations for performance
// Without synchronization, you get UNDEFINED BEHAVIOR

// ❌ WRONG - Assuming immediate visibility
shared flag = false
shared data = 0

function writer_thread():
    data = 42           // Write data
    flag = true         // Set flag

function reader_thread():
    while not flag:     // Wait for flag
        pass
    print(data)         // Might print 0! Not guaranteed to see 42

// ✅ CORRECT - Use proper synchronization

// Option 1: Locks (guarantee visibility)
lock = new Lock()
shared data = 0

function writer():
    lock.acquire()
    data = 42
    lock.release()  // All writes visible after release

function reader():
    lock.acquire()  // See all writes before acquire
    print(data)     // Guaranteed to see 42
    lock.release()

// Option 2: Atomic variables (lighter weight)
atomic flag = Atomic(false)
shared data = 0

function writer():
    data = 42
    flag.store(true, memory_order_release)  // Release semantics

function reader():
    while not flag.load(memory_order_acquire):  // Acquire semantics
        pass
    print(data)  // Guaranteed to see 42

// Option 3: Memory barriers/fences
shared data = 0
shared flag = false

function writer():
    data = 42
    memory_fence_release()  // Ensure data write visible
    flag = true

function reader():
    while not flag:
        pass
    memory_fence_acquire()  // Ensure we see data write
    print(data)
```

### B. Atomic Operations

**CRITICAL: Atomic operations are indivisible - they cannot be interrupted.**

#### ✅ CORRECT - Atomic Operations

```pseudocode
// Atomic operations for simple cases

// 1. Atomic counter increment
atomic counter = Atomic(0)

function increment_counter():
    // Atomic read-modify-write
    counter.fetch_add(1)  // Thread-safe, no lock needed

// 2. Compare-and-swap (CAS) for lock-free algorithms
atomic value = Atomic(0)

function try_update(old_value, new_value):
    // Atomically: if value == old_value, set to new_value
    return value.compare_and_swap(old_value, new_value)

// 3. Atomic flags for signaling
atomic ready = Atomic(false)

function producer():
    // Do work
    ready.store(true)

function consumer():
    while not ready.load():
        spin_or_yield()
    // Proceed

// 4. Atomic pointers for lock-free structures
atomic head = Atomic(null)

function push_lock_free(item):
    loop:
        old_head = head.load()
        item.next = old_head
        if head.compare_and_swap(old_head, item):
            break  // Success
        // Retry if CAS failed
```

#### When to Use Atomic vs Locks

```
Use Atomics When:
✅ Single variable update
✅ Simple read-modify-write
✅ Minimal contention expected
✅ Need wait-free guarantees

Use Locks When:
✅ Multiple variables must stay consistent
✅ Complex operations
✅ Code clarity is important
✅ Moderate contention is OK
```

---

## 5. Common Concurrency Patterns (MANDATORY)

### A. Producer-Consumer Pattern

```pseudocode
// Thread-safe queue with backpressure

class BoundedQueue:
    function __init__(max_size):
        this.queue = []
        this.max_size = max_size
        this.lock = new Lock()
        this.not_full = new Condition(this.lock)
        this.not_empty = new Condition(this.lock)

    function put(item):
        this.lock.acquire()
        try:
            // Wait if queue is full (backpressure)
            while len(this.queue) >= this.max_size:
                this.not_full.wait()

            this.queue.append(item)
            this.not_empty.notify()  // Wake up consumers
        finally:
            this.lock.release()

    function get():
        this.lock.acquire()
        try:
            // Wait if queue is empty
            while len(this.queue) == 0:
                this.not_empty.wait()

            item = this.queue.pop(0)
            this.not_full.notify()  // Wake up producers
            return item
        finally:
            this.lock.release()

// Usage
queue = new BoundedQueue(100)

function producer():
    for item in data_source:
        queue.put(item)  // Blocks if queue full

function consumer():
    while true:
        item = queue.get()  // Blocks if queue empty
        process(item)
```

### B. Thread Pool Pattern

```pseudocode
// Reusable thread pool for CPU-bound work

class ThreadPool:
    function __init__(num_threads):
        this.task_queue = new BoundedQueue(1000)
        this.threads = []
        this.shutdown_flag = Atomic(false)

        // Start worker threads
        for i in range(num_threads):
            thread = new Thread(this.worker)
            thread.start()
            this.threads.append(thread)

    function worker():
        while not this.shutdown_flag.load():
            try:
                task = this.task_queue.get(timeout=1)
                result = task.execute()
                task.set_result(result)
            catch TimeoutError:
                continue  // Check shutdown flag
            catch error:
                task.set_error(error)

    function submit(task):
        if this.shutdown_flag.load():
            throw Error("Pool is shut down")
        this.task_queue.put(task)
        return task.future

    function shutdown():
        this.shutdown_flag.store(true)
        for thread in this.threads:
            thread.join()

// Usage
pool = new ThreadPool(num_cpus())

futures = []
for item in data:
    future = pool.submit(Task(process_item, item))
    futures.append(future)

// Wait for all results
results = futures.map(f => f.get())
pool.shutdown()
```

### C. Read-Write Lock Pattern

```pseudocode
// Allow multiple readers OR one writer

class ReadWriteLock:
    function __init__():
        this.lock = new Lock()
        this.readers = 0
        this.writer = false
        this.reader_cond = new Condition(this.lock)
        this.writer_cond = new Condition(this.lock)

    function acquire_read():
        this.lock.acquire()
        try:
            // Wait while writer is active
            while this.writer:
                this.reader_cond.wait()
            this.readers += 1
        finally:
            this.lock.release()

    function release_read():
        this.lock.acquire()
        try:
            this.readers -= 1
            if this.readers == 0:
                this.writer_cond.notify()  // Wake up waiting writer
        finally:
            this.lock.release()

    function acquire_write():
        this.lock.acquire()
        try:
            // Wait while readers or writer active
            while this.readers > 0 or this.writer:
                this.writer_cond.wait()
            this.writer = true
        finally:
            this.lock.release()

    function release_write():
        this.lock.acquire()
        try:
            this.writer = false
            this.reader_cond.notify_all()  // Wake up all waiting readers
            this.writer_cond.notify()       // Wake up one waiting writer
        finally:
            this.lock.release()

// Usage: Shared cache
cache = {}
rw_lock = new ReadWriteLock()

function read_cache(key):
    rw_lock.acquire_read()
    try:
        return cache.get(key)
    finally:
        rw_lock.release_read()

function write_cache(key, value):
    rw_lock.acquire_write()
    try:
        cache.set(key, value)
    finally:
        rw_lock.release_write()
```

---

## 6. Testing Concurrent Code (MANDATORY)

### A. Testing Strategies

**CRITICAL: Concurrent bugs are non-deterministic - extensive testing required.**

#### Testing Approaches

```pseudocode
// 1. Stress testing - expose race conditions
function stress_test_counter():
    counter = new AtomicCounter()
    num_threads = 10
    increments_per_thread = 10000

    threads = []
    for i in range(num_threads):
        thread = new Thread(() => {
            for j in range(increments_per_thread):
                counter.increment()
        })
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    expected = num_threads * increments_per_thread
    actual = counter.get()

    assert(actual == expected, f"Race condition! Expected {expected}, got {actual}")

// 2. Property-based testing
function test_queue_fifo_property():
    queue = new ConcurrentQueue()
    items = [1, 2, 3, 4, 5]

    // Producer thread
    producer = new Thread(() => {
        for item in items:
            queue.enqueue(item)
    })

    // Consumer thread
    received = []
    consumer = new Thread(() => {
        for i in range(len(items)):
            item = queue.dequeue()
            received.append(item)
    })

    producer.start()
    consumer.start()
    producer.join()
    consumer.join()

    // Property: Items should be in FIFO order
    assert(received == items, "FIFO property violated!")

// 3. Invariant checking
class BankAccount:
    invariant: balance >= 0

    function transfer(to_account, amount):
        // Check invariant before
        assert(this.balance >= 0)
        assert(to_account.balance >= 0)

        this.lock.acquire()
        to_account.lock.acquire()
        try:
            if this.balance >= amount:
                this.balance -= amount
                to_account.balance += amount
            else:
                throw InsufficientFundsError()
        finally:
            to_account.lock.release()
            this.lock.release()

        // Check invariant after
        assert(this.balance >= 0)
        assert(to_account.balance >= 0)

// 4. Happens-before testing
function test_happens_before():
    shared_value = 0
    barrier = new Barrier(2)

    function writer():
        shared_value = 42
        barrier.wait()  // Synchronization point

    function reader():
        barrier.wait()  // Synchronization point
        // After barrier, write MUST be visible
        assert(shared_value == 42)

    thread1 = new Thread(writer)
    thread2 = new Thread(reader)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

// 5. Thread sanitizer / race detector
// Many languages provide tools to detect races
// Example: Run with -race flag (Go), ThreadSanitizer (C/C++)
//
// These tools instrument code to detect:
// - Data races
// - Deadlocks
// - Lock order violations
```

### B. Concurrency Testing Best Practices

```
1. Test with Different Thread Counts
   - Test with 1, 2, 4, 8, 16+ threads
   - Expose different race conditions

2. Repeat Tests Many Times
   - Race conditions are non-deterministic
   - Run test 1000+ times to increase confidence

3. Add Random Delays
   - Sleep random amounts to vary timing
   - Expose races that only occur with specific timing

4. Use Thread Sanitizers
   - Enable race detectors in CI/CD
   - Fail builds on detected races

5. Test Under Load
   - Test with high CPU usage
   - Test with memory pressure
   - Test with I/O contention

6. Test Shutdown/Cleanup
   - Ensure threads terminate cleanly
   - No resource leaks
   - Proper error propagation

7. Test Error Paths
   - What happens on timeout?
   - What happens on exception?
   - Ensure resources are released
```

---

## 7. Security Considerations (MANDATORY)

### A. Concurrency-Related Security Issues

**CRITICAL: Concurrent code can introduce security vulnerabilities.**

#### Time-of-Check Time-of-Use (TOCTOU) Races

```pseudocode
// ❌ WRONG - TOCTOU vulnerability
function unsafe_file_operation(filename):
    // Check if file exists
    if file_exists(filename):  // TIME OF CHECK
        // DANGER: File could be replaced here by attacker!
        // With a symlink to sensitive file

        content = read_file(filename)  // TIME OF USE
        // Might read sensitive file!

// ✅ CORRECT - Atomic operations
function safe_file_operation(filename):
    // Open file atomically with appropriate flags
    file = open_file(filename, flags=O_RDONLY | O_NOFOLLOW)
    // O_NOFOLLOW prevents symlink attacks

    content = read_file(file)
    close_file(file)
```

#### Resource Exhaustion Attacks

```pseudocode
// ❌ WRONG - Unbounded concurrency
async function handle_request(request):
    // No limit on concurrent operations!
    spawn async {
        // Attacker can create unlimited async tasks
        await process_request(request)
    }

// ✅ CORRECT - Bounded concurrency with semaphore
semaphore = new Semaphore(MAX_CONCURRENT_REQUESTS)

async function handle_request(request):
    // Limit concurrent operations
    if not await semaphore.try_acquire(timeout=100):
        return error("Service busy, try again later")

    try:
        await process_request(request)
    finally:
        semaphore.release()
```

#### Denial of Service via Locks

```pseudocode
// ❌ WRONG - Lock held during I/O
lock = new Lock()

function vulnerable_operation():
    lock.acquire()
    try:
        // DANGER: Holding lock during I/O
        data = read_from_network()  // Can block indefinitely!
        process(data)
    finally:
        lock.release()
    // Attacker can DoS by making network slow

// ✅ CORRECT - Minimize lock scope
lock = new Lock()

function safe_operation():
    // Do I/O without lock
    data = read_from_network_with_timeout(timeout=5)

    // Only lock for memory operations
    lock.acquire()
    try:
        process(data)
    finally:
        lock.release()
```

---

## 8. Performance Optimization (MANDATORY)

### A. Avoiding Common Performance Pitfalls

```pseudocode
// PITFALL 1: Lock Contention
// ❌ WRONG - Single lock for everything
global_lock = new Lock()
data = {}

function get(key):
    global_lock.acquire()
    value = data.get(key)
    global_lock.release()
    return value

// ✅ CORRECT - Sharded locks
NUM_SHARDS = 16
locks = [new Lock() for i in range(NUM_SHARDS)]
data = [{} for i in range(NUM_SHARDS)]

function get_shard(key):
    return hash(key) % NUM_SHARDS

function get(key):
    shard = get_shard(key)
    locks[shard].acquire()
    value = data[shard].get(key)
    locks[shard].release()
    return value

// PITFALL 2: False Sharing
// ❌ WRONG - Adjacent data accessed by different threads
struct SharedCounters:
    counter1: int  // CPU cache line
    counter2: int  // Same cache line!
    // Threads modifying counter1 and counter2 fight over cache line

// ✅ CORRECT - Pad to separate cache lines
CACHE_LINE_SIZE = 64

struct PaddedCounter:
    counter: int
    padding: byte[CACHE_LINE_SIZE - sizeof(int)]

counters = [PaddedCounter() for i in range(num_threads)]
// Each counter in its own cache line

// PITFALL 3: Excessive Synchronization
// ❌ WRONG - Lock per operation
function sum_array(array):
    total = 0
    lock = new Lock()

    parallel_for element in array:
        lock.acquire()
        total += element  // Synchronize each addition!
        lock.release()

    return total

// ✅ CORRECT - Per-thread accumulation
function sum_array(array):
    num_threads = num_cpus()
    partial_sums = [0 for i in range(num_threads)]

    parallel_for (i, element) in enumerate(array):
        thread_id = i % num_threads
        // No synchronization needed!
        partial_sums[thread_id] += element

    // Single reduction at end
    return sum(partial_sums)
```

---

## 9. Summary

**CRITICAL Requirements for All Concurrent Code:**

**CONCURRENCY HIERARCHY (PREFER HIGHER → LOWER):**
1. 🟢 **Sequential Code**: Default choice (safest)
2. 🟢 **Async/Await**: Preferred for I/O-bound work
3. 🟡 **Thread Pool + Immutable Data**: Good for CPU-bound work
4. 🟡 **Message Passing**: Good when threads must communicate
5. 🟠 **Locks**: Use sparingly, minimize critical sections
6. 🔴 **Lock-Free**: Expert only, when proven bottleneck
7. 🔴 **Multiprocessing**: When isolation needed, highest overhead

**CORE PRINCIPLES:**
- **Immutability**: Prefer immutable data over locks
- **Message Passing**: Share by communicating, not memory
- **Minimize Scope**: Keep critical sections small
- **Lock Ordering**: Prevent deadlocks with consistent order
- **Testing**: Test extensively with race detectors
- **Backpressure**: Always handle resource limits
- **Timeouts**: Never wait indefinitely

**SAFETY CHECKLIST:**
- [ ] No shared mutable state (or properly synchronized)
- [ ] No race conditions possible
- [ ] No deadlock possible (lock ordering)
- [ ] Memory visibility guaranteed (proper synchronization)
- [ ] Bounded resource usage (no DoS)
- [ ] Timeout on all blocking operations
- [ ] Clean shutdown implemented
- [ ] Error handling in all threads
- [ ] Thoroughly tested with thread sanitizers

**REMEMBER:**
🔴 **Make races impossible by design, not careful coding**
🔴 **Prefer async/await for I/O, thread pools for CPU work**
🔴 **Test concurrent code extensively - bugs are non-deterministic**
🔴 **Correctness > Performance - safe code first**

**End of Parallel and Concurrent Programming Guidelines**
