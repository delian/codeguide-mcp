# Verilog/SystemVerilog Development Guidelines
Mandatory standards for synthesizable, verifiable, timing-clean RTL. SystemVerilog (IEEE 1800-2017), Verilator, verible, Icarus (iverilog), cocotb.

---
name: verilog
title: Verilog/SystemVerilog Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [systemverilog@IEEE-1800-2017, verilator@5, verible, iverilog@12, cocotb@1.9]
requires:
  - tdd
recommends:
  - comments
  - performance
provides:
  - synthesizable-rtl
  - blocking-nonblocking
  - fsm-coding
  - hdl-verification
  - cdc
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Verilog/SystemVerilog RTL.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating RTL. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(HDL binding: the "tests" are **testbenches/simulation**; the runner is `verilator --binary` / `iverilog` / a `cocotb` pytest harness. RED = a failing simulation, GREEN = passing sim, REFACTOR = optimize area/timing while sims stay green.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`comments.md`](guides://comments.md) — doc/API-comment policy *(binding: header-block per module, `///<` port comments, Doxygen/Natural Docs generation).*
> - [`performance.md`](guides://performance.md) — the HDL analog is **timing & area** (Fmax, critical path, slack, LUT/FF/DSP/BRAM utilization); pipelining and retiming are the optimizations.

> 📎 **SEE ALSO:** [`error-handling.md`](guides://error-handling.md) *(reset/X-handling is the RTL analog)* · [`ci-cd.md`](guides://ci-cd.md) · [`code-review.md`](guides://code-review.md) · [`secure-coding.md`](guides://secure-coding.md) *(IP integrity, side-channel/trojan formal checks)*

---

## 1. Core Philosophies: SYNTH-FIRST

Verilog/SystemVerilog-specific principles only. TDD/coverage discipline comes from §0.

- **S**ynthesizable by default: production RTL maps to gates — no `#` delays, `fork/join`, `initial` (except testbench), `wait`, or unbounded loops. Simulation-only code is fenced under `` `ifndef SYNTHESIS ``.
- **Y**ield to SV-2017 idioms: `always_comb` / `always_ff` / `always_latch`, `logic` over `reg`/`wire`, `enum`, `typedef struct packed`, `interface`, packages — over legacy Verilog-1995/2001 forms.
- **N**on-blocking discipline: `<=` in clocked (`always_ff`) blocks, `=` in combinational (`always_comb`) blocks — never mix (the cardinal RTL footgun, §5.A).
- **T**estbench-first: every module ships with a simulation/testbench (directed + assertion + functional coverage); UVM for block/SoC-level (§6).
- **H**ierarchy & naming: acyclic module DAG; consistent `i_*`/`o_*` ports, `r_*` registered, `w_*`/`c_*` combinational, `*_n` active-low, `*_pkg` packages.

Cross-cutting **lint-clean, CDC-safe, timing-closed, assertion-checked** discipline is encoded as gates in §2.

**Verified Code**: Agent-generated RTL MUST compile, lint clean, and pass every testbench/gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `VLOG-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| VLOG-TST-01 | Every module MUST be testbench-first (see `tdd.md`) | `verilator --binary <m>.sv <m>_tb.sv && ./obj_dir/V<m>_tb` | exit 0, `$error` count 0 |
| VLOG-TST-02 | Each bug MUST get a regression testbench before the fix (see `tdd.md`) | re-run the bug's `*_tb` | failing→passing |
| VLOG-TST-03 | Functional + code coverage MUST meet target | `verilator --coverage … && verilator_coverage` | line/toggle ≥ 90%, func ≥ 80% |
| VLOG-FMT-01 | Code MUST be formatted | `verible-verilog-format --verify rtl/ tb/` | no diff |
| VLOG-LINT-01 | Lint MUST pass clean | `verilator --lint-only -Wall --sv rtl/*.sv` and `verible-verilog-lint` | exit 0, 0 errors |
| VLOG-SYNTH-01 | No non-synthesizable constructs in `rtl/` | lint / grep for `#`,`fork`,`initial`,`wait` outside `` `ifndef SYNTHESIS `` | none in RTL |
| VLOG-LATCH-01 | No inferred latches; no combinational loops; no multi-driven nets | `verilator --lint-only -Wall` (LATCH/UNOPTFLAT/MULTIDRIVEN) | 0 warnings |
| VLOG-NET-01 | `` `default_nettype none `` set; no implicit nets | grep header of each file | present |
| VLOG-BLK-01 | `<=` only in `always_ff`; `=` only in `always_comb` | lint / review (§5.A) | no mixing |
| VLOG-RST-01 | Every sequential element has a defined reset value & consistent scheme (§7) | review / lint | no unreset state |
| VLOG-CDC-01 | All clock-domain crossings use synchronizers; 0 unsafe CDC | CDC tool (`spyglass -cdc` / `verilator --lint-only` CDC checks) | 0 high/critical |
| VLOG-SVA-01 | Interface/protocol invariants MUST have SVA assertions | review / sim assertion pass | assertions present & passing |
| VLOG-DOC-01 | Public module interfaces documented (see `comments.md`) | doc build / review | header + port comments |

> **Forbidden**: shipping RTL before its testbench (violates `tdd.md`); fixing a bug without a regression testbench first; `=` in a clocked block or `<=` in a combinational block; delays/`fork`/`initial` in synthesizable RTL; implicit nets (missing `` `default_nettype none ``); uninitialized/unreset state; an async signal crossing a clock domain without a synchronizer.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
verible-verilog-format --verify rtl/ tb/        # VLOG-FMT-01
verible-verilog-lint rtl/*.sv                    # VLOG-LINT-01 (style)
verilator --lint-only -Wall --sv rtl/*.sv        # VLOG-LINT-01/SYNTH/LATCH/NET
verilator --binary --trace rtl/*.sv tb/<m>_tb.sv && ./obj_dir/V<m>_tb   # VLOG-TST-01
verilator --binary --coverage rtl/*.sv tb/*_tb.sv && verilator_coverage --annotate cov/ # VLOG-TST-03
# CDC + (optional) formal — owned by the flow's CDC/formal tool        # VLOG-CDC-01
```

`cocotb` alternative for Python-driven testbenches: `make SIM=verilator` (or `icarus`) drives a pytest harness — same gates apply.

The *why* behind each gate (test-first, coverage) lives in `tdd.md`; do not re-derive it here.

---

## 4. Project Structure

Idiomatic RTL layout. Synthesizable source is strictly separated from verification; the module dependency graph is an acyclic DAG (top → submodules → leaf cells).

```
project/
├── rtl/                 # synthesizable RTL ONLY (VLOG-SYNTH-01)
│   ├── top.sv
│   ├── core/            # group by FUNCTION not signal type (cpu/, memory/…)
│   └── pkg/             # *_pkg.sv: shared types, params, enums
├── tb/                  # testbenches (see tdd.md)
│   ├── unit/            # module-level directed/assertion tests
│   └── uvm/             # block/SoC UVM env, sequences
├── syn/constraints.sdc  # clocks, I/O timing, false/multicycle paths
├── lint/                # verible.rules, verilator waivers (*.vlt)
├── sim/Makefile         # verilator/iverilog/cocotb targets
└── README.md
```

- Group by functional domain, not by signal type (no `registers/` `combinational/` dirs).
- Keep modules single-responsibility; split when one file mixes unrelated FSMs/datapaths.
- Shared definitions go in a package, not duplicated `` `include ``s.

---

## 5. RTL Specifics — the unique value

### A. Blocking vs non-blocking (the cardinal rule)
Sequential logic uses **non-blocking `<=`**; combinational logic uses **blocking `=`**. Mixing them creates simulation/synthesis mismatches and races.

```systemverilog
always_ff @(posedge i_clk or negedge i_rst_n) begin   // sequential → <=
    if (!i_rst_n) q <= '0;
    else          q <= d;
end

always_comb begin                                      // combinational → =
    y = a & b;                                          // assign ALL outputs on ALL paths
end
```
Footguns: a combinational `always_comb` that does not assign an output on every path **infers a latch** (`VLOG-LATCH-01`); `<=` in `always_comb` reorders evaluation; `=` in `always_ff` creates shoot-through between pipeline stages.

### B. always-block selection & synthesizable subset
- `always_ff @(posedge clk …)` — registers/state. **Never** put `#` delays inside.
- `always_comb` — pure combinational; tool checks completeness (vs bare `always @*`).
- `always_latch` — only when a latch is intended (rare; justify it).
- Synthesizable subset excludes: `#delay`, `fork/join`, `wait`, `initial` (testbench only), real/time types, dynamic arrays/queues/classes, unbounded `for`. Fence any simulation aid under `` `ifndef SYNTHESIS … `endif ``.

### C. FSM coding style (two-/three-block)
Separate **state register** (`always_ff`), **next-state** (`always_comb`), and **output** logic. Use a `typedef enum logic […]` state type with a `default:` arm so unreachable states recover.

```systemverilog
typedef enum logic [1:0] { IDLE, RUN, DONE } state_t;
state_t state, next;

always_ff @(posedge i_clk or negedge i_rst_n)
    if (!i_rst_n) state <= IDLE; else state <= next;   // state register

always_comb begin                                       // next-state (latch-free default)
    next = state;
    case (state)
        IDLE: if (i_start) next = RUN;
        RUN:  if (i_done)  next = DONE;
        DONE:              next = IDLE;
        default:           next = IDLE;
    endcase
end
```
Binary encoding is area-efficient; one-hot is faster/timing-friendly on FPGAs (let the tool choose, or pin via a synthesis attribute). Moore outputs depend on state only; Mealy outputs depend on state+inputs (register them to avoid combinational paths to outputs).

### D. Reset strategy
Pick **one** scheme per clock domain and apply it consistently (`VLOG-RST-01`). Async-assert/sync-deassert is the common default; fully-synchronous reset suits some FPGA fabrics. Every sequential element gets a defined reset value.

```systemverilog
// Async assert, synchronous deassert of an active-low reset
logic rst_n_sync_q, rst_n_sync;
always_ff @(posedge i_clk or negedge i_rst_n_async)
    if (!i_rst_n_async) {rst_n_sync, rst_n_sync_q} <= '0;
    else                {rst_n_sync, rst_n_sync_q} <= {rst_n_sync_q, 1'b1};
```
Reset-domain crossings (RDC) need the same care as CDC — synchronize the deassertion edge.

### E. Clock-domain crossing (CDC)
Never let a signal cross clock domains combinationally. Single-bit control → 2-FF (or 3-FF) synchronizer; multi-bit data → handshake or async FIFO; never sample a multi-bit bus with parallel 2-FFs (bit skew).

```systemverilog
module cdc_sync (input logic i_clk, i_rst_n, i_d, output logic o_q);
    logic s1;
    always_ff @(posedge i_clk or negedge i_rst_n)
        if (!i_rst_n) {o_q, s1} <= '0;
        else          {o_q, s1} <= {s1, i_d};
endmodule
```
Mark synchronizer paths as false/max-delay in the SDC. Gate CDC correctness with a CDC tool (`VLOG-CDC-01`) — visual review is insufficient.

### F. Parametrization
Parameterize width/depth with `parameter` (overridable) and derive with `localparam`/`$clog2`; never hardcode magic widths. Prefer `'0`/`'1` fills and `$bits()` over literal counts.

```systemverilog
module fifo #(
    parameter int DATA_W = 32,
    parameter int DEPTH  = 16,
    localparam int ADDR_W = $clog2(DEPTH)
)( input logic [DATA_W-1:0] i_wdata, output logic [ADDR_W:0] o_count /* … */ );
```

### G. Common footguns → fix
- Width mismatch / truncation → size literals (`8'hFF`), enable Verilator `-Wall` WIDTH checks.
- Implicit net from a typo → `` `default_nettype none `` (`VLOG-NET-01`).
- `casex/casez` matching X → use `case` with explicit `default` (or `case … inside`).
- Blocking assignment ordering bugs → keep sequential code non-blocking (§5.A).
- Unintended priority/latch in `if` without `else` in combinational logic → assign a default first.

---

## 6. Verification — testbenches, assertions, UVM

Test-first discipline and coverage targets are owned by [`tdd.md`](guides://tdd.md). HDL specialization:

- **Directed testbenches** (`tb/unit/`): drive stimulus, `assert(…) else $error(…)`, include a `$dumpfile/$dumpvars` waveform and a timeout watchdog. RED = sim fails, GREEN = sim passes (`VLOG-TST-01`).
- **SVA (SystemVerilog Assertions)**: encode interface/protocol invariants as concurrent properties (`@(posedge clk) disable iff (!rst_n) …`) inside `` `ifndef SYNTHESIS ``. They are runtime + formal checks (`VLOG-SVA-01`).
- **Functional coverage**: `covergroup`/`coverpoint`/`cross` to measure that scenarios were exercised; meet `VLOG-TST-03` targets.
- **UVM** (block/SoC level): use the UVM 1.2/IEEE-1800.2 component model (`uvm_test`/`env`/`agent`/`sequencer`/`driver`/`monitor`/`scoreboard`) for constrained-random, reusable verification. Reach for UVM when directed tests no longer scale; keep unit modules on lightweight directed/assertion tests.
- **cocotb**: Python-coroutine testbenches over Verilator/Icarus — pairs naturally with a pytest gate and CI.

```systemverilog
// SVA: request must be granted within 3 cycles
assert property (@(posedge i_clk) disable iff (!i_rst_n)
    i_req |-> ##[1:3] o_gnt) else $error("grant timeout");
```

---

## 7. Timing & Area (performance binding)

The HDL analog of `performance.md` is timing/area closure. Policy is owned there; RTL bindings:

- **Constrain** clocks and I/O in the SDC; meet setup/hold with positive slack. Critical path determines Fmax.
- **Pipeline** long combinational paths (register intermediate results) to raise Fmax; balance stages; retiming may rebalance automatically.
- **Area/resource**: watch LUT/FF/DSP/BRAM utilization; share large operators (one multiplier behind a mux) instead of replicating.
- Avoid combinational paths straight from input port to output port; register at boundaries.

---

## 8. Tooling

```bash
verilator --lint-only -Wall --sv rtl/*.sv     # lint / synthesizable-subset checks
verilator --binary --trace rtl/*.sv tb/t.sv   # build+run sim, dump VCD/FST
iverilog -g2012 -o sim rtl/*.sv tb/t.sv && vvp sim   # alt open-source sim
verible-verilog-format --inplace rtl/ tb/     # format
verible-verilog-lint rtl/*.sv                 # style lint
make SIM=verilator                            # cocotb pytest-driven flow
```
Commercial flows (VCS/Questa, SpyGlass CDC, JasperGold formal, Design Compiler synthesis) plug into the same gates — keep the open-source path (Verilator + verible + Icarus + cocotb) authoritative for CI. IP cores are version-pinned submodules with checksum/SPDX integrity (see `secure-coding.md`).

---

## 9. Quick Reference

```bash
verible-verilog-format --inplace rtl/ tb/     # format
verilator --lint-only -Wall --sv rtl/*.sv     # lint
verilator --binary --trace rtl/*.sv tb/*_tb.sv && ./obj_dir/V*    # test
verilator --binary --coverage rtl/*.sv tb/*_tb.sv                 # coverage
make SIM=verilator                            # cocotb
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] VLOG-FMT-01 — `verible-verilog-format --verify` clean
- [ ] VLOG-LINT-01 — Verilator + verible lint, 0 errors
- [ ] VLOG-SYNTH-01 — no non-synthesizable constructs in `rtl/`
- [ ] VLOG-LATCH-01 — no inferred latches / comb loops / multi-driven nets
- [ ] VLOG-NET-01 — `` `default_nettype none ``, no implicit nets
- [ ] VLOG-BLK-01 — `<=` in `always_ff`, `=` in `always_comb` (no mixing)
- [ ] VLOG-RST-01 — every sequential element reset, one consistent scheme
- [ ] VLOG-CDC-01 — all clock crossings synchronized, 0 high/critical CDC
- [ ] VLOG-SVA-01 — interface/protocol assertions present & passing
- [ ] VLOG-TST-01/02/03 — testbenches pass, bugs have regression tests, coverage ≥ target
- [ ] VLOG-DOC-01 — public module interfaces documented
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Verilog/SystemVerilog Guidelines**
