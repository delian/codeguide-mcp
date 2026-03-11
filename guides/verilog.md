# Verilog/SystemVerilog Development Guidelines
Mandatory coding standards and development practices for Verilog/SystemVerilog RTL design. Build synthesizable, verifiable, timing-clean hardware designs. Tools: Verilog-2001, SystemVerilog-2017, Verilator, UVM, SpyGlass.

---

**Agent Profile**: The Verilog/SystemVerilog RTL Design Expert
**Role**: Senior Digital Design Engineer & Verification Architect
**Objective**: Generate production-ready, synthesizable, timing-clean RTL code.
**Tools**: SystemVerilog (IEEE 1800-2017), Verilator 5.x, Synopsys VCS, Questa/ModelSim, UVM 1.2, SpyGlass Lint, JasperGold

---

## 1. Core Philosophies: SYNTH-FIRST

The agent must adhere to the **SYNTH-FIRST** principles for every Verilog/SystemVerilog implementation:

**Test-Driven Development (TDD)**: ALWAYS write testbenches BEFORE RTL implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory lint checking, CDC/RDC verification, and timing analysis.

- **S**ynthesizable: All RTL code must be synthesizable to gates (no delays, fork/join, or behavioral constructs in production RTL)
- **Y**early: Use modern SystemVerilog-2017 features (always_comb, always_ff, logic, interfaces) over legacy Verilog-1995
- **N**on-blocking for Sequential: Always use non-blocking assignments (<=) in clocked always blocks
- **T**estable: Every module must have corresponding UVM/SystemVerilog testbench with functional coverage
- **H**ierarchical: Use clear module hierarchy with proper signal naming conventions (i_*, o_*, r_*, w_*)

**Additional Principles:**

- **Lint-Clean**: Code MUST pass lint checks (SpyGlass, Verilator --lint-only, Questa Lint) with zero errors
- **Timing-Aware**: Design with clock domain crossing (CDC) and reset domain crossing (RDC) in mind
- **Assertion-Based**: Use SVA (SystemVerilog Assertions) for formal verification and runtime checks
- **No X-Propagation**: Avoid unknown (X) values in simulation; use 4-state logic conservatively
- **Reset Strategy**: All sequential elements must have consistent reset (async reset recommended)

**Verified Code**: Agent-generated RTL MUST compile, lint-clean, synthesize, and pass all testbenches before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Verilog/SystemVerilog code compiles, lints clean, and passes simulation before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Verilog/SystemVerilog code, the agent MUST:**

1. **Compilation Check**:
   ```bash
   # Compile with SystemVerilog compiler (Verilator recommended for open-source)
   verilator --lint-only -Wall --sv -Wno-UNUSED <design>.sv
   # Exit code MUST be 0

   # Or with commercial simulator
   vlog -sv <design>.sv
   ```
   - **MUST** compile without errors
   - No syntax errors or undeclared signals
   - All module port connections must match
   - No implicit net declarations (use `default_nettype none)

2. **Lint Verification**:
   ```bash
   # Verilator lint (open-source)
   verilator --lint-only -Wall --sv <design>.sv

   # SpyGlass (commercial - if available)
   spyglass -batch lint_check.tcl
   ```
   - **MUST** have zero lint errors
   - Warnings should be justified or fixed
   - Check for combinational loops, latches, multi-driven signals

3. **Simulation & Testbench**:
   ```bash
   # Run testbench simulation
   vcs -sverilog <design>.sv <testbench>.sv -o simv
   ./simv +UVM_TESTNAME=base_test
   ```
   - All testbenches must pass (exit code 0)
   - Functional coverage must meet targets (>80%)
   - No unknown (X) or high-impedance (Z) values in critical signals
   - Assertions must all pass

4. **Security & Dependency Verification (MANDATORY)**:
   ```bash
   # Check for CDC violations
   spyglass -cdc <design>.sv

   # Check for security vulnerabilities (side-channel, trojan insertion points)
   jaspergold -prove <security_properties>.sva
   ```
   - **MUST** have 0 high/critical CDC violations
   - All clock domain crossings must use proper synchronizers
   - Security-critical modules must pass formal verification
   - IP integrity verified (no unauthorized RTL modifications)

5. **Documentation Verification**:
   ```bash
   # Generate documentation from inline comments
   doxygen Doxyfile
   # Or natural docs for Verilog
   naturaldocs -i . -o HTML docs/
   ```
   - All public module interfaces have documentation
   - Port descriptions follow naming conventions
   - Examples compile and simulate successfully

#### Error Correction Process

If verification fails:

1. **Compilation Errors**:
   - Read full error message from compiler
   - Check syntax, port mismatches, undeclared signals
   - Verify module instantiation parameter passing
   - Re-compile and verify

2. **Lint Errors**:
   - Fix inferred latches (ensure all outputs assigned in all branches)
   - Eliminate combinational loops
   - Fix multi-driven nets (resolve multiple drivers)
   - Address bit-width mismatches
   - Re-lint until clean

3. **Simulation Failures**:
   - Run failing test in isolation with waveform dump
   - Check test expectations vs actual RTL behavior
   - Verify reset sequence and initial conditions
   - Fix logic errors and re-run ALL tests for regression

### B. Agent Workflow Example

**Complete Verilog/SystemVerilog generation workflow:**

1. **Generate Module Structure**:
   ```
   project/
   ├── rtl/
   │   ├── top.sv
   │   ├── core/
   │   │   ├── alu.sv
   │   │   └── regfile.sv
   │   └── pkg/
   │       └── common_pkg.sv
   ├── tb/
   │   ├── top_tb.sv
   │   └── uvm/
   │       ├── base_test.sv
   │       └── sequences/
   ├── syn/
   │   └── constraints.sdc
   └── sim/
       └── Makefile
   ```

2. **Generate Initial RTL**:
   ```systemverilog
   // rtl/core/alu.sv
   `default_nettype none

   module alu #(
       parameter int WIDTH = 32
   ) (
       input  wire              i_clk,
       input  wire              i_rst_n,
       input  wire [WIDTH-1:0]  i_operand_a,
       input  wire [WIDTH-1:0]  i_operand_b,
       input  wire [3:0]        i_opcode,
       input  wire              i_valid,
       output logic [WIDTH-1:0] o_result,
       output logic             o_valid
   );

       // Sequential logic
       always_ff @(posedge i_clk or negedge i_rst_n) begin
           if (!i_rst_n) begin
               o_result <= '0;
               o_valid  <= '0;
           end else begin
               o_valid <= i_valid;
               // Combinational result registered
               case (i_opcode)
                   4'h0: o_result <= i_operand_a + i_operand_b;
                   4'h1: o_result <= i_operand_a - i_operand_b;
                   4'h2: o_result <= i_operand_a & i_operand_b;
                   4'h3: o_result <= i_operand_a | i_operand_b;
                   default: o_result <= '0;
               endcase
           end
       end

       // Assertions
       `ifndef SYNTHESIS
       assert property (@(posedge i_clk) disable iff (!i_rst_n)
           i_valid |-> ##1 o_valid)
           else $error("Valid signal not propagated");
       `endif

   endmodule

   `default_nettype wire
   ```

3. **Verify Compilation**:
   ```bash
   verilator --lint-only -Wall --sv rtl/core/alu.sv
   # ✓ Compilation successful, no lint errors
   ```

4. **Add Testbench**:
   ```systemverilog
   // tb/alu_tb.sv
   `timescale 1ns/1ps

   module alu_tb;
       parameter WIDTH = 32;
       parameter CLK_PERIOD = 10ns;

       logic              clk;
       logic              rst_n;
       logic [WIDTH-1:0]  operand_a;
       logic [WIDTH-1:0]  operand_b;
       logic [3:0]        opcode;
       logic              valid_in;
       logic [WIDTH-1:0]  result;
       logic              valid_out;

       // DUT instantiation
       alu #(.WIDTH(WIDTH)) dut (
           .i_clk(clk),
           .i_rst_n(rst_n),
           .i_operand_a(operand_a),
           .i_operand_b(operand_b),
           .i_opcode(opcode),
           .i_valid(valid_in),
           .o_result(result),
           .o_valid(valid_out)
       );

       // Clock generation
       initial begin
           clk = 0;
           forever #(CLK_PERIOD/2) clk = ~clk;
       end

       // Test stimulus
       initial begin
           // Reset sequence
           rst_n = 0;
           valid_in = 0;
           #(CLK_PERIOD*2);
           rst_n = 1;
           #(CLK_PERIOD);

           // Test addition
           operand_a = 32'd10;
           operand_b = 32'd20;
           opcode = 4'h0;
           valid_in = 1;
           @(posedge clk);
           @(posedge clk);
           assert(result == 32'd30) else $error("Addition failed");

           $display("All tests passed!");
           $finish;
       end

       // Waveform dump
       initial begin
           $dumpfile("alu_tb.vcd");
           $dumpvars(0, alu_tb);
       end
   endmodule
   ```

5. **Run Simulation**:
   ```bash
   verilator -Wall --trace --exe --build -j alu_tb.cpp rtl/core/alu.sv tb/alu_tb.sv
   ./obj_dir/Valu_tb
   # ✓ All tests pass
   ```

6. **Final Verification**:
   ```bash
   # Lint check
   verilator --lint-only -Wall --sv rtl/core/alu.sv
   # ✓ No errors

   # Run full regression
   make regression
   # ✓ All tests passed
   ```

7. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver Verilog/SystemVerilog code that:**
- [ ] Fails lint verification (latches, loops, multi-driven nets)
- [ ] Has failing testbenches or assertions
- [ ] Lacks testbenches for RTL modules
- [ ] Uses blocking assignments (=) in clocked always blocks
- [ ] Uses non-blocking assignments (<=) in combinational always blocks
- [ ] Has uninitialized registers without reset
- [ ] Uses delays (#) or fork/join in synthesizable RTL
- [ ] Contains implicit net declarations (missing `default_nettype none)
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes RTL before writing testbench (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new modules**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new RTL modules.**

### TDD Cycle for Hardware

```
1. RED: Write a failing testbench first
   ↓
2. GREEN: Write minimal RTL to make it pass
   ↓
3. REFACTOR: Optimize for area/timing while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Verilog

```systemverilog
// Step 1: RED - Write failing testbench first
module counter_tb;
    logic clk, rst_n, enable;
    logic [7:0] count;

    counter dut (.*);

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0;
        enable = 0;
        #20 rst_n = 1;
        #10 enable = 1;
        #100;
        assert(count == 8'd10) else $error("Count incorrect");
        $finish;
    end
endmodule

// Run: verilator --binary --trace counter.sv counter_tb.sv
// FAILS - Module 'counter' not found

// Step 2: GREEN - Write minimal RTL implementation
module counter (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       enable,
    output logic [7:0] count
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= '0;
        else if (enable)
            count <= count + 1;
    end
endmodule

// Run: verilator --binary --trace counter.sv counter_tb.sv && ./obj_dir/Vcounter_tb
// PASSES - testbench passes

// Step 3: REFACTOR - Add parameterization, optimize
module counter #(
    parameter int WIDTH = 8
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             enable,
    output logic [WIDTH-1:0] count
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= '0;
        else if (enable)
            count <= count + 1'b1;
    end

    // Add assertions
    assert property (@(posedge clk) disable iff (!rst_n)
        !enable |-> $stable(count));
endmodule
// Tests still pass with improved design
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for Hardware

```
1. Bug Reported/Discovered (synthesis mismatch, timing violation, functional bug)
   ↓
2. Write a testbench that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason (check waveforms)
   ↓
4. Fix the RTL bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug tracker ID)
   ↓
7. Synthesize and verify timing (regression prevented)
```

### Example Bug Fix

```systemverilog
// Bug Report #42: ALU produces incorrect result for subtract operation
// when operands are equal (should produce 0, produces garbage)

// Step 1-2: Write testbench that reproduces the bug
module alu_bug42_tb;
    logic [31:0] a, b, result;
    logic [3:0] opcode;

    alu dut (
        .i_operand_a(a),
        .i_operand_b(b),
        .i_opcode(opcode),
        .o_result(result)
    );

    initial begin
        // Bug #42: Subtract equal operands
        a = 32'h1234_5678;
        b = 32'h1234_5678;
        opcode = 4'h1; // Subtract
        #10;
        assert(result == 32'h0000_0000)
            else $error("Bug #42: Subtract equal operands failed, got %h", result);
        $finish;
    end
endmodule

// Run: verilator --binary alu.sv alu_bug42_tb.sv && ./obj_dir/Valu_bug42_tb
// FAILS - Bug #42: Subtract equal operands failed, got 32'hxxxxxxxx

// Step 3: Root cause - uninitialized variable in combinational logic

// Step 4: Fix the bug in RTL
module alu (
    input  logic [31:0] i_operand_a,
    input  logic [31:0] i_operand_b,
    input  logic [3:0]  i_opcode,
    output logic [31:0] o_result
);
    always_comb begin
        o_result = '0;  // BUG FIX: Initialize to prevent X propagation
        case (i_opcode)
            4'h0: o_result = i_operand_a + i_operand_b;
            4'h1: o_result = i_operand_a - i_operand_b;  // Now works correctly
            default: o_result = '0;
        endcase
    end
endmodule

// Run: verilator --binary alu.sv alu_bug42_tb.sv && ./obj_dir/Valu_bug42_tb
// PASSES - Bug #42 fixed, regression test prevents recurrence
```

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard RTL Project Layout

**Follow the standard Verilog/SystemVerilog project layout:**

```
project/
├── rtl/                      # Synthesizable RTL source
│   ├── top.sv               # Top-level module
│   ├── core/                # Core processing modules
│   │   ├── datapath.sv
│   │   ├── control.sv
│   │   └── alu.sv
│   ├── peripherals/         # Peripheral modules
│   │   ├── uart.sv
│   │   └── spi.sv
│   └── pkg/                 # SystemVerilog packages
│       ├── common_pkg.sv    # Common types/parameters
│       └── axi_pkg.sv       # Bus protocol definitions
├── tb/                      # Testbenches
│   ├── unit/               # Unit tests (module-level)
│   │   ├── alu_tb.sv
│   │   └── uart_tb.sv
│   └── integration/        # Integration tests (system-level)
│       ├── top_tb.sv
│       └── uvm/            # UVM verification environment
│           ├── base_test.sv
│           ├── env.sv
│           └── sequences/
├── sim/                    # Simulation scripts
│   ├── Makefile
│   └── run.do
├── syn/                    # Synthesis scripts
│   ├── constraints.sdc     # Timing constraints
│   └── synthesis.tcl
├── lint/                   # Lint configuration
│   ├── spyglass.prj
│   └── verilator.vlt       # Verilator waivers
├── docs/                   # Documentation
│   ├── specifications/
│   └── block_diagrams/
└── README.md
```

### B. Module Organization Principles

**Follow these principles for RTL organization:**

1. **Group by Function, Not by Type**:
   ```
   CORRECT - Group by functional domain
   rtl/
   ├── cpu/
   │   ├── fetch.sv
   │   ├── decode.sv
   │   └── execute.sv
   └── memory/
       ├── cache.sv
       └── arbiter.sv

   WRONG - Group by signal type
   rtl/
   ├── registers/
   ├── combinational/
   └── state_machines/
   ```

2. **Keep Modules Small and Focused**:
   - Each module should have a clear, single responsibility
   - Aim for < 500 lines per module (excluding testbenches)
   - Use hierarchical instantiation for large designs

3. **Avoid Circular Dependencies**:
   - Module dependency graph must be acyclic (DAG)
   - Use interfaces or packages for shared definitions
   - Top-down hierarchy: top → sub-modules → leaf cells

---

## 4. RTL Design Architecture (MANDATORY)

### A. Architecture Overview

**MANDATORY: Use Register-Transfer Level (RTL) architecture with clear separation:**

```
┌─────────────────────────────────────────┐
│          TOP MODULE (Integration)        │
│  ┌─────────────────────────────────┐    │
│  │   CONTROL PATH (FSM/Controller) │    │
│  │  • State machines               │    │
│  │  • Control signals              │    │
│  └──────────┬──────────────────────┘    │
│             │ control signals            │
│             ▼                            │
│  ┌─────────────────────────────────┐    │
│  │   DATAPATH (Processing)         │    │
│  │  • ALU, registers, muxes        │    │
│  │  • Arithmetic/logic operations  │    │
│  └─────────────────────────────────┘    │
│             ▲                            │
│             │ data                       │
│  ┌──────────┴──────────────────────┐    │
│  │   INTERFACES (I/O)              │    │
│  │  • Protocol converters          │    │
│  │  • Clock domain crossings       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### B. Implementation Example

```systemverilog
// Package: Common definitions
package common_pkg;
    typedef enum logic [1:0] {
        IDLE   = 2'b00,
        ACTIVE = 2'b01,
        DONE   = 2'b10
    } state_t;

    typedef struct packed {
        logic [31:0] data;
        logic        valid;
        logic        ready;
    } data_channel_t;
endpackage

// Control Path: FSM Controller
module controller (
    input  logic       i_clk,
    input  logic       i_rst_n,
    input  logic       i_start,
    input  logic       i_done,
    output logic       o_enable,
    output logic       o_busy
);
    import common_pkg::*;

    state_t state, next_state;

    // State register
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE:   if (i_start) next_state = ACTIVE;
            ACTIVE: if (i_done)  next_state = DONE;
            DONE:   next_state = IDLE;
        endcase
    end

    // Output logic
    assign o_enable = (state == ACTIVE);
    assign o_busy   = (state != IDLE);
endmodule

// Datapath: Processing unit
module datapath #(
    parameter int WIDTH = 32
) (
    input  logic              i_clk,
    input  logic              i_rst_n,
    input  logic              i_enable,
    input  logic [WIDTH-1:0]  i_data,
    output logic [WIDTH-1:0]  o_result,
    output logic              o_done
);
    logic [WIDTH-1:0] accumulator;
    logic [3:0]       counter;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            accumulator <= '0;
            counter     <= '0;
            o_done      <= '0;
        end else if (i_enable) begin
            accumulator <= accumulator + i_data;
            counter     <= counter + 1'b1;
            o_done      <= (counter == 4'd9);
        end else begin
            accumulator <= '0;
            counter     <= '0;
            o_done      <= '0;
        end
    end

    assign o_result = accumulator;
endmodule

// Top-Level: Integration
module top (
    input  logic        i_clk,
    input  logic        i_rst_n,
    input  logic        i_start,
    input  logic [31:0] i_data,
    output logic [31:0] o_result,
    output logic        o_busy
);
    logic enable, done;

    controller ctrl (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_start(i_start),
        .i_done(done),
        .o_enable(enable),
        .o_busy(o_busy)
    );

    datapath dp (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_enable(enable),
        .i_data(i_data),
        .o_result(o_result),
        .o_done(done)
    );
endmodule
```

**Benefits:**
- Clear separation of control and data flow
- Easier to verify and debug independently
- Synthesizes efficiently with good timing closure

---

## 5. Design Patterns (MANDATORY)

### A. State Machine Pattern (FSM)

**Use explicit state encoding for finite state machines:**

```systemverilog
module fsm (
    input  logic       i_clk,
    input  logic       i_rst_n,
    input  logic       i_input,
    output logic [1:0] o_output
);
    // Explicit state encoding (one-hot or binary)
    typedef enum logic [2:0] {
        S0 = 3'b001,  // One-hot encoding
        S1 = 3'b010,
        S2 = 3'b100
    } state_t;

    state_t state, next_state;

    // State register (sequential)
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            state <= S0;
        else
            state <= next_state;
    end

    // Next state logic (combinational)
    always_comb begin
        next_state = state;  // Default: stay in current state
        case (state)
            S0: if (i_input) next_state = S1;
            S1: next_state = S2;
            S2: next_state = S0;
            default: next_state = S0;  // Safe default
        endcase
    end

    // Output logic (combinational or registered)
    always_comb begin
        o_output = 2'b00;
        case (state)
            S0: o_output = 2'b00;
            S1: o_output = 2'b01;
            S2: o_output = 2'b10;
            default: o_output = 2'b00;
        endcase
    end
endmodule
```

**Benefits:**
- Clear separation of sequential and combinational logic
- Easier synthesis and timing analysis
- No inferred latches

### B. Pipeline Pattern

**Use pipeline registers for high-frequency designs:**

```systemverilog
module pipeline_multiplier (
    input  logic        i_clk,
    input  logic        i_rst_n,
    input  logic [15:0] i_a,
    input  logic [15:0] i_b,
    output logic [31:0] o_product
);
    // Stage 1: Partial products
    logic [31:0] partial_products [3:0];
    logic [31:0] stage1_pp0, stage1_pp1, stage1_pp2, stage1_pp3;

    always_comb begin
        partial_products[0] = i_b[0] ? {16'b0, i_a} : 32'b0;
        partial_products[1] = i_b[1] ? {15'b0, i_a, 1'b0} : 32'b0;
        partial_products[2] = i_b[2] ? {14'b0, i_a, 2'b0} : 32'b0;
        partial_products[3] = i_b[3] ? {13'b0, i_a, 3'b0} : 32'b0;
    end

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            stage1_pp0 <= '0;
            stage1_pp1 <= '0;
            stage1_pp2 <= '0;
            stage1_pp3 <= '0;
        end else begin
            stage1_pp0 <= partial_products[0];
            stage1_pp1 <= partial_products[1];
            stage1_pp2 <= partial_products[2];
            stage1_pp3 <= partial_products[3];
        end
    end

    // Stage 2: Sum partial products
    logic [31:0] stage2_sum;

    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            stage2_sum <= '0;
        else
            stage2_sum <= stage1_pp0 + stage1_pp1 + stage1_pp2 + stage1_pp3;
    end

    assign o_product = stage2_sum;
endmodule
```

---

## 6. Configuration & Environment (MANDATORY)

### A. Parameter Configuration

**Use parameters and localparams for configurability:**

```systemverilog
module configurable_fifo #(
    parameter int DATA_WIDTH = 32,
    parameter int DEPTH      = 16,
    localparam int ADDR_WIDTH = $clog2(DEPTH)
) (
    input  logic                  i_clk,
    input  logic                  i_rst_n,
    input  logic [DATA_WIDTH-1:0] i_wdata,
    input  logic                  i_wr_en,
    output logic                  o_full,
    output logic [DATA_WIDTH-1:0] o_rdata,
    input  logic                  i_rd_en,
    output logic                  o_empty
);
    logic [DATA_WIDTH-1:0] mem [DEPTH];
    logic [ADDR_WIDTH:0]   wr_ptr, rd_ptr;

    // Implementation...
endmodule
```

### B. Compile-Time Configuration

**Use `define for compile-time switches:**

| Macro | Description | Default | Usage |
|-------|-------------|---------|-------|
| `SYNTHESIS` | Synthesis mode (disables asserts, delays) | Undefined | Set during synthesis |
| `SIMULATION` | Simulation mode (enables debug) | Undefined | Set during simulation |
| `FORMAL` | Formal verification mode | Undefined | Set for formal tools |

```systemverilog
`ifdef SYNTHESIS
    // Synthesis-specific code (no asserts, no delays)
`else
    // Simulation-specific code
    initial begin
        $dumpfile("waves.vcd");
        $dumpvars(0, top);
    end
`endif
```

---

## 7. Logging & Observability (MANDATORY)

### A. Assertion-Based Verification

**Use SystemVerilog Assertions (SVA) for runtime checks:**

```systemverilog
module fifo_with_assertions #(
    parameter int DEPTH = 16
) (
    input  logic        i_clk,
    input  logic        i_rst_n,
    input  logic        i_wr_en,
    input  logic        i_rd_en,
    output logic        o_full,
    output logic        o_empty
);
    // FIFO implementation...

    // Assertions for verification
    `ifndef SYNTHESIS

    // Property: Can't write when full
    property p_no_write_when_full;
        @(posedge i_clk) disable iff (!i_rst_n)
        o_full |-> !i_wr_en;
    endproperty
    assert property (p_no_write_when_full)
        else $error("Write attempted when FIFO full");

    // Property: Can't read when empty
    property p_no_read_when_empty;
        @(posedge i_clk) disable iff (!i_rst_n)
        o_empty |-> !i_rd_en;
    endproperty
    assert property (p_no_read_when_empty)
        else $error("Read attempted when FIFO empty");

    // Property: Full and empty are mutually exclusive
    assert property (@(posedge i_clk) !(o_full && o_empty))
        else $error("FIFO cannot be both full and empty");

    `endif
endmodule
```

### B. Coverage Metrics

**Implement functional coverage:**

```systemverilog
module alu_with_coverage (
    input  logic [31:0] i_a,
    input  logic [31:0] i_b,
    input  logic [3:0]  i_opcode,
    output logic [31:0] o_result
);
    // ALU implementation...

    `ifndef SYNTHESIS
    covergroup cg_alu @(i_opcode);
        opcode_cp: coverpoint i_opcode {
            bins add    = {4'h0};
            bins sub    = {4'h1};
            bins and_op = {4'h2};
            bins or_op  = {4'h3};
            bins xor_op = {4'h4};
        }

        operand_a_cp: coverpoint i_a {
            bins zero       = {32'h0};
            bins max        = {32'hFFFF_FFFF};
            bins mid_range  = {[32'h1:32'hFFFF_FFFE]};
        }

        cross opcode_cp, operand_a_cp;
    endgroup

    cg_alu cg = new();
    `endif
endmodule
```

---

## 8. Testing (MANDATORY)

### A. Unit Tests (Module-Level)

**Use directed tests for basic functionality:**

```systemverilog
`timescale 1ns/1ps

module counter_unit_tb;
    logic       clk;
    logic       rst_n;
    logic       enable;
    logic [7:0] count;

    // DUT
    counter dut (.*);

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz clock
    end

    // Test sequence
    initial begin
        $display("Starting counter unit test");

        // Test 1: Reset behavior
        rst_n = 0;
        enable = 0;
        #20;
        assert(count == 8'd0) else $error("Reset failed");

        // Test 2: Counting
        rst_n = 1;
        #10;
        enable = 1;
        repeat(10) @(posedge clk);
        assert(count == 8'd10) else $error("Count incorrect: expected 10, got %d", count);

        // Test 3: Disable
        enable = 0;
        repeat(5) @(posedge clk);
        assert(count == 8'd10) else $error("Count changed when disabled");

        $display("All unit tests passed!");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #1000;
        $error("Testbench timeout!");
        $finish;
    end
endmodule
```

### B. UVM Integration Tests

```systemverilog
class base_test extends uvm_test;
    `uvm_component_utils(base_test)

    env_c env;

    function new(string name = "base_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        env = env_c::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
        basic_sequence seq;
        phase.raise_objection(this);

        seq = basic_sequence::type_id::create("seq");
        seq.start(env.agent.sequencer);

        phase.drop_objection(this);
    endtask
endclass
```

### C. Test Coverage Requirements

- Minimum coverage: **90%** for RTL code coverage (line, toggle, FSM)
- Critical paths: **100%** coverage (reset, error handling)
- All public module interfaces must have directed tests
- Functional coverage: **80%** of specified features

---

## 9. Error Handling (MANDATORY)

### A. Reset Strategy

**Use consistent async reset, sync deassert:**

```systemverilog
module safe_register (
    input  logic       i_clk,
    input  logic       i_rst_n,  // Active-low async reset
    input  logic [7:0] i_data,
    output logic [7:0] o_q
);
    always_ff @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            o_q <= '0;  // Reset value
        else
            o_q <= i_data;
    end
endmodule
```

### B. Error Detection & Handling

| Error Type | Description | Handling |
|------------|-------------|----------|
| X-propagation | Unknown values in simulation | Initialize all variables; use 4-state carefully |
| Clock domain crossing | Metastability risk | Use synchronizers (2-FF or 3-FF) |
| Reset domain crossing | Async reset metastability | Synchronize reset deassertion |
| Protocol violations | AXI/AHB handshake errors | Add assertions; checker modules |

```systemverilog
// CDC synchronizer (2-FF)
module cdc_sync (
    input  logic i_clk_dst,
    input  logic i_rst_n,
    input  logic i_async_signal,
    output logic o_sync_signal
);
    logic sync_ff1, sync_ff2;

    always_ff @(posedge i_clk_dst or negedge i_rst_n) begin
        if (!i_rst_n) begin
            sync_ff1 <= '0;
            sync_ff2 <= '0;
        end else begin
            sync_ff1 <= i_async_signal;
            sync_ff2 <= sync_ff1;
        end
    end

    assign o_sync_signal = sync_ff2;

    `ifndef SYNTHESIS
    // Assert: No X propagation
    assert property (@(posedge i_clk_dst) !$isunknown(o_sync_signal))
        else $error("X detected in synchronized signal");
    `endif
endmodule
```

---

## 10. Documentation (MANDATORY)

### A. Module Documentation

**Follow Doxygen-style documentation:**

```systemverilog
/**
 * @module uart_tx
 * @brief UART transmitter module with configurable baud rate
 *
 * @details Implements a simple UART transmitter with:
 * - Configurable baud rate via parameter
 * - 8 data bits, 1 stop bit, no parity
 * - Ready/valid handshake interface
 *
 * @param CLK_FREQ System clock frequency in Hz
 * @param BAUD_RATE UART baud rate (e.g., 115200)
 *
 * @author Design Team
 * @date 2026-03-11
 * @version 1.0
 */
module uart_tx #(
    parameter int CLK_FREQ  = 100_000_000,  ///< System clock frequency (Hz)
    parameter int BAUD_RATE = 115200        ///< UART baud rate (bps)
) (
    input  logic       i_clk,        ///< System clock
    input  logic       i_rst_n,      ///< Active-low async reset
    input  logic [7:0] i_data,       ///< Data byte to transmit
    input  logic       i_valid,      ///< Data valid signal
    output logic       o_ready,      ///< Ready to accept data
    output logic       o_tx          ///< UART TX line
);
    // Implementation...
endmodule
```

### B. Generate Documentation

```bash
# Generate documentation using Doxygen
doxygen rtl/Doxyfile

# View documentation
firefox docs/html/index.html

# Or use Natural Docs
naturaldocs -i rtl/ -o HTML docs/
```

---

## 11. Security & Dependency Management (MANDATORY)

### A. IP Core Management

**Use version-controlled IP cores with checksums:**

```bash
# Add IP core with integrity verification
git submodule add https://github.com/vendor/axi_interconnect.git ip/axi

# Verify integrity (SHA256 checksum)
sha256sum -c ip/checksums.txt

# Update to specific version
cd ip/axi && git checkout v2.1.0
```

### B. Vulnerability Scanning & Security

**Mandatory security checks for all RTL:**

1. **Lint Check for Security**:
   ```bash
   # Check for security issues (latches, X-propagation, CDC violations)
   spyglass -batch security_lint.tcl

   # Verilator security checks
   verilator --lint-only -Wall --timing rtl/*.sv
   ```
   - Agents MUST fix all CDC violations
   - No inferred latches (causes unpredictable behavior)
   - All clock domain crossings verified

2. **Formal Verification**:
   ```bash
   # Prove security properties with JasperGold
   jaspergold -batch prove_security.tcl
   ```
   - Verify no data leakage between security domains
   - Prove access control properties
   - Check for side-channel vulnerabilities

### C. License Management

```text
# File: rtl/LICENSE
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Your Organization
```

---

## 12. Deployment Checklist

### Agent-Generated RTL Verification (MANDATORY)

**If RTL was generated/modified by an agent, verify BEFORE delivery:**

#### Compilation & Lint
- [ ] Code compiles: `verilator --lint-only --sv rtl/*.sv` returns exit code 0
- [ ] No compilation errors or warnings
- [ ] No inferred latches or combinational loops
- [ ] All signals properly declared with `default_nettype none`
- [ ] Code formatted: `verible-verilog-format --inplace rtl/*.sv`

#### Testing
- [ ] All unit tests pass: `make test` returns exit code 0
- [ ] Reasonable coverage: `make coverage` shows >90% line coverage
- [ ] Integration/UVM tests pass (if applicable)
- [ ] Functional coverage meets targets (>80%)

#### Security & Timing
- [ ] Lint scan passes: 0 errors, justified warnings only
- [ ] CDC verification: All clock crossings use synchronizers
- [ ] Timing: Static timing analysis passes (if synthesis run)
- [ ] No hardcoded secrets or sensitive data in RTL

#### Code Quality
- [ ] Follows naming conventions (i_*, o_*, r_*, w_*)
- [ ] No unused parameters or signals
- [ ] No combinational loops or multi-driven nets
- [ ] Project structure follows standard layout

#### Documentation
- [ ] All public module interfaces have documentation
- [ ] Port descriptions included (Doxygen/Natural Docs style)
- [ ] Block diagrams for complex modules
- [ ] Examples provided for instantiation

#### Architecture
- [ ] RTL architecture followed (control/datapath separation)
- [ ] Parameterization used appropriately
- [ ] No global state or uninitialized registers

#### Agent Workflow Completed
- [ ] Agent verified code compiles successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran lint and fixed all errors
- [ ] Agent verified synthesizability (no behavioral constructs)
- [ ] Agent documented any fixes made during verification

---

## 13. Why This Configuration Works

**Synthesizable-First Approach**:
- Ensures RTL can be converted to gates for ASIC/FPGA implementation
- Avoids simulation-only constructs that cause synthesis failures
- Lint-clean code synthesizes predictably with good QoR

**Test-Driven Development for Hardware**:
- Catches functional bugs early before synthesis
- Regression tests prevent bug reintroduction
- Waveform-driven debug accelerates root cause analysis

**Timing-Aware Design**:
- Clock domain crossing synchronizers prevent metastability
- Pipeline registers improve Fmax for high-frequency designs
- Static timing analysis ensures setup/hold time requirements met

**Assertion-Based Verification**:
- SVA catches protocol violations and corner cases
- Formal verification proves correctness mathematically
- Runtime assertions act as "hardware unit tests"

---

## 14. Quick Reference

### Common Commands

```bash
# Compile & Lint (Verilator)
verilator --lint-only -Wall --sv rtl/*.sv

# Simulate (Verilator)
verilator --binary --trace rtl/top.sv tb/top_tb.sv
./obj_dir/Vtop

# Simulate (Commercial - VCS)
vcs -sverilog +v2k -debug_all rtl/*.sv tb/*_tb.sv
./simv

# Format
verible-verilog-format --inplace rtl/*.sv

# Lint (SpyGlass)
spyglass -batch lint.tcl

# Synthesis (Synopsys Design Compiler)
dc_shell -f syn/synthesis.tcl

# Coverage
make coverage
firefox coverage/html/index.html
```

### Makefile Template

```makefile
# Verilog/SystemVerilog Makefile

RTL_SRCS = $(wildcard rtl/*.sv rtl/**/*.sv)
TB_SRCS  = $(wildcard tb/*_tb.sv)

.PHONY: all clean lint sim coverage

all: lint sim

lint:
	verilator --lint-only -Wall --sv $(RTL_SRCS)

sim: $(TB_SRCS)
	verilator --binary --trace $(RTL_SRCS) tb/top_tb.sv
	./obj_dir/Vtop

coverage:
	verilator --binary --trace --coverage $(RTL_SRCS) tb/top_tb.sv
	./obj_dir/Vtop
	verilator_coverage --write coverage.dat obj_dir/*.dat
	verilator_coverage --annotate coverage/ coverage.dat

clean:
	rm -rf obj_dir/ *.vcd coverage/ coverage.dat
```

---

**End of Verilog/SystemVerilog Guidelines**
