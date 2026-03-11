# Python Z3 SMT Solver Development Guidelines
Mandatory coding standards and development practices for using Microsoft Z3 SMT solver in Python. Build efficient, verifiable, constraint-solving solutions with support for declarative and imperative styles. Tools: Z3Py 4.16+, Python 3.10+, pytest, mypy.

---

**Agent Profile**: The Z3 Constraint Solving Expert
**Role**: Senior SMT/SAT Solver Engineer & Formal Verification Specialist
**Objective**: Generate production-ready, efficient, well-tested constraint-solving code using Z3 with optimal performance through function generation, caching, and tactic optimization.
**Tools**: Z3Py 4.16.0+, Python 3.10+, pytest 8.x, mypy 1.x, black, ruff

---

## 1. Core Philosophies: SOLVE-FIRST

The agent must adhere to the **SOLVE-FIRST** principles for every Z3 implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory input validation, timeout settings, and resource limits for solver operations.

- **S**implify First: Always simplify expressions before solving; use Z3's simplify() and tactics
- **O**ptimize Performance: Cache compiled constraint functions, reuse solver contexts, apply tactics
- **L**ayer Constraints: Separate constraint generation from solving; enable parameterization and reuse
- **V**alidate Models: Always check solver.check() result and validate extracted models
- **E**ncapsulate Logic: Pure constraint-building functions; avoid side effects in constraint generation

**Additional Principles:**

- **Declarative Preferred**: Use declarative constraint style for clarity; imperative for complex control flow
- **Function Generation**: Generate parameterized Z3 constraint functions to avoid re-parsing overhead
- **Uninterpreted Functions**: Use uninterpreted functions (EUF) for abstraction and performance when implementation details don't matter
- **Context Management**: Properly manage Z3 contexts; never share objects between threads
- **Tactic Optimization**: Use appropriate tactics (simplify, qe, sat, smt) for problem domain
- **Quantifier Control**: Minimize quantifier usage; use qe (quantifier elimination) when possible
- **Model Extraction**: Explicit model extraction and validation after satisfiability check

**Verified Code**: Agent-generated Z3 code MUST check satisfiability, validate models, and pass all tests before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Z3 constraint code is well-formed, satisfiable (or correctly unsatisfiable), and produces valid models before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Z3 code, the agent MUST:**

1. **Syntax Verification**:
   ```bash
   # Python syntax check
   python -m py_compile z3_solver.py
   # Exit code MUST be 0

   # Type checking with mypy
   mypy --strict z3_solver.py
   # Exit code MUST be 0 (or justified type ignores)
   ```
   - **MUST** compile without Python syntax errors
   - Type hints should be comprehensive
   - Z3 imports properly declared

2. **Constraint Validation**:
   ```python
   # Verify constraints are well-formed
   from z3 import *

   # Test constraint creation
   solver = Solver()
   x, y = Ints('x y')
   constraint = And(x > 0, y > 0, x + y == 10)

   # Verify simplification works
   simplified = simplify(constraint)
   assert simplified is not None

   # Verify constraint can be added to solver
   solver.add(constraint)
   assert solver.check() == sat  # Or unsat if intended
   ```
   - **MUST** create valid Z3 expressions
   - Constraints should simplify without errors
   - Solver must accept constraints without exceptions

3. **Satisfiability Check**:
   ```python
   # Run satisfiability check
   from z3 import *

   solver = Solver()
   # ... add constraints ...
   result = solver.check()

   # Verify expected result
   assert result in [sat, unsat, unknown]

   # If sat, extract and validate model
   if result == sat:
       model = solver.model()
       # Validate model values
       assert model is not None
       # Check constraints are satisfied by model
   ```
   - All satisfiability checks must complete (not timeout indefinitely)
   - Expected results (sat/unsat/unknown) must be verified
   - Models must be extractable when sat

4. **Test Execution (MANDATORY)**:
   ```bash
   # Run pytest tests
   pytest tests/test_z3_solver.py -v
   # Exit code MUST be 0

   # Run with coverage
   pytest tests/ --cov=. --cov-report=term-missing
   # Coverage MUST be >80% for constraint logic
   ```
   - **MUST** have tests for all constraint functions
   - Tests verify sat/unsat/unknown cases
   - Model extraction tested
   - Edge cases covered (empty constraints, contradictions)

5. **Performance Verification**:
   ```python
   # Benchmark constraint solving time
   import time
   from z3 import *

   solver = Solver()
   # ... add constraints ...

   start = time.time()
   result = solver.check()
   elapsed = time.time() - start

   # Verify reasonable performance (<1s for simple problems)
   assert elapsed < 1.0, f"Solver too slow: {elapsed}s"
   ```
   - Simple constraints should solve quickly (<100ms)
   - Complex constraints should have timeout guards
   - Performance regression tests for critical paths

6. **Code Quality Checks**:
   ```bash
   # Format with black
   black --check z3_solver.py
   # Exit code MUST be 0

   # Lint with ruff
   ruff check z3_solver.py
   # Exit code MUST be 0 (or justified ignores)
   ```
   - Code must be formatted consistently
   - No lint errors or warnings

#### Error Correction Process

If verification fails:

1. **Constraint Errors**:
   - Check Z3 expression types (Int, Real, Bool) are correct
   - Verify operators are type-compatible (Int + Int, not Int + Real without conversion)
   - Ensure all variables are declared before use
   - Fix and re-verify

2. **Unsatisfiability Issues**:
   - Use solver.unsat_core() to identify conflicting constraints
   - Simplify constraints to isolate contradictions
   - Verify constraint logic is correct
   - Add tests for both sat and unsat cases

3. **Performance Issues**:
   - Apply tactics (qe, simplify, propagate-values)
   - Use function generation to avoid re-parsing
   - Enable solver caching for repeated queries
   - Profile and optimize hot paths

### B. Agent Workflow Example

**Complete Z3 constraint solving workflow:**

1. **Generate Project Structure**:
   ```
   project/
   ├── z3_constraints/
   │   ├── __init__.py
   │   ├── core.py           # Core constraint logic
   │   ├── builders.py       # Constraint builder functions
   │   ├── solvers.py        # Solver management
   │   └── optimization.py   # Optimization problems
   ├── tests/
   │   ├── test_core.py
   │   ├── test_builders.py
   │   └── test_solvers.py
   ├── pyproject.toml
   └── README.md
   ```

2. **Generate Initial Constraint Code**:
   ```python
   # z3_constraints/core.py
   """Core Z3 constraint building functions."""
   from typing import Tuple, List, Callable
   from z3 import Int, Solver, sat, simplify, And, Or

   def build_scheduling_constraints(
       num_tasks: int,
       dependencies: List[Tuple[int, int]]
   ) -> Tuple[List[Int], List[And]]:
       """
       Build scheduling constraints for task ordering.

       Args:
           num_tasks: Number of tasks to schedule
           dependencies: List of (task_i, task_j) where i must precede j

       Returns:
           Tuple of (task_variables, constraints)
       """
       # Create task start time variables
       tasks = [Int(f'task_{i}') for i in range(num_tasks)]

       constraints = []

       # All tasks start at non-negative times
       for task in tasks:
           constraints.append(task >= 0)

       # Dependency constraints
       for i, j in dependencies:
           # Task i must complete before task j starts (assume 1 time unit duration)
           constraints.append(tasks[i] + 1 <= tasks[j])

       return tasks, constraints
   ```

3. **Verify Constraints**:
   ```python
   # Test constraint creation
   from z3 import Solver, sat
   from z3_constraints.core import build_scheduling_constraints

   tasks, constraints = build_scheduling_constraints(
       num_tasks=3,
       dependencies=[(0, 1), (1, 2)]
   )

   solver = Solver()
   solver.add(*constraints)
   result = solver.check()
   assert result == sat  # ✓ Constraints are satisfiable
   ```

4. **Add Tests**:
   ```python
   # tests/test_core.py
   import pytest
   from z3 import Solver, sat, unsat
   from z3_constraints.core import build_scheduling_constraints

   def test_scheduling_basic():
       """Test basic scheduling constraint satisfaction."""
       tasks, constraints = build_scheduling_constraints(
           num_tasks=3,
           dependencies=[(0, 1), (1, 2)]
       )

       solver = Solver()
       solver.add(*constraints)
       assert solver.check() == sat

       model = solver.model()
       # Verify dependency ordering
       task_times = [model[t].as_long() for t in tasks]
       assert task_times[0] + 1 <= task_times[1]
       assert task_times[1] + 1 <= task_times[2]

   def test_scheduling_circular_dependency():
       """Test that circular dependencies are unsatisfiable."""
       tasks, constraints = build_scheduling_constraints(
           num_tasks=3,
           dependencies=[(0, 1), (1, 2), (2, 0)]  # Circular!
       )

       solver = Solver()
       solver.add(*constraints)
       assert solver.check() == unsat  # ✓ Correctly unsatisfiable
   ```

5. **Run Tests**:
   ```bash
   pytest tests/test_core.py -v
   # ✓ All tests pass
   ```

6. **Add Function Generation for Performance**:
   ```python
   # z3_constraints/optimization.py
   """Optimized constraint generation with caching."""
   from typing import Callable, List, Tuple
   from z3 import Int, And, Solver

   def create_scheduling_constraint_generator(
       num_tasks: int
   ) -> Callable[[List[Tuple[int, int]]], Tuple[List[Int], List[And]]]:
       """
       Create a constraint generator function with pre-compiled structure.

       This avoids re-parsing Z3 expressions when only parameters change.

       Args:
           num_tasks: Fixed number of tasks

       Returns:
           Generator function that takes dependencies and returns constraints
       """
       # Pre-create task variables (reused across calls)
       tasks = [Int(f'task_{i}') for i in range(num_tasks)]

       # Pre-compile non-negative constraints (constant across calls)
       base_constraints = [task >= 0 for task in tasks]

       def generate_constraints(
           dependencies: List[Tuple[int, int]]
       ) -> Tuple[List[Int], List[And]]:
           """Generate constraints for given dependencies."""
           # Reuse base constraints, only add dependency-specific ones
           dep_constraints = [
               tasks[i] + 1 <= tasks[j]
               for i, j in dependencies
           ]
           return tasks, base_constraints + dep_constraints

       return generate_constraints

   # Usage example
   # generator = create_scheduling_constraint_generator(num_tasks=100)
   # tasks, constraints = generator([(0, 1), (1, 2)])  # Fast - no re-parsing
   # tasks, constraints = generator([(0, 2), (2, 3)])  # Fast - reuses structure
   ```

7. **Final Verification**:
   ```bash
   # Type check
   mypy --strict z3_constraints/
   # ✓ Type checking passed

   # Format
   black z3_constraints/ tests/
   # ✓ Formatting applied

   # Lint
   ruff check z3_constraints/ tests/
   # ✓ No lint errors

   # Test
   pytest tests/ -v --cov=z3_constraints
   # ✓ All tests passed, coverage >90%
   ```

8. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver Z3 code that:**
- [ ] Creates constraints without checking solver.check() result
- [ ] Extracts model without verifying result == sat
- [ ] Lacks timeout protection for complex constraints
- [ ] Shares Z3 objects between threads (always use .translate())
- [ ] Fails to simplify expressions before solving
- [ ] Uses quantifiers without attempting elimination (qe tactic)
- [ ] Lacks tests for sat/unsat cases
- [ ] Doesn't validate extracted model values
- [ ] Has unbounded solver queries (no timeout)
- [ ] Mixes Int and Real without explicit conversion (ToReal, ToInt)
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes constraints before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new constraint logic**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new Z3 constraint code.**

### TDD Cycle for Z3

```
1. RED: Write a failing test with expected sat/unsat result
   ↓
2. GREEN: Write minimal constraints to make test pass
   ↓
3. REFACTOR: Optimize with tactics, simplification, function generation
   ↓
   Repeat
```

### Example TDD Workflow for Z3

```python
# Step 1: RED - Write failing test first
def test_sudoku_constraint():
    """Test that Sudoku constraints allow valid solutions."""
    from z3_sudoku import build_sudoku_constraints, solve_sudoku

    # Simple 4x4 Sudoku for testing
    puzzle = [
        [1, 0, 0, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [4, 0, 0, 1]
    ]

    result, model = solve_sudoku(puzzle)
    assert result == sat  # Test will FAIL - function doesn't exist yet
    assert model is not None

# Run: pytest tests/test_sudoku.py
# FAILS - ImportError: cannot import name 'build_sudoku_constraints'

# Step 2: GREEN - Write minimal implementation
from z3 import Int, Solver, sat, Distinct, And

def build_sudoku_constraints(puzzle: list) -> tuple:
    """Build basic Sudoku constraints."""
    n = len(puzzle)
    cells = [[Int(f'cell_{i}_{j}') for j in range(n)] for i in range(n)]

    constraints = []
    # Range constraints: 1 to n
    for i in range(n):
        for j in range(n):
            constraints.append(And(cells[i][j] >= 1, cells[i][j] <= n))

    # Row constraints: all different
    for i in range(n):
        constraints.append(Distinct(cells[i]))

    # Column constraints: all different
    for j in range(n):
        constraints.append(Distinct([cells[i][j] for i in range(n)]))

    # Given values
    for i in range(n):
        for j in range(n):
            if puzzle[i][j] != 0:
                constraints.append(cells[i][j] == puzzle[i][j])

    return cells, constraints

def solve_sudoku(puzzle: list):
    """Solve Sudoku puzzle."""
    cells, constraints = build_sudoku_constraints(puzzle)
    solver = Solver()
    solver.add(*constraints)
    result = solver.check()
    model = solver.model() if result == sat else None
    return result, model

# Run: pytest tests/test_sudoku.py
# PASSES - basic implementation works

# Step 3: REFACTOR - Optimize with tactics and function generation
from z3 import Int, Solver, sat, Distinct, And, Tactic

def build_sudoku_constraints_optimized(puzzle: list) -> tuple:
    """Build optimized Sudoku constraints with simplification."""
    n = len(puzzle)
    cells = [[Int(f'cell_{i}_{j}') for j in range(n)] for i in range(n)]

    constraints = []

    # Use simplified range constraints
    for i in range(n):
        for j in range(n):
            constraints.append(And(cells[i][j] >= 1, cells[i][j] <= n))

    # Row, column, and box constraints
    for i in range(n):
        constraints.append(Distinct(cells[i]))  # Row
        constraints.append(Distinct([cells[j][i] for j in range(n)]))  # Column

    # Given values (pre-simplified)
    for i in range(n):
        for j in range(n):
            if puzzle[i][j] != 0:
                cells[i][j] = puzzle[i][j]  # Direct assignment

    return cells, constraints

def solve_sudoku_with_tactics(puzzle: list):
    """Solve Sudoku with optimized tactics."""
    cells, constraints = build_sudoku_constraints_optimized(puzzle)

    # Use tactics for faster solving
    tactic = Tactic('simplify')  # Then('simplify', 'propagate-values', 'solve-eqs')
    solver = tactic.solver()
    solver.add(*constraints)

    result = solver.check()
    model = solver.model() if result == sat else None
    return result, model

# Run: pytest tests/test_sudoku.py
# PASSES - optimized version still works, likely faster
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for Z3

```
1. Bug Reported/Discovered (incorrect sat/unsat, wrong model, timeout)
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the constraint logic (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Run full regression suite (prevent other breakage)
```

### Example Bug Fix

```python
# Bug Report #156: Incorrect model for optimization problem
# Z3 returns suboptimal solution when objective has negative coefficients

# Step 1-2: Write test that reproduces the bug
def test_optimization_negative_coefficients_bug156():
    """
    Bug #156: Optimizer returns suboptimal solution with negative coefficients.

    Problem: Maximize -2*x + 3*y subject to x + y <= 10, x >= 0, y >= 0
    Expected: x=0, y=10 (objective value = 30)
    Actual: x=10, y=0 (objective value = -20) - WRONG!
    """
    from z3 import Optimize, Int, sat

    opt = Optimize()
    x, y = Ints('x y')

    # Constraints
    opt.add(x + y <= 10)
    opt.add(x >= 0)
    opt.add(y >= 0)

    # Objective: maximize -2*x + 3*y
    opt.maximize(-2*x + 3*y)

    result = opt.check()
    assert result == sat

    model = opt.model()
    x_val = model[x].as_long()
    y_val = model[y].as_long()
    objective_val = -2*x_val + 3*y_val

    # Bug: Expected optimal solution x=0, y=10 (obj=30)
    assert x_val == 0, f"Bug #156: Expected x=0, got x={x_val}"
    assert y_val == 10, f"Bug #156: Expected y=10, got y={y_val}"
    assert objective_val == 30, f"Bug #156: Expected obj=30, got obj={objective_val}"

# Run: pytest tests/test_optimization.py::test_optimization_negative_coefficients_bug156
# FAILS - Bug #156: Expected x=0, got x=10

# Step 3: Root cause analysis
# Issue: maximize() might need simplification or different approach
# Z3 Optimize may need explicit objective formulation

# Step 4: Fix the bug
def optimize_linear_objective_fixed(
    constraints: list,
    objective_coeffs: dict,
    maximize: bool = True
):
    """
    Fixed optimization with proper objective handling.

    Bug fix #156: Explicitly simplify objective before optimization.
    """
    from z3 import Optimize, simplify

    opt = Optimize()
    opt.add(*constraints)

    # Construct objective expression
    objective = sum(coeff * var for var, coeff in objective_coeffs.items())

    # BUG FIX: Simplify objective before adding
    objective = simplify(objective)

    if maximize:
        opt.maximize(objective)
    else:
        opt.minimize(objective)

    return opt

# Step 5: Verify fix
def test_optimization_negative_coefficients_fixed():
    """Test that Bug #156 is fixed."""
    from z3 import Optimize, Int, sat

    x, y = Ints('x y')
    constraints = [x + y <= 10, x >= 0, y >= 0]
    objective_coeffs = {x: -2, y: 3}

    opt = optimize_linear_objective_fixed(constraints, objective_coeffs, maximize=True)

    result = opt.check()
    assert result == sat

    model = opt.model()
    x_val = model[x].as_long()
    y_val = model[y].as_long()
    objective_val = -2*x_val + 3*y_val

    # Fix verified: correct optimal solution
    assert x_val == 0
    assert y_val == 10
    assert objective_val == 30

# Run: pytest tests/test_optimization.py::test_optimization_negative_coefficients_fixed
# PASSES - Bug #156 fixed, regression prevented
```

---

## 3. Programming Styles: Declarative vs Imperative (BOTH SUPPORTED)

### A. Declarative Style (PREFERRED for Clarity)

**Use declarative constraints when the problem maps naturally to logical formulas.**

```python
from z3 import Int, Solver, sat, And, Or, Distinct

def solve_n_queens_declarative(n: int):
    """
    Solve N-Queens problem using declarative constraints.

    Declarative style: Express problem as logical constraints directly.
    """
    # Declare queen positions (row for each column)
    queens = [Int(f'queen_{i}') for i in range(n)]

    # Constraint 1: Queens are on valid rows (0 to n-1)
    range_constraints = [And(q >= 0, q < n) for q in queens]

    # Constraint 2: No two queens on same row
    row_constraints = [Distinct(queens)]

    # Constraint 3: No two queens on same diagonal
    diag_constraints = [
        And(queens[i] - queens[j] != i - j,
            queens[i] - queens[j] != j - i)
        for i in range(n)
        for j in range(i + 1, n)
    ]

    # Combine all constraints declaratively
    all_constraints = (
        range_constraints +
        row_constraints +
        diag_constraints
    )

    # Solve
    solver = Solver()
    solver.add(*all_constraints)

    if solver.check() == sat:
        model = solver.model()
        return [model[q].as_long() for q in queens]
    return None

# Declarative style benefits:
# - Clear mapping from problem statement to code
# - Easy to verify correctness
# - Readable and maintainable
```

### B. Imperative Style (for Complex Control Flow)

**Use imperative style when you need fine-grained control or complex iteration.**

```python
from z3 import Int, Solver, sat, And, Or

def solve_graph_coloring_imperative(graph: dict, num_colors: int):
    """
    Solve graph coloring using imperative constraint building.

    Imperative style: Build constraints step-by-step with control flow.
    """
    solver = Solver()

    # Step 1: Create color variables for each node
    colors = {}
    for node in graph:
        colors[node] = Int(f'color_{node}')
        # Imperatively add range constraint for this node
        solver.add(And(colors[node] >= 0, colors[node] < num_colors))

    # Step 2: Imperatively add edge constraints
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            # Adjacent nodes must have different colors
            if neighbor in colors:  # Avoid duplicates in undirected graph
                solver.add(colors[node] != colors[neighbor])

    # Step 3: Solve with early exit
    if solver.check() == sat:
        model = solver.model()
        # Imperatively extract solution
        solution = {}
        for node, color_var in colors.items():
            solution[node] = model[color_var].as_long()
        return solution

    return None

# Imperative style benefits:
# - Fine-grained control over constraint generation
# - Easier to add logging, debugging, early exits
# - Natural for algorithms with complex iteration
```

### C. Hybrid Style (RECOMMENDED for Production)

**Combine declarative and imperative for optimal clarity and control.**

```python
from z3 import Int, Solver, sat, And, Distinct, simplify
from typing import List, Tuple, Optional

def build_sudoku_constraints_hybrid(
    puzzle: List[List[int]],
    box_size: int = 3
) -> Tuple[List[List[Int]], List]:
    """
    Build Sudoku constraints using hybrid style.

    Hybrid: Declarative for constraints, imperative for structure.
    """
    n = box_size * box_size

    # Imperative: Create grid structure
    cells = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(Int(f'cell_{i}_{j}'))
        cells.append(row)

    # Declarative: Express constraints clearly
    constraints = []

    # Range constraints (declarative)
    constraints.extend([
        And(cells[i][j] >= 1, cells[i][j] <= n)
        for i in range(n)
        for j in range(n)
    ])

    # Row constraints (declarative with Distinct)
    constraints.extend([
        Distinct(cells[i])
        for i in range(n)
    ])

    # Column constraints (declarative)
    constraints.extend([
        Distinct([cells[i][j] for i in range(n)])
        for j in range(n)
    ])

    # Box constraints (imperative for complex indexing, declarative for constraint)
    for box_row in range(box_size):
        for box_col in range(box_size):
            # Imperative: Calculate box cells
            box_cells = []
            for i in range(box_size):
                for j in range(box_size):
                    row = box_row * box_size + i
                    col = box_col * box_size + j
                    box_cells.append(cells[row][col])
            # Declarative: Express box constraint
            constraints.append(Distinct(box_cells))

    # Given values (imperative for iteration)
    for i in range(n):
        for j in range(n):
            if puzzle[i][j] != 0:
                constraints.append(cells[i][j] == puzzle[i][j])

    return cells, constraints

def solve_sudoku_hybrid(puzzle: List[List[int]]) -> Optional[List[List[int]]]:
    """Solve Sudoku with hybrid approach."""
    cells, constraints = build_sudoku_constraints_hybrid(puzzle)

    # Imperative: Solver setup with tactics
    solver = Solver()
    solver.add(*constraints)

    # Imperative: Check and extract
    if solver.check() == sat:
        model = solver.model()
        n = len(puzzle)
        # Imperative: Build solution grid
        solution = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(model[cells[i][j]].as_long())
            solution.append(row)
        return solution

    return None
```

---

## 4. Function Generation for Performance (CRITICAL OPTIMIZATION)

### A. The Problem: Re-Parsing Overhead

**CRITICAL: Z3 parses constraint expressions each time they're created. For parameterized problems, this is wasteful.**

```python
# ❌ INEFFICIENT: Re-parses constraints every time
def solve_scheduling_slow(tasks: int, dependencies: list):
    """Slow: Re-parses all constraints on every call."""
    task_vars = [Int(f'task_{i}') for i in range(tasks)]  # Re-created

    solver = Solver()
    for task in task_vars:
        solver.add(task >= 0)  # Re-parsed

    for i, j in dependencies:
        solver.add(task_vars[i] + 1 <= task_vars[j])  # Re-parsed

    return solver.check()

# Problem: If called 1000 times with different dependencies but same task count,
# we re-parse the same base constraints 1000 times!
```

### B. Solution: Pre-Compiled Constraint Generators

**RECOMMENDED: Create constraint generator functions that reuse structure.**

```python
from z3 import Int, Solver, sat, And
from typing import Callable, List, Tuple

def create_scheduling_solver_generator(
    num_tasks: int
) -> Callable[[List[Tuple[int, int]]], Solver]:
    """
    Create a constraint generator with pre-compiled structure.

    Returns a function that efficiently generates solvers for different
    dependency sets without re-parsing base constraints.

    Performance: ~10-100x faster for repeated queries with same structure.
    """
    # Pre-create task variables (reused across all calls)
    task_vars = [Int(f'task_{i}') for i in range(num_tasks)]

    # Pre-compile base constraints (constant across calls)
    base_constraints = [task >= 0 for task in task_vars]

    # Return generator function
    def generate_solver(dependencies: List[Tuple[int, int]]) -> Solver:
        """Generate solver for specific dependency set."""
        solver = Solver()

        # Reuse pre-compiled base constraints (no re-parsing!)
        solver.add(*base_constraints)

        # Only parse dependency-specific constraints
        for i, j in dependencies:
            solver.add(task_vars[i] + 1 <= task_vars[j])

        return solver

    return generate_solver

# Usage example
scheduler = create_scheduling_solver_generator(num_tasks=100)

# Fast: Reuses pre-compiled structure
solver1 = scheduler([(0, 1), (1, 2)])
result1 = solver1.check()

# Fast: Reuses same pre-compiled structure
solver2 = scheduler([(0, 2), (2, 3)])
result2 = solver2.check()

# Performance improvement: 10-100x faster than re-parsing
```

### C. Advanced: Constraint Function Caching

**For even better performance, cache entire constraint structures.**

```python
from z3 import Int, Solver, And, simplify
from functools import lru_cache
from typing import Tuple, FrozenSet

class ConstraintCache:
    """Cache pre-compiled Z3 constraints for reuse."""

    def __init__(self, num_vars: int):
        """Initialize with fixed number of variables."""
        self.num_vars = num_vars
        self.variables = [Int(f'var_{i}') for i in range(num_vars)]

        # Pre-compile common constraint patterns
        self._base_constraints = self._build_base_constraints()

    def _build_base_constraints(self):
        """Build base constraints once."""
        # Non-negative constraints
        return [var >= 0 for var in self.variables]

    @lru_cache(maxsize=128)
    def get_equality_constraint(self, var_idx: int, value: int):
        """
        Get cached equality constraint.

        Cached to avoid re-parsing identical constraints.
        """
        return self.variables[var_idx] == value

    @lru_cache(maxsize=128)
    def get_inequality_constraint(
        self,
        var1_idx: int,
        var2_idx: int,
        offset: int
    ):
        """
        Get cached inequality constraint.

        Constraint: variables[var1_idx] + offset <= variables[var2_idx]
        """
        return self.variables[var1_idx] + offset <= self.variables[var2_idx]

    def create_solver(
        self,
        equalities: FrozenSet[Tuple[int, int]],
        inequalities: FrozenSet[Tuple[int, int, int]]
    ) -> Solver:
        """
        Create solver with cached constraints.

        Args:
            equalities: Set of (var_idx, value) tuples
            inequalities: Set of (var1_idx, var2_idx, offset) tuples
        """
        solver = Solver()

        # Add pre-compiled base constraints
        solver.add(*self._base_constraints)

        # Add cached equality constraints
        for var_idx, value in equalities:
            solver.add(self.get_equality_constraint(var_idx, value))

        # Add cached inequality constraints
        for var1_idx, var2_idx, offset in inequalities:
            solver.add(self.get_inequality_constraint(var1_idx, var2_idx, offset))

        return solver

# Usage example
cache = ConstraintCache(num_vars=50)

# Create multiple solvers with cached constraints (fast!)
solver1 = cache.create_solver(
    equalities=frozenset([(0, 5), (1, 10)]),
    inequalities=frozenset([(0, 1, 1), (1, 2, 2)])
)

solver2 = cache.create_solver(
    equalities=frozenset([(0, 5)]),  # Reuses cached constraint
    inequalities=frozenset([(0, 1, 1)])  # Reuses cached constraint
)

# Performance: Cached constraints avoid re-parsing overhead
```

### D. Uninterpreted Functions (EUF) for Abstraction and Performance

**CRITICAL: Use uninterpreted functions to model abstract operations and improve performance.**

#### What are Uninterpreted Functions?

Uninterpreted functions are functions that Z3 reasons about without knowing their implementation. Z3 only enforces that the function is **consistent** - if `f(x) = y`, then every occurrence of `f(x)` equals `y`.

This provides significant benefits:
1. **Performance**: No need to expand function definitions
2. **Abstraction**: Model operations without implementation details
3. **Reusability**: Define once, use many times without re-parsing
4. **Parameterization**: Build constraints over abstract functions

Reference: [Programming Z3 - EUF](https://theory.stanford.edu/~nikolaj/programmingz3.html#sec-euf--equality-and-uninterpreted-functions)

#### Declaring Uninterpreted Functions

```python
from z3 import *

# Define custom sorts (types)
S = DeclareSort('S')
T = DeclareSort('T')

# Uninterpreted function: S → S
f = Function('f', S, S)

# Uninterpreted function: S × S → T
g = Function('g', S, S, T)

# Uninterpreted function with built-in sorts
hash_fn = Function('hash', IntSort(), IntSort())

# Example: Using uninterpreted functions
x, y = Consts('x y', S)

solver = Solver()
solver.add(f(x) == f(y))  # If f(x) = f(y), Z3 doesn't need to know what f is
solver.add(x != y)        # But x and y are different

# This is satisfiable - f could be a constant function
assert solver.check() == sat
```

#### Benefits Over Interpreted Functions

```python
# ❌ SLOWER: Interpreted function (lambda or explicit definition)
def interpreted_hash_example():
    """Using interpreted functions - Z3 must expand definitions."""
    x, y = Ints('x y')

    # Define hash function explicitly
    hash_x = (x * 31) % 1000
    hash_y = (y * 31) % 1000

    solver = Solver()
    solver.add(hash_x == hash_y)
    solver.add(x != y)

    # Z3 must reason about arithmetic to solve
    return solver.check()

# ✅ FASTER: Uninterpreted function
def uninterpreted_hash_example():
    """Using uninterpreted functions - Z3 reasons about equality only."""
    hash_fn = Function('hash', IntSort(), IntSort())
    x, y = Ints('x y')

    solver = Solver()
    solver.add(hash_fn(x) == hash_fn(y))  # Don't care about implementation
    solver.add(x != y)                     # Just reason about equality

    # Z3 reasons about function equality, not arithmetic
    # Much faster for complex functions!
    return solver.check()
```

#### Use Case 1: Modeling Abstract Operations

```python
from z3 import *

def model_cache_coherence():
    """
    Model cache coherence protocol using uninterpreted functions.

    We don't need to know how read/write work, just their properties.
    """
    # Define memory address and value sorts
    Addr = DeclareSort('Addr')
    Value = DeclareSort('Value')

    # Uninterpreted functions for memory operations
    read = Function('read', Addr, Value)
    write = Function('write', Addr, Value, Addr)  # write(addr, val) → new_memory

    # Memory addresses
    addr1, addr2 = Consts('addr1 addr2', Addr)
    val1, val2 = Consts('val1 val2', Value)

    solver = Solver()

    # Property 1: Reading from an address after writing returns written value
    mem_after_write = write(addr1, val1)
    solver.add(read(mem_after_write) == val1)

    # Property 2: Writing to different addresses doesn't affect each other
    solver.add(addr1 != addr2)
    mem_after_write1 = write(addr1, val1)
    solver.add(read(mem_after_write1) == val1)  # addr1 has val1

    # We can verify cache coherence without implementing memory!
    assert solver.check() == sat
    return solver.model()
```

#### Use Case 2: Parameterized Constraint Generation (CRITICAL for Performance)

```python
from z3 import *
from typing import Callable

def create_dataflow_constraint_generator() -> Callable:
    """
    Create generator for dataflow constraints using uninterpreted functions.

    This is the KEY use case for performance - reuse function definitions
    across many constraint-solving queries.
    """
    # Define sorts
    Node = DeclareSort('Node')
    Data = DeclareSort('Data')

    # Uninterpreted functions (defined once, reused forever)
    transform = Function('transform', Data, Data)  # Data transformation
    combines = Function('combine', Data, Data, Data)  # Combine two data items

    def generate_constraints(
        num_nodes: int,
        edges: list[tuple[int, int]]
    ):
        """
        Generate dataflow constraints for a specific graph.

        Performance benefit: transform and combine are NOT re-parsed!
        """
        # Create node data variables
        nodes = [Const(f'node_{i}', Data) for i in range(num_nodes)]

        solver = Solver()

        # Dataflow constraints using pre-defined functions
        for src, dst in edges:
            # Data flows from src to dst through transform
            solver.add(nodes[dst] == transform(nodes[src]))

        # Additional constraints (example: combine multiple inputs)
        # No re-parsing of combine function!

        return solver, nodes

    return generate_constraints

# Usage: Generate constraints for different graphs (FAST!)
generator = create_dataflow_constraint_generator()

# Graph 1: Linear pipeline
solver1, nodes1 = generator(
    num_nodes=5,
    edges=[(0, 1), (1, 2), (2, 3), (3, 4)]
)
result1 = solver1.check()

# Graph 2: Different topology (FAST - reuses transform function!)
solver2, nodes2 = generator(
    num_nodes=10,
    edges=[(0, 1), (0, 2), (1, 3), (2, 3)]
)
result2 = solver2.check()

# Performance: No re-parsing of transform/combine functions!
# 10-100x faster than redefining functions each time
```

#### Use Case 3: Equivalence Checking

```python
from z3 import *

def check_function_equivalence():
    """
    Check if two implementations are equivalent using uninterpreted functions.
    """
    # Define abstract function we're implementing
    spec = Function('spec', IntSort(), IntSort())

    # Two proposed implementations
    x = Int('x')

    impl1 = (x * 2) + 1
    impl2 = x + x + 1

    solver = Solver()

    # Both implementations should match the spec
    solver.add(ForAll(x, spec(x) == impl1))

    # Check if impl2 is equivalent to spec
    # (i.e., impl1 ≡ impl2)
    x_test = Int('x_test')
    solver.add(spec(x_test) != impl2)  # Try to find counterexample

    # If unsat, implementations are equivalent
    result = solver.check()
    if result == unsat:
        print("Implementations are equivalent!")
    else:
        print(f"Counterexample: {solver.model()}")
```

#### Use Case 4: Array Theory Replacement

```python
from z3 import *

# Sometimes uninterpreted functions are cleaner than arrays
def compare_array_vs_function():
    """Compare Array theory vs uninterpreted functions."""

    # Using Array theory (more complex)
    def with_arrays():
        A = Array('A', IntSort(), IntSort())
        i, j = Ints('i j')

        solver = Solver()
        solver.add(A[i] == 10)
        solver.add(A[j] == 20)
        solver.add(i == j)  # Contradiction!

        return solver.check()  # unsat

    # Using uninterpreted function (simpler)
    def with_function():
        f = Function('f', IntSort(), IntSort())
        i, j = Ints('i j')

        solver = Solver()
        solver.add(f(i) == 10)
        solver.add(f(j) == 20)
        solver.add(i == j)  # Contradiction!

        return solver.check()  # unsat

    # Both work, but uninterpreted function is often clearer
    assert with_arrays() == unsat
    assert with_function() == unsat
```

#### Performance Comparison

```python
import time
from z3 import *

def benchmark_interpreted_vs_uninterpreted():
    """Benchmark: interpreted vs uninterpreted functions."""

    # Interpreted function (SLOW)
    def with_interpretation(n: int):
        x = [Int(f'x_{i}') for i in range(n)]

        solver = Solver()

        # Define function via expansion (re-parsed each time)
        for i in range(n - 1):
            # Complex "hash" function
            hash_i = (x[i] * 31 + 17) % 1000
            hash_i1 = (x[i+1] * 31 + 17) % 1000
            solver.add(hash_i == hash_i1)

        start = time.time()
        solver.check()
        return time.time() - start

    # Uninterpreted function (FAST)
    def with_uninterpreted(n: int):
        hash_fn = Function('hash', IntSort(), IntSort())
        x = [Int(f'x_{i}') for i in range(n)]

        solver = Solver()

        # Use function without expansion
        for i in range(n - 1):
            solver.add(hash_fn(x[i]) == hash_fn(x[i+1]))

        start = time.time()
        solver.check()
        return time.time() - start

    # Run benchmark
    n = 50
    time_interpreted = with_interpretation(n)
    time_uninterpreted = with_uninterpreted(n)

    print(f"Interpreted:     {time_interpreted:.4f}s")
    print(f"Uninterpreted:   {time_uninterpreted:.4f}s")
    print(f"Speedup:         {time_interpreted / time_uninterpreted:.2f}x")

# Expected output: 10-100x speedup with uninterpreted functions
```

#### When to Use Uninterpreted Functions

**Use uninterpreted functions when:**
1. ✅ You need to model abstract operations (hash, encrypt, transform)
2. ✅ Function implementation details don't matter for correctness
3. ✅ You're building parameterized constraint generators
4. ✅ You need to reuse function structure across many queries
5. ✅ You're checking equivalence or properties of functions

**Don't use uninterpreted functions when:**
1. ❌ You need Z3 to reason about the function's implementation
2. ❌ The function's behavior is critical to the constraint
3. ❌ You need specific arithmetic properties (use constraints instead)

#### Summary: EUF Performance Benefits

**Key Advantages:**
- **10-100x faster** than expanding function definitions
- **Reusable** across multiple constraint-solving queries
- **Cleaner** abstraction for complex systems
- **Parameterization** - define once, use with different parameters
- **No re-parsing** overhead when structure stays the same

**Best Practices:**
1. Define uninterpreted functions at module level for reuse
2. Use with constraint generator pattern (Section 4B)
3. Combine with tactics (Section 5) for maximum performance
4. Cache solver instances that use the same functions
5. Test both sat and unsat cases to verify correctness

---

## 5. Tactics and Optimization (MANDATORY)

### A. Understanding Z3 Tactics

**Z3 tactics are strategies for simplifying and solving constraints.**

```python
from z3 import *

# Available tactics (use help_simplifier() for full list)
tactics_overview = """
Key Tactics:
- simplify: Algebraic simplification
- propagate-values: Constant propagation
- solve-eqs: Solve equations and substitute
- qe: Quantifier elimination
- sat: SAT solver
- smt: SMT solver
- qfnra: Quantifier-free nonlinear real arithmetic
- qflia: Quantifier-free linear integer arithmetic
"""

# Example: Using tactics
def solve_with_tactics(constraints: list):
    """Solve constraints with optimized tactics."""
    # Create a tactic pipeline
    t = Then(
        'simplify',           # First: Simplify expressions
        'propagate-values',   # Second: Propagate constants
        'solve-eqs',          # Third: Solve and substitute
        'smt'                 # Finally: Use SMT solver
    )

    # Convert tactic to solver
    solver = t.solver()
    solver.add(*constraints)

    return solver.check()
```

### B. Tactic Selection by Problem Domain

```python
from z3 import *

def solve_linear_integer_arithmetic(constraints: list):
    """Optimized solver for linear integer arithmetic."""
    # Use specialized tactic for QF_LIA
    tactic = Tactic('qflia')  # Quantifier-free linear integer arithmetic
    solver = tactic.solver()
    solver.add(*constraints)
    return solver.check()

def solve_with_quantifier_elimination(constraints: list):
    """Solve with quantifier elimination."""
    # Use qe tactic to eliminate quantifiers
    tactic = Then('qe', 'simplify', 'smt')
    solver = tactic.solver()
    solver.add(*constraints)
    return solver.check()

def solve_sat_problem(constraints: list):
    """Optimized for pure SAT problems (Boolean)."""
    # Use SAT solver directly
    tactic = Tactic('sat')
    solver = tactic.solver()
    solver.add(*constraints)
    return solver.check()

def solve_nonlinear_real(constraints: list):
    """Solve nonlinear real arithmetic."""
    # Use specialized tactic
    tactic = Tactic('qfnra')  # Quantifier-free nonlinear real arithmetic
    solver = tactic.solver()
    solver.add(*constraints)
    return solver.check()
```

### C. Simplification Before Solving

**CRITICAL: Always simplify constraints before solving.**

```python
from z3 import *

def solve_with_simplification(constraints: list):
    """
    Solve constraints with aggressive simplification.

    Simplification can reduce problem size by orders of magnitude.
    """
    # Simplify each constraint
    simplified_constraints = [simplify(c) for c in constraints]

    # Further simplify the conjunction
    combined = And(*simplified_constraints)
    final_constraint = simplify(combined)

    solver = Solver()
    solver.add(final_constraint)

    return solver.check()

# Example: Simplification power
x = Int('x')
constraint = And(x + 5 > 10, x > 5, x > 3)  # Redundant constraints

simplified = simplify(constraint)
# Result: x > 5 (eliminates redundancy)

print(f"Original: {constraint}")
print(f"Simplified: {simplified}")
```

---

## 6. Context Management and Thread Safety (MANDATORY)

### A. Context Management

**Z3 objects belong to contexts. Proper context management is critical.**

```python
from z3 import *

# ✅ CORRECT: Default context (simplest)
def use_default_context():
    """Use default global context (fine for single-threaded)."""
    x = Int('x')
    y = Int('y')
    solver = Solver()
    solver.add(x + y == 10)
    return solver.check()

# ✅ CORRECT: Explicit context (better for isolation)
def use_explicit_context():
    """Use explicit context for isolation."""
    ctx = Context()
    x = Int('x', ctx)
    y = Int('y', ctx)
    solver = Solver(ctx=ctx)
    solver.add(x + y == 10)
    return solver.check()

# ❌ WRONG: Mixing contexts
def mixing_contexts_wrong():
    """Never mix objects from different contexts!"""
    ctx1 = Context()
    ctx2 = Context()

    x = Int('x', ctx1)
    y = Int('y', ctx2)  # Different context!

    solver = Solver(ctx=ctx1)
    # This will fail: y is from different context
    # solver.add(x + y == 10)  # ERROR!

    # CORRECT: Translate y to ctx1
    y_translated = y.translate(ctx1)
    solver.add(x + y_translated == 10)
    return solver.check()
```

### B. Thread Safety (CRITICAL)

**NEVER share Z3 objects between threads. Always use translation.**

```python
from z3 import *
import threading
from typing import List

# ❌ WRONG: Sharing Z3 objects between threads
def parallel_solve_wrong(constraints_list: List[List]):
    """WRONG: Sharing solver between threads."""
    solver = Solver()  # Shared - BAD!

    def solve_constraints(constraints):
        solver.add(*constraints)  # NOT THREAD-SAFE!
        return solver.check()

    threads = []
    for constraints in constraints_list:
        thread = threading.Thread(target=solve_constraints, args=(constraints,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

# ✅ CORRECT: Separate context per thread
def parallel_solve_correct(constraints_list: List[List]):
    """CORRECT: Each thread has own context."""
    results = [None] * len(constraints_list)

    def solve_constraints(idx, constraints):
        # Create thread-local context
        ctx = Context()
        solver = Solver(ctx=ctx)

        # Translate constraints to thread-local context
        translated = [c.translate(ctx) if hasattr(c, 'translate') else c
                      for c in constraints]

        solver.add(*translated)
        results[idx] = solver.check()

    threads = []
    for idx, constraints in enumerate(constraints_list):
        thread = threading.Thread(
            target=solve_constraints,
            args=(idx, constraints)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results

# ✅ CORRECT: Using concurrent.futures with proper isolation
from concurrent.futures import ThreadPoolExecutor

def parallel_solve_with_executor(constraints_list: List[List]):
    """Best practice: Use executor with proper context isolation."""

    def solve_in_thread(constraints):
        # Each future gets its own context
        ctx = Context()
        solver = Solver(ctx=ctx)

        # Recreate constraints in local context
        # (assumes constraints can be reconstructed)
        solver.add(*constraints)

        return solver.check()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(solve_in_thread, constraints_list))

    return results
```

---

## 7. Quantifier Management (ADVANCED)

### A. Avoiding Quantifiers When Possible

**Quantifiers significantly increase solving complexity. Avoid when possible.**

```python
from z3 import *

# ❌ INEFFICIENT: Using quantifiers unnecessarily
def check_array_property_with_quantifier(array_size: int):
    """Inefficient: Uses quantifier when not needed."""
    A = Array('A', IntSort(), IntSort())

    # Quantified formula: ∀i. 0 ≤ i < size → A[i] ≥ 0
    i = Int('i')
    constraint = ForAll(
        i,
        Implies(And(i >= 0, i < array_size), A[i] >= 0)
    )

    solver = Solver()
    solver.add(constraint)
    return solver.check()  # Slow with quantifiers

# ✅ EFFICIENT: Avoid quantifiers by unrolling
def check_array_property_unrolled(array_size: int):
    """Efficient: Unroll instead of using quantifiers."""
    A = Array('A', IntSort(), IntSort())

    # Unroll the constraint for each index
    constraints = [A[i] >= 0 for i in range(array_size)]

    solver = Solver()
    solver.add(*constraints)
    return solver.check()  # Much faster
```

### B. Quantifier Elimination

**When quantifiers are necessary, use qe tactic to eliminate them.**

```python
from z3 import *

def solve_with_quantifier_elimination():
    """Use quantifier elimination tactic."""
    x, y = Ints('x y')

    # Formula with quantifier: ∃x. x + y > 10
    constraint = Exists(x, x + y > 10)

    # Eliminate quantifier using qe tactic
    goal = Goal()
    goal.add(constraint)

    qe_tactic = Tactic('qe')
    result = qe_tactic(goal)

    print("After quantifier elimination:")
    print(result)

    # Solve the simplified formula
    solver = Solver()
    solver.add(result.as_expr())
    return solver.check()
```

---

## 8. Testing Z3 Code (MANDATORY)

### A. Test Structure

```python
import pytest
from z3 import *

class TestZ3Constraints:
    """Test suite for Z3 constraint solving."""

    def test_satisfiable_constraint(self):
        """Test that satisfiable constraints return sat."""
        x, y = Ints('x y')
        solver = Solver()
        solver.add(x + y == 10)
        solver.add(x > 0)
        solver.add(y > 0)

        result = solver.check()
        assert result == sat

        # Validate model
        model = solver.model()
        assert model[x].as_long() + model[y].as_long() == 10

    def test_unsatisfiable_constraint(self):
        """Test that unsatisfiable constraints return unsat."""
        x = Int('x')
        solver = Solver()
        solver.add(x > 10)
        solver.add(x < 5)

        result = solver.check()
        assert result == unsat

    def test_constraint_simplification(self):
        """Test that constraints simplify correctly."""
        x = Int('x')
        constraint = And(x > 5, x > 3, x > 1)

        simplified = simplify(constraint)

        # Should simplify to x > 5
        assert str(simplified) == "x > 5"

    def test_model_extraction(self):
        """Test that models can be extracted and validated."""
        x, y, z = Ints('x y z')
        solver = Solver()
        solver.add(x + y + z == 15)
        solver.add(x > y)
        solver.add(y > z)
        solver.add(z > 0)

        assert solver.check() == sat

        model = solver.model()
        x_val = model[x].as_long()
        y_val = model[y].as_long()
        z_val = model[z].as_long()

        # Verify constraints are satisfied
        assert x_val + y_val + z_val == 15
        assert x_val > y_val
        assert y_val > z_val
        assert z_val > 0

    def test_timeout_protection(self):
        """Test that solver respects timeout."""
        solver = Solver()
        solver.set("timeout", 1000)  # 1 second timeout

        # Add potentially hard constraints
        x = Int('x')
        solver.add(x**10 + x**9 == 42)

        result = solver.check()
        assert result in [sat, unsat, unknown]  # unknown if timeout
```

### B. Property-Based Testing

```python
import pytest
from hypothesis import given, strategies as st
from z3 import *

@given(st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=10))
def test_sum_constraint_property(values):
    """Property test: Sum of variables should equal sum of values."""
    n = len(values)
    variables = [Int(f'x_{i}') for i in range(n)]
    target_sum = sum(values)

    solver = Solver()
    # Constrain variables to specific values
    for var, val in zip(variables, values):
        solver.add(var == val)

    # Add sum constraint
    solver.add(sum(variables) == target_sum)

    # Should always be satisfiable
    assert solver.check() == sat

    # Verify model
    model = solver.model()
    model_sum = sum(model[var].as_long() for var in variables)
    assert model_sum == target_sum
```

---

## 9. Common Patterns and Examples

### A. SAT Problem Pattern

```python
from z3 import *

def solve_3sat_problem(clauses: list):
    """
    Solve 3-SAT problem.

    Args:
        clauses: List of 3-literal clauses, e.g., [(1, -2, 3), (-1, 2, -3)]

    Returns:
        Model if satisfiable, None otherwise
    """
    # Extract variables
    var_indices = set()
    for clause in clauses:
        var_indices.update(abs(lit) for lit in clause)

    # Create Boolean variables
    vars = {i: Bool(f'x{i}') for i in var_indices}

    # Build clauses
    solver = Solver()
    for clause in clauses:
        literals = []
        for lit in clause:
            if lit > 0:
                literals.append(vars[lit])
            else:
                literals.append(Not(vars[-lit]))
        solver.add(Or(*literals))

    if solver.check() == sat:
        model = solver.model()
        return {i: is_true(model[var]) for i, var in vars.items()}
    return None

# Example usage
clauses = [(1, -2, 3), (-1, 2, -3), (1, 2, 3)]
solution = solve_3sat_problem(clauses)
print(f"Solution: {solution}")
```

### B. Optimization Pattern

```python
from z3 import *

def solve_knapsack_problem(
    weights: list,
    values: list,
    capacity: int
) -> tuple:
    """
    Solve 0-1 knapsack problem using Z3 Optimize.

    Args:
        weights: Item weights
        values: Item values
        capacity: Knapsack capacity

    Returns:
        (selected_items, total_value)
    """
    n = len(weights)

    # Create Boolean variables for item selection
    items = [Bool(f'item_{i}') for i in range(n)]

    # Create optimizer
    opt = Optimize()

    # Capacity constraint
    total_weight = sum(
        If(items[i], weights[i], 0)
        for i in range(n)
    )
    opt.add(total_weight <= capacity)

    # Maximize total value
    total_value = sum(
        If(items[i], values[i], 0)
        for i in range(n)
    )
    opt.maximize(total_value)

    if opt.check() == sat:
        model = opt.model()
        selected = [i for i in range(n) if is_true(model[items[i]])]
        max_value = sum(values[i] for i in selected)
        return selected, max_value

    return [], 0

# Example usage
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

selected, value = solve_knapsack_problem(weights, values, capacity)
print(f"Selected items: {selected}, Total value: {value}")
```

### C. Constraint Satisfaction Pattern

```python
from z3 import *

def solve_sudoku(puzzle: list) -> list:
    """
    Solve Sudoku puzzle.

    Args:
        puzzle: 9x9 grid with 0 for empty cells

    Returns:
        Solved 9x9 grid
    """
    # Create 9x9 grid of integer variables
    cells = [[Int(f'cell_{i}_{j}') for j in range(9)] for i in range(9)]

    solver = Solver()

    # Cell constraints: 1-9
    for i in range(9):
        for j in range(9):
            solver.add(And(cells[i][j] >= 1, cells[i][j] <= 9))

    # Row constraints
    for i in range(9):
        solver.add(Distinct(cells[i]))

    # Column constraints
    for j in range(9):
        solver.add(Distinct([cells[i][j] for i in range(9)]))

    # Box constraints
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(cells[box_row*3 + i][box_col*3 + j])
            solver.add(Distinct(box))

    # Given values
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                solver.add(cells[i][j] == puzzle[i][j])

    # Solve
    if solver.check() == sat:
        model = solver.model()
        solution = [[model[cells[i][j]].as_long() for j in range(9)]
                    for i in range(9)]
        return solution

    return None
```

---

## 10. Quick Reference

### Common Commands

```python
# Imports
from z3 import *

# Creating variables
x = Int('x')              # Integer variable
y = Real('y')             # Real variable
b = Bool('b')             # Boolean variable
A = Array('A', IntSort(), IntSort())  # Array

# Uninterpreted functions (for abstraction & performance)
S = DeclareSort('S')      # Custom sort
f = Function('f', S, S)   # Uninterpreted function S → S
hash_fn = Function('hash', IntSort(), IntSort())  # Int → Int

# Creating solver
solver = Solver()         # Standard solver
opt = Optimize()          # Optimization solver

# Adding constraints
solver.add(x > 0)
solver.add(x + y == 10)

# Checking satisfiability
result = solver.check()   # Returns: sat, unsat, or unknown

# Extracting model
if result == sat:
    model = solver.model()
    x_value = model[x].as_long()

# Simplification
expr = And(x > 5, x > 3)
simplified = simplify(expr)

# Tactics
tactic = Then('simplify', 'solve-eqs', 'smt')
solver = tactic.solver()

# Timeout
solver.set("timeout", 5000)  # 5 seconds
```

### Testing Template

```python
import pytest
from z3 import *

def test_constraint_satisfiability():
    """Test that constraints are satisfiable."""
    x, y = Ints('x y')
    solver = Solver()
    solver.add(x + y == 10)
    solver.add(x > 0)

    result = solver.check()
    assert result == sat

    model = solver.model()
    assert model[x].as_long() + model[y].as_long() == 10
```

---

## 11. Summary

**CRITICAL Requirements for Z3 Code:**

1. **Verification First**: Always check solver.check() result before extracting models
2. **Simplify Before Solving**: Use simplify() and tactics to optimize performance
3. **Function Generation**: Pre-compile constraint structures for parameterized problems
4. **Uninterpreted Functions**: Use EUF for abstraction and 10-100x performance gains
5. **Tactic Optimization**: Use appropriate tactics (qe, simplify, smt) for problem domain
6. **Context Management**: Never share Z3 objects between threads; use .translate()
7. **Quantifier Avoidance**: Minimize quantifiers; use qe when necessary
8. **Timeout Protection**: Always set timeouts for potentially hard constraints
9. **Model Validation**: Verify extracted models satisfy original constraints
10. **Testing**: Comprehensive tests for sat/unsat cases and model extraction
11. **Both Styles**: Support declarative (clarity) and imperative (control) styles

**Performance Optimization Priorities:**
1. Use uninterpreted functions (EUF) for abstract operations (10-100x speedup)
2. Pre-compile constraint structures (function generation pattern)
3. Use tactics (simplify, propagate-values, solve-eqs)
4. Cache constraint objects with @lru_cache
5. Simplify expressions before solving
6. Eliminate quantifiers with qe tactic
7. Use specialized tactics for problem domain (qflia, qfnra, sat)

**Agent Verification Protocol:**
- Constraints must be well-formed
- Solver.check() must complete (with timeout)
- Models must be extractable when sat
- Tests must cover sat/unsat/unknown cases
- Type checking with mypy must pass
- Code formatting with black must pass
- Performance tests for repeated queries

**Remember**: Optimize for reuse through function generation. Use tactics for speed. Test both satisfiable and unsatisfiable cases. Keep constraints simple and well-typed.

---

**End of Python Z3 SMT Solver Development Guidelines**
