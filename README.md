# Mini-Project #1: n^2 x n^2 Sudoku Solver with Grover's Algorithm

This repository contains a quantum-computed Sudoku solver implemented using Qiskit, designed to solve constraint-satisfaction problems on quantum simulators using Grover's Algorithm. 

The implementation features two separate versions tailored for scalability and troubleshooting:
1. **`Quantum_sudoku_4x4.py`**: A dedicated 4x4 (n=2) optimized configuration solver.
2. **`Quantum_sudoku_n^2xn^2.py`**: A generalized formulation designed to scale dynamically up to 9x9 (n=3) and higher configurations using dynamic range checking.

---

## Design and Quantum Constraints

Per project guidelines, the construction relies strictly on a fault-tolerant structure:
* **Allowed Gates**: Multi-controlled Z (MCZ) gates are restricted entirely to the Diffuser Phase. 
* **Oracle Gate Set**: The circuit construction uses arbitrary 1-qubit gates, CX, and Toffoli gates to compute constraints.
* **No Mid-Circuit Execution**: No classical bits or mid-circuit measurements are permitted. The entire pipeline remains coherent until the final measurement register is read.

The puzzle layout accepts a standard 0-indexed matrix configuration. Empty cells are defined as `0`, while predetermined values are mapped directly to their corresponding bitwise states within the oracle engine.

---

## Constraint Generation and Oracle Computation

The core engine of this Sudoku solver relies on translating classic board game rules into structural pairs of coordinates, which are then evaluated by quantum gates inside the oracle.

### 1. How the Constraint List is Formed

A constraint in Sudoku dictates that two cells in the same row, column, or $n \times n$ block cannot share the same value. The `build_generalized_sudoku_solver` function generates these rules by scanning the entire grid topology and mapping out every mandatory interaction:

* **Row and Column Pairs**: The function runs nested loops across the matrix dimensions ($grid\_size = n^2$). For every row, it tracks every unique pair of column indices `(c1, c2)`. If at least one of the two cells is an unassigned variable (`0`), it appends the pair to the list. It applies the exact same pairing logic vertically down every column for row pairs `(r1, r2)`.
* **$n \times n$ Block Pairs**: To handle the sub-grid boxes, the code calculates the top-left boundary of every block, gathers all $n^2$ coordinates belonging to that specific block into a temporary list, and pairs them up uniquely.
* **Optimization Filtering**: The function only appends a constraint pair if at least one of the two cells is an unassigned variable (`0`). If both cells are already filled with hardcoded hints, their relationship is static, so it skips them to minimize the size of the quantum circuit. Finally, `list(set(constraints))` eliminates any duplicate pairs that naturally overlap between rows, columns, and blocks.

### 2. How the Oracle Computes Based on Constraints

Once the constraint list is populated, the forward pass of the Grover oracle iterates through every single pair in the list to determine if the quantum state violates any rules. 

For each constraint pair `(pos_A, pos_B)`, the helper function `get_cell_elements` checks whether the coordinates point to a static hint or an unassigned quantum variable. This splits the computation into two distinct logic branches:

#### Branch A: Comparing Two Unknown Variables
If both cells in the constraint pair are unassigned variables, they are both represented by quantum registers. The oracle calls `cell_inequality_comparator`:
* **State XORing**: It applies $CX$ gates between the corresponding qubits of `cell_A` and `cell_B`, storing the result in a shared `xor_ancilla` register. If the bits are identical, the ancilla bit stays `0`; if they differ, it becomes `1`.
* **Violation Flagging**: If *all* bit positions in the ancilla register end up as `0`, it means `cell_A == cell_B` (a direct Sudoku violation). The code flips the target clause qubit using a multi-controlled $X$ (`mcx`) gate to flag this state configuration as invalid.
* **Uncomputation**: It immediately reruns the $CX$ gates in reverse to clean the `xor_ancilla` register back to $|0\rangle$ so it can be reused safely by the next constraint in the loop.

#### Branch B: Comparing a Variable Against a Fixed Hint
If one cell is an unassigned variable and the other is a predetermined hint on the board, the oracle calls `hint_comparator`:
* **Direct Value Masking**: Instead of wasting qubits to hold the hint's value, the code checks the hint's classic bit configuration. Wherever the hint has a `0` bit, it temporarily applies an $X$ gate to the corresponding qubit of the variable cell.
* **Matching Check**: A multi-controlled $X$ gate targets the clause qubit. If the variable's state matches the hint's bit pattern exactly, the control conditions are met, and the clause qubit is flipped to signal a violation.
* **Clean Up**: The temporary $X$ gates are reapplied to cleanly uncompute the variable register's state.

### 3. Global Reflection and Symmetric Cleanup

Once every constraint pair from the list has checked its state, the overall validity is tied directly to the `clause_reg` register. 



### 4. Some core functions 
* `cell_inequality_comparator(qc, cell_A, cell_B, clause_qubit, ancilla_reg, k_bits)`: Uses a series of CX and multi-controlled operations to compare two unassigned variable cells. It flips an auxiliary clause qubit to |1> if and only if the cell states are unique ($A \neq B$).
* `_get_invalid_implicants(grid_size, k_bits)`: captures invalid out of bounds strings. For n=3 this targets values 9 through 15 this is enforced by `enforce_valid_range(qc, cell_var, clause_qubits, implicants)`
$\begin{matrix}a\end{matrix}$

### 5. Test Cases
Test puzzles are included. For the 4x4 puzzle, the test puzzle is given 2 unassigned cells; this program should resolve quickly when run locally. For 3 unassigned cells, the program resolved locally but had a run time of a few minutes. Similarly, for our test 9x9 grid with 1 unassigned cell, the program resolved and retruned a valid solution within a few minutes. For 4 unassigned cells my local machine did not have enough memory to run the program. The user is encouraged to edit the test puzzles and attempt to run the program with 1-3 missing positions for a 4x4 grid and 1 missing position for the 9x9 grid. 

```python
test_puzzle = [
    [1, 2, 4, 0],  # Missing position (0,3) -> Solves to 3
    [3, 0, 1, 2],  # Missing position (1,1) -> Solves to 4
    [4, 3, 2, 1],
    [2, 1, 3, 4]
]
test_9x9_puzzle = [
    [5, 3, 4,  6, 7, 8,  9, 1, 2],
    [6, 7, 2,  1, 9, 5,  3, 4, 8],
    [1, 9, 8,  3, 4, 2,  5, 6, 7],

    [8, 5, 9,  7, 6, 1,  0, 2, 3],  # Missing position (3,6) -> Solves to 4
    [4, 2, 6,  8, 5, 3,  7, 9, 1],
    [7, 1, 3,  9, 2, 4,  8, 5, 6],

    [9, 6, 1,  5, 3, 7,  2, 8, 4],
    [2, 8, 7,  4, 1, 9,  6, 3, 5],
    [3, 4, 5,  2, 8, 6,  1, 7, 9]
]
