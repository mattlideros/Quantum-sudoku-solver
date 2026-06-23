import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


def cell_inequality_comparator(qc, cell_A, cell_B, clause_qubit, ancilla_pair):
    """Compares two 2-qubit variable cells. Flips clause_qubit to 1 if cell_A != cell_B."""
    qc.cx(cell_A[0], ancilla_pair[0])
    qc.cx(cell_B[0], ancilla_pair[0])
    qc.cx(cell_A[1], ancilla_pair[1])
    qc.cx(cell_B[1], ancilla_pair[1])

    qc.x(ancilla_pair[0])
    qc.x(ancilla_pair[1])
    qc.ccx(ancilla_pair[0], ancilla_pair[1], clause_qubit)
    qc.x(clause_qubit)

    # Uncompute
    qc.x(ancilla_pair[0])
    qc.x(ancilla_pair[1])
    qc.cx(cell_B[1], ancilla_pair[1])
    qc.cx(cell_A[1], ancilla_pair[1])
    qc.cx(cell_B[0], ancilla_pair[0])
    qc.cx(cell_A[0], ancilla_pair[0])


def hint_comparator(qc, cell_var, hint_tuple, clause_qubit):
    """Compares a 2-qubit variable cell against a static hint tuple. (Low, High) ordering."""
    hint_low, hint_high, _ = hint_tuple

    if hint_low == 0:  qc.x(cell_var[0])
    if hint_high == 0: qc.x(cell_var[1])

    qc.ccx(cell_var[0], cell_var[1], clause_qubit)
    qc.x(clause_qubit)

    if hint_high == 0: qc.x(cell_var[1])
    if hint_low == 0:  qc.x(cell_var[0])


def build_sudoku_solver(puzzle_matrix):
    vars_positions = [(r, c) for r in range(4) for c in range(4) if puzzle_matrix[r][c] == 0]
    num_vars = len(vars_positions)

    if num_vars == 0:
        raise ValueError("No variable cells found to solve!")

    var_reg = QuantumRegister(num_vars * 2, name='var')
    constraints = []

    def get_cell_logic(r, c):
        if puzzle_matrix[r][c] != 0:
            val = puzzle_matrix[r][c] - 1
            # FIXED: Low bit first (val & 1), High bit second (val >> 1 & 1)
            return (val & 1, val >> 1 & 1, True)
        else:
            idx = vars_positions.index((r, c))
            return (var_reg[idx * 2], var_reg[idx * 2 + 1], False)

    # 1. Gather Row & Column constraints
    for r in range(4):
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
                if puzzle_matrix[r][c1] == 0 or puzzle_matrix[r][c2] == 0:
                    constraints.append(((r, c1), (r, c2)))
    for c in range(4):
        for r1 in range(4):
            for r2 in range(r1 + 1, 4):
                if puzzle_matrix[r1][c] == 0 or puzzle_matrix[r2][c] == 0:
                    constraints.append(((r1, c), (r2, c)))

    # 2. Gather 2x2 Sub-Grid Block Constraints
    for block_r in [0, 2]:
        for block_c in [0, 2]:
            block_cells = [(block_r + i, block_c + j) for i in range(2) for j in range(2)]
            for i in range(4):
                for j in range(i + 1, 4):
                    p1, p2 = block_cells[i], block_cells[j]
                    if puzzle_matrix[p1[0]][p1[1]] == 0 or puzzle_matrix[p2[0]][p2[1]] == 0:
                        pair = tuple(sorted([p1, p2]))
                        constraints.append(pair)

    constraints = list(set(constraints))
    num_constraints = len(constraints)

    clause_reg = QuantumRegister(num_constraints, name='clause')
    xor_ancilla = QuantumRegister(2, name='xor')
    phase_target = QuantumRegister(1, name='phase')
    c_reg = ClassicalRegister(num_vars * 2, name='meas')

    qc = QuantumCircuit(var_reg, clause_reg, xor_ancilla, phase_target, c_reg)

    # Initialization
    qc.h(var_reg)
    qc.x(phase_target)
    qc.h(phase_target)

    search_space = 2 ** (num_vars * 2)
    iterations = int(np.floor(np.pi / 4 * np.sqrt(search_space)))
    if iterations < 1: iterations = 1

    # Grover Iteration Loop
    for _ in range(iterations):
        # Forward Oracle Pass
        for idx, (pos_A, pos_B) in enumerate(constraints):
            cell_A_bits = get_cell_logic(pos_A[0], pos_A[1])
            cell_B_bits = get_cell_logic(pos_B[0], pos_B[1])

            if not cell_A_bits[2] and not cell_B_bits[2]:
                cell_inequality_comparator(qc, cell_A_bits[:2], cell_B_bits[:2], clause_reg[idx], xor_ancilla)
            elif not cell_A_bits[2] and cell_B_bits[2]:
                hint_comparator(qc, cell_A_bits[:2], cell_B_bits, clause_reg[idx])
            elif cell_A_bits[2] and not cell_B_bits[2]:
                hint_comparator(qc, cell_B_bits[:2], cell_A_bits, clause_reg[idx])

        # Global Conjunction Trigger
        qc.mcx(clause_reg, phase_target)

        # Backward Oracle Uncomputation
        for idx, (pos_A, pos_B) in reversed(list(enumerate(constraints))):
            cell_A_bits = get_cell_logic(pos_A[0], pos_A[1])
            cell_B_bits = get_cell_logic(pos_B[0], pos_B[1])

            if not cell_A_bits[2] and not cell_B_bits[2]:
                cell_inequality_comparator(qc, cell_A_bits[:2], cell_B_bits[:2], clause_reg[idx], xor_ancilla)
            elif not cell_A_bits[2] and cell_B_bits[2]:
                hint_comparator(qc, cell_A_bits[:2], cell_B_bits, clause_reg[idx])
            elif cell_A_bits[2] and not cell_B_bits[2]:
                hint_comparator(qc, cell_B_bits[:2], cell_A_bits, clause_reg[idx])

        # Diffuser Phase
        qc.h(var_reg)
        qc.x(var_reg)
        qc.h(var_reg[-1])
        qc.mcx(var_reg[:-1], var_reg[-1])
        qc.h(var_reg[-1])
        qc.x(var_reg)
        qc.h(var_reg)

    qc.measure(var_reg, c_reg)
    return qc, vars_positions


# --- RUN EXECUTION ---
test_puzzle = [
    [1, 2, 4, 0], # Missing position (0,3) -> Should solve to 3
    [3, 0, 1, 2], # Missing position (1,1) -> Should solve to 4
    [4, 3, 2, 1],
    [2, 1, 3, 4]
]

print("Assembling Sudoku Oracle and Diffuser tracks...")
sudoku_circuit, open_slots = build_sudoku_solver(test_puzzle)

print("Simulating circuit execution on AerSimulator backend...")
backend = AerSimulator()
job = backend.run(sudoku_circuit, shots=1024)
counts = job.result().get_counts()

winning_bitstring = max(counts, key=counts.get)
high_score = counts[winning_bitstring]

print(f"\nRaw winning bitstring from simulator: {winning_bitstring} (found {high_score}/1024 times)")

# Reverse the bitstring to read left-to-right sequentially from var_reg[0]
ordered_bits = winning_bitstring[::-1]

print(f"Tracked Open Slots: {open_slots}")
print("-" * 40)

for idx, (r, c) in enumerate(open_slots):
    low_bit = int(ordered_bits[idx * 2])
    high_bit = int(ordered_bits[idx * 2 + 1])

    decimal_val = (high_bit << 1) | low_bit
    sudoku_val = decimal_val + 1

    print(f"Solved Slot {open_slots[idx]} = {sudoku_val}")