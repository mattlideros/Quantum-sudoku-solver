import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


def cell_inequality_comparator(qc, cell_A, cell_B, clause_qubit, ancilla_reg, k_bits):
    """Flips clause_qubit to 1 if cell_A != cell_B."""
    for i in range(k_bits):
        qc.cx(cell_A[i], ancilla_reg[i])
        qc.cx(cell_B[i], ancilla_reg[i])
    for i in range(k_bits):
        qc.x(ancilla_reg[i])
    qc.mcx(ancilla_reg[:k_bits], clause_qubit)
    qc.x(clause_qubit)
    for i in range(k_bits):
        qc.x(ancilla_reg[i])
    for i in reversed(range(k_bits)):
        qc.cx(cell_B[i], ancilla_reg[i])
        qc.cx(cell_A[i], ancilla_reg[i])


def hint_comparator(qc, cell_var, hint_bits, clause_qubit, k_bits):
    """Flips clause_qubit to 1 if cell_var != hint_bits."""
    for i in range(k_bits):
        if hint_bits[i] == 1:
            qc.x(cell_var[i])
    for i in range(k_bits):
        qc.x(cell_var[i])
    qc.mcx(cell_var, clause_qubit)
    qc.x(clause_qubit)
    for i in reversed(range(k_bits)):
        qc.x(cell_var[i])
    for i in reversed(range(k_bits)):
        if hint_bits[i] == 1:
            qc.x(cell_var[i])


def _get_invalid_implicants(grid_size, k_bits):
    """
    Helper to programmatically find the minimal bit patterns (implicants)
    representing numbers out of bounds: [grid_size, 2**k_bits - 1].
    Returns a list of dicts: {bit_index: bit_value}.
    """
    invalid_states = set(range(grid_size, 1 << k_bits))
    if not invalid_states:
        return []

    implicants = []
    remaining = set(invalid_states)

    # Greedy sub-cube minimization from largest cubes to smallest
    for size in range(k_bits, 0, -1):
        for combination in itertools.combinations(range(k_bits), k_bits - size):
            for values in itertools.product([0, 1], repeat=len(combination)):
                cube_states = []
                for num in range(1 << k_bits):
                    match = True
                    for idx, val in zip(combination, values):
                        if ((num >> idx) & 1) != val:
                            match = False
                            break
                    if match:
                        cube_states.append(num)

                if cube_states and all(s in invalid_states for s in cube_states):
                    if any(s in remaining for s in cube_states):
                        implicants.append({idx: val for idx, val in zip(combination, values)})
                        remaining.difference_update(cube_states)
    return implicants


def enforce_valid_range(qc, cell_var, clause_qubits, implicants):
    """Penalizes states outside the valid Sudoku range using calculated implicants."""
    for idx, imp in enumerate(implicants):
        control_qubits = []
        for bit_idx, bit_val in imp.items():
            # If the condition requires a 0 bit, wrap with X gates to activate MCX control
            if bit_val == 0:
                qc.x(cell_var[bit_idx])
            control_qubits.append(cell_var[bit_idx])

        qc.mcx(control_qubits, clause_qubits[idx])
        qc.x(clause_qubits[idx])

        # Uncompute the temporary X gates
        for bit_idx, bit_val in imp.items():
            if bit_val == 0:
                qc.x(cell_var[bit_idx])


def build_generalized_sudoku_solver(puzzle_matrix, n):
    grid_size = n ** 2
    k_bits = int(np.ceil(np.log2(grid_size)))
    vars_positions = [(r, c) for r in range(grid_size) for c in range(grid_size) if puzzle_matrix[r][c] == 0]
    num_vars = len(vars_positions)

    if num_vars == 0:
        raise ValueError("No variable cells found to solve!")

    var_reg = QuantumRegister(num_vars * k_bits, name='var')
    constraints = []

    # Gather standard Sudoku constraints
    for r in range(grid_size):
        for c1 in range(grid_size):
            for c2 in range(c1 + 1, grid_size):
                if puzzle_matrix[r][c1] == 0 or puzzle_matrix[r][c2] == 0:
                    constraints.append(((r, c1), (r, c2)))
    for c in range(grid_size):
        for r1 in range(grid_size):
            for r2 in range(r1 + 1, grid_size):
                if puzzle_matrix[r1][c] == 0 or puzzle_matrix[r2][c] == 0:
                    constraints.append(((r1, c), (r2, c)))
    for block_r in range(0, grid_size, n):
        for block_c in range(0, grid_size, n):
            block_cells = [(block_r + i, block_c + j) for i in range(n) for j in range(n)]
            for i in range(grid_size):
                for j in range(i + 1, grid_size):
                    p1, p2 = block_cells[i], block_cells[j]
                    if puzzle_matrix[p1[0]][p1[1]] == 0 or puzzle_matrix[p2[0]][p2[1]] == 0:
                        constraints.append(tuple(sorted([p1, p2])))

    constraints = list(set(constraints))
    num_constraints = len(constraints)

    # Compute minimal implicants for out-of-bounds states dynamically based on n
    implicants = _get_invalid_implicants(grid_size, k_bits)
    range_clauses_per_var = len(implicants)
    total_range_clauses = num_vars * range_clauses_per_var

    clause_reg = QuantumRegister(num_constraints + total_range_clauses, name='clause')
    xor_ancilla = QuantumRegister(k_bits, name='xor')
    phase_target = QuantumRegister(1, name='phase')
    c_reg = ClassicalRegister(num_vars * k_bits, name='meas')

    qc = QuantumCircuit(var_reg, clause_reg, xor_ancilla, phase_target, c_reg)

    qc.h(var_reg)
    qc.x(phase_target)
    qc.h(phase_target)

    def get_cell_elements(r, c):
        """Returns (is_hint, data) where data is either a list of qubits or a bit tuple."""
        if puzzle_matrix[r][c] != 0:
            val = puzzle_matrix[r][c] - 1
            bit_tuple = tuple((val >> i) & 1 for i in range(k_bits))
            return True, bit_tuple
        else:
            idx = vars_positions.index((r, c))
            cell_qubits = [var_reg[idx * k_bits + i] for i in range(k_bits)]
            return False, cell_qubits

    search_space = 2 ** (num_vars * k_bits)
    iterations = int(np.floor(np.pi / 4 * np.sqrt(search_space)))
    if iterations < 1: iterations = 1

    for _ in range(iterations):
        # --- Forward Oracle Pass ---
        for idx, (pos_A, pos_B) in enumerate(constraints):
            is_hint_A, data_A = get_cell_elements(pos_A[0], pos_A[1])
            is_hint_B, data_B = get_cell_elements(pos_B[0], pos_B[1])

            if not is_hint_A and not is_hint_B:
                cell_inequality_comparator(qc, data_A, data_B, clause_reg[idx], xor_ancilla, k_bits)
            elif not is_hint_A and is_hint_B:
                hint_comparator(qc, data_A, data_B, clause_reg[idx], k_bits)
            elif is_hint_A and not is_hint_B:
                hint_comparator(qc, data_B, data_A, clause_reg[idx], k_bits)

        if range_clauses_per_var > 0:
            for idx in range(num_vars):
                _, cell_qubits = get_cell_elements(vars_positions[idx][0], vars_positions[idx][1])
                r_start = num_constraints + (idx * range_clauses_per_var)
                enforce_valid_range(qc, cell_qubits, clause_reg[r_start: r_start + range_clauses_per_var], implicants)

        # Global Reflection
        qc.mcx(clause_reg, phase_target)

        # --- Backward Uncomputation Pass ---
        if range_clauses_per_var > 0:
            for idx in range(num_vars):
                _, cell_qubits = get_cell_elements(vars_positions[idx][0], vars_positions[idx][1])
                r_start = num_constraints + (idx * range_clauses_per_var)
                enforce_valid_range(qc, cell_qubits, clause_reg[r_start: r_start + range_clauses_per_var], implicants)

        for idx, (pos_A, pos_B) in reversed(list(enumerate(constraints))):
            is_hint_A, data_A = get_cell_elements(pos_A[0], pos_A[1])
            is_hint_B, data_B = get_cell_elements(pos_B[0], pos_B[1])

            if not is_hint_A and not is_hint_B:
                cell_inequality_comparator(qc, data_A, data_B, clause_reg[idx], xor_ancilla, k_bits)
            elif not is_hint_A and is_hint_B:
                hint_comparator(qc, data_A, data_B, clause_reg[idx], k_bits)
            elif is_hint_A and not is_hint_B:
                hint_comparator(qc, data_B, data_A, clause_reg[idx], k_bits)

        # --- Diffuser Pass ---
        qc.h(var_reg)
        qc.x(var_reg)
        qc.h(var_reg[-1])
        qc.mcx(var_reg[:-1], var_reg[-1])
        qc.h(var_reg[-1])
        qc.x(var_reg)
        qc.h(var_reg)

    qc.measure(var_reg, c_reg)
    return qc


if __name__ == "__main__":
    test_9x9_puzzle = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],

        [8, 5, 9, 7, 6, 1, 0, 2, 3],  # Missing position (3,6) -> Should solve to 4
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],

        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]

    open_slots = [(r, c) for r in range(9) for c in range(9) if test_9x9_puzzle[r][c] == 0]
    num_vars = len(open_slots)

    print("Assembling generalized solver layout for a 9x9 configuration (n=3)...")
    sudoku_circuit = build_generalized_sudoku_solver(test_9x9_puzzle, n=3)

    print(f"Circuit compiled successfully with {sudoku_circuit.num_qubits} qubits.")
    backend = AerSimulator()
    job = backend.run(sudoku_circuit, shots=1024)
    counts = job.result().get_counts()

    winning_bitstring = max(counts, key=counts.get)
    high_score = counts[winning_bitstring]

    print(f"\nRaw winning bitstring from simulator: {winning_bitstring} (found {high_score}/1024 times)")

    ordered_bits = winning_bitstring[::-1]
    k_bits = int(np.ceil(np.log2(9)))

    print(f"Tracked Open Slots: {open_slots}")
    print("-" * 40)

    for idx in range(num_vars):
        cell_bits = ordered_bits[idx * k_bits: (idx + 1) * k_bits]
        decimal_val = 0
        for bit_position, bit_char in enumerate(cell_bits):
            decimal_val |= (int(bit_char) << bit_position)

        sudoku_val = decimal_val + 1
        print(f"Solved Slot {open_slots[idx]} = {sudoku_val}")