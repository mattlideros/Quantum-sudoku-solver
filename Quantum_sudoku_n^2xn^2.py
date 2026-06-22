import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


def cell_inequality_comparator(qc, cell_A, cell_B, clause_qubit, ancilla_reg, k_bits):
    """
    Dynamically compares two k-bit variable cells.
    Flips clause_qubit to 1 if cell_A != cell_B.
    """
    # 1. Bitwise XOR via CX gates across all k bits
    for i in range(k_bits):
        qc.cx(cell_A[i], ancilla_reg[i])
        qc.cx(cell_B[i], ancilla_reg[i])

    # 2. Invert bits to check for a perfect match (|11...1>)
    for i in range(k_bits):
        qc.x(ancilla_reg[i])

    # Multi-controlled NOT on the ancillas flags EQUALITY
    qc.mcx(ancilla_reg[:k_bits], clause_qubit)

    # Invert the clause bit so 1 means NOT EQUAL
    qc.x(clause_qubit)

    # 3. Clean uncomputation pass
    for i in range(k_bits):
        qc.x(ancilla_reg[i])
    for i in reversed(range(k_bits)):
        qc.cx(cell_B[i], ancilla_reg[i])
        qc.cx(cell_A[i], ancilla_reg[i])


def hint_comparator(qc, cell_var, hint_tuple, clause_qubit, k_bits):
    """
    Compares a k-bit variable cell against a static k-bit classical hint tuple.
    Flips clause_qubit to 1 if they are NOT equal.
    """
    hint_bits = hint_tuple[:k_bits]

    # Flip the variable bits where the hint expected a 0
    for i in range(k_bits):
        if hint_bits[i] == 0:
            qc.x(cell_var[i])

    # If the adjusted variable state matches |11...1>, they are equal
    qc.mcx(cell_var, clause_qubit)
    qc.x(clause_qubit)

    # Uncompute basis modifications
    for i in reversed(range(k_bits)):
        if hint_bits[i] == 0:
            qc.x(cell_var[i])


def build_generalized_sudoku_solver(puzzle_matrix, n=2):
    """
    Generates a pure Qiskit circuit blueprint for any n^2 x n^2 Sudoku puzzle.
    n=2 -> 4x4 grid (2 bits per cell)
    n=3 -> 9x9 grid (4 bits per cell)
    """
    grid_size = n ** 2

    # Determine the number of qubits required to represent a single cell value
    k_bits = int(np.ceil(np.log2(grid_size)))

    # Find coordinates of all empty slots (0)
    vars_positions = [(r, c) for r in range(grid_size) for c in range(grid_size) if puzzle_matrix[r][c] == 0]
    num_vars = len(vars_positions)

    if num_vars == 0:
        raise ValueError("No variable cells found to solve!")

    # Allocate registers dynamically based on grid constraints
    var_reg = QuantumRegister(num_vars * k_bits, name='var')
    constraints = []

    # Helper function to track cell logic formats
    def get_cell_logic(r, c):
        if puzzle_matrix[r][c] != 0:
            val = puzzle_matrix[r][c] - 1
            # Extract k-bit binary representation array (Low bit to High bit)
            bit_array = [(val >> i) & 1 for i in range(k_bits)]
            bit_array.append(True)  # True flags it as a static constant hint
            return tuple(bit_array)
        else:
            idx = vars_positions.index((r, c))
            # Slice the exact k-qubit chunk dedicated to this variable position
            cell_qubits = [var_reg[idx * k_bits + i] for i in range(k_bits)]
            return (cell_qubits, False)

    # 1. Sweep Row and Column pairs containing at least one variable
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

    # 2. Sweep Sub-Grid Block pairs using block step intervals of size n
    for block_r in range(0, grid_size, n):
        for block_c in range(0, grid_size, n):
            block_cells = [(block_r + i, block_c + j) for i in range(n) for j in range(n)]
            for i in range(grid_size):
                for j in range(i + 1, grid_size):
                    p1, p2 = block_cells[i], block_cells[j]
                    if puzzle_matrix[p1[0]][p1[1]] == 0 or puzzle_matrix[p2[0]][p2[1]] == 0:
                        pair = tuple(sorted([p1, p2]))
                        constraints.append(pair)

    # De-duplicate the structural constraints list
    constraints = list(set(constraints))
    num_constraints = len(constraints)

    # Construct the final scale-free register environment
    clause_reg = QuantumRegister(num_constraints, name='clause')
    xor_ancilla = QuantumRegister(k_bits, name='xor')  # Scales to track k bits
    phase_target = QuantumRegister(1, name='phase')
    c_reg = ClassicalRegister(num_vars * k_bits, name='meas')

    qc = QuantumCircuit(var_reg, clause_reg, xor_ancilla, phase_target, c_reg)

    # --- Initialization ---
    qc.h(var_reg)
    qc.x(phase_target)
    qc.h(phase_target)

    # Calculate exact geometric Grover rotation threshold
    search_space = 2 ** (num_vars * k_bits)
    iterations = int(np.floor(np.pi / 4 * np.sqrt(search_space)))
    if iterations < 1: iterations = 1

    # --- Grover Iteration Loop ---
    for _ in range(iterations):
        # A. Dynamic Forward Oracle
        for idx, (pos_A, pos_B) in enumerate(constraints):
            res_A = get_cell_logic(pos_A[0], pos_A[1])
            res_B = get_cell_logic(pos_B[0], pos_B[1])

            is_hint_A, is_hint_B = res_A[-1], res_B[-1]

            if not is_hint_A and not is_hint_B:
                cell_inequality_comparator(qc, res_A[0], res_B[0], clause_reg[idx], xor_ancilla, k_bits)
            elif not is_hint_A and is_hint_B:
                hint_comparator(qc, res_A[0], res_B[:-1], clause_reg[idx], k_bits)
            elif is_hint_A and not is_hint_B:
                hint_comparator(qc, res_B[0], res_A[:-1], clause_reg[idx], k_bits)

        # Global Conjunction State Evaluation
        qc.mcx(clause_reg, phase_target)

        # B. Dynamic Backward Oracle Uncomputation
        for idx, (pos_A, pos_B) in reversed(list(enumerate(constraints))):
            res_A = get_cell_logic(pos_A[0], pos_A[1])
            res_B = get_cell_logic(pos_B[0], pos_B[1])

            is_hint_A, is_hint_B = res_A[-1], res_B[-1]

            if not is_hint_A and not is_hint_B:
                cell_inequality_comparator(qc, res_A[0], res_B[0], clause_reg[idx], xor_ancilla, k_bits)
            elif not is_hint_A and is_hint_B:
                hint_comparator(qc, res_A[0], res_B[:-1], clause_reg[idx], k_bits)
            elif is_hint_A and not is_hint_B:
                hint_comparator(qc, res_B[0], res_A[:-1], clause_reg[idx], k_bits)

        # C. Diffuser Phase
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
    # Define a 4x4 puzzle matrix (n=2)
    test_4x4_puzzle = [
        [1, 0, 4, 3],
        [3, 4, 1, 2],  # Missing position (1,2) -> Should solve to 1
        [4, 3, 2, 1],  # Missing position (2,1) -> Should solve to 3
        [2, 1, 0, 4]
    ]

    # Track where the open positions are for interpretation later
    open_slots = [(r, c) for r in range(4) for c in range(4) if test_4x4_puzzle[r][c] == 0]
    num_vars = len(open_slots)

    print("Assembling generalized solver layout for a 4x4 configuration (n=2)...")
    # Call the scale-free factory function explicitly specifying n=2
    sudoku_circuit = build_generalized_sudoku_solver(test_4x4_puzzle, n=2)

    print("Simulating circuit execution on AerSimulator backend...")
    backend = AerSimulator()
    job = backend.run(sudoku_circuit, shots=1024)
    counts = job.result().get_counts()

    # Extract the winning bitstring configuration
    winning_bitstring = max(counts, key=counts.get)
    high_score = counts[winning_bitstring]

    print(f"\nRaw winning bitstring from simulator: {winning_bitstring} (found {high_score}/1024 times)")

    # --- SCALE-FREE LITTLE-ENDIAN INTERPRETATION ---
    # Reverse the bitstring so character index maps cleanly to sequential var_reg allocation
    ordered_bits = winning_bitstring[::-1]

    # Dynamic bit-width for n=2 is 2 bits per cell
    k_bits = int(np.ceil(np.log2(4)))

    print(f"Tracked Open Slots: {open_slots}")
    print("-" * 40)

    for idx in range(num_vars):
        # Slice out the exact k_bits chunk for this specific variable
        cell_bits = ordered_bits[idx * k_bits: (idx + 1) * k_bits]

        # Reconstruct the value from binary bits (ordered low-to-high)
        decimal_val = 0
        for bit_position, bit_char in enumerate(cell_bits):
            decimal_val |= (int(bit_char) << bit_position)

        sudoku_val = decimal_val + 1
        print(f"Solved Slot {open_slots[idx]} = {sudoku_val}")