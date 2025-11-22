"""
Generate test CSP problems: N-Queens, Sudoku, etc.
"""

import sys
from pathlib import Path
from typing import List, Tuple
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from benchmark.csp.csp_solver_livnium import CSPProblem


def generate_n_queens(n: int) -> CSPProblem:
    """
    Generate N-Queens problem.
    
    Variables: Q1, Q2, ..., QN (queen in each row)
    Domain: [0, 1, ..., N-1] (column positions)
    Constraints:
        - All different columns
        - No two queens on same diagonal
    """
    variables = {}
    for i in range(n):
        variables[f"Q{i+1}"] = list(range(n))
    
    constraints = []
    
    # All different columns
    constraints.append({
        'type': 'all_different',
        'vars': list(variables.keys())
    })
    
    # No two queens on same diagonal
    for i in range(n):
        for j in range(i + 1, n):
            qi = f"Q{i+1}"
            qj = f"Q{j+1}"
            
            # Diagonal constraint: |Qi - Qj| != |i - j|
            constraints.append({
                'type': 'diagonal',
                'vars': [qi, qj],
                'row_i': i,
                'row_j': j
            })
    
    return CSPProblem(variables, constraints)


def generate_sudoku_4x4() -> CSPProblem:
    """
    Generate a simple 4x4 Sudoku problem.
    
    Variables: C11, C12, ..., C44 (cells)
    Domain: [1, 2, 3, 4]
    Constraints:
        - All different in each row
        - All different in each column
        - All different in each 2x2 block
    """
    variables = {}
    for row in range(1, 5):
        for col in range(1, 5):
            variables[f"C{row}{col}"] = [1, 2, 3, 4]
    
    constraints = []
    
    # All different in each row
    for row in range(1, 5):
        row_vars = [f"C{row}{col}" for col in range(1, 5)]
        constraints.append({
            'type': 'all_different',
            'vars': row_vars
        })
    
    # All different in each column
    for col in range(1, 5):
        col_vars = [f"C{row}{col}" for row in range(1, 5)]
        constraints.append({
            'type': 'all_different',
            'vars': col_vars
        })
    
    # All different in each 2x2 block
    for block_row in [1, 3]:
        for block_col in [1, 3]:
            block_vars = []
            for r in range(block_row, block_row + 2):
                for c in range(block_col, block_col + 2):
                    block_vars.append(f"C{r}{c}")
            constraints.append({
                'type': 'all_different',
                'vars': block_vars
            })
    
    return CSPProblem(variables, constraints)


def generate_graph_coloring(num_vertices: int, num_colors: int, edges: List[Tuple[int, int]]) -> CSPProblem:
    """
    Generate graph coloring problem.
    
    Variables: V1, V2, ..., VN (vertices)
    Domain: [0, 1, ..., num_colors-1] (colors)
    Constraints: Adjacent vertices must have different colors
    """
    variables = {}
    for i in range(1, num_vertices + 1):
        variables[f"V{i}"] = list(range(num_colors))
    
    constraints = []
    for v1, v2 in edges:
        constraints.append({
            'type': 'not_equal',
            'vars': [f"V{v1}", f"V{v2}"]
        })
    
    return CSPProblem(variables, constraints)


def save_csp_problem(csp: CSPProblem, output_path: Path):
    """Save CSP problem to JSON file."""
    # Convert constraints to JSON-serializable format
    constraints_serializable = []
    for constraint in csp.constraints:
        constraint_copy = constraint.copy()
        # Remove function references (can't serialize)
        if 'fn' in constraint_copy and callable(constraint_copy['fn']):
            constraint_copy['fn'] = None  # Will need to be reconstructed
        constraints_serializable.append(constraint_copy)
    
    problem_data = {
        'variables': csp.variables,
        'constraints': constraints_serializable
    }
    
    with open(output_path, 'w') as f:
        json.dump(problem_data, f, indent=2)
    
    print(f"Saved: {output_path}")


def main():
    """Generate test CSP problems."""
    output_dir = Path(__file__).parent / "csplib" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating test CSP problems...")
    
    # N-Queens problems
    for n in [4, 5, 6, 8]:
        csp = generate_n_queens(n)
        output_path = output_dir / f"nqueens_{n}.json"
        save_csp_problem(csp, output_path)
    
    # Sudoku
    csp = generate_sudoku_4x4()
    output_path = output_dir / "sudoku_4x4.json"
    save_csp_problem(csp, output_path)
    
    # Graph coloring
    # Simple 3-vertex triangle
    csp = generate_graph_coloring(3, 2, [(1, 2), (2, 3), (3, 1)])
    output_path = output_dir / "graph_coloring_triangle.json"
    save_csp_problem(csp, output_path)
    
    print(f"\nGenerated test CSP problems in {output_dir}")
    print(f"You can now run: python benchmark/csp/run_csp_benchmark.py --csp-dir {output_dir}")


if __name__ == '__main__':
    main()

