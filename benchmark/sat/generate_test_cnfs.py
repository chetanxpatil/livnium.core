"""
Generate simple test CNF files for benchmarking when SATLIB download fails.
"""

from pathlib import Path
import random


def generate_random_3sat(num_vars: int, num_clauses: int, output_dir: Path):
    """
    Generate a random 3-SAT CNF file.
    
    Args:
        num_vars: Number of variables
        num_clauses: Number of clauses
        output_dir: Directory to save CNF files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cnf_content = f"c Random 3-SAT: {num_vars} variables, {num_clauses} clauses\n"
    cnf_content += f"p cnf {num_vars} {num_clauses}\n"
    
    for _ in range(num_clauses):
        # Generate 3 random literals
        literals = []
        for _ in range(3):
            var = random.randint(1, num_vars)
            negate = random.choice([True, False])
            literal = -var if negate else var
            literals.append(literal)
        
        # Write clause
        cnf_content += " ".join(str(l) for l in literals) + " 0\n"
    
    # Save file
    output_file = output_dir / f"test_{num_vars}v_{num_clauses}c.cnf"
    with open(output_file, 'w') as f:
        f.write(cnf_content)
    
    print(f"Generated: {output_file}")
    return output_file


def generate_simple_satisfiable():
    """Generate a simple satisfiable CNF for testing."""
    cnf_content = """c Simple satisfiable 3-SAT
p cnf 3 3
1 2 3 0
-1 2 -3 0
1 -2 3 0
"""
    output_dir = Path(__file__).parent / "satlib" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "simple_sat.cnf"
    with open(output_file, 'w') as f:
        f.write(cnf_content)
    
    print(f"Generated: {output_file}")
    return output_file


def main():
    """Generate a set of test CNF files."""
    output_dir = Path(__file__).parent / "satlib" / "test"
    
    print("Generating test CNF files...")
    
    # Simple satisfiable
    generate_simple_satisfiable()
    
    # Small random 3-SAT instances
    for num_vars in [10, 20, 30]:
        for num_clauses in [num_vars * 3, num_vars * 4, num_vars * 5]:
            generate_random_3sat(num_vars, num_clauses, output_dir)
    
    print(f"\nGenerated test CNF files in {output_dir}")
    print(f"You can now run: python benchmark/sat/run_sat_benchmark.py --cnf-dir {output_dir}")


if __name__ == '__main__':
    main()

