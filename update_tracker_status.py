#!/usr/bin/env python3
"""
Update SageMath to RustMath tracker CSV files with current implementation status.

This script reads each tracker CSV file and updates the Status column based on
what functionality has been implemented in RustMath.
"""

import csv
import os
from typing import List, Tuple

# Define mapping rules from SageMath modules to RustMath implementation status
# Format: (module_pattern, status, comment)
IMPLEMENTATION_RULES = [
    # Fully Implemented modules
    ("sage.rings.integer_ring", "Implemented", "rustmath-integers"),
    ("sage.rings.integer.", "Implemented", "rustmath-integers"),
    ("sage.rings.rational_field", "Implemented", "rustmath-rationals"),
    ("sage.rings.rational.", "Implemented", "rustmath-rationals"),
    ("sage.rings.real_mpfr", "Implemented", "rustmath-reals (MPFR)"),
    ("sage.rings.real_double", "Implemented", "rustmath-reals"),
    ("sage.rings.real_lazy", "Partial", "rustmath-reals (f64 only)"),
    ("sage.rings.complex_mpfr", "Implemented", "rustmath-complex"),
    ("sage.rings.complex_double", "Implemented", "rustmath-complex"),
    ("sage.rings.complex_field", "Implemented", "rustmath-complex"),
    ("sage.rings.padic", "Implemented", "rustmath-padics"),
    ("sage.rings.padics", "Implemented", "rustmath-padics"),
    ("sage.rings.power_series_ring", "Implemented", "rustmath-powerseries"),
    ("sage.rings.finite_rings.finite_field", "Implemented", "rustmath-finitefields"),
    ("sage.rings.finite_rings.integer_mod_ring", "Implemented", "rustmath-integers (modular)"),

    # Matrix operations
    ("sage.matrix.matrix", "Implemented", "rustmath-matrix"),
    ("sage.matrix.constructor", "Implemented", "rustmath-matrix"),
    ("sage.modules.free_module", "Implemented", "rustmath-matrix (vector spaces)"),
    ("sage.modules.vector", "Implemented", "rustmath-matrix (vectors)"),

    # Combinatorics - Fully implemented
    ("sage.combinat.permutation", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.partition", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.combination", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.tableau", "Implemented", "rustmath-combinatorics (Young tableaux)"),
    ("sage.combinat.posets", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.composition", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.set_partition", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.dyck_word", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.perfect_matching", "Implemented", "rustmath-combinatorics"),
    ("sage.combinat.latin_square", "Implemented", "rustmath-combinatorics"),

    # Graphs
    ("sage.graphs.graph.", "Implemented", "rustmath-graphs"),
    ("sage.graphs.digraph", "Implemented", "rustmath-graphs"),
    ("sage.graphs.bipartite_graph", "Implemented", "rustmath-graphs"),

    # Geometry
    ("sage.geometry.polyhedron", "Implemented", "rustmath-geometry"),
    ("sage.geometry.polytope", "Implemented", "rustmath-geometry"),
    ("sage.geometry.triangulation", "Implemented", "rustmath-geometry"),
    ("sage.geometry.toric", "Implemented", "rustmath-geometry"),
    ("sage.geometry.cone", "Implemented", "rustmath-geometry"),
    ("sage.geometry.fan", "Implemented", "rustmath-geometry"),
    ("sage.geometry.voronoi", "Implemented", "rustmath-geometry"),

    # Quadratic forms
    ("sage.quadratic_forms", "Implemented", "rustmath-quadraticforms"),

    # Groups
    ("sage.groups.perm_gps", "Implemented", "rustmath-groups"),
    ("sage.groups.matrix_gps", "Implemented", "rustmath-groups"),
    ("sage.groups.abelian", "Implemented", "rustmath-groups"),

    # Partially implemented modules
    ("sage.symbolic", "Partial", "rustmath-symbolic (expression trees, diff, integration, limits, series)"),
    ("sage.calculus", "Partial", "rustmath-calculus + rustmath-symbolic"),
    ("sage.rings.polynomial", "Partial", "rustmath-polynomials (basic operations)"),
    ("sage.functions.trig", "Partial", "rustmath-functions"),
    ("sage.functions.hyperbolic", "Partial", "rustmath-functions"),
    ("sage.functions.log", "Partial", "rustmath-functions"),
    ("sage.functions.exp", "Partial", "rustmath-functions"),
    ("sage.functions.other", "Partial", "rustmath-functions"),
    ("sage.arith", "Partial", "rustmath-integers (factorial, binomial, gcd)"),
    ("sage.categories.rings", "Partial", "rustmath-core"),
    ("sage.categories.fields", "Partial", "rustmath-core"),
    ("sage.categories.groups", "Partial", "rustmath-core"),
    ("sage.categories.modules", "Partial", "rustmath-core"),
]

def should_mark_implemented(module: str, entity_name: str, entity_type: str) -> Tuple[str, str]:
    """
    Determine if a SageMath entity should be marked as implemented in RustMath.

    Returns:
        Tuple of (status, reason) where status is "", "Partial", or "Implemented"
    """
    full_name = f"{module}.{entity_name}".lower()
    module_lower = module.lower()

    # Check each rule
    for pattern, status, comment in IMPLEMENTATION_RULES:
        if pattern.lower() in module_lower or pattern.lower() in full_name:
            return (status, comment)

    return ("", "")

def update_csv_file(input_path: str, output_path: str) -> Tuple[int, int, int]:
    """
    Update a single CSV file with implementation status.

    Returns:
        Tuple of (total_rows, implemented_count, partial_count)
    """
    rows = []
    implemented_count = 0
    partial_count = 0
    total_rows = 0

    # Read the CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            total_rows += 1
            module = row.get('module', '')
            entity_name = row.get('entity_name', '')
            entity_type = row.get('type', '')

            # Determine status
            status, reason = should_mark_implemented(module, entity_name, entity_type)

            if status == "Implemented":
                row['Status'] = "Implemented"
                implemented_count += 1
            elif status == "Partial":
                row['Status'] = "Partial"
                partial_count += 1
            else:
                row['Status'] = ""

            rows.append(row)

    # Write the updated CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return (total_rows, implemented_count, partial_count)

def main():
    """Update all tracker CSV files."""
    base_dir = "/home/user/RustMath"

    # Process all 14 parts
    total_overall = 0
    implemented_overall = 0
    partial_overall = 0

    for i in range(1, 15):
        part_num = f"{i:02d}"
        input_file = os.path.join(base_dir, f"sagemath_to_rustmath_tracker_part_{part_num}.csv")
        output_file = input_file  # Overwrite the original file

        if os.path.exists(input_file):
            print(f"Processing part {part_num}...", end=" ")
            total, implemented, partial = update_csv_file(input_file, output_file)
            total_overall += total
            implemented_overall += implemented
            partial_overall += partial
            print(f"Done. ({total} rows, {implemented} implemented, {partial} partial)")
        else:
            print(f"Warning: {input_file} not found")

    print(f"\n{'='*60}")
    print(f"Overall Summary:")
    print(f"  Total entries: {total_overall}")
    print(f"  Implemented: {implemented_overall} ({100*implemented_overall/total_overall:.1f}%)")
    print(f"  Partial: {partial_overall} ({100*partial_overall/total_overall:.1f}%)")
    print(f"  Not implemented: {total_overall - implemented_overall - partial_overall} ({100*(total_overall - implemented_overall - partial_overall)/total_overall:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
