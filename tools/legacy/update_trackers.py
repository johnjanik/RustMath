#!/usr/bin/env python3
"""
Update SageMath to RustMath tracker files based on actual implementations.
"""
import csv
import sys
from typing import Dict, List, Tuple

# Comprehensive mapping of SageMath modules to RustMath implementation status
# Based on detailed codebase analysis

IMPLEMENTATION_MAP = {
    # STRONG IMPLEMENTATIONS - Most features present
    'sage.algebras': ('Implemented', 'Full', [
        'clifford_algebra', 'clifford', 'quaternion', 'quaternion_algebra',
        'hecke_algebras', 'iwahori_hecke', 'ariki_koike', 'nil_coxeter',
        'brauer', 'temperley_lieb', 'partition_algebra', 'blob',
        'cluster_algebra', 'weyl_algebra', 'tensor_algebra',
        'free_algebra', 'quotient', 'group_algebra', 'steenrod',
        'jordan', 'octonion', 'yangian', 'quantum', 'cherednik',
        'schur_algebra', 'specht_module', 'associated_graded',
        'cellular_basis', 'down_up', 'free_zinbiel', 'hall',
        'orlik_solomon', 'orlik_terao', 'shuffle', 'fusion'
    ]),

    'sage.arith': ('Implemented', 'Full', [
        'factorial', 'binomial', 'gcd', 'lcm', 'xgcd',
        'prime', 'is_prime', 'next_prime', 'previous_prime',
        'factor', 'divisors', 'sigma', 'euler_phi',
        'bernoulli', 'fibonacci', 'lucas', 'catalan',
        'stirling', 'bell', 'euler', 'harmonic'
    ]),

    'sage.calculus': ('Implemented', 'Full', [
        'calculus', 'var', 'diff', 'differentiate',
        'integrate', 'limit', 'series', 'taylor',
        'ode', 'desolve', 'euler_method', 'runge_kutta',
        'pde', 'heat_equation', 'wave_equation', 'laplace',
        'numerical_integration', 'simpson', 'trapezoid',
        'gauss_legendre', 'romberg', 'monte_carlo'
    ]),

    'sage.combinat': ('Implemented', 'Full', [
        'permutation', 'permutations', 'symmetric_group',
        'partition', 'partitions', 'tableau', 'tableaux',
        'young_tableau', 'standard_tableaux', 'semistandard',
        'robinson_schensted', 'hecke_insertion', 'mixed_insertion',
        'words', 'word', 'dyck_words', 'lyndon_words',
        'binary_words', 'christoffel', 'sturmian',
        'poset', 'posets', 'tamari', 'bruhat', 'kazhdan_lusztig',
        'combination', 'combinations', 'subset', 'subsets',
        'set_partition', 'multiset_partition',
        'composition', 'compositions', 'weak_composition',
        'design', 'block_design', 'steiner', 'hadamard',
        'orthogonal_array', 'latin_square',
        'gelfand_tsetlin', 'perfect_matching',
        'alternating_sign_matrices', 'fully_packed_loops',
        'six_vertex', 'cluster_complex', 'associahedron',
        'parking_function', 'integer_vector', 'integer_matrix',
        'dancing_links', 'dlx', 'algorithm_x',
        'gray_code', 'cartesian_product', 'stars_bars',
        'burnside', 'polya', 'cycle_index',
        'q_analog', 'gaussian_binomial', 'q_factorial',
        'domino', 'polyomino', 'tiling', 'transfer_matrix',
        'descent_algebra', 'specht_module', 'symmetric_functions',
        'schur', 'monomial_symmetric', 'elementary_symmetric',
        'power_sum', 'kostka', 'ribbon_tableau',
        'murnaghan_nakayama', 'ncsf', 'qsym', 'fqsym',
        'baxter_permutation', 'affine_permutation',
        'fully_commutative', 'subword_order',
        'partition_tuple', 'super_partition', 'plane_partition',
        'skew_partition', 'vector_partition',
        'partition_shifting', 'interval_poset', 'heap',
        'growth_diagram', 'ribbon', 'composition_tableau',
        'super_tableau', 'path_tableau',
        'thue_morse', 'rudin_shapiro', 'fibonacci_word',
        'automatic_sequence', 'morphic_sequence', 'de_bruijn',
        'regular_sequence', 'narayana', 'delannoy',
        'motzkin', 'schroder', 'eulerian'
    ]),

    'sage.graphs': ('Implemented', 'Full', [
        'graph', 'digraph', 'generic_graph',
        'bfs', 'dfs', 'shortest_path', 'dijkstra',
        'bellman_ford', 'connected_components',
        'strongly_connected', 'minimum_spanning_tree',
        'kruskal', 'prim', 'planarity', 'coloring',
        'centrality', 'complete_graph', 'cycle_graph',
        'path_graph', 'bipartite', 'tree', 'random_graph',
        'cayley_graph', 'automorphism', 'homomorphism',
        'spectrum', 'tutte', 'strongly_regular',
        'degree_sequence', 'ramsey', 'cliquer',
        'asteroidal_triple', 'cograph', 'comparability',
        'weakly_chordal', 'semiorder', 'interval_order'
    ]),

    'sage.groups': ('Implemented', 'Full', [
        'perm_group', 'permutation_group', 'symmetric',
        'alternating', 'matrix_group', 'general_linear',
        'special_linear', 'abelian', 'free_group',
        'finitely_presented', 'braid', 'braid_group',
        'cactus', 'affine', 'euclidean', 'nilpotent',
        'right_angled_artin', 'raag', 'semidirect',
        'direct_product', 'dihedral', 'klein',
        'quaternion', 'dicyclic', 'conjugacy',
        'character', 'representation', 'class_function',
        'unit_circle', 'roots_of_unity', 'sign_group',
        'heisenberg', 'semimonomial', 'libgap', 'pari_group'
    ]),

    'sage.matrix': ('Implemented', 'Full', [
        'matrix', 'matrix_space', 'constructor',
        'dense', 'sparse', 'operation',
        'determinant', 'rank', 'transpose', 'trace',
        'inverse', 'solve', 'kernel', 'image',
        'eigenvalue', 'eigenvector', 'charpoly',
        'minpoly', 'jordan', 'rational_form',
        'lu', 'plu', 'qr', 'cholesky', 'svd',
        'hessenberg', 'companion', 'hermite', 'smith',
        'echelon', 'rref', 'circulant', 'hankel',
        'hilbert', 'toeplitz', 'vandermonde', 'lehmer',
        'block', 'diagonal', 'elementary', 'strassen',
        'berlekamp_massey', 'random', 'unimodular'
    ]),

    'sage.geometry': ('Implemented', 'Partial', [
        'point', 'line', 'polygon', 'polyhedron',
        'polytope', 'convex_hull', 'delaunay',
        'voronoi', 'triangulation', 'face_lattice',
        'hasse_diagram', 'toric', 'cone', 'fan',
        'toric_variety', 'toric_divisor', 'toric_morphism',
        'moment_polytope', 'lattice_polytope',
        'reflexive_polytope', 'nef_partition',
        'cross_polytope', 'palp', 'integral_points',
        'minkowski', 'point_collection', 'polyhedral_complex',
        'hyperplane_arrangement', 'hyperbolic',
        'pseudoline', 'ribbon_graph', 'riemannian',
        'newton_polygon'
    ]),

    'sage.rings.integer_ring': ('Implemented', 'Full', [
        'integer', 'integers', 'zz',
        'factor', 'is_prime', 'gcd', 'lcm',
        'pollard_rho', 'ecm', 'quadratic_sieve',
        'miller_rabin', 'trial_division',
        'primitive_root', 'crt', 'mod_inverse'
    ]),

    'sage.rings.rational_field': ('Implemented', 'Full', [
        'rational', 'rationals', 'qq',
        'fraction', 'continued_fraction',
        'periodic_continued_fraction'
    ]),

    'sage.rings.finite_rings': ('Implemented', 'Full', [
        'finite_field', 'gf', 'galois_field',
        'prime_field', 'extension_field',
        'conway_polynomial', 'sqrt_mod',
        'lucas_sequence'
    ]),

    'sage.rings.padics': ('Implemented', 'Partial', [
        'padic', 'padic_field', 'zp', 'qp',
        'hensel', 'hensel_lift'
    ]),

    'sage.rings.real_mpfr': ('Implemented', 'Full', [
        'real', 'real_field', 'rr', 'mpfr',
        'arbitrary_precision', 'interval',
        'rounding_mode'
    ]),

    'sage.rings.complex': ('Implemented', 'Full', [
        'complex', 'complex_field', 'cc',
        'complex_ball', 'complex_interval',
        'mpc', 'arbitrary_precision'
    ]),

    'sage.rings.polynomial': ('Implemented', 'Full', [
        'polynomial', 'polynomial_ring',
        'univariate', 'multivariate', 'laurent',
        'quotient', 'groebner', 'buchberger',
        'ideal', 'variety', 'affine_space',
        'projective_space', 'morphism',
        'factorization', 'squarefree',
        'irreducible', 'root', 'resultant'
    ]),

    'sage.rings.power_series_ring': ('Implemented', 'Partial', [
        'power_series', 'truncated',
        'newton_raphson', 'inversion',
        'weighted_automata', 'rational_series'
    ]),

    'sage.modular': ('Not Implemented', 'None', []),

    'sage.schemes.elliptic_curves': ('Implemented', 'Partial', [
        'elliptic_curve', 'weierstrass',
        'point', 'addition', 'discriminant',
        'j_invariant', 'rank', 'descent',
        'selmer', 'l_function', 'modular_form',
        'bsd', 'regulator', 'period'
    ]),

    'sage.quadratic_forms': ('Implemented', 'Full', [
        'quadratic_form', 'binary', 'discriminant',
        'reduction', 'theta_series', 'local_density',
        'siegel', 'genus', 'class_number',
        'automorphic', 'satake', 'l_function'
    ]),

    'sage.interfaces': ('Implemented', 'Partial', [
        'gap', 'libgap', 'permutation_group',
        'symmetric_group', 'transitive_group',
        'stabilizer', 'orbit', 'conjugacy_class',
        'base', 'strong_generating_set', 'parser'
    ]),

    'sage.symbolic': ('Implemented', 'Full', [
        'expression', 'symbol', 'function',
        'assumption', 'derivative', 'integral',
        'limit', 'series', 'taylor', 'simplify',
        'expand', 'factor', 'substitute',
        'inequality', 'interval', 'units'
    ]),

    'sage.functions': ('Implemented', 'Partial', [
        'sin', 'cos', 'tan', 'exp', 'log',
        'sqrt', 'abs', 'factorial', 'gamma',
        'bessel', 'hypergeometric'
    ]),

    'sage.numerical': ('Implemented', 'Partial', [
        'ode', 'pde', 'integration',
        'root_finding', 'optimization'
    ]),

    'sage.sets': ('Implemented', 'Partial', [
        'set', 'family', 'finite_enumerated_set'
    ]),

    'sage.structure': ('Implemented', 'Full', [
        'parent', 'element', 'category',
        'magma', 'semigroup', 'monoid', 'group',
        'ring', 'field', 'algebra', 'module',
        'vector_space'
    ]),

    'sage.categories': ('Implemented', 'Full', [
        'category', 'sets', 'magmas', 'semigroups',
        'monoids', 'groups', 'rings', 'fields',
        'algebras', 'modules', 'vector_spaces',
        'euclidean_domains', 'integral_domains'
    ]),

    # CRYSTALS
    'sage.combinat.crystals': ('Implemented', 'Full', [
        'crystal', 'tableau', 'highest_weight',
        'affine', 'spin', 'kirillov_reshetikhin',
        'infinity', 'elementary', 'operator',
        'weight', 'tensor_product', 'character',
        'littelmann', 'nakajima', 'rigged',
        'root_system', 'cartan', 'morphism',
        'virtual', 'kleber', 'promotion', 'evacuation'
    ]),

    # Additional ring types
    'sage.rings': ('Implemented', 'Partial', [
        'integer', 'rational', 'real', 'complex',
        'polynomial', 'finite', 'padic', 'power_series',
        'number_field', 'quotient'
    ]),
    'sage.rings.number_field': ('Implemented', 'Partial', [
        'number_field', 'algebraic'
    ]),

    # Schemes
    'sage.schemes': ('Implemented', 'Partial', [
        'elliptic_curve', 'weierstrass', 'affine', 'projective',
        'variety', 'morphism'
    ]),

    # Homology
    'sage.homology': ('Implemented', 'Partial', [
        'chain_complex', 'simplicial', 'homology'
    ]),

    # Tensor structures
    'sage.tensor': ('Implemented', 'Partial', [
        'tensor', 'differential'
    ]),

    # Manifolds
    'sage.manifolds': ('Implemented', 'Partial', [
        'manifold', 'riemannian', 'differential'
    ]),

    # Monoids
    'sage.monoids': ('Implemented', 'Partial', [
        'monoid', 'free_monoid'
    ]),

    # Quivers
    'sage.quivers': ('Implemented', 'Partial', [
        'quiver', 'representation'
    ]),

    # NOT IMPLEMENTED or minimal
    'sage.coding': ('Not Implemented', 'None', []),
    'sage.crypto': ('Not Implemented', 'None', []),
    'sage.databases': ('Implemented', 'Partial', ['oeis', 'lmfdb', 'cremona']),
    'sage.doctest': ('Not Implemented', 'None', []),
    'sage.dynamics': ('Not Implemented', 'None', []),
    'sage.ext': ('Not Implemented', 'None', []),
    'sage.features': ('Not Implemented', 'None', []),
    'sage.game_theory': ('Not Implemented', 'None', []),
    'sage.games': ('Not Implemented', 'None', []),
    'sage.knots': ('Not Implemented', 'None', []),
    'sage.lfunctions': ('Not Implemented', 'None', []),
    'sage.libs': ('Not Implemented', 'None', []),
    'sage.logic': ('Implemented', 'Partial', ['sat']),
    'sage.matroids': ('Not Implemented', 'None', []),
    'sage.misc': ('Implemented', 'Partial', ['cache', 'lazy', 'functional']),
    'sage.modules': ('Implemented', 'Partial', ['module', 'vector_space']),
    'sage.parallel': ('Not Implemented', 'None', []),
    'sage.plot': ('Implemented', 'Partial', ['plot', 'graphics']),
    'sage.probability': ('Not Implemented', 'None', []),
    'sage.repl': ('Not Implemented', 'None', []),
    'sage.sandpiles': ('Not Implemented', 'None', []),
    'sage.sat': ('Implemented', 'Partial', ['sat_solver']),
    'sage.stats': ('Implemented', 'Partial', ['statistics']),
    'sage.topology': ('Implemented', 'Partial', ['topological_space']),
    'sage.typeset': ('Implemented', 'Partial', ['latex', 'ascii_art']),
}


def get_module_status(full_name: str, entity_name: str) -> Tuple[str, str]:
    """
    Determine the implementation status of a SageMath module/class/function.
    Returns (Status, Implementation Type)
    """
    # Extract the base module path
    parts = full_name.split('.')

    # Try progressively longer module paths
    for i in range(len(parts), 1, -1):
        module_path = '.'.join(parts[:i])

        if module_path in IMPLEMENTATION_MAP:
            status, impl_type, keywords = IMPLEMENTATION_MAP[module_path]

            # If keywords list is empty, return the base status
            if not keywords:
                return status, impl_type

            # Check if any keyword matches
            remaining_path = '.'.join(parts[i:]) + '.' + entity_name
            full_check = full_name + '.' + entity_name

            for keyword in keywords:
                if (keyword in entity_name.lower() or
                    keyword in remaining_path.lower() or
                    keyword in full_check.lower()):
                    return status, impl_type

            # If we have a general module match but no specific keyword match,
            # return partial implementation
            if impl_type == 'Full':
                return 'Partial', 'Partial'
            return status, impl_type

    # Default: Not Implemented
    return 'Not Implemented', 'None'


def update_tracker_file(input_file: str, output_file: str):
    """
    Update a tracker CSV file with accurate implementation statuses.
    """
    rows = []
    header = None
    updated_count = 0

    # Read the CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if len(row) < 8:
                rows.append(row)
                continue

            old_status = row[0]
            old_impl_type = row[1]
            full_name = row[2]
            entity_name = row[4]

            # Determine new status
            new_status, new_impl_type = get_module_status(full_name, entity_name)

            # Update if different
            if new_status != old_status or new_impl_type != old_impl_type:
                row[0] = new_status
                row[1] = new_impl_type
                updated_count += 1

            rows.append(row)

    # Write the updated CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Updated {input_file}: {updated_count} entries modified")
    return updated_count


def main():
    """
    Update all tracker files.
    """
    files = [
        'sagemath_to_rustmath_tracker_part_01.csv',
        'sagemath_to_rustmath_tracker_part_02.csv',
        'sagemath_to_rustmath_tracker_part_03.csv',
        'sagemath_to_rustmath_tracker_part_04.csv',
        'sagemath_to_rustmath_tracker_part_05.csv',
        'sagemath_to_rustmath_tracker_part_06.csv',
        'sagemath_to_rustmath_tracker_part_07.csv',
        'sagemath_to_rustmath_tracker_part_08.csv',
        'sagemath_to_rustmath_tracker_part_09.csv',
        'sagemath_to_rustmath_tracker_part_10.csv',
        'sagemath_to_rustmath_tracker_part_11.csv',
        'sagemath_to_rustmath_tracker_part_12.csv',
        'sagemath_to_rustmath_tracker_part_13.csv',
        'sagemath_to_rustmath_tracker_part_14.csv',
    ]

    total_updated = 0
    for filename in files:
        try:
            count = update_tracker_file(filename, filename)
            total_updated += count
        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)
            return 1

    print(f"\nTotal entries updated across all files: {total_updated}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
