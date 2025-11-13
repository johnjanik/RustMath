#!/usr/bin/env python3
"""
Analyze the SageMath to RustMath tracker to identify critical core modules for MVP.
"""

import csv
import os
from collections import defaultdict, Counter
from pathlib import Path

def analyze_tracker_parts():
    """Analyze all tracker parts and generate comprehensive statistics."""

    # Data structures for analysis
    module_stats = defaultdict(lambda: defaultdict(int))
    top_level_domains = Counter()
    second_level_breakdown = defaultdict(Counter)
    all_entries = []

    # Read all tracker parts
    for i in range(1, 15):
        filename = f"sagemath_to_rustmath_tracker_part_{i:02d}.csv"
        if not os.path.exists(filename):
            continue

        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_entries.append(row)

                # Extract module hierarchy
                full_name = row['full_name']
                entity_type = row['type']

                if not full_name or full_name.startswith('sage.') is False:
                    continue

                # Parse module hierarchy
                parts = full_name.split('.')
                if len(parts) < 2:
                    continue

                # Top level domain (e.g., sage.rings, sage.algebras)
                if len(parts) >= 2:
                    top_level = f"{parts[0]}.{parts[1]}"
                    top_level_domains[top_level] += 1

                    # Track by type
                    if entity_type:
                        module_stats[top_level][entity_type] += 1
                    module_stats[top_level]["total"] += 1

                # Second level breakdown
                if len(parts) >= 3:
                    second_level = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    second_level_breakdown[top_level][second_level] += 1

    return module_stats, top_level_domains, second_level_breakdown, all_entries


def identify_core_domains(module_stats, top_level_domains):
    """Identify the most critical core domains for MVP."""

    # Core mathematical domains that are fundamental for any CAS
    core_categories = {
        "Foundation": [
            "sage.structure",      # Base classes, parents, elements
            "sage.categories",     # Category framework
            "sage.cpython",        # Python integration
        ],
        "Number Systems": [
            "sage.rings",          # All ring structures (integers, rationals, reals, complex, finite fields, polynomials)
            "sage.arith",          # Arithmetic functions
        ],
        "Linear Algebra": [
            "sage.matrix",         # Matrix operations
            "sage.modules",        # Modules and vector spaces
        ],
        "Symbolic": [
            "sage.symbolic",       # Symbolic expressions and calculus
            "sage.calculus",       # Calculus operations
        ],
        "Polynomials": [
            "sage.rings.polynomial",  # Will be in sage.rings
        ],
        "Core Utilities": [
            "sage.misc",           # Utility functions
            "sage.functions",      # Mathematical functions
        ]
    }

    # Advanced/specialized domains (lower priority for MVP)
    advanced_categories = {
        "Algebra": ["sage.algebras", "sage.groups", "sage.monoids"],
        "Geometry": ["sage.geometry", "sage.schemes"],
        "Combinatorics": ["sage.combinat", "sage.graphs"],
        "Number Theory": ["sage.modular", "sage.quadratic_forms"],
        "Advanced Analysis": ["sage.manifolds", "sage.dynamics"],
        "Cryptography": ["sage.crypto"],
        "Coding Theory": ["sage.coding"],
        "Game Theory": ["sage.game_theory"],
        "Databases": ["sage.databases"],
    }

    return core_categories, advanced_categories


def calculate_priorities(module_stats, top_level_domains):
    """Calculate priority scores for each module based on various factors."""

    priorities = {}

    for module, count in top_level_domains.items():
        score = 0
        stats = module_stats[module]

        # Factor 1: Size (number of entities) - larger = more comprehensive
        size_score = min(count / 100, 10)  # Cap at 10 points

        # Factor 2: Balance of classes and functions (sign of mature API)
        if stats.get('class', 0) > 0 and stats.get('function', 0) > 0:
            balance_score = 5
        else:
            balance_score = 0

        # Factor 3: Core domain bonus
        core_bonus = 0
        if any(module.startswith(core) for cores in [
            ["sage.rings", "sage.matrix", "sage.symbolic", "sage.structure",
             "sage.categories", "sage.modules", "sage.arith", "sage.functions"]
        ] for core in cores):
            core_bonus = 20

        # Factor 4: Dependency indicator (structure and categories are foundational)
        if module in ["sage.structure", "sage.categories"]:
            core_bonus += 15

        score = size_score + balance_score + core_bonus
        priorities[module] = {
            "score": score,
            "count": count,
            "stats": stats
        }

    return priorities


def print_analysis(module_stats, top_level_domains, second_level_breakdown, priorities):
    """Print comprehensive analysis report."""

    print("=" * 80)
    print("SAGEMATH TO RUSTMATH TRACKER ANALYSIS")
    print("=" * 80)
    print()

    # Overall statistics
    total_entries = sum(top_level_domains.values())
    print(f"Total Entries: {total_entries:,}")
    print(f"Top-Level Domains: {len(top_level_domains)}")
    print()

    # Top domains by size
    print("=" * 80)
    print("TOP 20 DOMAINS BY SIZE")
    print("=" * 80)
    sorted_domains = sorted(top_level_domains.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Domain':<40} {'Count':>8} {'Classes':>8} {'Functions':>8} {'Modules':>8}")
    print("-" * 80)
    for domain, count in sorted_domains[:20]:
        stats = module_stats[domain]
        print(f"{domain:<40} {count:>8,} {stats.get('class', 0):>8,} {stats.get('function', 0):>8,} {stats.get('module', 0):>8,}")
    print()

    # Priority ranking for MVP
    print("=" * 80)
    print("PRIORITY RANKING FOR MVP (Top 15)")
    print("=" * 80)
    sorted_priorities = sorted(priorities.items(), key=lambda x: x[1]['score'], reverse=True)
    print(f"{'Rank':<6} {'Domain':<40} {'Score':>8} {'Entities':>10}")
    print("-" * 80)
    for rank, (domain, data) in enumerate(sorted_priorities[:15], 1):
        print(f"{rank:<6} {domain:<40} {data['score']:>8.1f} {data['count']:>10,}")
    print()

    # Core domains breakdown
    print("=" * 80)
    print("CORE DOMAINS DETAILED BREAKDOWN")
    print("=" * 80)

    core_domains = ["sage.rings", "sage.matrix", "sage.symbolic", "sage.structure",
                   "sage.categories", "sage.modules", "sage.functions"]

    for domain in core_domains:
        if domain in top_level_domains:
            print(f"\n{domain.upper()}")
            print("-" * 40)
            print(f"Total entities: {top_level_domains[domain]:,}")
            stats = module_stats[domain]
            print(f"  Classes:   {stats.get('class', 0):>6,}")
            print(f"  Functions: {stats.get('function', 0):>6,}")
            print(f"  Modules:   {stats.get('module', 0):>6,}")

            # Show top second-level modules
            if domain in second_level_breakdown:
                second_level = second_level_breakdown[domain]
                sorted_second = sorted(second_level.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top sub-modules:")
                for submod, count in sorted_second:
                    print(f"    {submod}: {count:,}")

    print()


def generate_mvp_recommendation():
    """Generate MVP recommendation based on analysis."""

    print("=" * 80)
    print("MVP RECOMMENDATION FOR RUSTMATH")
    print("=" * 80)
    print()

    print("PHASE 1: FOUNDATION (Highest Priority)")
    print("-" * 80)
    print("These are CRITICAL and must be implemented first:")
    print()
    print("1. sage.structure - Base algebraic structures (Ring, Field, etc.)")
    print("   → Maps to: rustmath-core (ALREADY DONE)")
    print()
    print("2. sage.rings.integer - Integer arithmetic")
    print("   → Maps to: rustmath-integers (ALREADY DONE)")
    print()
    print("3. sage.rings.rational - Rational numbers")
    print("   → Maps to: rustmath-rationals (ALREADY DONE)")
    print()
    print("4. sage.rings.polynomial - Polynomial rings")
    print("   → Maps to: rustmath-polynomials (ALREADY DONE)")
    print()
    print("5. sage.matrix - Matrix operations")
    print("   → Maps to: rustmath-matrix (ALREADY DONE)")
    print()

    print("PHASE 2: CORE MATHEMATICAL OPERATIONS (High Priority)")
    print("-" * 80)
    print("Essential for basic mathematical computing:")
    print()
    print("6. sage.symbolic - Symbolic expressions and calculus")
    print("   → Maps to: rustmath-symbolic (PARTIALLY DONE)")
    print()
    print("7. sage.rings.real_mpfr / real_double - Real number arithmetic")
    print("   → Maps to: rustmath-reals (BASIC DONE, needs arbitrary precision)")
    print()
    print("8. sage.rings.complex - Complex numbers")
    print("   → Maps to: rustmath-complex (ALREADY DONE)")
    print()
    print("9. sage.functions - Elementary functions (sin, cos, exp, log, etc.)")
    print("   → NEEDS IMPLEMENTATION")
    print()
    print("10. sage.calculus - Differentiation, integration, limits")
    print("    → Differentiation in rustmath-symbolic, integration MISSING")
    print()

    print("PHASE 3: SPECIALIZED STRUCTURES (Medium Priority)")
    print("-" * 80)
    print("Important for comprehensive testing:")
    print()
    print("11. sage.rings.finite_rings - Finite fields GF(p), GF(p^n)")
    print("    → Maps to: rustmath-finitefields (ALREADY DONE)")
    print()
    print("12. sage.rings.padics - p-adic numbers")
    print("    → Maps to: rustmath-padics (ALREADY DONE)")
    print()
    print("13. sage.modules - Modules and vector spaces")
    print("    → NEEDS IMPLEMENTATION")
    print()
    print("14. sage.combinat - Combinatorics (permutations, partitions)")
    print("    → Maps to: rustmath-combinatorics (ALREADY DONE)")
    print()

    print("BENCHMARKING TARGETS FOR MVP")
    print("-" * 80)
    print("To prove viability, RustMath should benchmark against SageMath on:")
    print()
    print("1. Integer operations:")
    print("   - Large integer arithmetic (addition, multiplication, division)")
    print("   - Primality testing (Miller-Rabin)")
    print("   - Integer factorization (Pollard's Rho)")
    print("   - GCD computation")
    print()
    print("2. Matrix operations:")
    print("   - Matrix multiplication (various sizes)")
    print("   - Determinant calculation")
    print("   - Matrix inversion")
    print("   - LU/PLU decomposition")
    print()
    print("3. Polynomial operations:")
    print("   - Polynomial multiplication")
    print("   - Polynomial division with remainder")
    print("   - GCD of polynomials")
    print("   - Factorization")
    print()
    print("4. Symbolic differentiation:")
    print("   - Expression tree construction")
    print("   - Differentiation of complex expressions")
    print("   - Simplification")
    print()
    print("5. Finite field arithmetic:")
    print("   - Operations in GF(p) and GF(p^n)")
    print("   - Polynomial arithmetic over finite fields")
    print()

    print("GAPS TO FILL FOR MVP")
    print("-" * 80)
    print("Critical missing pieces:")
    print()
    print("1. Expression parsing - Cannot parse strings like 'x^2 + 3*x + 2'")
    print("2. Arbitrary precision reals - Currently using f64")
    print("3. Elementary functions - sin, cos, exp, log, etc.")
    print("4. Symbolic integration - Only differentiation works")
    print("5. Better simplification - Current simplification is basic")
    print("6. Vector spaces and modules - Not implemented")
    print()


if __name__ == "__main__":
    print("Analyzing SageMath to RustMath tracker...")
    print()

    module_stats, top_level_domains, second_level_breakdown, all_entries = analyze_tracker_parts()
    priorities = calculate_priorities(module_stats, top_level_domains)

    print_analysis(module_stats, top_level_domains, second_level_breakdown, priorities)
    generate_mvp_recommendation()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)
