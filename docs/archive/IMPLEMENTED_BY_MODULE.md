# Implemented Modules Grouped by SageMath Top-Level Module

Total implemented entries: **4262**

Number of top-level modules with implementations: **28**

## Summary Table

| Top-Level Module | Count |
|------------------|-------|
| sage.rings | 952 |
| sage.modular | 571 |
| sage.combinat | 373 |
| sage.schemes | 271 |
| sage.modules | 265 |
| sage.geometry | 208 |
| sage.manifolds | 202 |
| sage.groups | 201 |
| sage.coding | 192 |
| sage.symbolic | 120 |
| sage.matrix | 116 |
| sage.arith | 102 |
| sage.misc | 95 |
| sage.numerical | 81 |
| sage.quadratic_forms | 75 |
| sage.functions | 70 |
| sage.monoids | 54 |
| sage.calculus | 52 |
| sage.dynamics | 51 |
| sage.homology | 51 |
| sage.stats | 43 |
| sage.logic | 40 |
| sage.databases | 22 |
| sage.crypto | 16 |
| sage.categories | 15 |
| sage.probability | 15 |
| sage.graphs | 8 |
| sage.libs | 1 |

## Detailed Breakdown

### sage.rings (952 entries)

**By Type:**
- attribute: 9
- class: 439
- function: 365
- module: 139

**By Part:**
- Part 11: 39
- Part 12: 741
- Part 13: 172

**Sub-modules:**
- sage.rings.complex_double: 7 entries
- sage.rings.complex_mpfr: 11 entries
- sage.rings.finite_rings.finite_field_base: 5 entries
- sage.rings.finite_rings.finite_field_constructor: 3 entries
- sage.rings.finite_rings.finite_field_givaro: 2 entries
- sage.rings.finite_rings.finite_field_ntl_gf2e: 3 entries
- sage.rings.finite_rings.finite_field_pari_ffelt: 2 entries
- sage.rings.finite_rings.finite_field_prime_modn: 2 entries
- sage.rings.finite_rings.integer_mod_ring: 4 entries
- sage.rings.ideal: 10 entries
- sage.rings.ideal_monoid: 3 entries
- sage.rings.integer: 8 entries
- sage.rings.integer_ring: 5 entries
- sage.rings.number_field.S_unit_solver: 42 entries
- sage.rings.number_field.bdd_height: 6 entries
- sage.rings.number_field.class_group: 5 entries
- sage.rings.number_field.galois_group: 5 entries
- sage.rings.number_field.homset: 4 entries
- sage.rings.number_field.maps: 11 entries
- sage.rings.number_field.morphism: 4 entries
- sage.rings.number_field.number_field: 22 entries
- sage.rings.number_field.number_field_base: 3 entries
- sage.rings.number_field.number_field_element: 8 entries
- sage.rings.number_field.number_field_element_quadratic: 8 entries
- sage.rings.number_field.number_field_ideal: 9 entries
- sage.rings.number_field.number_field_ideal_rel: 3 entries
- sage.rings.number_field.number_field_morphisms: 11 entries
- sage.rings.number_field.number_field_rel: 5 entries
- sage.rings.number_field.order: 16 entries
- sage.rings.number_field.order_ideal: 4 entries
- sage.rings.number_field.selmer_group: 4 entries
- sage.rings.number_field.small_primes_of_degree_one: 2 entries
- sage.rings.number_field.splitting_field: 4 entries
- sage.rings.number_field.structure: 6 entries
- sage.rings.number_field.totallyreal: 4 entries
- sage.rings.number_field.totallyreal_data: 6 entries
- sage.rings.number_field.totallyreal_phc: 2 entries
- sage.rings.number_field.totallyreal_rel: 5 entries
- sage.rings.number_field.unit_group: 2 entries
- sage.rings.padics.common_conversion: 1 entries
- sage.rings.padics.eisenstein_extension_generic: 2 entries
- sage.rings.padics.factory: 30 entries
- sage.rings.padics.generic_nodes: 21 entries
- sage.rings.padics.local_generic: 2 entries
- sage.rings.padics.local_generic_element: 2 entries
- sage.rings.padics.misc: 6 entries
- sage.rings.padics.morphism: 2 entries
- sage.rings.padics.padic_ZZ_pX_CA_element: 3 entries
- sage.rings.padics.padic_ZZ_pX_CR_element: 3 entries
- sage.rings.padics.padic_ZZ_pX_FM_element: 3 entries
- sage.rings.padics.padic_ZZ_pX_element: 2 entries
- sage.rings.padics.padic_base_generic: 2 entries
- sage.rings.padics.padic_base_leaves: 11 entries
- sage.rings.padics.padic_capped_absolute_element: 14 entries
- sage.rings.padics.padic_capped_relative_element: 17 entries
- sage.rings.padics.padic_ext_element: 2 entries
- sage.rings.padics.padic_extension_generic: 8 entries
- sage.rings.padics.padic_extension_leaves: 11 entries
- sage.rings.padics.padic_fixed_mod_element: 14 entries
- sage.rings.padics.padic_generic: 5 entries
- sage.rings.padics.padic_generic_element: 5 entries
- sage.rings.padics.padic_printing: 4 entries
- sage.rings.padics.padic_valuation: 6 entries
- sage.rings.padics.pow_computer: 4 entries
- sage.rings.padics.pow_computer_ext: 11 entries
- sage.rings.padics.precision_error: 1 entries
- sage.rings.padics.tutorial: 1 entries
- sage.rings.padics.unramified_extension_generic: 2 entries
- sage.rings.polynomial.complex_roots: 4 entries
- sage.rings.polynomial.convolution: 2 entries
- sage.rings.polynomial.cyclotomic: 4 entries
- sage.rings.polynomial.flatten: 5 entries
- sage.rings.polynomial.groebner_fan: 12 entries
- sage.rings.polynomial.hilbert: 4 entries
- sage.rings.polynomial.ideal: 2 entries
- sage.rings.polynomial.infinite_polynomial_element: 4 entries
- sage.rings.polynomial.infinite_polynomial_ring: 7 entries
- sage.rings.polynomial.integer_valued_polynomials: 8 entries
- sage.rings.polynomial.laurent_polynomial: 3 entries
- sage.rings.polynomial.laurent_polynomial_ring: 6 entries
- sage.rings.polynomial.laurent_polynomial_ring_base: 2 entries
- sage.rings.polynomial.msolve: 3 entries
- sage.rings.polynomial.multi_polynomial: 4 entries
- sage.rings.polynomial.multi_polynomial_element: 4 entries
- sage.rings.polynomial.multi_polynomial_ideal: 12 entries
- sage.rings.polynomial.multi_polynomial_ideal_libsingular: 5 entries
- sage.rings.polynomial.multi_polynomial_libsingular: 5 entries
- sage.rings.polynomial.multi_polynomial_ring: 4 entries
- sage.rings.polynomial.multi_polynomial_ring_base: 6 entries
- sage.rings.polynomial.multi_polynomial_sequence: 6 entries
- sage.rings.polynomial.omega: 4 entries
- sage.rings.polynomial.ore_function_element: 5 entries
- sage.rings.polynomial.ore_function_field: 5 entries
- sage.rings.polynomial.ore_polynomial_element: 5 entries
- sage.rings.polynomial.ore_polynomial_ring: 2 entries
- sage.rings.polynomial.padics.polynomial_padic: 2 entries
- sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense: 3 entries
- sage.rings.polynomial.padics.polynomial_padic_flat: 2 entries
- sage.rings.polynomial.pbori.pbori: 57 entries
- sage.rings.polynomial.plural: 11 entries
- sage.rings.polynomial.polydict: 7 entries
- sage.rings.polynomial.polynomial_compiled: 14 entries
- sage.rings.polynomial.polynomial_element: 11 entries
- sage.rings.polynomial.polynomial_element_generic: 15 entries
- sage.rings.polynomial.polynomial_fateman: 1 entries
- sage.rings.polynomial.polynomial_gf2x: 7 entries
- sage.rings.polynomial.polynomial_integer_dense_flint: 2 entries
- sage.rings.polynomial.polynomial_integer_dense_ntl: 2 entries
- sage.rings.polynomial.polynomial_modn_dense_ntl: 7 entries
- sage.rings.polynomial.polynomial_number_field: 3 entries
- sage.rings.polynomial.polynomial_quotient_ring: 7 entries
- sage.rings.polynomial.polynomial_quotient_ring_element: 2 entries
- sage.rings.polynomial.polynomial_rational_flint: 2 entries
- sage.rings.polynomial.polynomial_real_mpfr_dense: 3 entries
- sage.rings.polynomial.polynomial_ring: 19 entries
- sage.rings.polynomial.polynomial_ring_constructor: 5 entries
- sage.rings.polynomial.polynomial_ring_homomorphism: 2 entries
- sage.rings.polynomial.polynomial_singular_interface: 4 entries
- sage.rings.polynomial.polynomial_zmod_flint: 4 entries
- sage.rings.polynomial.polynomial_zz_pex: 4 entries
- sage.rings.polynomial.q_integer_valued_polynomials: 10 entries
- sage.rings.polynomial.real_roots: 51 entries
- sage.rings.polynomial.refine_root: 2 entries
- sage.rings.polynomial.skew_polynomial_element: 2 entries
- sage.rings.polynomial.skew_polynomial_finite_field: 2 entries
- sage.rings.polynomial.skew_polynomial_finite_order: 2 entries
- sage.rings.polynomial.skew_polynomial_ring: 6 entries
- sage.rings.polynomial.symmetric_ideal: 2 entries
- sage.rings.polynomial.symmetric_reduction: 2 entries
- sage.rings.polynomial.term_order: 5 entries
- sage.rings.polynomial.toy_buchberger: 10 entries
- sage.rings.polynomial.toy_d_basis: 8 entries
- sage.rings.polynomial.toy_variety: 7 entries
- sage.rings.power_series_ring: 7 entries
- sage.rings.power_series_ring_element: 5 entries
- sage.rings.rational: 9 entries
- sage.rings.rational_field: 4 entries
- sage.rings.real_double: 6 entries
- sage.rings.real_mpfr: 22 entries

---

### sage.modular (571 entries)

**By Type:**
- attribute: 8
- class: 229
- function: 219
- module: 115

**By Part:**
- Part 10: 571

**Sub-modules:**
- sage.modular.abvar.abvar: 12 entries
- sage.modular.abvar.abvar_ambient_jacobian: 3 entries
- sage.modular.abvar.abvar_newform: 2 entries
- sage.modular.abvar.constructor: 5 entries
- sage.modular.abvar.cuspidal_subgroup: 6 entries
- sage.modular.abvar.finite_subgroup: 3 entries
- sage.modular.abvar.homology: 7 entries
- sage.modular.abvar.homspace: 3 entries
- sage.modular.abvar.lseries: 4 entries
- sage.modular.abvar.morphism: 5 entries
- sage.modular.abvar.torsion_subgroup: 3 entries
- sage.modular.arithgroup.arithgroup_element: 2 entries
- sage.modular.arithgroup.arithgroup_generic: 3 entries
- sage.modular.arithgroup.arithgroup_perm: 10 entries
- sage.modular.arithgroup.congroup: 4 entries
- sage.modular.arithgroup.congroup_gamma: 4 entries
- sage.modular.arithgroup.congroup_gamma0: 4 entries
- sage.modular.arithgroup.congroup_gamma1: 4 entries
- sage.modular.arithgroup.congroup_gammaH: 5 entries
- sage.modular.arithgroup.congroup_generic: 6 entries
- sage.modular.arithgroup.congroup_sl2z: 3 entries
- sage.modular.arithgroup.farey_symbol: 2 entries
- sage.modular.btquotients.btquotient: 6 entries
- sage.modular.btquotients.pautomorphicform: 6 entries
- sage.modular.buzzard: 2 entries
- sage.modular.cusps: 3 entries
- sage.modular.cusps_nf: 9 entries
- sage.modular.dims: 10 entries
- sage.modular.dirichlet: 10 entries
- sage.modular.drinfeld_modform.element: 2 entries
- sage.modular.drinfeld_modform.ring: 2 entries
- sage.modular.drinfeld_modform.tutorial: 1 entries
- sage.modular.etaproducts: 10 entries
- sage.modular.hecke.algebra: 5 entries
- sage.modular.hecke.ambient_module: 3 entries
- sage.modular.hecke.degenmap: 2 entries
- sage.modular.hecke.element: 3 entries
- sage.modular.hecke.hecke_operator: 7 entries
- sage.modular.hecke.homspace: 3 entries
- sage.modular.hecke.module: 4 entries
- sage.modular.hecke.morphism: 5 entries
- sage.modular.hecke.submodule: 3 entries
- sage.modular.hypergeometric_motive: 10 entries
- sage.modular.local_comp.liftings: 7 entries
- sage.modular.local_comp.local_comp: 10 entries
- sage.modular.local_comp.smoothchar: 7 entries
- sage.modular.local_comp.type_space: 4 entries
- sage.modular.modform.ambient: 2 entries
- sage.modular.modform.ambient_R: 2 entries
- sage.modular.modform.ambient_eps: 2 entries
- sage.modular.modform.ambient_g0: 2 entries
- sage.modular.modform.ambient_g1: 3 entries
- sage.modular.modform.constructor: 9 entries
- sage.modular.modform.cuspidal_submodule: 11 entries
- sage.modular.modform.eis_series: 4 entries
- sage.modular.modform.eis_series_cython: 3 entries
- sage.modular.modform.eisenstein_submodule: 9 entries
- sage.modular.modform.element: 9 entries
- sage.modular.modform.half_integral: 2 entries
- sage.modular.modform.hecke_operator_on_qexp: 3 entries
- sage.modular.modform.j_invariant: 2 entries
- sage.modular.modform.notes: 1 entries
- sage.modular.modform.numerical: 3 entries
- sage.modular.modform.ring: 2 entries
- sage.modular.modform.space: 4 entries
- sage.modular.modform.submodule: 3 entries
- sage.modular.modform.theta: 3 entries
- sage.modular.modform.vm_basis: 3 entries
- sage.modular.modform_hecketriangle.abstract_ring: 2 entries
- sage.modular.modform_hecketriangle.abstract_space: 2 entries
- sage.modular.modform_hecketriangle.analytic_type: 3 entries
- sage.modular.modform_hecketriangle.constructor: 4 entries
- sage.modular.modform_hecketriangle.element: 2 entries
- sage.modular.modform_hecketriangle.functors: 7 entries
- sage.modular.modform_hecketriangle.graded_ring: 10 entries
- sage.modular.modform_hecketriangle.graded_ring_element: 2 entries
- sage.modular.modform_hecketriangle.hecke_triangle_group_element: 4 entries
- sage.modular.modform_hecketriangle.hecke_triangle_groups: 2 entries
- sage.modular.modform_hecketriangle.readme: 1 entries
- sage.modular.modform_hecketriangle.series_constructor: 2 entries
- sage.modular.modform_hecketriangle.space: 11 entries
- sage.modular.modform_hecketriangle.subspace: 4 entries
- sage.modular.modsym.ambient: 7 entries
- sage.modular.modsym.apply: 3 entries
- sage.modular.modsym.boundary: 7 entries
- sage.modular.modsym.element: 4 entries
- sage.modular.modsym.g1list: 2 entries
- sage.modular.modsym.ghlist: 2 entries
- sage.modular.modsym.hecke_operator: 2 entries
- sage.modular.modsym.heilbronn: 10 entries
- sage.modular.modsym.manin_symbol: 6 entries
- sage.modular.modsym.manin_symbol_list: 7 entries
- sage.modular.modsym.modsym: 4 entries
- sage.modular.modsym.modular_symbols: 2 entries
- sage.modular.modsym.p1list: 12 entries
- sage.modular.modsym.p1list_nf: 10 entries
- sage.modular.modsym.relation_matrix: 8 entries
- sage.modular.modsym.relation_matrix_pyx: 2 entries
- sage.modular.modsym.space: 6 entries
- sage.modular.modsym.subspace: 2 entries
- sage.modular.multiple_zeta: 21 entries
- sage.modular.overconvergent.genus0: 4 entries
- sage.modular.overconvergent.hecke_series: 19 entries
- sage.modular.overconvergent.weightspace: 6 entries
- sage.modular.pollack_stevens.distributions: 6 entries
- sage.modular.pollack_stevens.fund_domain: 5 entries
- sage.modular.pollack_stevens.manin_map: 4 entries
- sage.modular.pollack_stevens.modsym: 5 entries
- sage.modular.pollack_stevens.padic_lseries: 3 entries
- sage.modular.pollack_stevens.space: 6 entries
- sage.modular.quasimodform.element: 2 entries
- sage.modular.quasimodform.ring: 2 entries
- sage.modular.quatalg.brandt: 12 entries
- sage.modular.ssmod.ssmod: 7 entries

---

### sage.combinat (373 entries)

**By Type:**
- attribute: 13
- class: 251
- function: 76
- module: 33

**By Part:**
- Part 03: 42
- Part 04: 262
- Part 05: 69

**Sub-modules:**
- sage.combinat.combination: 9 entries
- sage.combinat.composition: 7 entries
- sage.combinat.composition_signed: 2 entries
- sage.combinat.composition_tableau: 7 entries
- sage.combinat.dyck_word: 17 entries
- sage.combinat.partition: 30 entries
- sage.combinat.partition_algebra: 49 entries
- sage.combinat.partition_kleshchev: 9 entries
- sage.combinat.partition_shifting_algebras: 4 entries
- sage.combinat.partition_tuple: 13 entries
- sage.combinat.partitions: 10 entries
- sage.combinat.perfect_matching: 3 entries
- sage.combinat.permutation: 52 entries
- sage.combinat.permutation_cython: 8 entries
- sage.combinat.posets.all: 1 entries
- sage.combinat.posets.cartesian_product: 3 entries
- sage.combinat.posets.d_complete: 2 entries
- sage.combinat.posets.elements: 5 entries
- sage.combinat.posets.forest: 2 entries
- sage.combinat.posets.hasse_diagram: 3 entries
- sage.combinat.posets.incidence_algebras: 4 entries
- sage.combinat.posets.lattices: 7 entries
- sage.combinat.posets.linear_extensions: 6 entries
- sage.combinat.posets.mobile: 2 entries
- sage.combinat.posets.moebius_algebra: 15 entries
- sage.combinat.posets.poset_examples: 5 entries
- sage.combinat.posets.posets: 5 entries
- sage.combinat.set_partition: 10 entries
- sage.combinat.set_partition_iterator: 3 entries
- sage.combinat.set_partition_ordered: 11 entries
- sage.combinat.tableau: 39 entries
- sage.combinat.tableau_residues: 3 entries
- sage.combinat.tableau_tuple: 27 entries

---

### sage.schemes (271 entries)

**By Type:**
- attribute: 5
- class: 67
- function: 152
- module: 47

**By Part:**
- Part 13: 271

**Sub-modules:**
- sage.schemes.elliptic_curves.Qcurves: 4 entries
- sage.schemes.elliptic_curves.cm: 11 entries
- sage.schemes.elliptic_curves.constructor: 13 entries
- sage.schemes.elliptic_curves.descent_two_isogeny: 7 entries
- sage.schemes.elliptic_curves.ec_database: 2 entries
- sage.schemes.elliptic_curves.ell_curve_isogeny: 16 entries
- sage.schemes.elliptic_curves.ell_egros: 8 entries
- sage.schemes.elliptic_curves.ell_field: 4 entries
- sage.schemes.elliptic_curves.ell_finite_field: 12 entries
- sage.schemes.elliptic_curves.ell_generic: 3 entries
- sage.schemes.elliptic_curves.ell_local_data: 3 entries
- sage.schemes.elliptic_curves.ell_modular_symbols: 5 entries
- sage.schemes.elliptic_curves.ell_number_field: 2 entries
- sage.schemes.elliptic_curves.ell_padic_field: 2 entries
- sage.schemes.elliptic_curves.ell_point: 5 entries
- sage.schemes.elliptic_curves.ell_rational_field: 6 entries
- sage.schemes.elliptic_curves.ell_tate_curve: 2 entries
- sage.schemes.elliptic_curves.ell_torsion: 3 entries
- sage.schemes.elliptic_curves.ell_wp: 6 entries
- sage.schemes.elliptic_curves.formal_group: 2 entries
- sage.schemes.elliptic_curves.gal_reps: 2 entries
- sage.schemes.elliptic_curves.gal_reps_number_field: 11 entries
- sage.schemes.elliptic_curves.gp_simon: 2 entries
- sage.schemes.elliptic_curves.heegner: 40 entries
- sage.schemes.elliptic_curves.height: 8 entries
- sage.schemes.elliptic_curves.hom: 5 entries
- sage.schemes.elliptic_curves.hom_composite: 2 entries
- sage.schemes.elliptic_curves.hom_frobenius: 2 entries
- sage.schemes.elliptic_curves.hom_scalar: 2 entries
- sage.schemes.elliptic_curves.hom_sum: 2 entries
- sage.schemes.elliptic_curves.hom_velusqrt: 3 entries
- sage.schemes.elliptic_curves.isogeny_class: 6 entries
- sage.schemes.elliptic_curves.isogeny_small_degree: 21 entries
- sage.schemes.elliptic_curves.jacobian: 4 entries
- sage.schemes.elliptic_curves.kodaira_symbol: 3 entries
- sage.schemes.elliptic_curves.lseries_ell: 2 entries
- sage.schemes.elliptic_curves.mod5family: 2 entries
- sage.schemes.elliptic_curves.mod_poly: 2 entries
- sage.schemes.elliptic_curves.mod_sym_num: 2 entries
- sage.schemes.elliptic_curves.modular_parametrization: 2 entries
- sage.schemes.elliptic_curves.padic_lseries: 4 entries
- sage.schemes.elliptic_curves.period_lattice: 7 entries
- sage.schemes.elliptic_curves.period_lattice_region: 6 entries
- sage.schemes.elliptic_curves.saturation: 4 entries
- sage.schemes.elliptic_curves.sha_tate: 2 entries
- sage.schemes.elliptic_curves.weierstrass_morphism: 5 entries
- sage.schemes.elliptic_curves.weierstrass_transform: 4 entries

---

### sage.modules (265 entries)

**By Type:**
- attribute: 1
- class: 132
- function: 68
- module: 64

**By Part:**
- Part 10: 265

**Sub-modules:**
- sage.modules.complex_double_vector: 1 entries
- sage.modules.diamond_cutting: 5 entries
- sage.modules.fg_pid.fgp_element: 2 entries
- sage.modules.fg_pid.fgp_module: 6 entries
- sage.modules.fg_pid.fgp_morphism: 4 entries
- sage.modules.filtered_vector_space: 8 entries
- sage.modules.finite_submodule_iter: 4 entries
- sage.modules.fp_graded.element: 2 entries
- sage.modules.fp_graded.free_element: 2 entries
- sage.modules.fp_graded.free_homspace: 2 entries
- sage.modules.fp_graded.free_module: 2 entries
- sage.modules.fp_graded.free_morphism: 2 entries
- sage.modules.fp_graded.homspace: 2 entries
- sage.modules.fp_graded.module: 2 entries
- sage.modules.fp_graded.morphism: 2 entries
- sage.modules.fp_graded.steenrod.module: 4 entries
- sage.modules.fp_graded.steenrod.morphism: 3 entries
- sage.modules.free_module: 24 entries
- sage.modules.free_module_element: 14 entries
- sage.modules.free_module_homspace: 3 entries
- sage.modules.free_module_integer: 4 entries
- sage.modules.free_module_morphism: 6 entries
- sage.modules.free_module_pseudohomspace: 2 entries
- sage.modules.free_module_pseudomorphism: 2 entries
- sage.modules.free_quadratic_module: 16 entries
- sage.modules.free_quadratic_module_integer_symmetric: 6 entries
- sage.modules.matrix_morphism: 4 entries
- sage.modules.misc: 2 entries
- sage.modules.module: 4 entries
- sage.modules.multi_filtered_vector_space: 3 entries
- sage.modules.ore_module: 7 entries
- sage.modules.ore_module_element: 2 entries
- sage.modules.ore_module_homspace: 2 entries
- sage.modules.ore_module_morphism: 4 entries
- sage.modules.quotient_module: 3 entries
- sage.modules.real_double_vector: 1 entries
- sage.modules.submodule: 2 entries
- sage.modules.tensor_operations: 5 entries
- sage.modules.torsion_quadratic_module: 4 entries
- sage.modules.tutorial_free_modules: 1 entries
- sage.modules.vector_callable_symbolic_dense: 2 entries
- sage.modules.vector_complex_double_dense: 4 entries
- sage.modules.vector_double_dense: 2 entries
- sage.modules.vector_integer_dense: 4 entries
- sage.modules.vector_integer_sparse: 1 entries
- sage.modules.vector_mod2_dense: 3 entries
- sage.modules.vector_modn_dense: 4 entries
- sage.modules.vector_modn_sparse: 1 entries
- sage.modules.vector_numpy_dense: 2 entries
- sage.modules.vector_numpy_integer_dense: 2 entries
- sage.modules.vector_rational_dense: 4 entries
- sage.modules.vector_rational_sparse: 1 entries
- sage.modules.vector_real_double_dense: 4 entries
- sage.modules.vector_space_homspace: 3 entries
- sage.modules.vector_space_morphism: 4 entries
- sage.modules.vector_symbolic_dense: 3 entries
- sage.modules.vector_symbolic_sparse: 3 entries
- sage.modules.with_basis.all: 1 entries
- sage.modules.with_basis.cell_module: 4 entries
- sage.modules.with_basis.indexed_element: 2 entries
- sage.modules.with_basis.invariant: 4 entries
- sage.modules.with_basis.morphism: 11 entries
- sage.modules.with_basis.representation: 19 entries
- sage.modules.with_basis.subquotient: 3 entries

---

### sage.geometry (208 entries)

**By Type:**
- attribute: 3
- class: 88
- function: 66
- module: 51

**By Part:**
- Part 06: 208

**Sub-modules:**
- sage.geometry.cone: 9 entries
- sage.geometry.cone_catalog: 6 entries
- sage.geometry.cone_critical_angles: 7 entries
- sage.geometry.fan: 9 entries
- sage.geometry.fan_isomorphism: 7 entries
- sage.geometry.fan_morphism: 2 entries
- sage.geometry.polyhedron.backend_cdd: 3 entries
- sage.geometry.polyhedron.backend_cdd_rdf: 2 entries
- sage.geometry.polyhedron.backend_field: 2 entries
- sage.geometry.polyhedron.backend_normaliz: 4 entries
- sage.geometry.polyhedron.backend_number_field: 2 entries
- sage.geometry.polyhedron.backend_polymake: 4 entries
- sage.geometry.polyhedron.backend_ppl: 4 entries
- sage.geometry.polyhedron.base: 3 entries
- sage.geometry.polyhedron.base0: 2 entries
- sage.geometry.polyhedron.base1: 2 entries
- sage.geometry.polyhedron.base2: 2 entries
- sage.geometry.polyhedron.base3: 2 entries
- sage.geometry.polyhedron.base4: 2 entries
- sage.geometry.polyhedron.base5: 2 entries
- sage.geometry.polyhedron.base6: 2 entries
- sage.geometry.polyhedron.base7: 2 entries
- sage.geometry.polyhedron.base_QQ: 2 entries
- sage.geometry.polyhedron.base_RDF: 2 entries
- sage.geometry.polyhedron.base_ZZ: 2 entries
- sage.geometry.polyhedron.cdd_file_format: 3 entries
- sage.geometry.polyhedron.combinatorial_polyhedron.base: 2 entries
- sage.geometry.polyhedron.combinatorial_polyhedron.combinatorial_face: 2 entries
- sage.geometry.polyhedron.combinatorial_polyhedron.conversions: 5 entries
- sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator: 5 entries
- sage.geometry.polyhedron.combinatorial_polyhedron.list_of_faces: 2 entries
- sage.geometry.polyhedron.combinatorial_polyhedron.polyhedron_face_lattice: 3 entries
- sage.geometry.polyhedron.constructor: 2 entries
- sage.geometry.polyhedron.double_description: 7 entries
- sage.geometry.polyhedron.double_description_inhomogeneous: 4 entries
- sage.geometry.polyhedron.face: 3 entries
- sage.geometry.polyhedron.generating_function: 2 entries
- sage.geometry.polyhedron.lattice_euclidean_group_element: 2 entries
- sage.geometry.polyhedron.library: 6 entries
- sage.geometry.polyhedron.modules.formal_polyhedra_module: 2 entries
- sage.geometry.polyhedron.palp_database: 3 entries
- sage.geometry.polyhedron.parent: 14 entries
- sage.geometry.polyhedron.plot: 6 entries
- sage.geometry.polyhedron.ppl_lattice_polygon: 9 entries
- sage.geometry.polyhedron.ppl_lattice_polytope: 3 entries
- sage.geometry.polyhedron.representation: 10 entries
- sage.geometry.toric_lattice: 10 entries
- sage.geometry.toric_plotter: 7 entries
- sage.geometry.triangulation.base: 4 entries
- sage.geometry.triangulation.element: 4 entries
- sage.geometry.triangulation.point_configuration: 2 entries

---

### sage.manifolds (202 entries)

**By Type:**
- attribute: 5
- class: 119
- function: 13
- module: 65

**By Part:**
- Part 08: 202

**Sub-modules:**
- sage.manifolds.calculus_method: 2 entries
- sage.manifolds.catalog: 5 entries
- sage.manifolds.chart: 4 entries
- sage.manifolds.chart_func: 4 entries
- sage.manifolds.continuous_map: 2 entries
- sage.manifolds.continuous_map_image: 2 entries
- sage.manifolds.differentiable.affine_connection: 2 entries
- sage.manifolds.differentiable.automorphismfield: 3 entries
- sage.manifolds.differentiable.automorphismfield_group: 3 entries
- sage.manifolds.differentiable.bundle_connection: 2 entries
- sage.manifolds.differentiable.characteristic_cohomology_class: 11 entries
- sage.manifolds.differentiable.chart: 4 entries
- sage.manifolds.differentiable.curve: 2 entries
- sage.manifolds.differentiable.de_rham_cohomology: 3 entries
- sage.manifolds.differentiable.degenerate: 3 entries
- sage.manifolds.differentiable.degenerate_submanifold: 3 entries
- sage.manifolds.differentiable.diff_form: 3 entries
- sage.manifolds.differentiable.diff_form_module: 4 entries
- sage.manifolds.differentiable.diff_map: 2 entries
- sage.manifolds.differentiable.differentiable_submanifold: 2 entries
- sage.manifolds.differentiable.examples.euclidean: 4 entries
- sage.manifolds.differentiable.examples.real_line: 3 entries
- sage.manifolds.differentiable.examples.sphere: 2 entries
- sage.manifolds.differentiable.examples.symplectic_space: 2 entries
- sage.manifolds.differentiable.integrated_curve: 4 entries
- sage.manifolds.differentiable.levi_civita_connection: 2 entries
- sage.manifolds.differentiable.manifold: 2 entries
- sage.manifolds.differentiable.manifold_homset: 6 entries
- sage.manifolds.differentiable.metric: 5 entries
- sage.manifolds.differentiable.mixed_form: 2 entries
- sage.manifolds.differentiable.mixed_form_algebra: 2 entries
- sage.manifolds.differentiable.multivector_module: 3 entries
- sage.manifolds.differentiable.multivectorfield: 3 entries
- sage.manifolds.differentiable.poisson_tensor: 3 entries
- sage.manifolds.differentiable.pseudo_riemannian: 2 entries
- sage.manifolds.differentiable.pseudo_riemannian_submanifold: 2 entries
- sage.manifolds.differentiable.scalarfield: 2 entries
- sage.manifolds.differentiable.scalarfield_algebra: 2 entries
- sage.manifolds.differentiable.symplectic_form: 3 entries
- sage.manifolds.differentiable.tangent_space: 2 entries
- sage.manifolds.differentiable.tangent_vector: 2 entries
- sage.manifolds.differentiable.tensorfield: 2 entries
- sage.manifolds.differentiable.tensorfield_module: 3 entries
- sage.manifolds.differentiable.tensorfield_paral: 2 entries
- sage.manifolds.differentiable.vector_bundle: 3 entries
- sage.manifolds.differentiable.vectorfield: 3 entries
- sage.manifolds.differentiable.vectorfield_module: 3 entries
- sage.manifolds.differentiable.vectorframe: 5 entries
- sage.manifolds.family: 3 entries
- sage.manifolds.local_frame: 5 entries
- sage.manifolds.manifold: 4 entries
- sage.manifolds.manifold_homset: 2 entries
- sage.manifolds.operators: 6 entries
- sage.manifolds.point: 2 entries
- sage.manifolds.scalarfield: 2 entries
- sage.manifolds.scalarfield_algebra: 2 entries
- sage.manifolds.section: 3 entries
- sage.manifolds.section_module: 3 entries
- sage.manifolds.structure: 13 entries
- sage.manifolds.subset: 2 entries
- sage.manifolds.subsets.closure: 2 entries
- sage.manifolds.subsets.pullback: 2 entries
- sage.manifolds.topological_submanifold: 2 entries
- sage.manifolds.trivialization: 3 entries

---

### sage.groups (201 entries)

**By Type:**
- class: 92
- function: 66
- module: 43

**By Part:**
- Part 07: 137
- Part 08: 64

**Sub-modules:**
- sage.groups.abelian_gps.abelian_aut: 5 entries
- sage.groups.abelian_gps.abelian_group: 6 entries
- sage.groups.abelian_gps.abelian_group_element: 3 entries
- sage.groups.abelian_gps.abelian_group_gap: 7 entries
- sage.groups.abelian_gps.abelian_group_morphism: 4 entries
- sage.groups.abelian_gps.dual_abelian_group: 3 entries
- sage.groups.abelian_gps.dual_abelian_group_element: 3 entries
- sage.groups.abelian_gps.element_base: 2 entries
- sage.groups.abelian_gps.values: 5 entries
- sage.groups.matrix_gps.binary_dihedral: 2 entries
- sage.groups.matrix_gps.catalog: 1 entries
- sage.groups.matrix_gps.coxeter_group: 3 entries
- sage.groups.matrix_gps.finitely_generated: 5 entries
- sage.groups.matrix_gps.finitely_generated_gap: 2 entries
- sage.groups.matrix_gps.group_element: 3 entries
- sage.groups.matrix_gps.group_element_gap: 2 entries
- sage.groups.matrix_gps.heisenberg: 2 entries
- sage.groups.matrix_gps.isometries: 4 entries
- sage.groups.matrix_gps.linear: 4 entries
- sage.groups.matrix_gps.linear_gap: 2 entries
- sage.groups.matrix_gps.matrix_group: 4 entries
- sage.groups.matrix_gps.matrix_group_gap: 2 entries
- sage.groups.matrix_gps.named_group: 4 entries
- sage.groups.matrix_gps.named_group_gap: 2 entries
- sage.groups.matrix_gps.orthogonal: 5 entries
- sage.groups.matrix_gps.orthogonal_gap: 2 entries
- sage.groups.matrix_gps.symplectic: 3 entries
- sage.groups.matrix_gps.symplectic_gap: 2 entries
- sage.groups.matrix_gps.unitary: 5 entries
- sage.groups.matrix_gps.unitary_gap: 2 entries
- sage.groups.perm_gps.constructor: 4 entries
- sage.groups.perm_gps.cubegroup: 15 entries
- sage.groups.perm_gps.partn_ref.canonical_augmentation: 1 entries
- sage.groups.perm_gps.partn_ref.data_structures: 4 entries
- sage.groups.perm_gps.partn_ref.refinement_graphs: 11 entries
- sage.groups.perm_gps.partn_ref.refinement_lists: 2 entries
- sage.groups.perm_gps.partn_ref.refinement_matrices: 3 entries
- sage.groups.perm_gps.permgroup: 9 entries
- sage.groups.perm_gps.permgroup_element: 6 entries
- sage.groups.perm_gps.permgroup_morphism: 6 entries
- sage.groups.perm_gps.permgroup_named: 34 entries
- sage.groups.perm_gps.permutation_groups_catalog: 1 entries
- sage.groups.perm_gps.symgp_conjugacy_class: 6 entries

---

### sage.coding (192 entries)

**By Type:**
- attribute: 1
- class: 80
- function: 68
- module: 43

**By Part:**
- Part 02: 192

**Sub-modules:**
- sage.coding.abstract_code: 2 entries
- sage.coding.ag_code: 5 entries
- sage.coding.ag_code_decoders: 12 entries
- sage.coding.bch_code: 3 entries
- sage.coding.binary_code: 8 entries
- sage.coding.bounds_catalog: 1 entries
- sage.coding.channel: 7 entries
- sage.coding.channels_catalog: 1 entries
- sage.coding.code_bounds: 19 entries
- sage.coding.code_constructions: 13 entries
- sage.coding.codecan.autgroup_can_label: 2 entries
- sage.coding.codecan.codecan: 3 entries
- sage.coding.codes_catalog: 1 entries
- sage.coding.cyclic_code: 7 entries
- sage.coding.databases: 5 entries
- sage.coding.decoder: 2 entries
- sage.coding.decoders_catalog: 1 entries
- sage.coding.delsarte_bounds: 7 entries
- sage.coding.encoder: 2 entries
- sage.coding.encoders_catalog: 1 entries
- sage.coding.extended_code: 4 entries
- sage.coding.gabidulin_code: 5 entries
- sage.coding.golay_code: 2 entries
- sage.coding.goppa_code: 3 entries
- sage.coding.grs_code: 9 entries
- sage.coding.guava: 3 entries
- sage.coding.guruswami_sudan.gs_decoder: 5 entries
- sage.coding.guruswami_sudan.interpolation: 4 entries
- sage.coding.guruswami_sudan.utils: 6 entries
- sage.coding.hamming_code: 2 entries
- sage.coding.information_set_decoder: 4 entries
- sage.coding.kasami_codes: 2 entries
- sage.coding.linear_code: 6 entries
- sage.coding.linear_code_no_metric: 3 entries
- sage.coding.linear_rank_metric: 8 entries
- sage.coding.parity_check_code: 4 entries
- sage.coding.punctured_code: 4 entries
- sage.coding.reed_muller_code: 6 entries
- sage.coding.self_dual_codes: 2 entries
- sage.coding.source_coding.huffman: 3 entries
- sage.coding.subfield_subcode: 3 entries
- sage.coding.two_weight_db: 1 entries

---

### sage.symbolic (120 entries)

**By Type:**
- attribute: 20
- class: 26
- function: 70
- module: 4

**By Part:**
- Part 14: 120

**Sub-modules:**
- sage.symbolic.expression: 67 entries
- sage.symbolic.expression_conversions: 36 entries
- sage.symbolic.relation: 8 entries
- sage.symbolic.ring: 9 entries

---

### sage.matrix (116 entries)

**By Type:**
- attribute: 1
- class: 31
- function: 52
- module: 32

**By Part:**
- Part 09: 116

**Sub-modules:**
- sage.matrix.constructor: 4 entries
- sage.matrix.matrix0: 5 entries
- sage.matrix.matrix1: 2 entries
- sage.matrix.matrix2: 4 entries
- sage.matrix.matrix_complex_ball_dense: 2 entries
- sage.matrix.matrix_complex_double_dense: 2 entries
- sage.matrix.matrix_cyclo_dense: 2 entries
- sage.matrix.matrix_dense: 2 entries
- sage.matrix.matrix_double_dense: 2 entries
- sage.matrix.matrix_generic_dense: 2 entries
- sage.matrix.matrix_generic_sparse: 3 entries
- sage.matrix.matrix_gf2e_dense: 4 entries
- sage.matrix.matrix_integer_dense: 2 entries
- sage.matrix.matrix_integer_dense_hnf: 26 entries
- sage.matrix.matrix_integer_dense_saturation: 6 entries
- sage.matrix.matrix_integer_sparse: 2 entries
- sage.matrix.matrix_misc: 4 entries
- sage.matrix.matrix_mod2_dense: 8 entries
- sage.matrix.matrix_modn_dense_double: 3 entries
- sage.matrix.matrix_modn_dense_float: 3 entries
- sage.matrix.matrix_modn_sparse: 3 entries
- sage.matrix.matrix_mpolynomial_dense: 2 entries
- sage.matrix.matrix_polynomial_dense: 2 entries
- sage.matrix.matrix_rational_dense: 3 entries
- sage.matrix.matrix_rational_sparse: 2 entries
- sage.matrix.matrix_real_double_dense: 2 entries
- sage.matrix.matrix_space: 5 entries
- sage.matrix.matrix_sparse: 2 entries
- sage.matrix.matrix_symbolic_dense: 2 entries
- sage.matrix.matrix_symbolic_sparse: 2 entries
- sage.matrix.matrix_window: 2 entries

---

### sage.arith (102 entries)

**By Type:**
- class: 6
- function: 90
- module: 6

**By Part:**
- Part 01: 102

**Sub-modules:**
- sage.arith.functions: 3 entries
- sage.arith.misc: 87 entries
- sage.arith.multi_modular: 4 entries
- sage.arith.power: 2 entries
- sage.arith.srange: 5 entries

---

### sage.misc (95 entries)

**By Type:**
- attribute: 2
- class: 14
- function: 62
- module: 17

**By Part:**
- Part 10: 95

**Sub-modules:**
- sage.misc.sage_unittest: 1 entries
- sage.misc.sagedoc: 15 entries
- sage.misc.sagedoc_conf: 11 entries
- sage.misc.sageinspect: 16 entries
- sage.misc.search: 2 entries
- sage.misc.session: 5 entries
- sage.misc.sh: 2 entries
- sage.misc.sphinxify: 2 entries
- sage.misc.stopgap: 3 entries
- sage.misc.superseded: 7 entries
- sage.misc.table: 2 entries
- sage.misc.temporary_file: 6 entries
- sage.misc.test_nested_class: 1 entries
- sage.misc.trace: 2 entries
- sage.misc.unknown: 2 entries
- sage.misc.verbose: 7 entries
- sage.misc.viewer: 7 entries
- sage.misc.weak_dict: 4 entries

---

### sage.numerical (81 entries)

**By Type:**
- class: 30
- function: 30
- module: 21

**By Part:**
- Part 10: 14
- Part 11: 67

**Sub-modules:**
- sage.numerical.backends.cvxopt_backend: 2 entries
- sage.numerical.backends.cvxopt_sdp_backend: 2 entries
- sage.numerical.backends.generic_backend: 4 entries
- sage.numerical.backends.generic_sdp_backend: 4 entries
- sage.numerical.backends.glpk_backend: 2 entries
- sage.numerical.backends.glpk_exact_backend: 2 entries
- sage.numerical.backends.glpk_graph_backend: 2 entries
- sage.numerical.backends.interactivelp_backend: 2 entries
- sage.numerical.backends.logging_backend: 3 entries
- sage.numerical.backends.ppl_backend: 2 entries
- sage.numerical.gauss_legendre: 6 entries
- sage.numerical.interactive_simplex_method: 10 entries
- sage.numerical.knapsack: 3 entries
- sage.numerical.linear_functions: 10 entries
- sage.numerical.linear_tensor: 4 entries
- sage.numerical.linear_tensor_constraints: 5 entries
- sage.numerical.linear_tensor_element: 2 entries
- sage.numerical.mip: 3 entries
- sage.numerical.optimize: 8 entries
- sage.numerical.sdp: 4 entries

---

### sage.quadratic_forms (75 entries)

**By Type:**
- attribute: 1
- class: 8
- function: 51
- module: 15

**By Part:**
- Part 11: 75

**Sub-modules:**
- sage.quadratic_forms.binary_qf: 3 entries
- sage.quadratic_forms.bqf_class_group: 4 entries
- sage.quadratic_forms.constructions: 3 entries
- sage.quadratic_forms.count_local_2: 4 entries
- sage.quadratic_forms.extras: 4 entries
- sage.quadratic_forms.genera.genus: 19 entries
- sage.quadratic_forms.genera.normal_form: 3 entries
- sage.quadratic_forms.qfsolve: 4 entries
- sage.quadratic_forms.quadratic_form: 5 entries
- sage.quadratic_forms.quadratic_form__evaluate: 3 entries
- sage.quadratic_forms.random_quadraticform: 5 entries
- sage.quadratic_forms.special_values: 6 entries
- sage.quadratic_forms.ternary: 6 entries
- sage.quadratic_forms.ternary_qf: 5 entries

---

### sage.functions (70 entries)

**By Type:**
- class: 61
- function: 3
- module: 6

**By Part:**
- Part 06: 70

**Sub-modules:**
- sage.functions.bessel: 15 entries
- sage.functions.error: 7 entries
- sage.functions.exp_integral: 11 entries
- sage.functions.hyperbolic: 13 entries
- sage.functions.log: 10 entries
- sage.functions.trig: 14 entries

---

### sage.monoids (54 entries)

**By Type:**
- class: 24
- function: 17
- module: 13

**By Part:**
- Part 10: 54

**Sub-modules:**
- sage.monoids.automatic_semigroup: 4 entries
- sage.monoids.free_abelian_monoid: 5 entries
- sage.monoids.free_abelian_monoid_element: 3 entries
- sage.monoids.free_monoid: 3 entries
- sage.monoids.free_monoid_element: 3 entries
- sage.monoids.hecke_monoid: 2 entries
- sage.monoids.indexed_free_monoid: 7 entries
- sage.monoids.monoid: 3 entries
- sage.monoids.string_monoid: 7 entries
- sage.monoids.string_monoid_element: 8 entries
- sage.monoids.string_ops: 5 entries
- sage.monoids.trace_monoid: 3 entries

---

### sage.calculus (52 entries)

**By Type:**
- class: 2
- function: 45
- module: 5

**By Part:**
- Part 01: 52

**Sub-modules:**
- sage.calculus.calculus: 21 entries
- sage.calculus.desolvers: 15 entries
- sage.calculus.functional: 10 entries
- sage.calculus.integration: 5 entries

---

### sage.dynamics (51 entries)

**By Type:**
- class: 31
- function: 6
- module: 14

**By Part:**
- Part 06: 51

**Sub-modules:**
- sage.dynamics.arithmetic_dynamics.affine_ds: 4 entries
- sage.dynamics.arithmetic_dynamics.berkovich_ds: 4 entries
- sage.dynamics.arithmetic_dynamics.dynamical_semigroup: 8 entries
- sage.dynamics.arithmetic_dynamics.generic_ds: 2 entries
- sage.dynamics.arithmetic_dynamics.product_projective_ds: 4 entries
- sage.dynamics.arithmetic_dynamics.projective_ds: 4 entries
- sage.dynamics.arithmetic_dynamics.wehlerK3: 6 entries
- sage.dynamics.cellular_automata.catalog: 1 entries
- sage.dynamics.cellular_automata.elementary: 2 entries
- sage.dynamics.cellular_automata.glca: 2 entries
- sage.dynamics.cellular_automata.solitons: 3 entries
- sage.dynamics.complex_dynamics.mandel_julia: 5 entries
- sage.dynamics.finite_dynamical_system: 5 entries

---

### sage.homology (51 entries)

**By Type:**
- class: 27
- function: 9
- module: 15

**By Part:**
- Part 08: 51

**Sub-modules:**
- sage.homology.algebraic_topological_model: 3 entries
- sage.homology.chain_complex: 4 entries
- sage.homology.chain_complex_homspace: 3 entries
- sage.homology.chain_complex_morphism: 3 entries
- sage.homology.chain_homotopy: 3 entries
- sage.homology.chains: 5 entries
- sage.homology.free_resolution: 5 entries
- sage.homology.graded_resolution: 4 entries
- sage.homology.hochschild_complex: 3 entries
- sage.homology.homology_group: 3 entries
- sage.homology.homology_morphism: 2 entries
- sage.homology.homology_vector_space_with_basis: 8 entries
- sage.homology.koszul_complex: 2 entries
- sage.homology.matrix_utils: 2 entries

---

### sage.stats (43 entries)

**By Type:**
- attribute: 5
- class: 13
- function: 14
- module: 11

**By Part:**
- Part 14: 43

**Sub-modules:**
- sage.stats.basic_stats: 7 entries
- sage.stats.distributions.discrete_gaussian_integer: 7 entries
- sage.stats.distributions.discrete_gaussian_lattice: 2 entries
- sage.stats.distributions.discrete_gaussian_polynomial: 2 entries
- sage.stats.hmm.chmm: 6 entries
- sage.stats.hmm.distributions: 6 entries
- sage.stats.hmm.hmm: 5 entries
- sage.stats.hmm.util: 2 entries
- sage.stats.intlist: 3 entries
- sage.stats.r: 2 entries

---

### sage.logic (40 entries)

**By Type:**
- class: 3
- function: 30
- module: 7

**By Part:**
- Part 08: 40

**Sub-modules:**
- sage.logic.booleval: 4 entries
- sage.logic.boolformula: 3 entries
- sage.logic.logic: 14 entries
- sage.logic.logicparser: 12 entries
- sage.logic.logictable: 2 entries
- sage.logic.propcalc: 4 entries

---

### sage.databases (22 entries)

**By Type:**
- class: 5
- function: 14
- module: 3

**By Part:**
- Part 05: 17
- Part 06: 5

**Sub-modules:**
- sage.databases.cremona: 15 entries
- sage.databases.cunningham_tables: 2 entries
- sage.databases.oeis: 5 entries

---

### sage.crypto (16 entries)

**By Type:**
- class: 13
- module: 3

**By Part:**
- Part 05: 16

**Sub-modules:**
- sage.crypto.classical: 7 entries
- sage.crypto.classical_cipher: 7 entries
- sage.crypto.public_key.blum_goldwasser: 2 entries

---

### sage.categories (15 entries)

**By Type:**
- class: 9
- function: 4
- module: 2

**By Part:**
- Part 02: 15

**Sub-modules:**
- sage.categories.functor: 7 entries
- sage.categories.morphism: 8 entries

---

### sage.probability (15 entries)

**By Type:**
- class: 8
- function: 4
- module: 3

**By Part:**
- Part 11: 15

**Sub-modules:**
- sage.probability.probability_distribution: 5 entries
- sage.probability.random_variable: 9 entries

---

### sage.graphs (8 entries)

**By Type:**
- class: 4
- module: 4

**By Part:**
- Part 07: 8

**Sub-modules:**
- sage.graphs.bipartite_graph: 2 entries
- sage.graphs.digraph: 2 entries
- sage.graphs.digraph_generators: 2 entries
- sage.graphs.graph: 2 entries

---

### sage.libs (1 entries)

**By Type:**
- module: 1

**By Part:**
- Part 08: 1

**Sub-modules:**
- sage.libs.flint.arith_sage: 1 entries

---

