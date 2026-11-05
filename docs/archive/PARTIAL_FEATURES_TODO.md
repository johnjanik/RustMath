# SageMath Features - Partial Implementation TODO List

**Total Partial Features:** 970

This list contains all SageMath features marked as 'Partial' in the tracker files.
These represent features where some related functionality exists in RustMath, but the implementation is incomplete.

---

## sage.arith (102 features)

  - [ ] `sage.arith` (module)
  - [ ] `sage.arith.functions` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/functions.pyx#L127
  - [ ] `sage.arith.functions.LCM_list` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/functions.pyx#L127
  - [ ] `sage.arith.functions.lcm` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/functions.pyx#L22
  - [ ] `sage.arith.misc` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3354
  - [ ] `sage.arith.misc.CRT` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3354
  - [ ] `sage.arith.misc.CRT_basis` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3641
  - [ ] `sage.arith.misc.CRT_list` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3500
  - [ ] `sage.arith.misc.CRT_vectors` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3688
  - [ ] `sage.arith.misc.Euler_Phi` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3073
  - [ ] `sage.arith.misc.GCD` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1718
  - [ ] `sage.arith.misc.Moebius` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4566
  - [ ] `sage.arith.misc.Sigma` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1580
  - [ ] `sage.arith.misc.XGCD` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1939
  - [ ] `sage.arith.misc.algdep` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L43
  - [ ] `sage.arith.misc.algebraic_dependency` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L43
  - [ ] `sage.arith.misc.bernoulli` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L280
  - [ ] `sage.arith.misc.binomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3724
  - [ ] `sage.arith.misc.binomial_coefficients` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4063
  - [ ] `sage.arith.misc.carmichael_lambda` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3206
  - [ ] `sage.arith.misc.continuant` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4749
  - [ ] `sage.arith.misc.coprime_part` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6470
  - [ ] `sage.arith.misc.crt` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3354
  - [ ] `sage.arith.misc.dedekind_psi` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6393
  - [ ] `sage.arith.misc.dedekind_sum` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6167
  - [ ] `sage.arith.misc.differences` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5933
  - [ ] `sage.arith.misc.divisors` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1489
  - [ ] `sage.arith.misc.eratosthenes` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L963
  - [ ] `sage.arith.misc.factor` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2563
  - [ ] `sage.arith.misc.factorial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L399
  - [ ] `sage.arith.misc.falling_factorial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5123
  - [ ] `sage.arith.misc.four_squares` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5677
  - [ ] `sage.arith.misc.fundamental_discriminant` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6072
  - [ ] `sage.arith.misc.gauss_sum` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6279
  - [ ] `sage.arith.misc.gcd` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1718
  - [ ] `sage.arith.misc.get_gcd` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2231
  - [ ] `sage.arith.misc.get_inverse_mod` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2253
  - [ ] `sage.arith.misc.hilbert_conductor` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4981
  - [ ] `sage.arith.misc.hilbert_conductor_inverse` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5025
  - [ ] `sage.arith.misc.hilbert_symbol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4866
  - [ ] `sage.arith.misc.integer_ceil` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5316
  - [ ] `sage.arith.misc.integer_floor` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5348
  - [ ] `sage.arith.misc.integer_trunc` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5394
  - [ ] `sage.arith.misc.inverse_mod` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2195
  - [ ] `sage.arith.misc.is_power_of_two` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5894
  - [ ] `sage.arith.misc.is_prime` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L479
  - [ ] `sage.arith.misc.is_prime_power` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L610
  - [ ] `sage.arith.misc.is_pseudoprime` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L575
  - [ ] `sage.arith.misc.is_pseudoprime_power` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L680
  - [ ] `sage.arith.misc.is_square` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2928
  - [ ] `sage.arith.misc.is_squarefree` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3007
  - [ ] `sage.arith.misc.jacobi_symbol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4318
  - [ ] `sage.arith.misc.kronecker` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4218
  - [ ] `sage.arith.misc.kronecker_symbol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4218
  - [ ] `sage.arith.misc.legendre_symbol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4266
  - [ ] `sage.arith.misc.mqrr_rational_reconstruction` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2458
  - [ ] `sage.arith.misc.multinomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3995
  - [ ] `sage.arith.misc.multinomial_coefficients` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4108
  - [ ] `sage.arith.misc.next_prime` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1200
  - [ ] `sage.arith.misc.next_prime_power` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1108
  - [ ] `sage.arith.misc.next_probable_prime` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1168
  - [ ] `sage.arith.misc.nth_prime` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4490
  - [ ] `sage.arith.misc.number_of_divisors` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4833
  - [ ] `sage.arith.misc.odd_part` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2858
  - [ ] `sage.arith.misc.power_mod` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2285
  - [ ] `sage.arith.misc.previous_prime` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1249
  - [ ] `sage.arith.misc.previous_prime_power` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1303
  - [ ] `sage.arith.misc.prime_divisors` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2802
  - [ ] `sage.arith.misc.prime_factors` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2802
  - [ ] `sage.arith.misc.prime_powers` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L809
  - [ ] `sage.arith.misc.prime_to_m_part` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2886
  - [ ] `sage.arith.misc.primes` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1020
  - [ ] `sage.arith.misc.primes_first_n` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L931
  - [ ] `sage.arith.misc.primitive_root` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4369
  - [ ] `sage.arith.misc.quadratic_residues` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4533
  - [ ] `sage.arith.misc.radical` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2762
  - [ ] `sage.arith.misc.random_prime` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1376
  - [ ] `sage.arith.misc.rational_reconstruction` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2360
  - [ ] `sage.arith.misc.rising_factorial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5224
  - [ ] `sage.arith.misc.smooth_part` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6421
  - [ ] `sage.arith.misc.sort_complex_numbers_for_display` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6011
  - [ ] `sage.arith.misc.squarefree_divisors` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6110
  - [ ] `sage.arith.misc.subfactorial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5858
  - [ ] `sage.arith.misc.sum_of_k_squares` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5752
  - [ ] `sage.arith.misc.three_squares` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5531
  - [ ] `sage.arith.misc.trial_division` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2518
  - [ ] `sage.arith.misc.two_squares` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5412
  - [ ] `sage.arith.misc.valuation` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L738
  - [ ] `sage.arith.misc.xgcd` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1939
  - [ ] `sage.arith.misc.xkcd` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2136
  - [ ] `sage.arith.misc.xlcm` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1899
  - [ ] `sage.arith.multi_modular` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L851
  - [ ] `sage.arith.multi_modular.MultiModularBasis` (class) extends MultiModularBasis_base
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L851
  - [ ] `sage.arith.multi_modular.MultiModularBasis_base` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L37
  - [ ] `sage.arith.multi_modular.MutableMultiModularBasis` (class) extends MultiModularBasis
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L917
  - [ ] `sage.arith.power` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/power.pyx#L24
  - [ ] `sage.arith.power.generic_power` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/power.pyx#L24
  - [ ] `sage.arith.srange` (module)
  - [ ] `sage.arith.srange.ellipsis_iter` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L321
  - [ ] `sage.arith.srange.ellipsis_range` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L449
  - [ ] `sage.arith.srange.srange` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L175
  - [ ] `sage.arith.srange.xsrange` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L30

## sage.calculus (98 features)

  - [ ] `sage.calculus` (module)
  - [ ] `sage.calculus.calculus` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1964
  - [ ] `sage.calculus.calculus.at` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1964
  - [ ] `sage.calculus.calculus.dummy_diff` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2042
  - [ ] `sage.calculus.calculus.dummy_integrate` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2067
  - [ ] `sage.calculus.calculus.dummy_inverse_laplace` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2104
  - [ ] `sage.calculus.calculus.dummy_laplace` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2088
  - [ ] `sage.calculus.calculus.dummy_pochhammer` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2120
  - [ ] `sage.calculus.calculus.inverse_laplace` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1779
  - [ ] `sage.calculus.calculus.laplace` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1537
  - [ ] `sage.calculus.calculus.lim` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1159
  - [ ] `sage.calculus.calculus.limit` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1159
  - [ ] `sage.calculus.calculus.mapped_opts` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2427
  - [ ] `sage.calculus.calculus.maxima_options` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2455
  - [ ] `sage.calculus.calculus.minpoly` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L936
  - [ ] `sage.calculus.calculus.mma_free_limit` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1488
  - [ ] `sage.calculus.calculus.nintegral` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L683
  - [ ] `sage.calculus.calculus.nintegrate` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L683
  - [ ] `sage.calculus.calculus.symbolic_expression_from_maxima_string` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2236
  - [ ] `sage.calculus.calculus.symbolic_expression_from_string` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2564
  - [ ] `sage.calculus.calculus.symbolic_product` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L828
  - [ ] `sage.calculus.calculus.symbolic_sum` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L445
  - [ ] `sage.calculus.desolvers` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L179
  - [ ] `sage.calculus.desolvers.desolve` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L179
  - [ ] `sage.calculus.desolvers.desolve_laplace` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L661
  - [ ] `sage.calculus.desolvers.desolve_mintides` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1684
  - [ ] `sage.calculus.desolvers.desolve_odeint` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1486
  - [ ] `sage.calculus.desolvers.desolve_rk4` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1228
  - [ ] `sage.calculus.desolvers.desolve_rk4_determine_bounds` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1184
  - [ ] `sage.calculus.desolvers.desolve_system` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L789
  - [ ] `sage.calculus.desolvers.desolve_system_rk4` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1382
  - [ ] `sage.calculus.desolvers.desolve_tides_mpfr` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1771
  - [ ] `sage.calculus.desolvers.eulers_method` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L950
  - [ ] `sage.calculus.desolvers.eulers_method_2x2` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1038
  - [ ] `sage.calculus.desolvers.eulers_method_2x2_plot` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1141
  - [ ] `sage.calculus.desolvers.fricas_desolve` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L88
  - [ ] `sage.calculus.desolvers.fricas_desolve_system` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L125
  - [ ] `sage.calculus.expr` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/expr.py#L14
  - [ ] `sage.calculus.expr.symbolic_expression` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/expr.py#L14
  - [ ] `sage.calculus.functional` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L69
  - [ ] `sage.calculus.functional.derivative` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L69
  - [ ] `sage.calculus.functional.diff` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L69
  - [ ] `sage.calculus.functional.expand` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L425
  - [ ] `sage.calculus.functional.integral` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L175
  - [ ] `sage.calculus.functional.integrate` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L175
  - [ ] `sage.calculus.functional.lim` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L325
  - [ ] `sage.calculus.functional.limit` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L325
  - [ ] `sage.calculus.functional.simplify` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L32
  - [ ] `sage.calculus.functional.taylor` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L381
  - [ ] `sage.calculus.functions` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functions.py#L109
  - [ ] `sage.calculus.functions.jacobian` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functions.py#L109
  - [ ] `sage.calculus.functions.wronskian` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functions.py#L13
  - [ ] `sage.calculus.integration` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/integration.pyx#L443
  - [ ] `sage.calculus.integration.PyFunctionWrapper` (class) extends object
  - [ ] `sage.calculus.integration.compiled_integrand` (class) extends object
  - [ ] `sage.calculus.integration.monte_carlo_integral` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/integration.pyx#L443
  - [ ] `sage.calculus.integration.numerical_integral` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/integration.pyx#L74
  - [ ] `sage.calculus.interpolation` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolation.pyx#L9
  - [ ] `sage.calculus.interpolation.Spline` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolation.pyx#L9
  - [ ] `sage.calculus.interpolation.spline` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolation.pyx#L9
  - [ ] `sage.calculus.interpolators` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L207
  - [ ] `sage.calculus.interpolators.CCSpline` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L207
  - [ ] `sage.calculus.interpolators.PSpline` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L71
  - [ ] `sage.calculus.interpolators.complex_cubic_spline` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L172
  - [ ] `sage.calculus.interpolators.polygon_spline` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L37
  - [ ] `sage.calculus.ode` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/ode.pyx#L106
  - [ ] `sage.calculus.ode.PyFunctionWrapper` (class) extends object
  - [ ] `sage.calculus.ode.ode_solver` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/ode.pyx#L106
  - [ ] `sage.calculus.ode.ode_system` (class) extends object
  - [ ] `sage.calculus.riemann` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L70
  - [ ] `sage.calculus.riemann.Riemann_Map` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L70
  - [ ] `sage.calculus.riemann.analytic_boundary` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1371
  - [ ] `sage.calculus.riemann.analytic_interior` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1467
  - [ ] `sage.calculus.riemann.cauchy_kernel` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1421
  - [ ] `sage.calculus.riemann.complex_to_rgb` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1266
  - [ ] `sage.calculus.riemann.complex_to_spiderweb` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1147
  - [ ] `sage.calculus.riemann.get_derivatives` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1092
  - [ ] `sage.calculus.test_sympy` (module)
  - [ ] `sage.calculus.tests` (module)
  - [ ] `sage.calculus.transforms.dft` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dft.py#L90
  - [ ] `sage.calculus.transforms.dft.IndexedSequence` (class) extends L, SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dft.py#L90
  - [ ] `sage.calculus.transforms.dwt` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L24
  - [ ] `sage.calculus.transforms.dwt.DWT` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L24
  - [ ] `sage.calculus.transforms.dwt.DiscreteWaveletTransform` (class) extends GSLDoubleArray
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L102
  - [ ] `sage.calculus.transforms.dwt.WaveletTransform` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L24
  - [ ] `sage.calculus.transforms.dwt.is2pow` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L156
  - [ ] `sage.calculus.transforms.fft` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L29
  - [ ] `sage.calculus.transforms.fft.FFT` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L29
  - [ ] `sage.calculus.transforms.fft.FastFourierTransform` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L29
  - [ ] `sage.calculus.transforms.fft.FastFourierTransform_base` (class) extends object
  - [ ] `sage.calculus.transforms.fft.FastFourierTransform_complex` (class) extends FastFourierTransform_base
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L85
  - [ ] `sage.calculus.transforms.fft.FourierTransform_complex` (class) extends object
  - [ ] `sage.calculus.transforms.fft.FourierTransform_real` (class) extends object
  - [ ] `sage.calculus.var` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L362
  - [ ] `sage.calculus.var.clear_vars` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L362
  - [ ] `sage.calculus.var.function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L133
  - [ ] `sage.calculus.var.var` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L10
  - [ ] `sage.calculus.wester` (module)

## sage.categories (37 features)

  - [ ] `sage.categories.fields` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L27
  - [ ] `sage.categories.fields.ElementMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L652
  - [ ] `sage.categories.fields.Fields` (class) extends CategoryWithAxiom_singleton
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L27
  - [ ] `sage.categories.fields.ParentMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L194
  - [ ] `sage.categories.groups` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L24
  - [ ] `sage.categories.groups.CartesianProducts` (class) extends CartesianProductsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L548
  - [ ] `sage.categories.groups.Commutative` (class) extends CategoryWithAxiom
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L496
  - [ ] `sage.categories.groups.ElementMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L454
  - [ ] `sage.categories.groups.Groups` (class) extends CategoryWithAxiom_singleton
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L24
  - [ ] `sage.categories.groups.ParentMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L573
  - [ ] `sage.categories.groups.Topological` (class) extends TopologicalSpacesCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L655
  - [ ] `sage.categories.modules` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L34
  - [ ] `sage.categories.modules.CartesianProducts` (class) extends CartesianProductsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L835
  - [ ] `sage.categories.modules.ElementMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L918
  - [ ] `sage.categories.modules.Endset` (class) extends CategoryWithAxiom_over_base_ring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L813
  - [ ] `sage.categories.modules.FiniteDimensional` (class) extends CategoryWithAxiom_over_base_ring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L515
  - [ ] `sage.categories.modules.FinitelyPresented` (class) extends CategoryWithAxiom_over_base_ring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L564
  - [ ] `sage.categories.modules.Homsets` (class) extends HomsetsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L724
  - [ ] `sage.categories.modules.Modules` (class) extends Category_module
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L34
  - [ ] `sage.categories.modules.ParentMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L860
  - [ ] `sage.categories.modules.SubcategoryMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L199
  - [ ] `sage.categories.modules.TensorProducts` (class) extends TensorProductsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L547
  - [ ] `sage.categories.modules_with_basis` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L45
  - [ ] `sage.categories.modules_with_basis.CartesianProducts` (class) extends CartesianProductsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2547
  - [ ] `sage.categories.modules_with_basis.DualObjects` (class) extends DualObjectsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2753
  - [ ] `sage.categories.modules_with_basis.ElementMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L1441
  - [ ] `sage.categories.modules_with_basis.Homsets` (class) extends HomsetsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2429
  - [ ] `sage.categories.modules_with_basis.ModulesWithBasis` (class) extends CategoryWithAxiom_over_base_ring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L45
  - [ ] `sage.categories.modules_with_basis.MorphismMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2500
  - [ ] `sage.categories.modules_with_basis.ParentMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2566
  - [ ] `sage.categories.modules_with_basis.TensorProducts` (class) extends TensorProductsCategory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2592
  - [ ] `sage.categories.rings` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L27
  - [ ] `sage.categories.rings.ElementMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L1598
  - [ ] `sage.categories.rings.MorphismMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L66
  - [ ] `sage.categories.rings.ParentMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L317
  - [ ] `sage.categories.rings.Rings` (class) extends CategoryWithAxiom_singleton
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L27
  - [ ] `sage.categories.rings.SubcategoryMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L266

## sage.functions (68 features)

  - [ ] `sage.functions.exp_integral` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L882
  - [ ] `sage.functions.exp_integral.Function_cos_integral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L882
  - [ ] `sage.functions.exp_integral.Function_cosh_integral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1168
  - [ ] `sage.functions.exp_integral.Function_exp_integral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1302
  - [ ] `sage.functions.exp_integral.Function_exp_integral_e` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L72
  - [ ] `sage.functions.exp_integral.Function_exp_integral_e1` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L263
  - [ ] `sage.functions.exp_integral.Function_log_integral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L370
  - [ ] `sage.functions.exp_integral.Function_log_integral_offset` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L499
  - [ ] `sage.functions.exp_integral.Function_sin_integral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L691
  - [ ] `sage.functions.exp_integral.Function_sinh_integral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1021
  - [ ] `sage.functions.exp_integral.exponential_integral_1` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1412
  - [ ] `sage.functions.hyperbolic` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L432
  - [ ] `sage.functions.hyperbolic.Function_arccosh` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L432
  - [ ] `sage.functions.hyperbolic.Function_arccoth` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L584
  - [ ] `sage.functions.hyperbolic.Function_arccsch` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L694
  - [ ] `sage.functions.hyperbolic.Function_arcsech` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L646
  - [ ] `sage.functions.hyperbolic.Function_arcsinh` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L369
  - [ ] `sage.functions.hyperbolic.Function_arctanh` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L523
  - [ ] `sage.functions.hyperbolic.Function_cosh` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L87
  - [ ] `sage.functions.hyperbolic.Function_coth` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L193
  - [ ] `sage.functions.hyperbolic.Function_csch` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L309
  - [ ] `sage.functions.hyperbolic.Function_sech` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L252
  - [ ] `sage.functions.hyperbolic.Function_sinh` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L49
  - [ ] `sage.functions.hyperbolic.Function_tanh` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L125
  - [ ] `sage.functions.log` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L449
  - [ ] `sage.functions.log.Function_dilog` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L449
  - [ ] `sage.functions.log.Function_exp` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L41
  - [ ] `sage.functions.log.Function_exp_polar` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L916
  - [ ] `sage.functions.log.Function_harmonic_number` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L1306
  - [ ] `sage.functions.log.Function_harmonic_number_generalized` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L1027
  - [ ] `sage.functions.log.Function_lambert_w` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L564
  - [ ] `sage.functions.log.Function_log1` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L184
  - [ ] `sage.functions.log.Function_log2` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L266
  - [ ] `sage.functions.log.Function_polylog` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L301
  - [ ] `sage.functions.other` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L641
  - [ ] `sage.functions.other.Function_Order` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L641
  - [ ] `sage.functions.other.Function_abs` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L42
  - [ ] `sage.functions.other.Function_arg` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L964
  - [ ] `sage.functions.other.Function_binomial` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1567
  - [ ] `sage.functions.other.Function_cases` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L2006
  - [ ] `sage.functions.other.Function_ceil` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L300
  - [ ] `sage.functions.other.Function_conjugate` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1283
  - [ ] `sage.functions.other.Function_crootof` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L2126
  - [ ] `sage.functions.other.Function_elementof` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L2219
  - [ ] `sage.functions.other.Function_factorial` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1372
  - [ ] `sage.functions.other.Function_floor` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L473
  - [ ] `sage.functions.other.Function_frac` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L706
  - [ ] `sage.functions.other.Function_imag_part` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1216
  - [ ] `sage.functions.other.Function_limit` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1900
  - [ ] `sage.functions.other.Function_prod` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1835
  - [ ] `sage.functions.other.Function_real_nth_root` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L789
  - [ ] `sage.functions.other.Function_real_part` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1119
  - [ ] `sage.functions.other.Function_sqrt` (class) extends object
  - [ ] `sage.functions.other.Function_sum` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1778
  - [ ] `sage.functions.trig` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L589
  - [ ] `sage.functions.trig.Function_arccos` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L589
  - [ ] `sage.functions.trig.Function_arccot` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L732
  - [ ] `sage.functions.trig.Function_arccsc` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L797
  - [ ] `sage.functions.trig.Function_arcsec` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L858
  - [ ] `sage.functions.trig.Function_arcsin` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L525
  - [ ] `sage.functions.trig.Function_arctan` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L658
  - [ ] `sage.functions.trig.Function_arctan2` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L921
  - [ ] `sage.functions.trig.Function_cos` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L138
  - [ ] `sage.functions.trig.Function_cot` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L272
  - [ ] `sage.functions.trig.Function_csc` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L447
  - [ ] `sage.functions.trig.Function_sec` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L373
  - [ ] `sage.functions.trig.Function_sin` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L9
  - [ ] `sage.functions.trig.Function_tan` (class) extends GinacFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L206

## sage.libs (1 features)

  - [ ] `sage.libs.flint.arith_sage` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/libs/flint/arith_sage.pyx#L28

## sage.rings (453 features)

  - [ ] `sage.rings.polynomial.complex_roots` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L159
  - [ ] `sage.rings.polynomial.complex_roots.complex_roots` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L159
  - [ ] `sage.rings.polynomial.complex_roots.interval_roots` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L51
  - [ ] `sage.rings.polynomial.complex_roots.intervals_disjoint` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L94
  - [ ] `sage.rings.polynomial.convolution` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/convolution.py#L57
  - [ ] `sage.rings.polynomial.convolution.convolution` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/convolution.py#L57
  - [ ] `sage.rings.polynomial.cyclotomic` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L387
  - [ ] `sage.rings.polynomial.cyclotomic.bateman_bound` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L387
  - [ ] `sage.rings.polynomial.cyclotomic.cyclotomic_coeffs` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L44
  - [ ] `sage.rings.polynomial.cyclotomic.cyclotomic_value` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L201
  - [ ] `sage.rings.polynomial.flatten` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L48
  - [ ] `sage.rings.polynomial.flatten.FlatteningMorphism` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L48
  - [ ] `sage.rings.polynomial.flatten.FractionSpecializationMorphism` (class) extends D, Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L663
  - [ ] `sage.rings.polynomial.flatten.SpecializationMorphism` (class) extends D, Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L417
  - [ ] `sage.rings.polynomial.flatten.UnflatteningMorphism` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L286
  - [ ] `sage.rings.polynomial.groebner_fan` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L772
  - [ ] `sage.rings.polynomial.groebner_fan.GroebnerFan` (class) extends I, SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L772
  - [ ] `sage.rings.polynomial.groebner_fan.InitialForm` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L534
  - [ ] `sage.rings.polynomial.groebner_fan.PolyhedralCone` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L157
  - [ ] `sage.rings.polynomial.groebner_fan.PolyhedralFan` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L284
  - [ ] `sage.rings.polynomial.groebner_fan.ReducedGroebnerBasis` (class) extends SageObject, list
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L1762
  - [ ] `sage.rings.polynomial.groebner_fan.TropicalPrevariety` (class) extends PolyhedralFan
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L645
  - [ ] `sage.rings.polynomial.groebner_fan.ideal_to_gfan_format` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L744
  - [ ] `sage.rings.polynomial.groebner_fan.max_degree` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L104
  - [ ] `sage.rings.polynomial.groebner_fan.prefix_check` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L83
  - [ ] `sage.rings.polynomial.groebner_fan.ring_to_gfan_format` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L721
  - [ ] `sage.rings.polynomial.groebner_fan.verts_for_normal` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L623
  - [ ] `sage.rings.polynomial.hilbert` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L36
  - [ ] `sage.rings.polynomial.hilbert.Node` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L36
  - [ ] `sage.rings.polynomial.hilbert.first_hilbert_series` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L425
  - [ ] `sage.rings.polynomial.hilbert.hilbert_poincare_series` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L556
  - [ ] `sage.rings.polynomial.ideal` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ideal.py#L23
  - [ ] `sage.rings.polynomial.ideal.Ideal_1poly_field` (class) extends Ideal_pid
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ideal.py#L23
  - [ ] `sage.rings.polynomial.infinite_polynomial_element` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L114
  - [ ] `sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial` (class) extends A, CommutativePolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L114
  - [ ] `sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial_dense` (class) extends A, InfinitePolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L1582
  - [ ] `sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial_sparse` (class) extends A, InfinitePolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L1286
  - [ ] `sage.rings.polynomial.infinite_polynomial_ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L497
  - [ ] `sage.rings.polynomial.infinite_polynomial_ring.GenDictWithBasering` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L497
  - [ ] `sage.rings.polynomial.infinite_polynomial_ring.InfiniteGenDict` (class) extends Gens, object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L390
  - [ ] `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialGen` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L1376
  - [ ] `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRingFactory` (class) extends UniqueFactory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L279
  - [ ] `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRing_dense` (class) extends R, InfinitePolynomialRing_sparse
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L1537
  - [ ] `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRing_sparse` (class) extends R, CommutativeRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L621
  - [ ] `sage.rings.polynomial.integer_valued_polynomials` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L34
  - [ ] `sage.rings.polynomial.integer_valued_polynomials.Bases` (class) extends Category_realization_of_parent
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L119
  - [ ] `sage.rings.polynomial.integer_valued_polynomials.Binomial` (class) extends A, CombinatorialFreeModule, BindableClass
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L898
  - [ ] `sage.rings.polynomial.integer_valued_polynomials.Element` (class) extends IndexedFreeModuleElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L1188
  - [ ] `sage.rings.polynomial.integer_valued_polynomials.ElementMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L282
  - [ ] `sage.rings.polynomial.integer_valued_polynomials.IntegerValuedPolynomialRing` (class) extends R, UniqueRepresentation, Parent
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L34
  - [ ] `sage.rings.polynomial.integer_valued_polynomials.ParentMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L144
  - [ ] `sage.rings.polynomial.integer_valued_polynomials.Shifted` (class) extends A, CombinatorialFreeModule, BindableClass
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L393
  - [ ] `sage.rings.polynomial.laurent_polynomial` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial.pyx#L21
  - [ ] `sage.rings.polynomial.laurent_polynomial.LaurentPolynomial` (class) extends CommutativeAlgebraElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial.pyx#L21
  - [ ] `sage.rings.polynomial.laurent_polynomial.LaurentPolynomial_univariate` (class) extends LaurentPolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial.pyx#L303
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L82
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L82
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing_mpair` (class) extends R, LaurentPolynomialRing_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L590
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing_univariate` (class) extends R, LaurentPolynomialRing_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L432
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring.from_fraction_field` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L399
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring.is_LaurentPolynomialRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L52
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring_base` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring_base.py#L34
  - [ ] `sage.rings.polynomial.laurent_polynomial_ring_base.LaurentPolynomialRing_generic` (class) extends R, CommutativeRing, Parent
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring_base.py#L34
  - [ ] `sage.rings.polynomial.msolve` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/msolve.py#L68
  - [ ] `sage.rings.polynomial.msolve.groebner_basis_degrevlex` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/msolve.py#L68
  - [ ] `sage.rings.polynomial.msolve.variety` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/msolve.py#L117
  - [ ] `sage.rings.polynomial.multi_polynomial` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L369
  - [ ] `sage.rings.polynomial.multi_polynomial.MPolynomial` (class) extends CommutativePolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L369
  - [ ] `sage.rings.polynomial.multi_polynomial.MPolynomial_libsingular` (class) extends MPolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L2993
  - [ ] `sage.rings.polynomial.multi_polynomial.is_MPolynomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L24
  - [ ] `sage.rings.polynomial.multi_polynomial_element` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L77
  - [ ] `sage.rings.polynomial.multi_polynomial_element.MPolynomial_element` (class) extends MPolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L77
  - [ ] `sage.rings.polynomial.multi_polynomial_element.MPolynomial_polydict` (class) extends Polynomial_singular_repr, MPolynomial_element
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L419
  - [ ] `sage.rings.polynomial.multi_polynomial_element.degree_lowest_rational_function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L2517
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3876
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal` (class) extends MPolynomialIdeal_singular_repr, MPolynomialIdeal_macaulay2_repr, MPolynomialIdeal_magma_repr, Ideal_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3876
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_macaulay2_repr` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3382
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_magma_repr` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L349
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_quotient` (class) extends QuotientRingIdeal_generic, MPolynomialIdeal
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L5646
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_singular_base_repr` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L454
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_singular_repr` (class) extends MPolynomialIdeal_singular_base_repr
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L617
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.NCPolynomialIdeal` (class) extends MPolynomialIdeal_singular_repr, Ideal_nc
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3495
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.RequireField` (class) extends MethodDecorator
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L273
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.basis` (attribute)
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.is_MPolynomialIdeal` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L310
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal.require_field` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L273
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal_libsingular` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L278
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal_libsingular.interred_libsingular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L278
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal_libsingular.kbase_libsingular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L150
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal_libsingular.slimgb_libsingular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L238
  - [ ] `sage.rings.polynomial.multi_polynomial_ideal_libsingular.std_libsingular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L207
  - [ ] `sage.rings.polynomial.multi_polynomial_libsingular` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L264
  - [ ] `sage.rings.polynomial.multi_polynomial_libsingular.MPolynomialRing_libsingular` (class) extends MPolynomialRing_base
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L264
  - [ ] `sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular` (class) extends MPolynomial_libsingular
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L1896
  - [ ] `sage.rings.polynomial.multi_polynomial_libsingular.unpickle_MPolynomialRing_libsingular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L1880
  - [ ] `sage.rings.polynomial.multi_polynomial_libsingular.unpickle_MPolynomial_libsingular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L6179
  - [ ] `sage.rings.polynomial.multi_polynomial_ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L79
  - [ ] `sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_macaulay2_repr` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L79
  - [ ] `sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_polydict` (class) extends MPolynomialRing_macaulay2_repr, PolynomialRing_singular_repr, MPolynomialRing_base
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L97
  - [ ] `sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_polydict_domain` (class) extends MPolynomialRing_polydict
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L937
  - [ ] `sage.rings.polynomial.multi_polynomial_ring_base` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1805
  - [ ] `sage.rings.polynomial.multi_polynomial_ring_base.BooleanPolynomialRing_base` (class) extends MPolynomialRing_base
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1805
  - [ ] `sage.rings.polynomial.multi_polynomial_ring_base.MPolynomialRing_base` (class) extends CommutativeRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L40
  - [ ] `sage.rings.polynomial.multi_polynomial_ring_base.is_MPolynomialRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L32
  - [ ] `sage.rings.polynomial.multi_polynomial_ring_base.unpickle_MPolynomialRing_generic` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1835
  - [ ] `sage.rings.polynomial.multi_polynomial_ring_base.unpickle_MPolynomialRing_generic_v1` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1830
  - [ ] `sage.rings.polynomial.multi_polynomial_sequence` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L212
  - [ ] `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L212
  - [ ] `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_generic` (class) extends Sequence_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L397
  - [ ] `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_gf2` (class) extends PolynomialSequence_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L1311
  - [ ] `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_gf2e` (class) extends PolynomialSequence_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L1826
  - [ ] `sage.rings.polynomial.multi_polynomial_sequence.is_PolynomialSequence` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L186
  - [ ] `sage.rings.polynomial.omega` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L56
  - [ ] `sage.rings.polynomial.omega.MacMahonOmega` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L56
  - [ ] `sage.rings.polynomial.omega.Omega_ge` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L470
  - [ ] `sage.rings.polynomial.omega.homogeneous_symmetric_function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L949
  - [ ] `sage.rings.polynomial.ore_function_element` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L668
  - [ ] `sage.rings.polynomial.ore_function_element.ConstantOreFunctionSection` (class) extends Map
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L668
  - [ ] `sage.rings.polynomial.ore_function_element.OreFunction` (class) extends AlgebraElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L28
  - [ ] `sage.rings.polynomial.ore_function_element.OreFunctionBaseringInjection` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L718
  - [ ] `sage.rings.polynomial.ore_function_element.OreFunction_with_large_center` (class) extends OreFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L815
  - [ ] `sage.rings.polynomial.ore_function_field` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L773
  - [ ] `sage.rings.polynomial.ore_function_field.OreFunctionCenterInjection` (class) extends RingHomomorphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L773
  - [ ] `sage.rings.polynomial.ore_function_field.OreFunctionField` (class) extends Parent, UniqueRepresentation
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L186
  - [ ] `sage.rings.polynomial.ore_function_field.OreFunctionField_with_large_center` (class) extends OreFunctionField
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L876
  - [ ] `sage.rings.polynomial.ore_function_field.SectionOreFunctionCenterInjection` (class) extends Section
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L694
  - [ ] `sage.rings.polynomial.ore_polynomial_element` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L2968
  - [ ] `sage.rings.polynomial.ore_polynomial_element.ConstantOrePolynomialSection` (class) extends Map
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L2968
  - [ ] `sage.rings.polynomial.ore_polynomial_element.OrePolynomial` (class) extends AlgebraElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L50
  - [ ] `sage.rings.polynomial.ore_polynomial_element.OrePolynomialBaseringInjection` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L3021
  - [ ] `sage.rings.polynomial.ore_polynomial_element.OrePolynomial_generic_dense` (class) extends OrePolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L2212
  - [ ] `sage.rings.polynomial.ore_polynomial_ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_ring.py#L73
  - [ ] `sage.rings.polynomial.ore_polynomial_ring.OrePolynomialRing` (class) extends UniqueRepresentation, Parent
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_ring.py#L73
  - [ ] `sage.rings.polynomial.padics.polynomial_padic` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic.py#L30
  - [ ] `sage.rings.polynomial.padics.polynomial_padic.Polynomial_padic` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic.py#L30
  - [ ] `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_capped_relative_dense.py#L39
  - [ ] `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense.Polynomial_padic_capped_relative_dense` (class) extends Polynomial_generic_cdv, Polynomial_padic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_capped_relative_dense.py#L39
  - [ ] `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense.make_padic_poly` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_capped_relative_dense.py#L1319
  - [ ] `sage.rings.polynomial.padics.polynomial_padic_flat` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_flat.py#L18
  - [ ] `sage.rings.polynomial.padics.polynomial_padic_flat.Polynomial_padic_flat` (class) extends Polynomial_generic_dense, Polynomial_padic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_flat.py#L18
  - [ ] `sage.rings.polynomial.pbori.pbori` (module)
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleConstant` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7766
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleSet` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5237
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleSetIterator` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5863
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanMonomial` (class) extends MonoidElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L2211
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanMonomialIterator` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L2874
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanMonomialMonoid` (class) extends UniqueRepresentation, Monoid_class
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L1848
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanMonomialVariableIterator` (class) extends object
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanMulAction` (class) extends Action
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L78
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanPolynomial` (class) extends MPolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L2922
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanPolynomialEntry` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6425
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanPolynomialIdeal` (class) extends MPolynomialIdeal
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4801
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanPolynomialIterator` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4750
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanPolynomialRing` (class) extends BooleanPolynomialRing_base
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L254
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5960
  - [ ] `sage.rings.polynomial.pbori.pbori.BooleanPolynomialVectorIterator` (class) extends object
  - [ ] `sage.rings.polynomial.pbori.pbori.CCuddNavigator` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5928
  - [ ] `sage.rings.polynomial.pbori.pbori.FGLMStrategy` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6430
  - [ ] `sage.rings.polynomial.pbori.pbori.GroebnerStrategy` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6502
  - [ ] `sage.rings.polynomial.pbori.pbori.MonomialConstruct` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4687
  - [ ] `sage.rings.polynomial.pbori.pbori.MonomialFactory` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7955
  - [ ] `sage.rings.polynomial.pbori.pbori.PolynomialConstruct` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4636
  - [ ] `sage.rings.polynomial.pbori.pbori.PolynomialFactory` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L8028
  - [ ] `sage.rings.polynomial.pbori.pbori.ReductionStrategy` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6153
  - [ ] `sage.rings.polynomial.pbori.pbori.TermOrder_from_pb_order` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7891
  - [ ] `sage.rings.polynomial.pbori.pbori.VariableBlock` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7014
  - [ ] `sage.rings.polynomial.pbori.pbori.VariableConstruct` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4725
  - [ ] `sage.rings.polynomial.pbori.pbori.VariableFactory` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7907
  - [ ] `sage.rings.polynomial.pbori.pbori.add_up_polynomials` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7030
  - [ ] `sage.rings.polynomial.pbori.pbori.contained_vars` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7277
  - [ ] `sage.rings.polynomial.pbori.pbori.easy_linear_factors` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7715
  - [ ] `sage.rings.polynomial.pbori.pbori.gauss_on_polys` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7603
  - [ ] `sage.rings.polynomial.pbori.pbori.get_var_mapping` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L1794
  - [ ] `sage.rings.polynomial.pbori.pbori.if_then_else` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7412
  - [ ] `sage.rings.polynomial.pbori.pbori.interpolate` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7144
  - [ ] `sage.rings.polynomial.pbori.pbori.interpolate_smallest_lex` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7199
  - [ ] `sage.rings.polynomial.pbori.pbori.ll_red_nf_noredsb` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7340
  - [ ] `sage.rings.polynomial.pbori.pbori.ll_red_nf_noredsb_single_recursive_call` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7367
  - [ ] `sage.rings.polynomial.pbori.pbori.ll_red_nf_redsb` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7297
  - [ ] `sage.rings.polynomial.pbori.pbori.map_every_x_to_x_plus_one` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7080
  - [ ] `sage.rings.polynomial.pbori.pbori.mod_mon_set` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7400
  - [ ] `sage.rings.polynomial.pbori.pbori.mod_var_set` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7282
  - [ ] `sage.rings.polynomial.pbori.pbori.mult_fact_sim_C` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7286
  - [ ] `sage.rings.polynomial.pbori.pbori.nf3` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7051
  - [ ] `sage.rings.polynomial.pbori.pbori.p` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L69
  - [ ] `sage.rings.polynomial.pbori.pbori.parallel_reduce` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7406
  - [ ] `sage.rings.polynomial.pbori.pbori.random_set` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7695
  - [ ] `sage.rings.polynomial.pbori.pbori.recursively_insert` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7290
  - [ ] `sage.rings.polynomial.pbori.pbori.red_tail` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7056
  - [ ] `sage.rings.polynomial.pbori.pbori.reduction_strategy` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L84
  - [ ] `sage.rings.polynomial.pbori.pbori.set_random_seed` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7670
  - [ ] `sage.rings.polynomial.pbori.pbori.substitute_variables` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7631
  - [ ] `sage.rings.polynomial.pbori.pbori.top_index` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7495
  - [ ] `sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7720
  - [ ] `sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomial0` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7735
  - [ ] `sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomialRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7751
  - [ ] `sage.rings.polynomial.pbori.pbori.zeros` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7100
  - [ ] `sage.rings.polynomial.plural` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3115
  - [ ] `sage.rings.polynomial.plural.ExteriorAlgebra` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3115
  - [ ] `sage.rings.polynomial.plural.ExteriorAlgebra_plural` (class) extends NCPolynomialRing_plural
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L246
  - [ ] `sage.rings.polynomial.plural.G_AlgFactory` (class) extends UniqueFactory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L150
  - [ ] `sage.rings.polynomial.plural.NCPolynomialRing_plural` (class) extends Ring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L246
  - [ ] `sage.rings.polynomial.plural.NCPolynomial_plural` (class) extends RingElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L1422
  - [ ] `sage.rings.polynomial.plural.SCA` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3061
  - [ ] `sage.rings.polynomial.plural.new_CRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L2865
  - [ ] `sage.rings.polynomial.plural.new_NRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L2936
  - [ ] `sage.rings.polynomial.plural.new_Ring` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3005
  - [ ] `sage.rings.polynomial.plural.unpickle_NCPolynomial_plural` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L1383
  - [ ] `sage.rings.polynomial.polydict` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L1425
  - [ ] `sage.rings.polynomial.polydict.ETuple` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L1425
  - [ ] `sage.rings.polynomial.polydict.PolyDict` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L96
  - [ ] `sage.rings.polynomial.polydict.gen_index` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L47
  - [ ] `sage.rings.polynomial.polydict.make_ETuple` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L2696
  - [ ] `sage.rings.polynomial.polydict.make_PolyDict` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L2689
  - [ ] `sage.rings.polynomial.polydict.monomial_exponent` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L72
  - [ ] `sage.rings.polynomial.polynomial_compiled` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L21
  - [ ] `sage.rings.polynomial.polynomial_compiled.CompiledPolynomialFunction` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L21
  - [ ] `sage.rings.polynomial.polynomial_compiled.abc_pd` (class) extends binary_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L495
  - [ ] `sage.rings.polynomial.polynomial_compiled.add_pd` (class) extends binary_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L455
  - [ ] `sage.rings.polynomial.polynomial_compiled.binary_pd` (class) extends generic_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L455
  - [ ] `sage.rings.polynomial.polynomial_compiled.coeff_pd` (class) extends generic_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L404
  - [ ] `sage.rings.polynomial.polynomial_compiled.dummy_pd` (class) extends generic_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L371
  - [ ] `sage.rings.polynomial.polynomial_compiled.generic_pd` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L354
  - [ ] `sage.rings.polynomial.polynomial_compiled.mul_pd` (class) extends binary_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L455
  - [ ] `sage.rings.polynomial.polynomial_compiled.pow_pd` (class) extends unary_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L441
  - [ ] `sage.rings.polynomial.polynomial_compiled.sqr_pd` (class) extends unary_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L417
  - [ ] `sage.rings.polynomial.polynomial_compiled.unary_pd` (class) extends generic_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L417
  - [ ] `sage.rings.polynomial.polynomial_compiled.univar_pd` (class) extends generic_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L394
  - [ ] `sage.rings.polynomial.polynomial_compiled.var_pd` (class) extends generic_pd
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L383
  - [ ] `sage.rings.polynomial.polynomial_element` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12874
  - [ ] `sage.rings.polynomial.polynomial_element.ConstantPolynomialSection` (class) extends Map
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12874
  - [ ] `sage.rings.polynomial.polynomial_element.Polynomial` (class) extends CommutativePolynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L187
  - [ ] `sage.rings.polynomial.polynomial_element.PolynomialBaseringInjection` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12928
  - [ ] `sage.rings.polynomial.polynomial_element.Polynomial_generic_dense` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L11952
  - [ ] `sage.rings.polynomial.polynomial_element.Polynomial_generic_dense_inexact` (class) extends Polynomial_generic_dense
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12736
  - [ ] `sage.rings.polynomial.polynomial_element.generic_power_trunc` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12632
  - [ ] `sage.rings.polynomial.polynomial_element.is_Polynomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L135
  - [ ] `sage.rings.polynomial.polynomial_element.make_generic_polynomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12588
  - [ ] `sage.rings.polynomial.polynomial_element.polynomial_is_variable` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L13124
  - [ ] `sage.rings.polynomial.polynomial_element.universal_discriminant` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12592
  - [ ] `sage.rings.polynomial.polynomial_element_generic` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1155
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdv` (class) extends Polynomial_generic_domain
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1155
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdvf` (class) extends Polynomial_generic_cdv, Polynomial_generic_field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1610
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdvr` (class) extends Polynomial_generic_cdv
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1598
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdv` (class) extends Polynomial_generic_dense_inexact, Polynomial_generic_cdv
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1590
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdvf` (class) extends Polynomial_generic_dense_cdv, Polynomial_generic_cdvf
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1614
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdvr` (class) extends Polynomial_generic_dense_cdv, Polynomial_generic_cdvr
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1602
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_field` (class) extends Polynomial_generic_dense, Polynomial_generic_field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1146
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_domain` (class) extends Polynomial, IntegralDomainElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1053
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_field` (class) extends Polynomial_singular_repr, Polynomial_generic_domain, EuclideanDomainElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1087
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L52
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdv` (class) extends Polynomial_generic_sparse, Polynomial_generic_cdv
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1594
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdvf` (class) extends Polynomial_generic_sparse_cdv, Polynomial_generic_cdvf
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1618
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdvr` (class) extends Polynomial_generic_sparse_cdv, Polynomial_generic_cdvr
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1606
  - [ ] `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_field` (class) extends Polynomial_generic_sparse, Polynomial_generic_field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1131
  - [ ] `sage.rings.polynomial.polynomial_fateman` (module)
  - [ ] `sage.rings.polynomial.polynomial_gf2x` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L295
  - [ ] `sage.rings.polynomial.polynomial_gf2x.GF2X_BuildIrred_list` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L295
  - [ ] `sage.rings.polynomial.polynomial_gf2x.GF2X_BuildRandomIrred_list` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L340
  - [ ] `sage.rings.polynomial.polynomial_gf2x.GF2X_BuildSparseIrred_list` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L319
  - [ ] `sage.rings.polynomial.polynomial_gf2x.Polynomial_GF2X` (class) extends Polynomial_template
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L36
  - [ ] `sage.rings.polynomial.polynomial_gf2x.Polynomial_template` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L60
  - [ ] `sage.rings.polynomial.polynomial_gf2x.make_element` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L28
  - [ ] `sage.rings.polynomial.polynomial_integer_dense_flint` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_flint.pyx#L81
  - [ ] `sage.rings.polynomial.polynomial_integer_dense_flint.Polynomial_integer_dense_flint` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_flint.pyx#L81
  - [ ] `sage.rings.polynomial.polynomial_integer_dense_ntl` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_ntl.pyx#L74
  - [ ] `sage.rings.polynomial.polynomial_integer_dense_ntl.Polynomial_integer_dense_ntl` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_ntl.pyx#L74
  - [ ] `sage.rings.polynomial.polynomial_modn_dense_ntl` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L66
  - [ ] `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_mod_n` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L66
  - [ ] `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_mod_p` (class) extends Polynomial_dense_mod_n
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L1868
  - [ ] `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_modn_ntl_ZZ` (class) extends Polynomial_dense_mod_n
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L1296
  - [ ] `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_modn_ntl_zz` (class) extends Polynomial_dense_mod_n
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L671
  - [ ] `sage.rings.polynomial.polynomial_modn_dense_ntl.make_element` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L60
  - [ ] `sage.rings.polynomial.polynomial_modn_dense_ntl.small_roots` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L472
  - [ ] `sage.rings.polynomial.polynomial_number_field` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_number_field.pyx#L81
  - [ ] `sage.rings.polynomial.polynomial_number_field.Polynomial_absolute_number_field_dense` (class) extends Polynomial_generic_dense_field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_number_field.pyx#L81
  - [ ] `sage.rings.polynomial.polynomial_number_field.Polynomial_relative_number_field_dense` (class) extends Polynomial_generic_dense_field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_number_field.pyx#L215
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L65
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRingFactory` (class) extends UniqueFactory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L65
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_coercion` (class) extends DefaultConvertMap_unique
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L2170
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_domain` (class) extends PolynomialQuotientRing_generic, CommutativeRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L2280
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_field` (class) extends PolynomialQuotientRing_domain, Field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L2417
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_generic` (class) extends QuotientRing_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L274
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring.is_PolynomialQuotientRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L266
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring_element` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring_element.py#L93
  - [ ] `sage.rings.polynomial.polynomial_quotient_ring_element.PolynomialQuotientRingElement` (class) extends Polynomial_singular_repr, CommutativeRingElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring_element.py#L93
  - [ ] `sage.rings.polynomial.polynomial_rational_flint` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_rational_flint.pyx#L83
  - [ ] `sage.rings.polynomial.polynomial_rational_flint.Polynomial_rational_flint` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_rational_flint.pyx#L83
  - [ ] `sage.rings.polynomial.polynomial_real_mpfr_dense` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_real_mpfr_dense.pyx#L49
  - [ ] `sage.rings.polynomial.polynomial_real_mpfr_dense.PolynomialRealDense` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_real_mpfr_dense.pyx#L49
  - [ ] `sage.rings.polynomial.polynomial_real_mpfr_dense.make_PolynomialRealDense` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_real_mpfr_dense.pyx#L779
  - [ ] `sage.rings.polynomial.polynomial_ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3056
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_cdvf` (class) extends PolynomialRing_cdvr, PolynomialRing_field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3056
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_cdvr` (class) extends PolynomialRing_integral_domain
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3024
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_commutative` (class) extends PolynomialRing_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L1800
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_finite_field` (class) extends PolynomialRing_field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L2601
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_mod_n` (class) extends PolynomialRing_commutative
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3228
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_mod_p` (class) extends PolynomialRing_dense_finite_field, PolynomialRing_dense_mod_n, PolynomialRing_singular_repr
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3404
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_field_capped_relative` (class) extends PolynomialRing_dense_padic_field_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3207
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_field_generic` (class) extends PolynomialRing_cdvf
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3116
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_capped_absolute` (class) extends PolynomialRing_dense_padic_ring_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3166
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_capped_relative` (class) extends PolynomialRing_dense_padic_ring_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3145
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_fixed_mod` (class) extends PolynomialRing_dense_padic_ring_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3186
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_generic` (class) extends PolynomialRing_cdvr
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3087
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_field` (class) extends PolynomialRing_integral_domain
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L2159
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_generic` (class) extends Ring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L240
  - [ ] `sage.rings.polynomial.polynomial_ring.PolynomialRing_integral_domain` (class) extends PolynomialRing_commutative, PolynomialRing_singular_repr, CommutativeRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L1929
  - [ ] `sage.rings.polynomial.polynomial_ring.is_PolynomialRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L184
  - [ ] `sage.rings.polynomial.polynomial_ring.polygen` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3700
  - [ ] `sage.rings.polynomial.polynomial_ring.polygens` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3742
  - [ ] `sage.rings.polynomial.polynomial_ring_constructor` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L973
  - [ ] `sage.rings.polynomial.polynomial_ring_constructor.BooleanPolynomialRing_constructor` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L973
  - [ ] `sage.rings.polynomial.polynomial_ring_constructor.PolynomialRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L60
  - [ ] `sage.rings.polynomial.polynomial_ring_constructor.polynomial_default_category` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L909
  - [ ] `sage.rings.polynomial.polynomial_ring_constructor.unpickle_PolynomialRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L731
  - [ ] `sage.rings.polynomial.polynomial_ring_homomorphism` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_homomorphism.pyx#L21
  - [ ] `sage.rings.polynomial.polynomial_ring_homomorphism.PolynomialRingHomomorphism_from_base` (class) extends RingHomomorphism_from_base
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_homomorphism.pyx#L21
  - [ ] `sage.rings.polynomial.polynomial_singular_interface` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L172
  - [ ] `sage.rings.polynomial.polynomial_singular_interface.PolynomialRing_singular_repr` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L172
  - [ ] `sage.rings.polynomial.polynomial_singular_interface.Polynomial_singular_repr` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L451
  - [ ] `sage.rings.polynomial.polynomial_singular_interface.can_convert_to_singular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L381
  - [ ] `sage.rings.polynomial.polynomial_zmod_flint` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L60
  - [ ] `sage.rings.polynomial.polynomial_zmod_flint.Polynomial_template` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L60
  - [ ] `sage.rings.polynomial.polynomial_zmod_flint.Polynomial_zmod_flint` (class) extends Polynomial_template
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L71
  - [ ] `sage.rings.polynomial.polynomial_zmod_flint.make_element` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L28
  - [ ] `sage.rings.polynomial.polynomial_zz_pex` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L68
  - [ ] `sage.rings.polynomial.polynomial_zz_pex.Polynomial_ZZ_pEX` (class) extends Polynomial_template
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L68
  - [ ] `sage.rings.polynomial.polynomial_zz_pex.Polynomial_template` (class) extends Polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L60
  - [ ] `sage.rings.polynomial.polynomial_zz_pex.make_element` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L28
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L117
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.Bases` (class) extends Category_realization_of_parent
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L265
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.Binomial` (class) extends A, CombinatorialFreeModule, BindableClass
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L986
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.Element` (class) extends IndexedFreeModuleElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L1221
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.ElementMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L447
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.ParentMethods` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L293
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.QuantumValuedPolynomialRing` (class) extends R, UniqueRepresentation, Parent
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L117
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.Shifted` (class) extends A, CombinatorialFreeModule, BindableClass
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L533
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.q_binomial_x` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L73
  - [ ] `sage.rings.polynomial.q_integer_valued_polynomials.q_int_x` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L42
  - [ ] `sage.rings.polynomial.real_roots` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2067
  - [ ] `sage.rings.polynomial.real_roots.bernstein_down` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2067
  - [ ] `sage.rings.polynomial.real_roots.bernstein_expand` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4463
  - [ ] `sage.rings.polynomial.real_roots.bernstein_polynomial_factory` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2530
  - [ ] `sage.rings.polynomial.real_roots.bernstein_polynomial_factory_ar` (class) extends bernstein_polynomial_factory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2734
  - [ ] `sage.rings.polynomial.real_roots.bernstein_polynomial_factory_intlist` (class) extends bernstein_polynomial_factory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2570
  - [ ] `sage.rings.polynomial.real_roots.bernstein_polynomial_factory_ratlist` (class) extends bernstein_polynomial_factory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2649
  - [ ] `sage.rings.polynomial.real_roots.bernstein_up` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2109
  - [ ] `sage.rings.polynomial.real_roots.bitsize_doctest` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1968
  - [ ] `sage.rings.polynomial.real_roots.cl_maximum_root` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2325
  - [ ] `sage.rings.polynomial.real_roots.cl_maximum_root_first_lambda` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2201
  - [ ] `sage.rings.polynomial.real_roots.cl_maximum_root_local_max` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2289
  - [ ] `sage.rings.polynomial.real_roots.context` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4318
  - [ ] `sage.rings.polynomial.real_roots.de_casteljau_doublevec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1712
  - [ ] `sage.rings.polynomial.real_roots.de_casteljau_intvec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L995
  - [ ] `sage.rings.polynomial.real_roots.degree_reduction_next_size` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1972
  - [ ] `sage.rings.polynomial.real_roots.dprod_imatrow_vec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4560
  - [ ] `sage.rings.polynomial.real_roots.get_realfield_rndu` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4295
  - [ ] `sage.rings.polynomial.real_roots.interval_bernstein_polynomial` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L168
  - [ ] `sage.rings.polynomial.real_roots.interval_bernstein_polynomial_float` (class) extends interval_bernstein_polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1335
  - [ ] `sage.rings.polynomial.real_roots.interval_bernstein_polynomial_integer` (class) extends interval_bernstein_polynomial
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L437
  - [ ] `sage.rings.polynomial.real_roots.intvec_to_doublevec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1270
  - [ ] `sage.rings.polynomial.real_roots.island` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3260
  - [ ] `sage.rings.polynomial.real_roots.linear_map` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3782
  - [ ] `sage.rings.polynomial.real_roots.max_abs_doublevec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1797
  - [ ] `sage.rings.polynomial.real_roots.max_bitsize_intvec_doctest` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4556
  - [ ] `sage.rings.polynomial.real_roots.maximum_root_first_lambda` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2174
  - [ ] `sage.rings.polynomial.real_roots.maximum_root_local_max` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2264
  - [ ] `sage.rings.polynomial.real_roots.min_max_delta_intvec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4609
  - [ ] `sage.rings.polynomial.real_roots.min_max_diff_doublevec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4675
  - [ ] `sage.rings.polynomial.real_roots.min_max_diff_intvec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4644
  - [ ] `sage.rings.polynomial.real_roots.mk_context` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4379
  - [ ] `sage.rings.polynomial.real_roots.mk_ibpf` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1683
  - [ ] `sage.rings.polynomial.real_roots.mk_ibpi` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L978
  - [ ] `sage.rings.polynomial.real_roots.ocean` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2971
  - [ ] `sage.rings.polynomial.real_roots.precompute_degree_reduction_cache` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2010
  - [ ] `sage.rings.polynomial.real_roots.pseudoinverse` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2104
  - [ ] `sage.rings.polynomial.real_roots.rational_root_bounds` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2418
  - [ ] `sage.rings.polynomial.real_roots.real_roots` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3835
  - [ ] `sage.rings.polynomial.real_roots.relative_bounds` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1918
  - [ ] `sage.rings.polynomial.real_roots.reverse_intvec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4270
  - [ ] `sage.rings.polynomial.real_roots.root_bounds` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2347
  - [ ] `sage.rings.polynomial.real_roots.rr_gap` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3763
  - [ ] `sage.rings.polynomial.real_roots.scale_intvec_var` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4211
  - [ ] `sage.rings.polynomial.real_roots.split_for_targets` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2846
  - [ ] `sage.rings.polynomial.real_roots.subsample_vec_doctest` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2170
  - [ ] `sage.rings.polynomial.real_roots.taylor_shift1_intvec` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4239
  - [ ] `sage.rings.polynomial.real_roots.to_bernstein` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4395
  - [ ] `sage.rings.polynomial.real_roots.to_bernstein_warp` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4445
  - [ ] `sage.rings.polynomial.real_roots.warp_map` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3806
  - [ ] `sage.rings.polynomial.real_roots.wordsize_rational` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1821
  - [ ] `sage.rings.polynomial.refine_root` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/refine_root.pyx#L28
  - [ ] `sage.rings.polynomial.refine_root.refine_root` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/refine_root.pyx#L28
  - [ ] `sage.rings.polynomial.skew_polynomial_element` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_element.pyx#L62
  - [ ] `sage.rings.polynomial.skew_polynomial_element.SkewPolynomial_generic_dense` (class) extends OrePolynomial_generic_dense
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_element.pyx#L62
  - [ ] `sage.rings.polynomial.skew_polynomial_finite_field` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_field.pyx#L28
  - [ ] `sage.rings.polynomial.skew_polynomial_finite_field.SkewPolynomial_finite_field_dense` (class) extends SkewPolynomial_finite_order_dense
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_field.pyx#L28
  - [ ] `sage.rings.polynomial.skew_polynomial_finite_order` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_order.pyx#L28
  - [ ] `sage.rings.polynomial.skew_polynomial_finite_order.SkewPolynomial_finite_order_dense` (class) extends SkewPolynomial_generic_dense
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_order.pyx#L28
  - [ ] `sage.rings.polynomial.skew_polynomial_ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L369
  - [ ] `sage.rings.polynomial.skew_polynomial_ring.SectionSkewPolynomialCenterInjection` (class) extends Section
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L369
  - [ ] `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialCenterInjection` (class) extends RingHomomorphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L446
  - [ ] `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing` (class) extends OrePolynomialRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L218
  - [ ] `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing_finite_field` (class) extends SkewPolynomialRing_finite_order
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L764
  - [ ] `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing_finite_order` (class) extends SkewPolynomialRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L561
  - [ ] `sage.rings.polynomial.symmetric_ideal` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_ideal.py#L68
  - [ ] `sage.rings.polynomial.symmetric_ideal.SymmetricIdeal` (class) extends Ideal_generic
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_ideal.py#L68
  - [ ] `sage.rings.polynomial.symmetric_reduction` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_reduction.pyx#L126
  - [ ] `sage.rings.polynomial.symmetric_reduction.SymmetricReductionStrategy` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_reduction.pyx#L126
  - [ ] `sage.rings.polynomial.term_order` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/term_order.py#L543
  - [ ] `sage.rings.polynomial.term_order.TermOrder` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/term_order.py#L543
  - [ ] `sage.rings.polynomial.term_order.greater_tuple` (attribute)
  - [ ] `sage.rings.polynomial.term_order.sortkey` (attribute)
  - [ ] `sage.rings.polynomial.term_order.termorder_from_singular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/term_order.py#L2158
  - [ ] `sage.rings.polynomial.toy_buchberger` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L149
  - [ ] `sage.rings.polynomial.toy_buchberger.LCM` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L149
  - [ ] `sage.rings.polynomial.toy_buchberger.LM` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L150
  - [ ] `sage.rings.polynomial.toy_buchberger.LT` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L151
  - [ ] `sage.rings.polynomial.toy_buchberger.buchberger` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L175
  - [ ] `sage.rings.polynomial.toy_buchberger.buchberger_improved` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L231
  - [ ] `sage.rings.polynomial.toy_buchberger.inter_reduction` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L398
  - [ ] `sage.rings.polynomial.toy_buchberger.select` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L375
  - [ ] `sage.rings.polynomial.toy_buchberger.spol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L154
  - [ ] `sage.rings.polynomial.toy_buchberger.update` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L292
  - [ ] `sage.rings.polynomial.toy_d_basis` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L196
  - [ ] `sage.rings.polynomial.toy_d_basis.LC` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L196
  - [ ] `sage.rings.polynomial.toy_d_basis.LM` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L192
  - [ ] `sage.rings.polynomial.toy_d_basis.d_basis` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L200
  - [ ] `sage.rings.polynomial.toy_d_basis.gpol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L160
  - [ ] `sage.rings.polynomial.toy_d_basis.select` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L273
  - [ ] `sage.rings.polynomial.toy_d_basis.spol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L127
  - [ ] `sage.rings.polynomial.toy_d_basis.update` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L305
  - [ ] `sage.rings.polynomial.toy_variety` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L82
  - [ ] `sage.rings.polynomial.toy_variety.coefficient_matrix` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L82
  - [ ] `sage.rings.polynomial.toy_variety.elim_pol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L300
  - [ ] `sage.rings.polynomial.toy_variety.is_linearly_dependent` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L128
  - [ ] `sage.rings.polynomial.toy_variety.is_triangular` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L33
  - [ ] `sage.rings.polynomial.toy_variety.linear_representation` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L182
  - [ ] `sage.rings.polynomial.toy_variety.triangular_factorization` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L230
  - [ ] `sage.rings.real_lazy` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L513
  - [ ] `sage.rings.real_lazy.ComplexLazyField` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L513
  - [ ] `sage.rings.real_lazy.ComplexLazyField_class` (class) extends LazyField
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L371
  - [ ] `sage.rings.real_lazy.LazyAlgebraic` (class) extends LazyFieldElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1588
  - [ ] `sage.rings.real_lazy.LazyBinop` (class) extends LazyFieldElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1064
  - [ ] `sage.rings.real_lazy.LazyConstant` (class) extends LazyFieldElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1445
  - [ ] `sage.rings.real_lazy.LazyField` (class) extends Field
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L82
  - [ ] `sage.rings.real_lazy.LazyFieldElement` (class) extends FieldElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L369
  - [ ] `sage.rings.real_lazy.LazyNamedUnop` (class) extends LazyUnop
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1305
  - [ ] `sage.rings.real_lazy.LazyUnop` (class) extends LazyFieldElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1201
  - [ ] `sage.rings.real_lazy.LazyWrapper` (class) extends LazyFieldElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L920
  - [ ] `sage.rings.real_lazy.LazyWrapperMorphism` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1713
  - [ ] `sage.rings.real_lazy.RealLazyField` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L357
  - [ ] `sage.rings.real_lazy.RealLazyField_class` (class) extends LazyField
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L220
  - [ ] `sage.rings.real_lazy.make_element` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L907

## sage.symbolic (211 features)

  - [ ] `sage.symbolic` (module)
  - [ ] `sage.symbolic.assumptions` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L90
  - [ ] `sage.symbolic.assumptions.GenericDeclaration` (class) extends UniqueRepresentation
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L90
  - [ ] `sage.symbolic.assumptions.assume` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L436
  - [ ] `sage.symbolic.assumptions.assuming` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L834
  - [ ] `sage.symbolic.assumptions.assumptions` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L726
  - [ ] `sage.symbolic.assumptions.forget` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L675
  - [ ] `sage.symbolic.assumptions.preprocess_assumptions` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L405
  - [ ] `sage.symbolic.benchmark` (module)
  - [ ] `sage.symbolic.callable` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L73
  - [ ] `sage.symbolic.callable.CallableSymbolicExpressionFunctor` (class) extends ConstructionFunctor
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L73
  - [ ] `sage.symbolic.callable.CallableSymbolicExpressionRingFactory` (class) extends UniqueFactory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L426
  - [ ] `sage.symbolic.callable.CallableSymbolicExpressionRing_class` (class) extends SymbolicRing, CallableSymbolicExpressionRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L227
  - [ ] `sage.symbolic.complexity_measures` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/complexity_measures.py#L10
  - [ ] `sage.symbolic.complexity_measures.string_length` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/complexity_measures.py#L10
  - [ ] `sage.symbolic.constants` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1014
  - [ ] `sage.symbolic.constants.Catalan` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1014
  - [ ] `sage.symbolic.constants.Constant` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L273
  - [ ] `sage.symbolic.constants.EulerGamma` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L935
  - [ ] `sage.symbolic.constants.Glaisher` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1233
  - [ ] `sage.symbolic.constants.GoldenRatio` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L754
  - [ ] `sage.symbolic.constants.Khinchin` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1084
  - [ ] `sage.symbolic.constants.Log2` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L858
  - [ ] `sage.symbolic.constants.Mertens` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1185
  - [ ] `sage.symbolic.constants.NotANumber` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L692
  - [ ] `sage.symbolic.constants.Pi` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L548
  - [ ] `sage.symbolic.constants.TwinPrime` (class) extends Constant
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1137
  - [ ] `sage.symbolic.constants.pi` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L697
  - [ ] `sage.symbolic.constants.unpickle_Constant` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L244
  - [ ] `sage.symbolic.expression` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L33
  - [ ] `sage.symbolic.expression.E` (class) extends Expression
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L33
  - [ ] `sage.symbolic.expression.Expression` (class) extends Expression
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L697
  - [ ] `sage.symbolic.expression.ExpressionIterator` (class) extends object
  - [ ] `sage.symbolic.expression.OperandsWrapper` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L65
  - [ ] `sage.symbolic.expression.PynacConstant` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L112
  - [ ] `sage.symbolic.expression.SubstitutionMap` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L36
  - [ ] `sage.symbolic.expression.SymbolicSeries` (class) extends Expression
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L129
  - [ ] `sage.symbolic.expression.call_registered_function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1
  - [ ] `sage.symbolic.expression.doublefactorial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1517
  - [ ] `sage.symbolic.expression.find_registered_function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L67
  - [ ] `sage.symbolic.expression.get_fn_serial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L270
  - [ ] `sage.symbolic.expression.get_ginac_serial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L251
  - [ ] `sage.symbolic.expression.get_sfunction_from_hash` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L190
  - [ ] `sage.symbolic.expression.get_sfunction_from_serial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L174
  - [ ] `sage.symbolic.expression.hold_class` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L14082
  - [ ] `sage.symbolic.expression.init_function_table` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2477
  - [ ] `sage.symbolic.expression.init_pynac_I` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2380
  - [ ] `sage.symbolic.expression.is_SymbolicEquation` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L412
  - [ ] `sage.symbolic.expression.make_map` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L76
  - [ ] `sage.symbolic.expression.math_sorted` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L216
  - [ ] `sage.symbolic.expression.mixed_order` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L240
  - [ ] `sage.symbolic.expression.mixed_sorted` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L403
  - [ ] `sage.symbolic.expression.new_Expression` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13696
  - [ ] `sage.symbolic.expression.new_Expression_from_pyobject` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13786
  - [ ] `sage.symbolic.expression.new_Expression_symbol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13885
  - [ ] `sage.symbolic.expression.new_Expression_wild` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13857
  - [ ] `sage.symbolic.expression.normalize_index_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L53
  - [ ] `sage.symbolic.expression.op` (attribute)
  - [ ] `sage.symbolic.expression.paramset_from_Expression` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L213
  - [ ] `sage.symbolic.expression.print_order` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L48
  - [ ] `sage.symbolic.expression.print_sorted` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L136
  - [ ] `sage.symbolic.expression.py_atan2_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1900
  - [ ] `sage.symbolic.expression.py_denom_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1318
  - [ ] `sage.symbolic.expression.py_eval_infinity_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2301
  - [ ] `sage.symbolic.expression.py_eval_neg_infinity_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2321
  - [ ] `sage.symbolic.expression.py_eval_unsigned_infinity_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2281
  - [ ] `sage.symbolic.expression.py_exp_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1715
  - [ ] `sage.symbolic.expression.py_factorial_py` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1495
  - [ ] `sage.symbolic.expression.py_float_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1394
  - [ ] `sage.symbolic.expression.py_imag_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1082
  - [ ] `sage.symbolic.expression.py_is_cinteger_for_doctest` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1335
  - [ ] `sage.symbolic.expression.py_is_crational_for_doctest` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1188
  - [ ] `sage.symbolic.expression.py_is_integer_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1155
  - [ ] `sage.symbolic.expression.py_latex_fderivative_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L726
  - [ ] `sage.symbolic.expression.py_latex_function_pystring` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L511
  - [ ] `sage.symbolic.expression.py_latex_variable_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L426
  - [ ] `sage.symbolic.expression.py_lgamma_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2003
  - [ ] `sage.symbolic.expression.py_li2_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2236
  - [ ] `sage.symbolic.expression.py_li_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2142
  - [ ] `sage.symbolic.expression.py_log_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1785
  - [ ] `sage.symbolic.expression.py_mod_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2066
  - [ ] `sage.symbolic.expression.py_numer_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1282
  - [ ] `sage.symbolic.expression.py_print_fderivative_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L650
  - [ ] `sage.symbolic.expression.py_print_function_pystring` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L442
  - [ ] `sage.symbolic.expression.py_psi2_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2207
  - [ ] `sage.symbolic.expression.py_psi_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2178
  - [ ] `sage.symbolic.expression.py_real_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1025
  - [ ] `sage.symbolic.expression.py_stieltjes_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1638
  - [ ] `sage.symbolic.expression.py_tgamma_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1455
  - [ ] `sage.symbolic.expression.py_zeta_for_doctests` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1673
  - [ ] `sage.symbolic.expression.register_or_update_function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L91
  - [ ] `sage.symbolic.expression.restore_op_wrapper` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L192
  - [ ] `sage.symbolic.expression.solve_diophantine` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13522
  - [ ] `sage.symbolic.expression.test_binomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L915
  - [ ] `sage.symbolic.expression.tolerant_is_symbol` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L601
  - [ ] `sage.symbolic.expression.unpack_operands` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L115
  - [ ] `sage.symbolic.expression_conversions` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L131
  - [ ] `sage.symbolic.expression_conversions.Converter` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L131
  - [ ] `sage.symbolic.expression_conversions.DeMoivre` (class) extends ExpressionTreeWalker
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1712
  - [ ] `sage.symbolic.expression_conversions.Exponentialize` (class) extends ExpressionTreeWalker
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1646
  - [ ] `sage.symbolic.expression_conversions.ExpressionTreeWalker` (class) extends Converter
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1448
  - [ ] `sage.symbolic.expression_conversions.FakeExpression` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L33
  - [ ] `sage.symbolic.expression_conversions.FastCallableConverter` (class) extends Converter
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1153
  - [ ] `sage.symbolic.expression_conversions.FriCASConverter` (class) extends InterfaceInit
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L660
  - [ ] `sage.symbolic.expression_conversions.HalfAngle` (class) extends ExpressionTreeWalker
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1774
  - [ ] `sage.symbolic.expression_conversions.HoldRemover` (class) extends ExpressionTreeWalker
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1841
  - [ ] `sage.symbolic.expression_conversions.InterfaceInit` (class) extends Converter
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L387
  - [ ] `sage.symbolic.expression_conversions.LaurentPolynomialConverter` (class) extends PolynomialConverter
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1083
  - [ ] `sage.symbolic.expression_conversions.PolynomialConverter` (class) extends Converter
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L844
  - [ ] `sage.symbolic.expression_conversions.RingConverter` (class) extends R, Converter
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1333
  - [ ] `sage.symbolic.expression_conversions.SubstituteFunction` (class) extends ExpressionTreeWalker
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1562
  - [ ] `sage.symbolic.expression_conversions.cos` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L138
  - [ ] `sage.symbolic.expression_conversions.cosh` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L87
  - [ ] `sage.symbolic.expression_conversions.cot` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L272
  - [ ] `sage.symbolic.expression_conversions.coth` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L193
  - [ ] `sage.symbolic.expression_conversions.csc` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L447
  - [ ] `sage.symbolic.expression_conversions.csch` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L309
  - [ ] `sage.symbolic.expression_conversions.e` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L33
  - [ ] `sage.symbolic.expression_conversions.fast_callable` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1312
  - [ ] `sage.symbolic.expression_conversions.half` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L414
  - [ ] `sage.symbolic.expression_conversions.halfx` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L697
  - [ ] `sage.symbolic.expression_conversions.laurent_polynomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1108
  - [ ] `sage.symbolic.expression_conversions.one` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L404
  - [ ] `sage.symbolic.expression_conversions.polynomial` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1021
  - [ ] `sage.symbolic.expression_conversions.sec` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L373
  - [ ] `sage.symbolic.expression_conversions.sech` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L252
  - [ ] `sage.symbolic.expression_conversions.sin` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L9
  - [ ] `sage.symbolic.expression_conversions.sinh` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L49
  - [ ] `sage.symbolic.expression_conversions.tan` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L206
  - [ ] `sage.symbolic.expression_conversions.tanh` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L125
  - [ ] `sage.symbolic.expression_conversions.two` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L404
  - [ ] `sage.symbolic.expression_conversions.x` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L697
  - [ ] `sage.symbolic.function` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L867
  - [ ] `sage.symbolic.function.BuiltinFunction` (class) extends Function
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L867
  - [ ] `sage.symbolic.function.Function` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L166
  - [ ] `sage.symbolic.function.GinacFunction` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L818
  - [ ] `sage.symbolic.function.SymbolicFunction` (class) extends Function
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L1174
  - [ ] `sage.symbolic.function.pickle_wrapper` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L1407
  - [ ] `sage.symbolic.function.unpickle_wrapper` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L1429
  - [ ] `sage.symbolic.function_factory` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L152
  - [ ] `sage.symbolic.function_factory.function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L152
  - [ ] `sage.symbolic.function_factory.function_factory` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L18
  - [ ] `sage.symbolic.function_factory.unpickle_function` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L113
  - [ ] `sage.symbolic.integration.external` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L131
  - [ ] `sage.symbolic.integration.external.fricas_integrator` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L131
  - [ ] `sage.symbolic.integration.external.libgiac_integrator` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L217
  - [ ] `sage.symbolic.integration.external.maxima_integrator` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L13
  - [ ] `sage.symbolic.integration.external.mma_free_integrator` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L72
  - [ ] `sage.symbolic.integration.external.sympy_integrator` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L50
  - [ ] `sage.symbolic.integration.integral` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L196
  - [ ] `sage.symbolic.integration.integral.DefiniteIntegral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L196
  - [ ] `sage.symbolic.integration.integral.IndefiniteIntegral` (class) extends BuiltinFunction
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L41
  - [ ] `sage.symbolic.integration.integral.integral` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L446
  - [ ] `sage.symbolic.integration.integral.integrate` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L446
  - [ ] `sage.symbolic.maxima_wrapper` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/maxima_wrapper.py#L16
  - [ ] `sage.symbolic.maxima_wrapper.MaximaFunctionElementWrapper` (class) extends InterfaceFunctionElement
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/maxima_wrapper.py#L16
  - [ ] `sage.symbolic.maxima_wrapper.MaximaWrapper` (class) extends SageObject
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/maxima_wrapper.py#L34
  - [ ] `sage.symbolic.operators` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L204
  - [ ] `sage.symbolic.operators.DerivativeOperator` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L204
  - [ ] `sage.symbolic.operators.DerivativeOperatorWithParameters` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L229
  - [ ] `sage.symbolic.operators.FDerivativeOperator` (class) extends object
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L75
  - [ ] `sage.symbolic.operators.add_vararg` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L8
  - [ ] `sage.symbolic.operators.mul_vararg` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L33
  - [ ] `sage.symbolic.random_tests` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L329
  - [ ] `sage.symbolic.random_tests.assert_strict_weak_order` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L329
  - [ ] `sage.symbolic.random_tests.choose_from_prob_list` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L130
  - [ ] `sage.symbolic.random_tests.normalize_prob_list` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L79
  - [ ] `sage.symbolic.random_tests.random_expr` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L270
  - [ ] `sage.symbolic.random_tests.random_expr_helper` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L224
  - [ ] `sage.symbolic.random_tests.random_integer_vector` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L169
  - [ ] `sage.symbolic.random_tests.test_symbolic_expression_order` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L421
  - [ ] `sage.symbolic.relation` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L584
  - [ ] `sage.symbolic.relation.solve` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L584
  - [ ] `sage.symbolic.relation.solve_ineq` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1829
  - [ ] `sage.symbolic.relation.solve_ineq_fourier` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1755
  - [ ] `sage.symbolic.relation.solve_ineq_univar` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1708
  - [ ] `sage.symbolic.relation.solve_mod` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1473
  - [ ] `sage.symbolic.relation.string_to_list_of_solutions` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L550
  - [ ] `sage.symbolic.relation.test_relation_maxima` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L363
  - [ ] `sage.symbolic.ring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1148
  - [ ] `sage.symbolic.ring.NumpyToSRMorphism` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1148
  - [ ] `sage.symbolic.ring.SymbolicRing` (class) extends SymbolicRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L66
  - [ ] `sage.symbolic.ring.TemporaryVariables` (class) extends tuple
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1374
  - [ ] `sage.symbolic.ring.UnderscoreSageMorphism` (class) extends Morphism
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1243
  - [ ] `sage.symbolic.ring.isidentifier` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1331
  - [ ] `sage.symbolic.ring.symbols` (attribute)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L4
  - [ ] `sage.symbolic.ring.the_SymbolicRing` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1282
  - [ ] `sage.symbolic.ring.var` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1300
  - [ ] `sage.symbolic.subring` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L249
  - [ ] `sage.symbolic.subring.GenericSymbolicSubring` (class) extends SymbolicRing
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L249
  - [ ] `sage.symbolic.subring.GenericSymbolicSubringFunctor` (class) extends ConstructionFunctor
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L476
  - [ ] `sage.symbolic.subring.SymbolicConstantsSubring` (class) extends SymbolicSubringAcceptingVars
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L968
  - [ ] `sage.symbolic.subring.SymbolicSubringAcceptingVars` (class) extends GenericSymbolicSubring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L613
  - [ ] `sage.symbolic.subring.SymbolicSubringAcceptingVarsFunctor` (class) extends GenericSymbolicSubringFunctor
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L714
  - [ ] `sage.symbolic.subring.SymbolicSubringFactory` (class) extends UniqueFactory
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L106
  - [ ] `sage.symbolic.subring.SymbolicSubringRejectingVars` (class) extends GenericSymbolicSubring
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L779
  - [ ] `sage.symbolic.subring.SymbolicSubringRejectingVarsFunctor` (class) extends GenericSymbolicSubringFunctor
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L903
  - [ ] `sage.symbolic.subring.coercion_reversed` (attribute)
  - [ ] `sage.symbolic.subring.rank` (attribute)
  - [ ] `sage.symbolic.units` (module)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L989
  - [ ] `sage.symbolic.units.UnitExpression` (class) extends Expression
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L989
  - [ ] `sage.symbolic.units.Units` (class) extends ExtraTabCompletion
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1042
  - [ ] `sage.symbolic.units.base_units` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1355
  - [ ] `sage.symbolic.units.convert` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1263
  - [ ] `sage.symbolic.units.convert_temperature` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1407
  - [ ] `sage.symbolic.units.evalunitdict` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L493
  - [ ] `sage.symbolic.units.is_unit` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1232
  - [ ] `sage.symbolic.units.str_to_unit` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1020
  - [ ] `sage.symbolic.units.unit_derivations_expr` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L952
  - [ ] `sage.symbolic.units.unitdocs` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1203
  - [ ] `sage.symbolic.units.vars_in_str` (function)
        Source: https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L933

---

## Summary by Module

| Module | Partial Features |
|--------|------------------|
| sage.arith | 102 |
| sage.calculus | 98 |
| sage.categories | 37 |
| sage.functions | 68 |
| sage.libs | 1 |
| sage.rings | 453 |
| sage.symbolic | 211 |
| **TOTAL** | **970** |
