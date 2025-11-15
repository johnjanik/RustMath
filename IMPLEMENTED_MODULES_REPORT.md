# RustMath Implemented Modules Report

This report lists all modules marked as 'implemented' in the SageMath to RustMath tracker files.

## Summary by Part

| Part | Implemented Entries |
|------|--------------------|
| Part 01 | 154 |
| Part 02 | 207 |
| Part 03 | 42 |
| Part 04 | 262 |
| Part 05 | 102 |
| Part 06 | 334 |
| Part 07 | 145 |
| Part 08 | 358 |
| Part 09 | 116 |
| Part 10 | 999 |
| Part 11 | 196 |
| Part 12 | 741 |
| Part 13 | 443 |
| Part 14 | 163 |
| **TOTAL** | **4262** |

## Statistics by Type

| Type | Count |
|------|-------|
| attribute | 74 |
| class | 1812 |
| function | 1594 |
| module | 782 |

## Detailed List by Part

### Part 01 (154 entries)

#### CLASS (8 entries)

- **sage.arith.misc.Euler_Phi**
  - Entity: `Euler_Phi`
  - Module: `sage.arith.misc`
  - Type: `class`

- **sage.arith.misc.Moebius**
  - Entity: `Moebius`
  - Module: `sage.arith.misc`
  - Type: `class`

- **sage.arith.misc.Sigma**
  - Entity: `Sigma`
  - Module: `sage.arith.misc`
  - Type: `class`

- **sage.arith.multi_modular.MultiModularBasis**
  - Entity: `MultiModularBasis`
  - Module: `sage.arith.multi_modular`
  - Type: `class`

- **sage.arith.multi_modular.MultiModularBasis_base**
  - Entity: `MultiModularBasis_base`
  - Module: `sage.arith.multi_modular`
  - Type: `class`

- **sage.arith.multi_modular.MutableMultiModularBasis**
  - Entity: `MutableMultiModularBasis`
  - Module: `sage.arith.multi_modular`
  - Type: `class`

- **sage.calculus.integration.PyFunctionWrapper**
  - Entity: `PyFunctionWrapper`
  - Module: `sage.calculus.integration`
  - Type: `class`

- **sage.calculus.integration.compiled_integrand**
  - Entity: `compiled_integrand`
  - Module: `sage.calculus.integration`
  - Type: `class`

#### FUNCTION (135 entries)

- **sage.arith.functions.LCM_list**
  - Entity: `LCM_list`
  - Module: `sage.arith.functions`
  - Type: `function`

- **sage.arith.functions.lcm**
  - Entity: `lcm`
  - Module: `sage.arith.functions`
  - Type: `function`

- **sage.arith.misc.CRT**
  - Entity: `CRT`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.CRT_basis**
  - Entity: `CRT_basis`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.CRT_list**
  - Entity: `CRT_list`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.CRT_vectors**
  - Entity: `CRT_vectors`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.GCD**
  - Entity: `GCD`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.XGCD**
  - Entity: `XGCD`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.algdep**
  - Entity: `algdep`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.algebraic_dependency**
  - Entity: `algebraic_dependency`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.bernoulli**
  - Entity: `bernoulli`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.binomial**
  - Entity: `binomial`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.binomial_coefficients**
  - Entity: `binomial_coefficients`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.carmichael_lambda**
  - Entity: `carmichael_lambda`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.continuant**
  - Entity: `continuant`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.coprime_part**
  - Entity: `coprime_part`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.crt**
  - Entity: `crt`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.dedekind_psi**
  - Entity: `dedekind_psi`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.dedekind_sum**
  - Entity: `dedekind_sum`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.differences**
  - Entity: `differences`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.divisors**
  - Entity: `divisors`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.eratosthenes**
  - Entity: `eratosthenes`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.factor**
  - Entity: `factor`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.factorial**
  - Entity: `factorial`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.falling_factorial**
  - Entity: `falling_factorial`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.four_squares**
  - Entity: `four_squares`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.fundamental_discriminant**
  - Entity: `fundamental_discriminant`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.gauss_sum**
  - Entity: `gauss_sum`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.gcd**
  - Entity: `gcd`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.get_gcd**
  - Entity: `get_gcd`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.get_inverse_mod**
  - Entity: `get_inverse_mod`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.hilbert_conductor**
  - Entity: `hilbert_conductor`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.hilbert_conductor_inverse**
  - Entity: `hilbert_conductor_inverse`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.hilbert_symbol**
  - Entity: `hilbert_symbol`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.integer_ceil**
  - Entity: `integer_ceil`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.integer_floor**
  - Entity: `integer_floor`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.integer_trunc**
  - Entity: `integer_trunc`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.inverse_mod**
  - Entity: `inverse_mod`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.is_power_of_two**
  - Entity: `is_power_of_two`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.is_prime**
  - Entity: `is_prime`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.is_prime_power**
  - Entity: `is_prime_power`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.is_pseudoprime**
  - Entity: `is_pseudoprime`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.is_pseudoprime_power**
  - Entity: `is_pseudoprime_power`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.is_square**
  - Entity: `is_square`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.is_squarefree**
  - Entity: `is_squarefree`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.jacobi_symbol**
  - Entity: `jacobi_symbol`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.kronecker**
  - Entity: `kronecker`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.kronecker_symbol**
  - Entity: `kronecker_symbol`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.legendre_symbol**
  - Entity: `legendre_symbol`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.mqrr_rational_reconstruction**
  - Entity: `mqrr_rational_reconstruction`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.multinomial**
  - Entity: `multinomial`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.multinomial_coefficients**
  - Entity: `multinomial_coefficients`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.next_prime**
  - Entity: `next_prime`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.next_prime_power**
  - Entity: `next_prime_power`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.next_probable_prime**
  - Entity: `next_probable_prime`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.nth_prime**
  - Entity: `nth_prime`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.number_of_divisors**
  - Entity: `number_of_divisors`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.odd_part**
  - Entity: `odd_part`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.power_mod**
  - Entity: `power_mod`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.previous_prime**
  - Entity: `previous_prime`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.previous_prime_power**
  - Entity: `previous_prime_power`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.prime_divisors**
  - Entity: `prime_divisors`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.prime_factors**
  - Entity: `prime_factors`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.prime_powers**
  - Entity: `prime_powers`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.prime_to_m_part**
  - Entity: `prime_to_m_part`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.primes**
  - Entity: `primes`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.primes_first_n**
  - Entity: `primes_first_n`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.primitive_root**
  - Entity: `primitive_root`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.quadratic_residues**
  - Entity: `quadratic_residues`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.radical**
  - Entity: `radical`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.random_prime**
  - Entity: `random_prime`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.rational_reconstruction**
  - Entity: `rational_reconstruction`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.rising_factorial**
  - Entity: `rising_factorial`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.smooth_part**
  - Entity: `smooth_part`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.sort_complex_numbers_for_display**
  - Entity: `sort_complex_numbers_for_display`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.squarefree_divisors**
  - Entity: `squarefree_divisors`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.subfactorial**
  - Entity: `subfactorial`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.sum_of_k_squares**
  - Entity: `sum_of_k_squares`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.three_squares**
  - Entity: `three_squares`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.trial_division**
  - Entity: `trial_division`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.two_squares**
  - Entity: `two_squares`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.valuation**
  - Entity: `valuation`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.xgcd**
  - Entity: `xgcd`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.xkcd**
  - Entity: `xkcd`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.misc.xlcm**
  - Entity: `xlcm`
  - Module: `sage.arith.misc`
  - Type: `function`

- **sage.arith.power.generic_power**
  - Entity: `generic_power`
  - Module: `sage.arith.power`
  - Type: `function`

- **sage.arith.srange.ellipsis_iter**
  - Entity: `ellipsis_iter`
  - Module: `sage.arith.srange`
  - Type: `function`

- **sage.arith.srange.ellipsis_range**
  - Entity: `ellipsis_range`
  - Module: `sage.arith.srange`
  - Type: `function`

- **sage.arith.srange.srange**
  - Entity: `srange`
  - Module: `sage.arith.srange`
  - Type: `function`

- **sage.arith.srange.xsrange**
  - Entity: `xsrange`
  - Module: `sage.arith.srange`
  - Type: `function`

- **sage.calculus.calculus.at**
  - Entity: `at`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.dummy_diff**
  - Entity: `dummy_diff`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.dummy_integrate**
  - Entity: `dummy_integrate`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.dummy_inverse_laplace**
  - Entity: `dummy_inverse_laplace`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.dummy_laplace**
  - Entity: `dummy_laplace`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.dummy_pochhammer**
  - Entity: `dummy_pochhammer`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.inverse_laplace**
  - Entity: `inverse_laplace`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.laplace**
  - Entity: `laplace`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.lim**
  - Entity: `lim`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.limit**
  - Entity: `limit`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.mapped_opts**
  - Entity: `mapped_opts`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.maxima_options**
  - Entity: `maxima_options`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.minpoly**
  - Entity: `minpoly`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.mma_free_limit**
  - Entity: `mma_free_limit`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.nintegral**
  - Entity: `nintegral`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.nintegrate**
  - Entity: `nintegrate`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.symbolic_expression_from_maxima_string**
  - Entity: `symbolic_expression_from_maxima_string`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.symbolic_expression_from_string**
  - Entity: `symbolic_expression_from_string`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.symbolic_product**
  - Entity: `symbolic_product`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.calculus.symbolic_sum**
  - Entity: `symbolic_sum`
  - Module: `sage.calculus.calculus`
  - Type: `function`

- **sage.calculus.desolvers.desolve**
  - Entity: `desolve`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_laplace**
  - Entity: `desolve_laplace`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_mintides**
  - Entity: `desolve_mintides`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_odeint**
  - Entity: `desolve_odeint`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_rk4**
  - Entity: `desolve_rk4`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_rk4_determine_bounds**
  - Entity: `desolve_rk4_determine_bounds`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_system**
  - Entity: `desolve_system`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_system_rk4**
  - Entity: `desolve_system_rk4`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.desolve_tides_mpfr**
  - Entity: `desolve_tides_mpfr`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.eulers_method**
  - Entity: `eulers_method`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.eulers_method_2x2**
  - Entity: `eulers_method_2x2`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.eulers_method_2x2_plot**
  - Entity: `eulers_method_2x2_plot`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.fricas_desolve**
  - Entity: `fricas_desolve`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.desolvers.fricas_desolve_system**
  - Entity: `fricas_desolve_system`
  - Module: `sage.calculus.desolvers`
  - Type: `function`

- **sage.calculus.functional.derivative**
  - Entity: `derivative`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.diff**
  - Entity: `diff`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.expand**
  - Entity: `expand`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.integral**
  - Entity: `integral`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.integrate**
  - Entity: `integrate`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.lim**
  - Entity: `lim`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.limit**
  - Entity: `limit`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.simplify**
  - Entity: `simplify`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.functional.taylor**
  - Entity: `taylor`
  - Module: `sage.calculus.functional`
  - Type: `function`

- **sage.calculus.integration.monte_carlo_integral**
  - Entity: `monte_carlo_integral`
  - Module: `sage.calculus.integration`
  - Type: `function`

- **sage.calculus.integration.numerical_integral**
  - Entity: `numerical_integral`
  - Module: `sage.calculus.integration`
  - Type: `function`

#### MODULE (11 entries)

- **sage.arith**
  - Entity: `arith`
  - Module: `sage.arith`
  - Type: `module`

- **sage.arith.functions**
  - Entity: `functions`
  - Module: `sage.arith.functions`
  - Type: `module`

- **sage.arith.misc**
  - Entity: `misc`
  - Module: `sage.arith.misc`
  - Type: `module`

- **sage.arith.multi_modular**
  - Entity: `multi_modular`
  - Module: `sage.arith.multi_modular`
  - Type: `module`

- **sage.arith.power**
  - Entity: `power`
  - Module: `sage.arith.power`
  - Type: `module`

- **sage.arith.srange**
  - Entity: `srange`
  - Module: `sage.arith.srange`
  - Type: `module`

- **sage.calculus**
  - Entity: `calculus`
  - Module: `sage.calculus`
  - Type: `module`

- **sage.calculus.calculus**
  - Entity: `calculus`
  - Module: `sage.calculus.calculus`
  - Type: `module`

- **sage.calculus.desolvers**
  - Entity: `desolvers`
  - Module: `sage.calculus.desolvers`
  - Type: `module`

- **sage.calculus.functional**
  - Entity: `functional`
  - Module: `sage.calculus.functional`
  - Type: `module`

- **sage.calculus.integration**
  - Entity: `integration`
  - Module: `sage.calculus.integration`
  - Type: `module`


### Part 02 (207 entries)

#### ATTRIBUTE (1 entries)

- **sage.coding.ag_code_decoders.info**
  - Entity: `info`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `attribute`

#### CLASS (89 entries)

- **sage.categories.functor.ForgetfulFunctor_generic**
  - Entity: `ForgetfulFunctor_generic`
  - Module: `sage.categories.functor`
  - Type: `class`

- **sage.categories.functor.Functor**
  - Entity: `Functor`
  - Module: `sage.categories.functor`
  - Type: `class`

- **sage.categories.functor.IdentityFunctor_generic**
  - Entity: `IdentityFunctor_generic`
  - Module: `sage.categories.functor`
  - Type: `class`

- **sage.categories.morphism.CallMorphism**
  - Entity: `CallMorphism`
  - Module: `sage.categories.morphism`
  - Type: `class`

- **sage.categories.morphism.FormalCoercionMorphism**
  - Entity: `FormalCoercionMorphism`
  - Module: `sage.categories.morphism`
  - Type: `class`

- **sage.categories.morphism.IdentityMorphism**
  - Entity: `IdentityMorphism`
  - Module: `sage.categories.morphism`
  - Type: `class`

- **sage.categories.morphism.Morphism**
  - Entity: `Morphism`
  - Module: `sage.categories.morphism`
  - Type: `class`

- **sage.categories.morphism.SetIsomorphism**
  - Entity: `SetIsomorphism`
  - Module: `sage.categories.morphism`
  - Type: `class`

- **sage.categories.morphism.SetMorphism**
  - Entity: `SetMorphism`
  - Module: `sage.categories.morphism`
  - Type: `class`

- **sage.coding.abstract_code.AbstractCode**
  - Entity: `AbstractCode`
  - Module: `sage.coding.abstract_code`
  - Type: `class`

- **sage.coding.ag_code.AGCode**
  - Entity: `AGCode`
  - Module: `sage.coding.ag_code`
  - Type: `class`

- **sage.coding.ag_code.CartierCode**
  - Entity: `CartierCode`
  - Module: `sage.coding.ag_code`
  - Type: `class`

- **sage.coding.ag_code.DifferentialAGCode**
  - Entity: `DifferentialAGCode`
  - Module: `sage.coding.ag_code`
  - Type: `class`

- **sage.coding.ag_code.EvaluationAGCode**
  - Entity: `EvaluationAGCode`
  - Module: `sage.coding.ag_code`
  - Type: `class`

- **sage.coding.ag_code_decoders.Decoder_K**
  - Entity: `Decoder_K`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.Decoder_K_extension**
  - Entity: `Decoder_K_extension`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.DifferentialAGCodeDecoder_K**
  - Entity: `DifferentialAGCodeDecoder_K`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.DifferentialAGCodeDecoder_K_extension**
  - Entity: `DifferentialAGCodeDecoder_K_extension`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.DifferentialAGCodeEncoder**
  - Entity: `DifferentialAGCodeEncoder`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.DifferentialAGCodeUniqueDecoder**
  - Entity: `DifferentialAGCodeUniqueDecoder`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.EvaluationAGCodeDecoder_K**
  - Entity: `EvaluationAGCodeDecoder_K`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.EvaluationAGCodeDecoder_K_extension**
  - Entity: `EvaluationAGCodeDecoder_K_extension`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.EvaluationAGCodeEncoder**
  - Entity: `EvaluationAGCodeEncoder`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.ag_code_decoders.EvaluationAGCodeUniqueDecoder**
  - Entity: `EvaluationAGCodeUniqueDecoder`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `class`

- **sage.coding.bch_code.BCHCode**
  - Entity: `BCHCode`
  - Module: `sage.coding.bch_code`
  - Type: `class`

- **sage.coding.bch_code.BCHUnderlyingGRSDecoder**
  - Entity: `BCHUnderlyingGRSDecoder`
  - Module: `sage.coding.bch_code`
  - Type: `class`

- **sage.coding.binary_code.BinaryCode**
  - Entity: `BinaryCode`
  - Module: `sage.coding.binary_code`
  - Type: `class`

- **sage.coding.binary_code.BinaryCodeClassifier**
  - Entity: `BinaryCodeClassifier`
  - Module: `sage.coding.binary_code`
  - Type: `class`

- **sage.coding.binary_code.OrbitPartition**
  - Entity: `OrbitPartition`
  - Module: `sage.coding.binary_code`
  - Type: `class`

- **sage.coding.binary_code.PartitionStack**
  - Entity: `PartitionStack`
  - Module: `sage.coding.binary_code`
  - Type: `class`

- **sage.coding.channel.Channel**
  - Entity: `Channel`
  - Module: `sage.coding.channel`
  - Type: `class`

- **sage.coding.channel.ErrorErasureChannel**
  - Entity: `ErrorErasureChannel`
  - Module: `sage.coding.channel`
  - Type: `class`

- **sage.coding.channel.QarySymmetricChannel**
  - Entity: `QarySymmetricChannel`
  - Module: `sage.coding.channel`
  - Type: `class`

- **sage.coding.channel.StaticErrorRateChannel**
  - Entity: `StaticErrorRateChannel`
  - Module: `sage.coding.channel`
  - Type: `class`

- **sage.coding.codecan.autgroup_can_label.LinearCodeAutGroupCanLabel**
  - Entity: `LinearCodeAutGroupCanLabel`
  - Module: `sage.coding.codecan.autgroup_can_label`
  - Type: `class`

- **sage.coding.codecan.codecan.InnerGroup**
  - Entity: `InnerGroup`
  - Module: `sage.coding.codecan.codecan`
  - Type: `class`

- **sage.coding.codecan.codecan.PartitionRefinementLinearCode**
  - Entity: `PartitionRefinementLinearCode`
  - Module: `sage.coding.codecan.codecan`
  - Type: `class`

- **sage.coding.cyclic_code.CyclicCode**
  - Entity: `CyclicCode`
  - Module: `sage.coding.cyclic_code`
  - Type: `class`

- **sage.coding.cyclic_code.CyclicCodePolynomialEncoder**
  - Entity: `CyclicCodePolynomialEncoder`
  - Module: `sage.coding.cyclic_code`
  - Type: `class`

- **sage.coding.cyclic_code.CyclicCodeSurroundingBCHDecoder**
  - Entity: `CyclicCodeSurroundingBCHDecoder`
  - Module: `sage.coding.cyclic_code`
  - Type: `class`

- **sage.coding.cyclic_code.CyclicCodeVectorEncoder**
  - Entity: `CyclicCodeVectorEncoder`
  - Module: `sage.coding.cyclic_code`
  - Type: `class`

- **sage.coding.decoder.Decoder**
  - Entity: `Decoder`
  - Module: `sage.coding.decoder`
  - Type: `class`

- **sage.coding.encoder.Encoder**
  - Entity: `Encoder`
  - Module: `sage.coding.encoder`
  - Type: `class`

- **sage.coding.extended_code.ExtendedCode**
  - Entity: `ExtendedCode`
  - Module: `sage.coding.extended_code`
  - Type: `class`

- **sage.coding.extended_code.ExtendedCodeExtendedMatrixEncoder**
  - Entity: `ExtendedCodeExtendedMatrixEncoder`
  - Module: `sage.coding.extended_code`
  - Type: `class`

- **sage.coding.extended_code.ExtendedCodeOriginalCodeDecoder**
  - Entity: `ExtendedCodeOriginalCodeDecoder`
  - Module: `sage.coding.extended_code`
  - Type: `class`

- **sage.coding.gabidulin_code.GabidulinCode**
  - Entity: `GabidulinCode`
  - Module: `sage.coding.gabidulin_code`
  - Type: `class`

- **sage.coding.gabidulin_code.GabidulinGaoDecoder**
  - Entity: `GabidulinGaoDecoder`
  - Module: `sage.coding.gabidulin_code`
  - Type: `class`

- **sage.coding.gabidulin_code.GabidulinPolynomialEvaluationEncoder**
  - Entity: `GabidulinPolynomialEvaluationEncoder`
  - Module: `sage.coding.gabidulin_code`
  - Type: `class`

- **sage.coding.gabidulin_code.GabidulinVectorEvaluationEncoder**
  - Entity: `GabidulinVectorEvaluationEncoder`
  - Module: `sage.coding.gabidulin_code`
  - Type: `class`

- **sage.coding.golay_code.GolayCode**
  - Entity: `GolayCode`
  - Module: `sage.coding.golay_code`
  - Type: `class`

- **sage.coding.goppa_code.GoppaCode**
  - Entity: `GoppaCode`
  - Module: `sage.coding.goppa_code`
  - Type: `class`

- **sage.coding.goppa_code.GoppaCodeEncoder**
  - Entity: `GoppaCodeEncoder`
  - Module: `sage.coding.goppa_code`
  - Type: `class`

- **sage.coding.grs_code.GRSBerlekampWelchDecoder**
  - Entity: `GRSBerlekampWelchDecoder`
  - Module: `sage.coding.grs_code`
  - Type: `class`

- **sage.coding.grs_code.GRSErrorErasureDecoder**
  - Entity: `GRSErrorErasureDecoder`
  - Module: `sage.coding.grs_code`
  - Type: `class`

- **sage.coding.grs_code.GRSEvaluationPolynomialEncoder**
  - Entity: `GRSEvaluationPolynomialEncoder`
  - Module: `sage.coding.grs_code`
  - Type: `class`

- **sage.coding.grs_code.GRSEvaluationVectorEncoder**
  - Entity: `GRSEvaluationVectorEncoder`
  - Module: `sage.coding.grs_code`
  - Type: `class`

- **sage.coding.grs_code.GRSGaoDecoder**
  - Entity: `GRSGaoDecoder`
  - Module: `sage.coding.grs_code`
  - Type: `class`

- **sage.coding.grs_code.GRSKeyEquationSyndromeDecoder**
  - Entity: `GRSKeyEquationSyndromeDecoder`
  - Module: `sage.coding.grs_code`
  - Type: `class`

- **sage.coding.grs_code.GeneralizedReedSolomonCode**
  - Entity: `GeneralizedReedSolomonCode`
  - Module: `sage.coding.grs_code`
  - Type: `class`

- **sage.coding.guruswami_sudan.gs_decoder.GRSGuruswamiSudanDecoder**
  - Entity: `GRSGuruswamiSudanDecoder`
  - Module: `sage.coding.guruswami_sudan.gs_decoder`
  - Type: `class`

- **sage.coding.hamming_code.HammingCode**
  - Entity: `HammingCode`
  - Module: `sage.coding.hamming_code`
  - Type: `class`

- **sage.coding.information_set_decoder.InformationSetAlgorithm**
  - Entity: `InformationSetAlgorithm`
  - Module: `sage.coding.information_set_decoder`
  - Type: `class`

- **sage.coding.information_set_decoder.LeeBrickellISDAlgorithm**
  - Entity: `LeeBrickellISDAlgorithm`
  - Module: `sage.coding.information_set_decoder`
  - Type: `class`

- **sage.coding.information_set_decoder.LinearCodeInformationSetDecoder**
  - Entity: `LinearCodeInformationSetDecoder`
  - Module: `sage.coding.information_set_decoder`
  - Type: `class`

- **sage.coding.kasami_codes.KasamiCode**
  - Entity: `KasamiCode`
  - Module: `sage.coding.kasami_codes`
  - Type: `class`

- **sage.coding.linear_code.AbstractLinearCode**
  - Entity: `AbstractLinearCode`
  - Module: `sage.coding.linear_code`
  - Type: `class`

- **sage.coding.linear_code.LinearCode**
  - Entity: `LinearCode`
  - Module: `sage.coding.linear_code`
  - Type: `class`

- **sage.coding.linear_code.LinearCodeGeneratorMatrixEncoder**
  - Entity: `LinearCodeGeneratorMatrixEncoder`
  - Module: `sage.coding.linear_code`
  - Type: `class`

- **sage.coding.linear_code.LinearCodeNearestNeighborDecoder**
  - Entity: `LinearCodeNearestNeighborDecoder`
  - Module: `sage.coding.linear_code`
  - Type: `class`

- **sage.coding.linear_code.LinearCodeSyndromeDecoder**
  - Entity: `LinearCodeSyndromeDecoder`
  - Module: `sage.coding.linear_code`
  - Type: `class`

- **sage.coding.linear_code_no_metric.AbstractLinearCodeNoMetric**
  - Entity: `AbstractLinearCodeNoMetric`
  - Module: `sage.coding.linear_code_no_metric`
  - Type: `class`

- **sage.coding.linear_code_no_metric.LinearCodeSystematicEncoder**
  - Entity: `LinearCodeSystematicEncoder`
  - Module: `sage.coding.linear_code_no_metric`
  - Type: `class`

- **sage.coding.linear_rank_metric.AbstractLinearRankMetricCode**
  - Entity: `AbstractLinearRankMetricCode`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `class`

- **sage.coding.linear_rank_metric.LinearRankMetricCode**
  - Entity: `LinearRankMetricCode`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `class`

- **sage.coding.linear_rank_metric.LinearRankMetricCodeNearestNeighborDecoder**
  - Entity: `LinearRankMetricCodeNearestNeighborDecoder`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `class`

- **sage.coding.parity_check_code.ParityCheckCode**
  - Entity: `ParityCheckCode`
  - Module: `sage.coding.parity_check_code`
  - Type: `class`

- **sage.coding.parity_check_code.ParityCheckCodeGeneratorMatrixEncoder**
  - Entity: `ParityCheckCodeGeneratorMatrixEncoder`
  - Module: `sage.coding.parity_check_code`
  - Type: `class`

- **sage.coding.parity_check_code.ParityCheckCodeStraightforwardEncoder**
  - Entity: `ParityCheckCodeStraightforwardEncoder`
  - Module: `sage.coding.parity_check_code`
  - Type: `class`

- **sage.coding.punctured_code.PuncturedCode**
  - Entity: `PuncturedCode`
  - Module: `sage.coding.punctured_code`
  - Type: `class`

- **sage.coding.punctured_code.PuncturedCodeOriginalCodeDecoder**
  - Entity: `PuncturedCodeOriginalCodeDecoder`
  - Module: `sage.coding.punctured_code`
  - Type: `class`

- **sage.coding.punctured_code.PuncturedCodePuncturedMatrixEncoder**
  - Entity: `PuncturedCodePuncturedMatrixEncoder`
  - Module: `sage.coding.punctured_code`
  - Type: `class`

- **sage.coding.reed_muller_code.BinaryReedMullerCode**
  - Entity: `BinaryReedMullerCode`
  - Module: `sage.coding.reed_muller_code`
  - Type: `class`

- **sage.coding.reed_muller_code.QAryReedMullerCode**
  - Entity: `QAryReedMullerCode`
  - Module: `sage.coding.reed_muller_code`
  - Type: `class`

- **sage.coding.reed_muller_code.ReedMullerPolynomialEncoder**
  - Entity: `ReedMullerPolynomialEncoder`
  - Module: `sage.coding.reed_muller_code`
  - Type: `class`

- **sage.coding.reed_muller_code.ReedMullerVectorEncoder**
  - Entity: `ReedMullerVectorEncoder`
  - Module: `sage.coding.reed_muller_code`
  - Type: `class`

- **sage.coding.source_coding.huffman.Huffman**
  - Entity: `Huffman`
  - Module: `sage.coding.source_coding.huffman`
  - Type: `class`

- **sage.coding.subfield_subcode.SubfieldSubcode**
  - Entity: `SubfieldSubcode`
  - Module: `sage.coding.subfield_subcode`
  - Type: `class`

- **sage.coding.subfield_subcode.SubfieldSubcodeOriginalCodeDecoder**
  - Entity: `SubfieldSubcodeOriginalCodeDecoder`
  - Module: `sage.coding.subfield_subcode`
  - Type: `class`

#### FUNCTION (72 entries)

- **sage.categories.functor.ForgetfulFunctor**
  - Entity: `ForgetfulFunctor`
  - Module: `sage.categories.functor`
  - Type: `function`

- **sage.categories.functor.IdentityFunctor**
  - Entity: `IdentityFunctor`
  - Module: `sage.categories.functor`
  - Type: `function`

- **sage.categories.functor.is_Functor**
  - Entity: `is_Functor`
  - Module: `sage.categories.functor`
  - Type: `function`

- **sage.categories.morphism.is_Morphism**
  - Entity: `is_Morphism`
  - Module: `sage.categories.morphism`
  - Type: `function`

- **sage.coding.binary_code.test_expand_to_ortho_basis**
  - Entity: `test_expand_to_ortho_basis`
  - Module: `sage.coding.binary_code`
  - Type: `function`

- **sage.coding.binary_code.test_word_perms**
  - Entity: `test_word_perms`
  - Module: `sage.coding.binary_code`
  - Type: `function`

- **sage.coding.binary_code.weight_dist**
  - Entity: `weight_dist`
  - Module: `sage.coding.binary_code`
  - Type: `function`

- **sage.coding.channel.format_interval**
  - Entity: `format_interval`
  - Module: `sage.coding.channel`
  - Type: `function`

- **sage.coding.channel.random_error_vector**
  - Entity: `random_error_vector`
  - Module: `sage.coding.channel`
  - Type: `function`

- **sage.coding.code_bounds.codesize_upper_bound**
  - Entity: `codesize_upper_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.dimension_upper_bound**
  - Entity: `dimension_upper_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.elias_bound_asymp**
  - Entity: `elias_bound_asymp`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.elias_upper_bound**
  - Entity: `elias_upper_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.entropy**
  - Entity: `entropy`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.entropy_inverse**
  - Entity: `entropy_inverse`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.gilbert_lower_bound**
  - Entity: `gilbert_lower_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.griesmer_upper_bound**
  - Entity: `griesmer_upper_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.gv_bound_asymp**
  - Entity: `gv_bound_asymp`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.gv_info_rate**
  - Entity: `gv_info_rate`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.hamming_bound_asymp**
  - Entity: `hamming_bound_asymp`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.hamming_upper_bound**
  - Entity: `hamming_upper_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.mrrw1_bound_asymp**
  - Entity: `mrrw1_bound_asymp`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.plotkin_bound_asymp**
  - Entity: `plotkin_bound_asymp`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.plotkin_upper_bound**
  - Entity: `plotkin_upper_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.singleton_bound_asymp**
  - Entity: `singleton_bound_asymp`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.singleton_upper_bound**
  - Entity: `singleton_upper_bound`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_bounds.volume_hamming**
  - Entity: `volume_hamming`
  - Module: `sage.coding.code_bounds`
  - Type: `function`

- **sage.coding.code_constructions.DuadicCodeEvenPair**
  - Entity: `DuadicCodeEvenPair`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.DuadicCodeOddPair**
  - Entity: `DuadicCodeOddPair`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.ExtendedQuadraticResidueCode**
  - Entity: `ExtendedQuadraticResidueCode`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.QuadraticResidueCode**
  - Entity: `QuadraticResidueCode`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.QuadraticResidueCodeEvenPair**
  - Entity: `QuadraticResidueCodeEvenPair`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.QuadraticResidueCodeOddPair**
  - Entity: `QuadraticResidueCodeOddPair`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.ToricCode**
  - Entity: `ToricCode`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.WalshCode**
  - Entity: `WalshCode`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.from_parity_check_matrix**
  - Entity: `from_parity_check_matrix`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.permutation_action**
  - Entity: `permutation_action`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.random_linear_code**
  - Entity: `random_linear_code`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.code_constructions.walsh_matrix**
  - Entity: `walsh_matrix`
  - Module: `sage.coding.code_constructions`
  - Type: `function`

- **sage.coding.cyclic_code.bch_bound**
  - Entity: `bch_bound`
  - Module: `sage.coding.cyclic_code`
  - Type: `function`

- **sage.coding.cyclic_code.find_generator_polynomial**
  - Entity: `find_generator_polynomial`
  - Module: `sage.coding.cyclic_code`
  - Type: `function`

- **sage.coding.databases.best_linear_code_in_codetables_dot_de**
  - Entity: `best_linear_code_in_codetables_dot_de`
  - Module: `sage.coding.databases`
  - Type: `function`

- **sage.coding.databases.best_linear_code_in_guava**
  - Entity: `best_linear_code_in_guava`
  - Module: `sage.coding.databases`
  - Type: `function`

- **sage.coding.databases.bounds_on_minimum_distance_in_guava**
  - Entity: `bounds_on_minimum_distance_in_guava`
  - Module: `sage.coding.databases`
  - Type: `function`

- **sage.coding.databases.self_orthogonal_binary_codes**
  - Entity: `self_orthogonal_binary_codes`
  - Module: `sage.coding.databases`
  - Type: `function`

- **sage.coding.delsarte_bounds.delsarte_bound_Q_matrix**
  - Entity: `delsarte_bound_Q_matrix`
  - Module: `sage.coding.delsarte_bounds`
  - Type: `function`

- **sage.coding.delsarte_bounds.delsarte_bound_additive_hamming_space**
  - Entity: `delsarte_bound_additive_hamming_space`
  - Module: `sage.coding.delsarte_bounds`
  - Type: `function`

- **sage.coding.delsarte_bounds.delsarte_bound_constant_weight_code**
  - Entity: `delsarte_bound_constant_weight_code`
  - Module: `sage.coding.delsarte_bounds`
  - Type: `function`

- **sage.coding.delsarte_bounds.delsarte_bound_hamming_space**
  - Entity: `delsarte_bound_hamming_space`
  - Module: `sage.coding.delsarte_bounds`
  - Type: `function`

- **sage.coding.delsarte_bounds.eberlein**
  - Entity: `eberlein`
  - Module: `sage.coding.delsarte_bounds`
  - Type: `function`

- **sage.coding.delsarte_bounds.krawtchouk**
  - Entity: `krawtchouk`
  - Module: `sage.coding.delsarte_bounds`
  - Type: `function`

- **sage.coding.grs_code.ReedSolomonCode**
  - Entity: `ReedSolomonCode`
  - Module: `sage.coding.grs_code`
  - Type: `function`

- **sage.coding.guava.QuasiQuadraticResidueCode**
  - Entity: `QuasiQuadraticResidueCode`
  - Module: `sage.coding.guava`
  - Type: `function`

- **sage.coding.guava.RandomLinearCodeGuava**
  - Entity: `RandomLinearCodeGuava`
  - Module: `sage.coding.guava`
  - Type: `function`

- **sage.coding.guruswami_sudan.gs_decoder.alekhnovich_root_finder**
  - Entity: `alekhnovich_root_finder`
  - Module: `sage.coding.guruswami_sudan.gs_decoder`
  - Type: `function`

- **sage.coding.guruswami_sudan.gs_decoder.n_k_params**
  - Entity: `n_k_params`
  - Module: `sage.coding.guruswami_sudan.gs_decoder`
  - Type: `function`

- **sage.coding.guruswami_sudan.gs_decoder.roth_ruckenstein_root_finder**
  - Entity: `roth_ruckenstein_root_finder`
  - Module: `sage.coding.guruswami_sudan.gs_decoder`
  - Type: `function`

- **sage.coding.guruswami_sudan.interpolation.gs_interpolation_lee_osullivan**
  - Entity: `gs_interpolation_lee_osullivan`
  - Module: `sage.coding.guruswami_sudan.interpolation`
  - Type: `function`

- **sage.coding.guruswami_sudan.interpolation.gs_interpolation_linalg**
  - Entity: `gs_interpolation_linalg`
  - Module: `sage.coding.guruswami_sudan.interpolation`
  - Type: `function`

- **sage.coding.guruswami_sudan.interpolation.lee_osullivan_module**
  - Entity: `lee_osullivan_module`
  - Module: `sage.coding.guruswami_sudan.interpolation`
  - Type: `function`

- **sage.coding.guruswami_sudan.utils.gilt**
  - Entity: `gilt`
  - Module: `sage.coding.guruswami_sudan.utils`
  - Type: `function`

- **sage.coding.guruswami_sudan.utils.johnson_radius**
  - Entity: `johnson_radius`
  - Module: `sage.coding.guruswami_sudan.utils`
  - Type: `function`

- **sage.coding.guruswami_sudan.utils.ligt**
  - Entity: `ligt`
  - Module: `sage.coding.guruswami_sudan.utils`
  - Type: `function`

- **sage.coding.guruswami_sudan.utils.polynomial_to_list**
  - Entity: `polynomial_to_list`
  - Module: `sage.coding.guruswami_sudan.utils`
  - Type: `function`

- **sage.coding.guruswami_sudan.utils.solve_degree2_to_integer_range**
  - Entity: `solve_degree2_to_integer_range`
  - Module: `sage.coding.guruswami_sudan.utils`
  - Type: `function`

- **sage.coding.linear_rank_metric.from_matrix_representation**
  - Entity: `from_matrix_representation`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `function`

- **sage.coding.linear_rank_metric.rank_distance**
  - Entity: `rank_distance`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `function`

- **sage.coding.linear_rank_metric.rank_weight**
  - Entity: `rank_weight`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `function`

- **sage.coding.linear_rank_metric.to_matrix_representation**
  - Entity: `to_matrix_representation`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `function`

- **sage.coding.reed_muller_code.ReedMullerCode**
  - Entity: `ReedMullerCode`
  - Module: `sage.coding.reed_muller_code`
  - Type: `function`

- **sage.coding.self_dual_codes.self_dual_binary_codes**
  - Entity: `self_dual_binary_codes`
  - Module: `sage.coding.self_dual_codes`
  - Type: `function`

- **sage.coding.source_coding.huffman.frequency_table**
  - Entity: `frequency_table`
  - Module: `sage.coding.source_coding.huffman`
  - Type: `function`

#### MODULE (45 entries)

- **sage.categories.functor**
  - Entity: `functor`
  - Module: `sage.categories.functor`
  - Type: `module`

- **sage.categories.morphism**
  - Entity: `morphism`
  - Module: `sage.categories.morphism`
  - Type: `module`

- **sage.coding**
  - Entity: `coding`
  - Module: `sage.coding`
  - Type: `module`

- **sage.coding.abstract_code**
  - Entity: `abstract_code`
  - Module: `sage.coding.abstract_code`
  - Type: `module`

- **sage.coding.ag_code**
  - Entity: `ag_code`
  - Module: `sage.coding.ag_code`
  - Type: `module`

- **sage.coding.ag_code_decoders**
  - Entity: `ag_code_decoders`
  - Module: `sage.coding.ag_code_decoders`
  - Type: `module`

- **sage.coding.bch_code**
  - Entity: `bch_code`
  - Module: `sage.coding.bch_code`
  - Type: `module`

- **sage.coding.binary_code**
  - Entity: `binary_code`
  - Module: `sage.coding.binary_code`
  - Type: `module`

- **sage.coding.bounds_catalog**
  - Entity: `bounds_catalog`
  - Module: `sage.coding.bounds_catalog`
  - Type: `module`

- **sage.coding.channel**
  - Entity: `channel`
  - Module: `sage.coding.channel`
  - Type: `module`

- **sage.coding.channels_catalog**
  - Entity: `channels_catalog`
  - Module: `sage.coding.channels_catalog`
  - Type: `module`

- **sage.coding.code_bounds**
  - Entity: `code_bounds`
  - Module: `sage.coding.code_bounds`
  - Type: `module`

- **sage.coding.code_constructions**
  - Entity: `code_constructions`
  - Module: `sage.coding.code_constructions`
  - Type: `module`

- **sage.coding.codecan.autgroup_can_label**
  - Entity: `autgroup_can_label`
  - Module: `sage.coding.codecan.autgroup_can_label`
  - Type: `module`

- **sage.coding.codecan.codecan**
  - Entity: `codecan`
  - Module: `sage.coding.codecan.codecan`
  - Type: `module`

- **sage.coding.codes_catalog**
  - Entity: `codes_catalog`
  - Module: `sage.coding.codes_catalog`
  - Type: `module`

- **sage.coding.cyclic_code**
  - Entity: `cyclic_code`
  - Module: `sage.coding.cyclic_code`
  - Type: `module`

- **sage.coding.databases**
  - Entity: `databases`
  - Module: `sage.coding.databases`
  - Type: `module`

- **sage.coding.decoder**
  - Entity: `decoder`
  - Module: `sage.coding.decoder`
  - Type: `module`

- **sage.coding.decoders_catalog**
  - Entity: `decoders_catalog`
  - Module: `sage.coding.decoders_catalog`
  - Type: `module`

- **sage.coding.delsarte_bounds**
  - Entity: `delsarte_bounds`
  - Module: `sage.coding.delsarte_bounds`
  - Type: `module`

- **sage.coding.encoder**
  - Entity: `encoder`
  - Module: `sage.coding.encoder`
  - Type: `module`

- **sage.coding.encoders_catalog**
  - Entity: `encoders_catalog`
  - Module: `sage.coding.encoders_catalog`
  - Type: `module`

- **sage.coding.extended_code**
  - Entity: `extended_code`
  - Module: `sage.coding.extended_code`
  - Type: `module`

- **sage.coding.gabidulin_code**
  - Entity: `gabidulin_code`
  - Module: `sage.coding.gabidulin_code`
  - Type: `module`

- **sage.coding.golay_code**
  - Entity: `golay_code`
  - Module: `sage.coding.golay_code`
  - Type: `module`

- **sage.coding.goppa_code**
  - Entity: `goppa_code`
  - Module: `sage.coding.goppa_code`
  - Type: `module`

- **sage.coding.grs_code**
  - Entity: `grs_code`
  - Module: `sage.coding.grs_code`
  - Type: `module`

- **sage.coding.guava**
  - Entity: `guava`
  - Module: `sage.coding.guava`
  - Type: `module`

- **sage.coding.guruswami_sudan.gs_decoder**
  - Entity: `gs_decoder`
  - Module: `sage.coding.guruswami_sudan.gs_decoder`
  - Type: `module`

- **sage.coding.guruswami_sudan.interpolation**
  - Entity: `interpolation`
  - Module: `sage.coding.guruswami_sudan.interpolation`
  - Type: `module`

- **sage.coding.guruswami_sudan.utils**
  - Entity: `utils`
  - Module: `sage.coding.guruswami_sudan.utils`
  - Type: `module`

- **sage.coding.hamming_code**
  - Entity: `hamming_code`
  - Module: `sage.coding.hamming_code`
  - Type: `module`

- **sage.coding.information_set_decoder**
  - Entity: `information_set_decoder`
  - Module: `sage.coding.information_set_decoder`
  - Type: `module`

- **sage.coding.kasami_codes**
  - Entity: `kasami_codes`
  - Module: `sage.coding.kasami_codes`
  - Type: `module`

- **sage.coding.linear_code**
  - Entity: `linear_code`
  - Module: `sage.coding.linear_code`
  - Type: `module`

- **sage.coding.linear_code_no_metric**
  - Entity: `linear_code_no_metric`
  - Module: `sage.coding.linear_code_no_metric`
  - Type: `module`

- **sage.coding.linear_rank_metric**
  - Entity: `linear_rank_metric`
  - Module: `sage.coding.linear_rank_metric`
  - Type: `module`

- **sage.coding.parity_check_code**
  - Entity: `parity_check_code`
  - Module: `sage.coding.parity_check_code`
  - Type: `module`

- **sage.coding.punctured_code**
  - Entity: `punctured_code`
  - Module: `sage.coding.punctured_code`
  - Type: `module`

- **sage.coding.reed_muller_code**
  - Entity: `reed_muller_code`
  - Module: `sage.coding.reed_muller_code`
  - Type: `module`

- **sage.coding.self_dual_codes**
  - Entity: `self_dual_codes`
  - Module: `sage.coding.self_dual_codes`
  - Type: `module`

- **sage.coding.source_coding.huffman**
  - Entity: `huffman`
  - Module: `sage.coding.source_coding.huffman`
  - Type: `module`

- **sage.coding.subfield_subcode**
  - Entity: `subfield_subcode`
  - Module: `sage.coding.subfield_subcode`
  - Type: `module`

- **sage.coding.two_weight_db**
  - Entity: `two_weight_db`
  - Module: `sage.coding.two_weight_db`
  - Type: `module`


### Part 03 (42 entries)

#### ATTRIBUTE (1 entries)

- **sage.combinat.dyck_word.options**
  - Entity: `options`
  - Module: `sage.combinat.dyck_word`
  - Type: `attribute`

#### CLASS (27 entries)

- **sage.combinat.combination.ChooseNK**
  - Entity: `ChooseNK`
  - Module: `sage.combinat.combination`
  - Type: `class`

- **sage.combinat.combination.Combinations_mset**
  - Entity: `Combinations_mset`
  - Module: `sage.combinat.combination`
  - Type: `class`

- **sage.combinat.combination.Combinations_msetk**
  - Entity: `Combinations_msetk`
  - Module: `sage.combinat.combination`
  - Type: `class`

- **sage.combinat.combination.Combinations_set**
  - Entity: `Combinations_set`
  - Module: `sage.combinat.combination`
  - Type: `class`

- **sage.combinat.combination.Combinations_setk**
  - Entity: `Combinations_setk`
  - Module: `sage.combinat.combination`
  - Type: `class`

- **sage.combinat.composition.Composition**
  - Entity: `Composition`
  - Module: `sage.combinat.composition`
  - Type: `class`

- **sage.combinat.composition.Compositions**
  - Entity: `Compositions`
  - Module: `sage.combinat.composition`
  - Type: `class`

- **sage.combinat.composition.Compositions_all**
  - Entity: `Compositions_all`
  - Module: `sage.combinat.composition`
  - Type: `class`

- **sage.combinat.composition.Compositions_constraints**
  - Entity: `Compositions_constraints`
  - Module: `sage.combinat.composition`
  - Type: `class`

- **sage.combinat.composition.Compositions_n**
  - Entity: `Compositions_n`
  - Module: `sage.combinat.composition`
  - Type: `class`

- **sage.combinat.composition_signed.SignedCompositions**
  - Entity: `SignedCompositions`
  - Module: `sage.combinat.composition_signed`
  - Type: `class`

- **sage.combinat.composition_tableau.CompositionTableau**
  - Entity: `CompositionTableau`
  - Module: `sage.combinat.composition_tableau`
  - Type: `class`

- **sage.combinat.composition_tableau.CompositionTableaux**
  - Entity: `CompositionTableaux`
  - Module: `sage.combinat.composition_tableau`
  - Type: `class`

- **sage.combinat.composition_tableau.CompositionTableauxBacktracker**
  - Entity: `CompositionTableauxBacktracker`
  - Module: `sage.combinat.composition_tableau`
  - Type: `class`

- **sage.combinat.composition_tableau.CompositionTableaux_all**
  - Entity: `CompositionTableaux_all`
  - Module: `sage.combinat.composition_tableau`
  - Type: `class`

- **sage.combinat.composition_tableau.CompositionTableaux_shape**
  - Entity: `CompositionTableaux_shape`
  - Module: `sage.combinat.composition_tableau`
  - Type: `class`

- **sage.combinat.composition_tableau.CompositionTableaux_size**
  - Entity: `CompositionTableaux_size`
  - Module: `sage.combinat.composition_tableau`
  - Type: `class`

- **sage.combinat.dyck_word.CompleteDyckWords**
  - Entity: `CompleteDyckWords`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.CompleteDyckWords_all**
  - Entity: `CompleteDyckWords_all`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.CompleteDyckWords_size**
  - Entity: `CompleteDyckWords_size`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.DyckWord**
  - Entity: `DyckWord`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.DyckWordBacktracker**
  - Entity: `DyckWordBacktracker`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.DyckWord_complete**
  - Entity: `DyckWord_complete`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.DyckWords**
  - Entity: `DyckWords`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.DyckWords_all**
  - Entity: `DyckWords_all`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.DyckWords_size**
  - Entity: `DyckWords_size`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

- **sage.combinat.dyck_word.height_poset**
  - Entity: `height_poset`
  - Module: `sage.combinat.dyck_word`
  - Type: `class`

#### FUNCTION (9 entries)

- **sage.combinat.combination.Combinations**
  - Entity: `Combinations`
  - Module: `sage.combinat.combination`
  - Type: `function`

- **sage.combinat.combination.from_rank**
  - Entity: `from_rank`
  - Module: `sage.combinat.combination`
  - Type: `function`

- **sage.combinat.combination.rank**
  - Entity: `rank`
  - Module: `sage.combinat.combination`
  - Type: `function`

- **sage.combinat.composition.composition_iterator_fast**
  - Entity: `composition_iterator_fast`
  - Module: `sage.combinat.composition`
  - Type: `function`

- **sage.combinat.dyck_word.is_a**
  - Entity: `is_a`
  - Module: `sage.combinat.dyck_word`
  - Type: `function`

- **sage.combinat.dyck_word.is_area_sequence**
  - Entity: `is_area_sequence`
  - Module: `sage.combinat.dyck_word`
  - Type: `function`

- **sage.combinat.dyck_word.pealing**
  - Entity: `pealing`
  - Module: `sage.combinat.dyck_word`
  - Type: `function`

- **sage.combinat.dyck_word.replace_parens**
  - Entity: `replace_parens`
  - Module: `sage.combinat.dyck_word`
  - Type: `function`

- **sage.combinat.dyck_word.replace_symbols**
  - Entity: `replace_symbols`
  - Module: `sage.combinat.dyck_word`
  - Type: `function`

#### MODULE (5 entries)

- **sage.combinat.combination**
  - Entity: `combination`
  - Module: `sage.combinat.combination`
  - Type: `module`

- **sage.combinat.composition**
  - Entity: `composition`
  - Module: `sage.combinat.composition`
  - Type: `module`

- **sage.combinat.composition_signed**
  - Entity: `composition_signed`
  - Module: `sage.combinat.composition_signed`
  - Type: `module`

- **sage.combinat.composition_tableau**
  - Entity: `composition_tableau`
  - Module: `sage.combinat.composition_tableau`
  - Type: `module`

- **sage.combinat.dyck_word**
  - Entity: `dyck_word`
  - Module: `sage.combinat.dyck_word`
  - Type: `module`


### Part 04 (262 entries)

#### ATTRIBUTE (9 entries)

- **sage.combinat.partition.options**
  - Entity: `options`
  - Module: `sage.combinat.partition`
  - Type: `attribute`

- **sage.combinat.partition_tuple.options**
  - Entity: `options`
  - Module: `sage.combinat.partition_tuple`
  - Type: `attribute`

- **sage.combinat.permutation.options**
  - Entity: `options`
  - Module: `sage.combinat.permutation`
  - Type: `attribute`

- **sage.combinat.posets.moebius_algebra.characteristic_basis**
  - Entity: `characteristic_basis`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `attribute`

- **sage.combinat.posets.moebius_algebra.idempotent**
  - Entity: `idempotent`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `attribute`

- **sage.combinat.posets.moebius_algebra.kazhdan_lusztig**
  - Entity: `kazhdan_lusztig`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `attribute`

- **sage.combinat.posets.moebius_algebra.natural**
  - Entity: `natural`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `attribute`

- **sage.combinat.posets.poset_examples.posets**
  - Entity: `posets`
  - Module: `sage.combinat.posets.poset_examples`
  - Type: `attribute`

- **sage.combinat.posets.poset_examples.sage**
  - Entity: `sage`
  - Module: `sage.combinat.posets.poset_examples`
  - Type: `attribute`

#### CLASS (165 entries)

- **sage.combinat.partition.OrderedPartitions**
  - Entity: `OrderedPartitions`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partition**
  - Entity: `Partition`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions**
  - Entity: `Partitions`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.PartitionsGreatestEQ**
  - Entity: `PartitionsGreatestEQ`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.PartitionsGreatestLE**
  - Entity: `PartitionsGreatestLE`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.PartitionsInBox**
  - Entity: `PartitionsInBox`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_all**
  - Entity: `Partitions_all`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_all_bounded**
  - Entity: `Partitions_all_bounded`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_all_constrained**
  - Entity: `Partitions_all_constrained`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_constraints**
  - Entity: `Partitions_constraints`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_ending**
  - Entity: `Partitions_ending`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_length_and_parts_constrained**
  - Entity: `Partitions_length_and_parts_constrained`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_n**
  - Entity: `Partitions_n`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_nk**
  - Entity: `Partitions_nk`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_parts_in**
  - Entity: `Partitions_parts_in`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_starting**
  - Entity: `Partitions_starting`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.Partitions_with_constraints**
  - Entity: `Partitions_with_constraints`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RegularPartitions**
  - Entity: `RegularPartitions`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RegularPartitions_all**
  - Entity: `RegularPartitions_all`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RegularPartitions_bounded**
  - Entity: `RegularPartitions_bounded`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RegularPartitions_n**
  - Entity: `RegularPartitions_n`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RegularPartitions_truncated**
  - Entity: `RegularPartitions_truncated`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RestrictedPartitions_all**
  - Entity: `RestrictedPartitions_all`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RestrictedPartitions_generic**
  - Entity: `RestrictedPartitions_generic`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition.RestrictedPartitions_n**
  - Entity: `RestrictedPartitions_n`
  - Module: `sage.combinat.partition`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_ak**
  - Entity: `PartitionAlgebraElement_ak`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_bk**
  - Entity: `PartitionAlgebraElement_bk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_generic**
  - Entity: `PartitionAlgebraElement_generic`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_pk**
  - Entity: `PartitionAlgebraElement_pk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_prk**
  - Entity: `PartitionAlgebraElement_prk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_rk**
  - Entity: `PartitionAlgebraElement_rk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_sk**
  - Entity: `PartitionAlgebraElement_sk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebraElement_tk**
  - Entity: `PartitionAlgebraElement_tk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_ak**
  - Entity: `PartitionAlgebra_ak`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_bk**
  - Entity: `PartitionAlgebra_bk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_generic**
  - Entity: `PartitionAlgebra_generic`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_pk**
  - Entity: `PartitionAlgebra_pk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_prk**
  - Entity: `PartitionAlgebra_prk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_rk**
  - Entity: `PartitionAlgebra_rk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_sk**
  - Entity: `PartitionAlgebra_sk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.PartitionAlgebra_tk**
  - Entity: `PartitionAlgebra_tk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsAk_k**
  - Entity: `SetPartitionsAk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsAkhalf_k**
  - Entity: `SetPartitionsAkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsBk_k**
  - Entity: `SetPartitionsBk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsBkhalf_k**
  - Entity: `SetPartitionsBkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsIk_k**
  - Entity: `SetPartitionsIk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsIkhalf_k**
  - Entity: `SetPartitionsIkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsPRk_k**
  - Entity: `SetPartitionsPRk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsPRkhalf_k**
  - Entity: `SetPartitionsPRkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsPk_k**
  - Entity: `SetPartitionsPk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsPkhalf_k**
  - Entity: `SetPartitionsPkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsRk_k**
  - Entity: `SetPartitionsRk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsRkhalf_k**
  - Entity: `SetPartitionsRkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsSk_k**
  - Entity: `SetPartitionsSk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsSkhalf_k**
  - Entity: `SetPartitionsSkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsTk_k**
  - Entity: `SetPartitionsTk_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsTkhalf_k**
  - Entity: `SetPartitionsTkhalf_k`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_algebra.SetPartitionsXkElement**
  - Entity: `SetPartitionsXkElement`
  - Module: `sage.combinat.partition_algebra`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevCrystalMixin**
  - Entity: `KleshchevCrystalMixin`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevPartition**
  - Entity: `KleshchevPartition`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevPartitionCrystal**
  - Entity: `KleshchevPartitionCrystal`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevPartitionTuple**
  - Entity: `KleshchevPartitionTuple`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevPartitionTupleCrystal**
  - Entity: `KleshchevPartitionTupleCrystal`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevPartitions**
  - Entity: `KleshchevPartitions`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevPartitions_all**
  - Entity: `KleshchevPartitions_all`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_kleshchev.KleshchevPartitions_size**
  - Entity: `KleshchevPartitions_size`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `class`

- **sage.combinat.partition_shifting_algebras.Element**
  - Entity: `Element`
  - Module: `sage.combinat.partition_shifting_algebras`
  - Type: `class`

- **sage.combinat.partition_shifting_algebras.ShiftingOperatorAlgebra**
  - Entity: `ShiftingOperatorAlgebra`
  - Module: `sage.combinat.partition_shifting_algebras`
  - Type: `class`

- **sage.combinat.partition_shifting_algebras.ShiftingSequenceSpace**
  - Entity: `ShiftingSequenceSpace`
  - Module: `sage.combinat.partition_shifting_algebras`
  - Type: `class`

- **sage.combinat.partition_tuple.PartitionTuple**
  - Entity: `PartitionTuple`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.PartitionTuples**
  - Entity: `PartitionTuples`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.PartitionTuples_all**
  - Entity: `PartitionTuples_all`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.PartitionTuples_level**
  - Entity: `PartitionTuples_level`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.PartitionTuples_level_size**
  - Entity: `PartitionTuples_level_size`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.PartitionTuples_size**
  - Entity: `PartitionTuples_size`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.RegularPartitionTuples**
  - Entity: `RegularPartitionTuples`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.RegularPartitionTuples_all**
  - Entity: `RegularPartitionTuples_all`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.RegularPartitionTuples_level**
  - Entity: `RegularPartitionTuples_level`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.RegularPartitionTuples_level_size**
  - Entity: `RegularPartitionTuples_level_size`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.partition_tuple.RegularPartitionTuples_size**
  - Entity: `RegularPartitionTuples_size`
  - Module: `sage.combinat.partition_tuple`
  - Type: `class`

- **sage.combinat.perfect_matching.PerfectMatching**
  - Entity: `PerfectMatching`
  - Module: `sage.combinat.perfect_matching`
  - Type: `class`

- **sage.combinat.perfect_matching.PerfectMatchings**
  - Entity: `PerfectMatchings`
  - Module: `sage.combinat.perfect_matching`
  - Type: `class`

- **sage.combinat.permutation.Arrangements**
  - Entity: `Arrangements`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Arrangements_msetk**
  - Entity: `Arrangements_msetk`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Arrangements_setk**
  - Entity: `Arrangements_setk`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.CyclicPermutations**
  - Entity: `CyclicPermutations`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.CyclicPermutationsOfPartition**
  - Entity: `CyclicPermutationsOfPartition`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Element**
  - Entity: `Element`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.PatternAvoider**
  - Entity: `PatternAvoider`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Permutation**
  - Entity: `Permutation`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Permutations**
  - Entity: `Permutations`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.PermutationsNK**
  - Entity: `PermutationsNK`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Permutations_mset**
  - Entity: `Permutations_mset`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Permutations_msetk**
  - Entity: `Permutations_msetk`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Permutations_nk**
  - Entity: `Permutations_nk`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Permutations_set**
  - Entity: `Permutations_set`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.Permutations_setk**
  - Entity: `Permutations_setk`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_all**
  - Entity: `StandardPermutations_all`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_all_avoiding**
  - Entity: `StandardPermutations_all_avoiding`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_12**
  - Entity: `StandardPermutations_avoiding_12`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_123**
  - Entity: `StandardPermutations_avoiding_123`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_132**
  - Entity: `StandardPermutations_avoiding_132`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_21**
  - Entity: `StandardPermutations_avoiding_21`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_213**
  - Entity: `StandardPermutations_avoiding_213`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_231**
  - Entity: `StandardPermutations_avoiding_231`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_312**
  - Entity: `StandardPermutations_avoiding_312`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_321**
  - Entity: `StandardPermutations_avoiding_321`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_avoiding_generic**
  - Entity: `StandardPermutations_avoiding_generic`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_bruhat_greater**
  - Entity: `StandardPermutations_bruhat_greater`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_bruhat_smaller**
  - Entity: `StandardPermutations_bruhat_smaller`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_descents**
  - Entity: `StandardPermutations_descents`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_n**
  - Entity: `StandardPermutations_n`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_n_abstract**
  - Entity: `StandardPermutations_n_abstract`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_recoils**
  - Entity: `StandardPermutations_recoils`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_recoilsfatter**
  - Entity: `StandardPermutations_recoilsfatter`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.permutation.StandardPermutations_recoilsfiner**
  - Entity: `StandardPermutations_recoilsfiner`
  - Module: `sage.combinat.permutation`
  - Type: `class`

- **sage.combinat.posets.cartesian_product.CartesianProductPoset**
  - Entity: `CartesianProductPoset`
  - Module: `sage.combinat.posets.cartesian_product`
  - Type: `class`

- **sage.combinat.posets.cartesian_product.Element**
  - Entity: `Element`
  - Module: `sage.combinat.posets.cartesian_product`
  - Type: `class`

- **sage.combinat.posets.d_complete.DCompletePoset**
  - Entity: `DCompletePoset`
  - Module: `sage.combinat.posets.d_complete`
  - Type: `class`

- **sage.combinat.posets.elements.JoinSemilatticeElement**
  - Entity: `JoinSemilatticeElement`
  - Module: `sage.combinat.posets.elements`
  - Type: `class`

- **sage.combinat.posets.elements.LatticePosetElement**
  - Entity: `LatticePosetElement`
  - Module: `sage.combinat.posets.elements`
  - Type: `class`

- **sage.combinat.posets.elements.MeetSemilatticeElement**
  - Entity: `MeetSemilatticeElement`
  - Module: `sage.combinat.posets.elements`
  - Type: `class`

- **sage.combinat.posets.elements.PosetElement**
  - Entity: `PosetElement`
  - Module: `sage.combinat.posets.elements`
  - Type: `class`

- **sage.combinat.posets.forest.ForestPoset**
  - Entity: `ForestPoset`
  - Module: `sage.combinat.posets.forest`
  - Type: `class`

- **sage.combinat.posets.hasse_diagram.HasseDiagram**
  - Entity: `HasseDiagram`
  - Module: `sage.combinat.posets.hasse_diagram`
  - Type: `class`

- **sage.combinat.posets.incidence_algebras.Element**
  - Entity: `Element`
  - Module: `sage.combinat.posets.incidence_algebras`
  - Type: `class`

- **sage.combinat.posets.incidence_algebras.IncidenceAlgebra**
  - Entity: `IncidenceAlgebra`
  - Module: `sage.combinat.posets.incidence_algebras`
  - Type: `class`

- **sage.combinat.posets.incidence_algebras.ReducedIncidenceAlgebra**
  - Entity: `ReducedIncidenceAlgebra`
  - Module: `sage.combinat.posets.incidence_algebras`
  - Type: `class`

- **sage.combinat.posets.lattices.FiniteJoinSemilattice**
  - Entity: `FiniteJoinSemilattice`
  - Module: `sage.combinat.posets.lattices`
  - Type: `class`

- **sage.combinat.posets.lattices.FiniteLatticePoset**
  - Entity: `FiniteLatticePoset`
  - Module: `sage.combinat.posets.lattices`
  - Type: `class`

- **sage.combinat.posets.lattices.FiniteMeetSemilattice**
  - Entity: `FiniteMeetSemilattice`
  - Module: `sage.combinat.posets.lattices`
  - Type: `class`

- **sage.combinat.posets.linear_extensions.LinearExtensionOfPoset**
  - Entity: `LinearExtensionOfPoset`
  - Module: `sage.combinat.posets.linear_extensions`
  - Type: `class`

- **sage.combinat.posets.linear_extensions.LinearExtensionsOfForest**
  - Entity: `LinearExtensionsOfForest`
  - Module: `sage.combinat.posets.linear_extensions`
  - Type: `class`

- **sage.combinat.posets.linear_extensions.LinearExtensionsOfMobile**
  - Entity: `LinearExtensionsOfMobile`
  - Module: `sage.combinat.posets.linear_extensions`
  - Type: `class`

- **sage.combinat.posets.linear_extensions.LinearExtensionsOfPoset**
  - Entity: `LinearExtensionsOfPoset`
  - Module: `sage.combinat.posets.linear_extensions`
  - Type: `class`

- **sage.combinat.posets.linear_extensions.LinearExtensionsOfPosetWithHooks**
  - Entity: `LinearExtensionsOfPosetWithHooks`
  - Module: `sage.combinat.posets.linear_extensions`
  - Type: `class`

- **sage.combinat.posets.mobile.MobilePoset**
  - Entity: `MobilePoset`
  - Module: `sage.combinat.posets.mobile`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.BasisAbstract**
  - Entity: `BasisAbstract`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.C**
  - Entity: `C`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.E**
  - Entity: `E`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.ElementMethods**
  - Entity: `ElementMethods`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.I**
  - Entity: `I`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.KL**
  - Entity: `KL`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.MoebiusAlgebra**
  - Entity: `MoebiusAlgebra`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.MoebiusAlgebraBases**
  - Entity: `MoebiusAlgebraBases`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.ParentMethods**
  - Entity: `ParentMethods`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.moebius_algebra.QuantumMoebiusAlgebra**
  - Entity: `QuantumMoebiusAlgebra`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `class`

- **sage.combinat.posets.poset_examples.Posets**
  - Entity: `Posets`
  - Module: `sage.combinat.posets.poset_examples`
  - Type: `class`

- **sage.combinat.posets.posets.FinitePoset**
  - Entity: `FinitePoset`
  - Module: `sage.combinat.posets.posets`
  - Type: `class`

- **sage.combinat.posets.posets.FinitePosets_n**
  - Entity: `FinitePosets_n`
  - Module: `sage.combinat.posets.posets`
  - Type: `class`

- **sage.combinat.set_partition.AbstractSetPartition**
  - Entity: `AbstractSetPartition`
  - Module: `sage.combinat.set_partition`
  - Type: `class`

- **sage.combinat.set_partition.SetPartition**
  - Entity: `SetPartition`
  - Module: `sage.combinat.set_partition`
  - Type: `class`

- **sage.combinat.set_partition.SetPartitions**
  - Entity: `SetPartitions`
  - Module: `sage.combinat.set_partition`
  - Type: `class`

- **sage.combinat.set_partition.SetPartitions_all**
  - Entity: `SetPartitions_all`
  - Module: `sage.combinat.set_partition`
  - Type: `class`

- **sage.combinat.set_partition.SetPartitions_set**
  - Entity: `SetPartitions_set`
  - Module: `sage.combinat.set_partition`
  - Type: `class`

- **sage.combinat.set_partition.SetPartitions_setn**
  - Entity: `SetPartitions_setn`
  - Module: `sage.combinat.set_partition`
  - Type: `class`

- **sage.combinat.set_partition.SetPartitions_setparts**
  - Entity: `SetPartitions_setparts`
  - Module: `sage.combinat.set_partition`
  - Type: `class`

- **sage.combinat.set_partition_ordered.Element**
  - Entity: `Element`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

- **sage.combinat.set_partition_ordered.OrderedSetPartition**
  - Entity: `OrderedSetPartition`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

- **sage.combinat.set_partition_ordered.OrderedSetPartitions**
  - Entity: `OrderedSetPartitions`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

- **sage.combinat.set_partition_ordered.OrderedSetPartitions_all**
  - Entity: `OrderedSetPartitions_all`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

- **sage.combinat.set_partition_ordered.OrderedSetPartitions_s**
  - Entity: `OrderedSetPartitions_s`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

- **sage.combinat.set_partition_ordered.OrderedSetPartitions_scomp**
  - Entity: `OrderedSetPartitions_scomp`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

- **sage.combinat.set_partition_ordered.OrderedSetPartitions_sn**
  - Entity: `OrderedSetPartitions_sn`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

- **sage.combinat.set_partition_ordered.SplitNK**
  - Entity: `SplitNK`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `class`

#### FUNCTION (63 entries)

- **sage.combinat.partition.number_of_partitions**
  - Entity: `number_of_partitions`
  - Module: `sage.combinat.partition`
  - Type: `function`

- **sage.combinat.partition.number_of_partitions_length**
  - Entity: `number_of_partitions_length`
  - Module: `sage.combinat.partition`
  - Type: `function`

- **sage.combinat.partition.number_of_partitions_max_length_max_part**
  - Entity: `number_of_partitions_max_length_max_part`
  - Module: `sage.combinat.partition`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsAk**
  - Entity: `SetPartitionsAk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsBk**
  - Entity: `SetPartitionsBk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsIk**
  - Entity: `SetPartitionsIk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsPRk**
  - Entity: `SetPartitionsPRk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsPk**
  - Entity: `SetPartitionsPk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsRk**
  - Entity: `SetPartitionsRk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsSk**
  - Entity: `SetPartitionsSk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.SetPartitionsTk**
  - Entity: `SetPartitionsTk`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.identity**
  - Entity: `identity`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.is_planar**
  - Entity: `is_planar`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.pair_to_graph**
  - Entity: `pair_to_graph`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.propagating_number**
  - Entity: `propagating_number`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.set_partition_composition**
  - Entity: `set_partition_composition`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.to_graph**
  - Entity: `to_graph`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partition_algebra.to_set_partition**
  - Entity: `to_set_partition`
  - Module: `sage.combinat.partition_algebra`
  - Type: `function`

- **sage.combinat.partitions.AccelAsc_iterator**
  - Entity: `AccelAsc_iterator`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.AccelAsc_next**
  - Entity: `AccelAsc_next`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.AccelDesc_iterator**
  - Entity: `AccelDesc_iterator`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.AccelDesc_next**
  - Entity: `AccelDesc_next`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.ZS1_iterator**
  - Entity: `ZS1_iterator`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.ZS1_iterator_nk**
  - Entity: `ZS1_iterator_nk`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.ZS1_next**
  - Entity: `ZS1_next`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.ZS2_iterator**
  - Entity: `ZS2_iterator`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.partitions.ZS2_next**
  - Entity: `ZS2_next`
  - Module: `sage.combinat.partitions`
  - Type: `function`

- **sage.combinat.permutation.bistochastic_as_sum_of_permutations**
  - Entity: `bistochastic_as_sum_of_permutations`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.bounded_affine_permutation**
  - Entity: `bounded_affine_permutation`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.bruhat_lequal**
  - Entity: `bruhat_lequal`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.descents_composition_first**
  - Entity: `descents_composition_first`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.descents_composition_last**
  - Entity: `descents_composition_last`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.descents_composition_list**
  - Entity: `descents_composition_list`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_cycles**
  - Entity: `from_cycles`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_inversion_vector**
  - Entity: `from_inversion_vector`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_lehmer_cocode**
  - Entity: `from_lehmer_cocode`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_lehmer_code**
  - Entity: `from_lehmer_code`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_major_code**
  - Entity: `from_major_code`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_permutation_group_element**
  - Entity: `from_permutation_group_element`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_rank**
  - Entity: `from_rank`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.from_reduced_word**
  - Entity: `from_reduced_word`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.permutohedron_lequal**
  - Entity: `permutohedron_lequal`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation.to_standard**
  - Entity: `to_standard`
  - Module: `sage.combinat.permutation`
  - Type: `function`

- **sage.combinat.permutation_cython.left_action_product**
  - Entity: `left_action_product`
  - Module: `sage.combinat.permutation_cython`
  - Type: `function`

- **sage.combinat.permutation_cython.left_action_same_n**
  - Entity: `left_action_same_n`
  - Module: `sage.combinat.permutation_cython`
  - Type: `function`

- **sage.combinat.permutation_cython.map_to_list**
  - Entity: `map_to_list`
  - Module: `sage.combinat.permutation_cython`
  - Type: `function`

- **sage.combinat.permutation_cython.next_perm**
  - Entity: `next_perm`
  - Module: `sage.combinat.permutation_cython`
  - Type: `function`

- **sage.combinat.permutation_cython.permutation_iterator_transposition_list**
  - Entity: `permutation_iterator_transposition_list`
  - Module: `sage.combinat.permutation_cython`
  - Type: `function`

- **sage.combinat.permutation_cython.right_action_product**
  - Entity: `right_action_product`
  - Module: `sage.combinat.permutation_cython`
  - Type: `function`

- **sage.combinat.permutation_cython.right_action_same_n**
  - Entity: `right_action_same_n`
  - Module: `sage.combinat.permutation_cython`
  - Type: `function`

- **sage.combinat.posets.hasse_diagram.LatticeError**
  - Entity: `LatticeError`
  - Module: `sage.combinat.posets.hasse_diagram`
  - Type: `function`

- **sage.combinat.posets.lattices.JoinSemilattice**
  - Entity: `JoinSemilattice`
  - Module: `sage.combinat.posets.lattices`
  - Type: `function`

- **sage.combinat.posets.lattices.LatticePoset**
  - Entity: `LatticePoset`
  - Module: `sage.combinat.posets.lattices`
  - Type: `function`

- **sage.combinat.posets.lattices.MeetSemilattice**
  - Entity: `MeetSemilattice`
  - Module: `sage.combinat.posets.lattices`
  - Type: `function`

- **sage.combinat.posets.poset_examples.check_int**
  - Entity: `check_int`
  - Module: `sage.combinat.posets.poset_examples`
  - Type: `function`

- **sage.combinat.posets.posets.Poset**
  - Entity: `Poset`
  - Module: `sage.combinat.posets.posets`
  - Type: `function`

- **sage.combinat.posets.posets.is_poset**
  - Entity: `is_poset`
  - Module: `sage.combinat.posets.posets`
  - Type: `function`

- **sage.combinat.set_partition.cyclic_permutations_of_set_partition**
  - Entity: `cyclic_permutations_of_set_partition`
  - Module: `sage.combinat.set_partition`
  - Type: `function`

- **sage.combinat.set_partition.cyclic_permutations_of_set_partition_iterator**
  - Entity: `cyclic_permutations_of_set_partition_iterator`
  - Module: `sage.combinat.set_partition`
  - Type: `function`

- **sage.combinat.set_partition_iterator.set_partition_iterator**
  - Entity: `set_partition_iterator`
  - Module: `sage.combinat.set_partition_iterator`
  - Type: `function`

- **sage.combinat.set_partition_iterator.set_partition_iterator_blocks**
  - Entity: `set_partition_iterator_blocks`
  - Module: `sage.combinat.set_partition_iterator`
  - Type: `function`

- **sage.combinat.set_partition_ordered.multiset_permutation_next_lex**
  - Entity: `multiset_permutation_next_lex`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `function`

- **sage.combinat.set_partition_ordered.multiset_permutation_to_ordered_set_partition**
  - Entity: `multiset_permutation_to_ordered_set_partition`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `function`

#### MODULE (25 entries)

- **sage.combinat.partition**
  - Entity: `partition`
  - Module: `sage.combinat.partition`
  - Type: `module`

- **sage.combinat.partition_algebra**
  - Entity: `partition_algebra`
  - Module: `sage.combinat.partition_algebra`
  - Type: `module`

- **sage.combinat.partition_kleshchev**
  - Entity: `partition_kleshchev`
  - Module: `sage.combinat.partition_kleshchev`
  - Type: `module`

- **sage.combinat.partition_shifting_algebras**
  - Entity: `partition_shifting_algebras`
  - Module: `sage.combinat.partition_shifting_algebras`
  - Type: `module`

- **sage.combinat.partition_tuple**
  - Entity: `partition_tuple`
  - Module: `sage.combinat.partition_tuple`
  - Type: `module`

- **sage.combinat.partitions**
  - Entity: `partitions`
  - Module: `sage.combinat.partitions`
  - Type: `module`

- **sage.combinat.perfect_matching**
  - Entity: `perfect_matching`
  - Module: `sage.combinat.perfect_matching`
  - Type: `module`

- **sage.combinat.permutation**
  - Entity: `permutation`
  - Module: `sage.combinat.permutation`
  - Type: `module`

- **sage.combinat.permutation_cython**
  - Entity: `permutation_cython`
  - Module: `sage.combinat.permutation_cython`
  - Type: `module`

- **sage.combinat.posets.all**
  - Entity: `all`
  - Module: `sage.combinat.posets.all`
  - Type: `module`

- **sage.combinat.posets.cartesian_product**
  - Entity: `cartesian_product`
  - Module: `sage.combinat.posets.cartesian_product`
  - Type: `module`

- **sage.combinat.posets.d_complete**
  - Entity: `d_complete`
  - Module: `sage.combinat.posets.d_complete`
  - Type: `module`

- **sage.combinat.posets.elements**
  - Entity: `elements`
  - Module: `sage.combinat.posets.elements`
  - Type: `module`

- **sage.combinat.posets.forest**
  - Entity: `forest`
  - Module: `sage.combinat.posets.forest`
  - Type: `module`

- **sage.combinat.posets.hasse_diagram**
  - Entity: `hasse_diagram`
  - Module: `sage.combinat.posets.hasse_diagram`
  - Type: `module`

- **sage.combinat.posets.incidence_algebras**
  - Entity: `incidence_algebras`
  - Module: `sage.combinat.posets.incidence_algebras`
  - Type: `module`

- **sage.combinat.posets.lattices**
  - Entity: `lattices`
  - Module: `sage.combinat.posets.lattices`
  - Type: `module`

- **sage.combinat.posets.linear_extensions**
  - Entity: `linear_extensions`
  - Module: `sage.combinat.posets.linear_extensions`
  - Type: `module`

- **sage.combinat.posets.mobile**
  - Entity: `mobile`
  - Module: `sage.combinat.posets.mobile`
  - Type: `module`

- **sage.combinat.posets.moebius_algebra**
  - Entity: `moebius_algebra`
  - Module: `sage.combinat.posets.moebius_algebra`
  - Type: `module`

- **sage.combinat.posets.poset_examples**
  - Entity: `poset_examples`
  - Module: `sage.combinat.posets.poset_examples`
  - Type: `module`

- **sage.combinat.posets.posets**
  - Entity: `posets`
  - Module: `sage.combinat.posets.posets`
  - Type: `module`

- **sage.combinat.set_partition**
  - Entity: `set_partition`
  - Module: `sage.combinat.set_partition`
  - Type: `module`

- **sage.combinat.set_partition_iterator**
  - Entity: `set_partition_iterator`
  - Module: `sage.combinat.set_partition_iterator`
  - Type: `module`

- **sage.combinat.set_partition_ordered**
  - Entity: `set_partition_ordered`
  - Module: `sage.combinat.set_partition_ordered`
  - Type: `module`


### Part 05 (102 entries)

#### ATTRIBUTE (3 entries)

- **sage.combinat.tableau.options**
  - Entity: `options`
  - Module: `sage.combinat.tableau`
  - Type: `attribute`

- **sage.combinat.tableau_tuple.level_one_parent_class**
  - Entity: `level_one_parent_class`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `attribute`

- **sage.combinat.tableau_tuple.options**
  - Entity: `options`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `attribute`

#### CLASS (74 entries)

- **sage.combinat.tableau.IncreasingTableau**
  - Entity: `IncreasingTableau`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux**
  - Entity: `IncreasingTableaux`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux_all**
  - Entity: `IncreasingTableaux_all`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux_shape**
  - Entity: `IncreasingTableaux_shape`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux_shape_inf**
  - Entity: `IncreasingTableaux_shape_inf`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux_shape_weight**
  - Entity: `IncreasingTableaux_shape_weight`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux_size**
  - Entity: `IncreasingTableaux_size`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux_size_inf**
  - Entity: `IncreasingTableaux_size_inf`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.IncreasingTableaux_size_weight**
  - Entity: `IncreasingTableaux_size_weight`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.RowStandardTableau**
  - Entity: `RowStandardTableau`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.RowStandardTableaux**
  - Entity: `RowStandardTableaux`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.RowStandardTableaux_all**
  - Entity: `RowStandardTableaux_all`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.RowStandardTableaux_shape**
  - Entity: `RowStandardTableaux_shape`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.RowStandardTableaux_size**
  - Entity: `RowStandardTableaux_size`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableau**
  - Entity: `SemistandardTableau`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux**
  - Entity: `SemistandardTableaux`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux_all**
  - Entity: `SemistandardTableaux_all`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux_shape**
  - Entity: `SemistandardTableaux_shape`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux_shape_inf**
  - Entity: `SemistandardTableaux_shape_inf`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux_shape_weight**
  - Entity: `SemistandardTableaux_shape_weight`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux_size**
  - Entity: `SemistandardTableaux_size`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux_size_inf**
  - Entity: `SemistandardTableaux_size_inf`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.SemistandardTableaux_size_weight**
  - Entity: `SemistandardTableaux_size_weight`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.StandardTableau**
  - Entity: `StandardTableau`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.StandardTableaux**
  - Entity: `StandardTableaux`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.StandardTableaux_all**
  - Entity: `StandardTableaux_all`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.StandardTableaux_shape**
  - Entity: `StandardTableaux_shape`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.StandardTableaux_size**
  - Entity: `StandardTableaux_size`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.Tableau**
  - Entity: `Tableau`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.Tableau_class**
  - Entity: `Tableau_class`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.Tableaux**
  - Entity: `Tableaux`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.Tableaux_all**
  - Entity: `Tableaux_all`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau.Tableaux_size**
  - Entity: `Tableaux_size`
  - Module: `sage.combinat.tableau`
  - Type: `class`

- **sage.combinat.tableau_residues.ResidueSequence**
  - Entity: `ResidueSequence`
  - Module: `sage.combinat.tableau_residues`
  - Type: `class`

- **sage.combinat.tableau_residues.ResidueSequences**
  - Entity: `ResidueSequences`
  - Module: `sage.combinat.tableau_residues`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuple**
  - Entity: `RowStandardTableauTuple`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples**
  - Entity: `RowStandardTableauTuples`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples_all**
  - Entity: `RowStandardTableauTuples_all`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples_level**
  - Entity: `RowStandardTableauTuples_level`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples_level_size**
  - Entity: `RowStandardTableauTuples_level_size`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples_residue**
  - Entity: `RowStandardTableauTuples_residue`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples_residue_shape**
  - Entity: `RowStandardTableauTuples_residue_shape`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples_shape**
  - Entity: `RowStandardTableauTuples_shape`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.RowStandardTableauTuples_size**
  - Entity: `RowStandardTableauTuples_size`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableauTuple**
  - Entity: `StandardTableauTuple`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableauTuples**
  - Entity: `StandardTableauTuples`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableauTuples_all**
  - Entity: `StandardTableauTuples_all`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableauTuples_level**
  - Entity: `StandardTableauTuples_level`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableauTuples_level_size**
  - Entity: `StandardTableauTuples_level_size`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableauTuples_shape**
  - Entity: `StandardTableauTuples_shape`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableauTuples_size**
  - Entity: `StandardTableauTuples_size`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableaux_residue**
  - Entity: `StandardTableaux_residue`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.StandardTableaux_residue_shape**
  - Entity: `StandardTableaux_residue_shape`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.TableauTuple**
  - Entity: `TableauTuple`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.TableauTuples**
  - Entity: `TableauTuples`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.TableauTuples_all**
  - Entity: `TableauTuples_all`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.TableauTuples_level**
  - Entity: `TableauTuples_level`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.TableauTuples_level_size**
  - Entity: `TableauTuples_level_size`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.combinat.tableau_tuple.TableauTuples_size**
  - Entity: `TableauTuples_size`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `class`

- **sage.crypto.classical.AffineCryptosystem**
  - Entity: `AffineCryptosystem`
  - Module: `sage.crypto.classical`
  - Type: `class`

- **sage.crypto.classical.HillCryptosystem**
  - Entity: `HillCryptosystem`
  - Module: `sage.crypto.classical`
  - Type: `class`

- **sage.crypto.classical.ShiftCryptosystem**
  - Entity: `ShiftCryptosystem`
  - Module: `sage.crypto.classical`
  - Type: `class`

- **sage.crypto.classical.SubstitutionCryptosystem**
  - Entity: `SubstitutionCryptosystem`
  - Module: `sage.crypto.classical`
  - Type: `class`

- **sage.crypto.classical.TranspositionCryptosystem**
  - Entity: `TranspositionCryptosystem`
  - Module: `sage.crypto.classical`
  - Type: `class`

- **sage.crypto.classical.VigenereCryptosystem**
  - Entity: `VigenereCryptosystem`
  - Module: `sage.crypto.classical`
  - Type: `class`

- **sage.crypto.classical_cipher.AffineCipher**
  - Entity: `AffineCipher`
  - Module: `sage.crypto.classical_cipher`
  - Type: `class`

- **sage.crypto.classical_cipher.HillCipher**
  - Entity: `HillCipher`
  - Module: `sage.crypto.classical_cipher`
  - Type: `class`

- **sage.crypto.classical_cipher.ShiftCipher**
  - Entity: `ShiftCipher`
  - Module: `sage.crypto.classical_cipher`
  - Type: `class`

- **sage.crypto.classical_cipher.SubstitutionCipher**
  - Entity: `SubstitutionCipher`
  - Module: `sage.crypto.classical_cipher`
  - Type: `class`

- **sage.crypto.classical_cipher.TranspositionCipher**
  - Entity: `TranspositionCipher`
  - Module: `sage.crypto.classical_cipher`
  - Type: `class`

- **sage.crypto.classical_cipher.VigenereCipher**
  - Entity: `VigenereCipher`
  - Module: `sage.crypto.classical_cipher`
  - Type: `class`

- **sage.crypto.public_key.blum_goldwasser.BlumGoldwasser**
  - Entity: `BlumGoldwasser`
  - Module: `sage.crypto.public_key.blum_goldwasser`
  - Type: `class`

- **sage.databases.cremona.LargeCremonaDatabase**
  - Entity: `LargeCremonaDatabase`
  - Module: `sage.databases.cremona`
  - Type: `class`

- **sage.databases.cremona.MiniCremonaDatabase**
  - Entity: `MiniCremonaDatabase`
  - Module: `sage.databases.cremona`
  - Type: `class`

#### FUNCTION (17 entries)

- **sage.combinat.tableau.from_chain**
  - Entity: `from_chain`
  - Module: `sage.combinat.tableau`
  - Type: `function`

- **sage.combinat.tableau.from_shape_and_word**
  - Entity: `from_shape_and_word`
  - Module: `sage.combinat.tableau`
  - Type: `function`

- **sage.combinat.tableau.symmetric_group_action_on_values**
  - Entity: `symmetric_group_action_on_values`
  - Module: `sage.combinat.tableau`
  - Type: `function`

- **sage.combinat.tableau.unmatched_places**
  - Entity: `unmatched_places`
  - Module: `sage.combinat.tableau`
  - Type: `function`

- **sage.databases.cremona.CremonaDatabase**
  - Entity: `CremonaDatabase`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.build**
  - Entity: `build`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.class_to_int**
  - Entity: `class_to_int`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.cremona_letter_code**
  - Entity: `cremona_letter_code`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.cremona_to_lmfdb**
  - Entity: `cremona_to_lmfdb`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.is_optimal_id**
  - Entity: `is_optimal_id`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.lmfdb_to_cremona**
  - Entity: `lmfdb_to_cremona`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.old_cremona_letter_code**
  - Entity: `old_cremona_letter_code`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.parse_cremona_label**
  - Entity: `parse_cremona_label`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.parse_lmfdb_label**
  - Entity: `parse_lmfdb_label`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.sort_key**
  - Entity: `sort_key`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cremona.split_code**
  - Entity: `split_code`
  - Module: `sage.databases.cremona`
  - Type: `function`

- **sage.databases.cunningham_tables.cunningham_prime_factors**
  - Entity: `cunningham_prime_factors`
  - Module: `sage.databases.cunningham_tables`
  - Type: `function`

#### MODULE (8 entries)

- **sage.combinat.tableau**
  - Entity: `tableau`
  - Module: `sage.combinat.tableau`
  - Type: `module`

- **sage.combinat.tableau_residues**
  - Entity: `tableau_residues`
  - Module: `sage.combinat.tableau_residues`
  - Type: `module`

- **sage.combinat.tableau_tuple**
  - Entity: `tableau_tuple`
  - Module: `sage.combinat.tableau_tuple`
  - Type: `module`

- **sage.crypto.classical**
  - Entity: `classical`
  - Module: `sage.crypto.classical`
  - Type: `module`

- **sage.crypto.classical_cipher**
  - Entity: `classical_cipher`
  - Module: `sage.crypto.classical_cipher`
  - Type: `module`

- **sage.crypto.public_key.blum_goldwasser**
  - Entity: `blum_goldwasser`
  - Module: `sage.crypto.public_key.blum_goldwasser`
  - Type: `module`

- **sage.databases.cremona**
  - Entity: `cremona`
  - Module: `sage.databases.cremona`
  - Type: `module`

- **sage.databases.cunningham_tables**
  - Entity: `cunningham_tables`
  - Module: `sage.databases.cunningham_tables`
  - Type: `module`


### Part 06 (334 entries)

#### ATTRIBUTE (3 entries)

- **sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator.dual**
  - Entity: `dual`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator`
  - Type: `attribute`

- **sage.geometry.polyhedron.combinatorial_polyhedron.polyhedron_face_lattice.dual**
  - Entity: `dual`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.polyhedron_face_lattice`
  - Type: `attribute`

- **sage.geometry.polyhedron.double_description.pair_class**
  - Entity: `pair_class`
  - Module: `sage.geometry.polyhedron.double_description`
  - Type: `attribute`

#### CLASS (183 entries)

- **sage.databases.oeis.FancyTuple**
  - Entity: `FancyTuple`
  - Module: `sage.databases.oeis`
  - Type: `class`

- **sage.databases.oeis.OEIS**
  - Entity: `OEIS`
  - Module: `sage.databases.oeis`
  - Type: `class`

- **sage.databases.oeis.OEISSequence**
  - Entity: `OEISSequence`
  - Module: `sage.databases.oeis`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.affine_ds.DynamicalSystem_affine**
  - Entity: `DynamicalSystem_affine`
  - Module: `sage.dynamics.arithmetic_dynamics.affine_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.affine_ds.DynamicalSystem_affine_field**
  - Entity: `DynamicalSystem_affine_field`
  - Module: `sage.dynamics.arithmetic_dynamics.affine_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.affine_ds.DynamicalSystem_affine_finite_field**
  - Entity: `DynamicalSystem_affine_finite_field`
  - Module: `sage.dynamics.arithmetic_dynamics.affine_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.berkovich_ds.DynamicalSystem_Berkovich**
  - Entity: `DynamicalSystem_Berkovich`
  - Module: `sage.dynamics.arithmetic_dynamics.berkovich_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.berkovich_ds.DynamicalSystem_Berkovich_affine**
  - Entity: `DynamicalSystem_Berkovich_affine`
  - Module: `sage.dynamics.arithmetic_dynamics.berkovich_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.berkovich_ds.DynamicalSystem_Berkovich_projective**
  - Entity: `DynamicalSystem_Berkovich_projective`
  - Module: `sage.dynamics.arithmetic_dynamics.berkovich_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup.DynamicalSemigroup**
  - Entity: `DynamicalSemigroup`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup.DynamicalSemigroup_affine**
  - Entity: `DynamicalSemigroup_affine`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup.DynamicalSemigroup_affine_field**
  - Entity: `DynamicalSemigroup_affine_field`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup.DynamicalSemigroup_affine_finite_field**
  - Entity: `DynamicalSemigroup_affine_finite_field`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup.DynamicalSemigroup_projective**
  - Entity: `DynamicalSemigroup_projective`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup.DynamicalSemigroup_projective_field**
  - Entity: `DynamicalSemigroup_projective_field`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup.DynamicalSemigroup_projective_finite_field**
  - Entity: `DynamicalSemigroup_projective_finite_field`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.generic_ds.DynamicalSystem**
  - Entity: `DynamicalSystem`
  - Module: `sage.dynamics.arithmetic_dynamics.generic_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.product_projective_ds.DynamicalSystem_product_projective**
  - Entity: `DynamicalSystem_product_projective`
  - Module: `sage.dynamics.arithmetic_dynamics.product_projective_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.product_projective_ds.DynamicalSystem_product_projective_field**
  - Entity: `DynamicalSystem_product_projective_field`
  - Module: `sage.dynamics.arithmetic_dynamics.product_projective_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.product_projective_ds.DynamicalSystem_product_projective_finite_field**
  - Entity: `DynamicalSystem_product_projective_finite_field`
  - Module: `sage.dynamics.arithmetic_dynamics.product_projective_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.projective_ds.DynamicalSystem_projective**
  - Entity: `DynamicalSystem_projective`
  - Module: `sage.dynamics.arithmetic_dynamics.projective_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.projective_ds.DynamicalSystem_projective_field**
  - Entity: `DynamicalSystem_projective_field`
  - Module: `sage.dynamics.arithmetic_dynamics.projective_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.projective_ds.DynamicalSystem_projective_finite_field**
  - Entity: `DynamicalSystem_projective_finite_field`
  - Module: `sage.dynamics.arithmetic_dynamics.projective_ds`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.wehlerK3.WehlerK3Surface_field**
  - Entity: `WehlerK3Surface_field`
  - Module: `sage.dynamics.arithmetic_dynamics.wehlerK3`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.wehlerK3.WehlerK3Surface_finite_field**
  - Entity: `WehlerK3Surface_finite_field`
  - Module: `sage.dynamics.arithmetic_dynamics.wehlerK3`
  - Type: `class`

- **sage.dynamics.arithmetic_dynamics.wehlerK3.WehlerK3Surface_ring**
  - Entity: `WehlerK3Surface_ring`
  - Module: `sage.dynamics.arithmetic_dynamics.wehlerK3`
  - Type: `class`

- **sage.dynamics.cellular_automata.elementary.ElementaryCellularAutomata**
  - Entity: `ElementaryCellularAutomata`
  - Module: `sage.dynamics.cellular_automata.elementary`
  - Type: `class`

- **sage.dynamics.cellular_automata.glca.GraftalLaceCellularAutomata**
  - Entity: `GraftalLaceCellularAutomata`
  - Module: `sage.dynamics.cellular_automata.glca`
  - Type: `class`

- **sage.dynamics.cellular_automata.solitons.PeriodicSolitonCellularAutomata**
  - Entity: `PeriodicSolitonCellularAutomata`
  - Module: `sage.dynamics.cellular_automata.solitons`
  - Type: `class`

- **sage.dynamics.cellular_automata.solitons.SolitonCellularAutomata**
  - Entity: `SolitonCellularAutomata`
  - Module: `sage.dynamics.cellular_automata.solitons`
  - Type: `class`

- **sage.dynamics.finite_dynamical_system.DiscreteDynamicalSystem**
  - Entity: `DiscreteDynamicalSystem`
  - Module: `sage.dynamics.finite_dynamical_system`
  - Type: `class`

- **sage.dynamics.finite_dynamical_system.FiniteDynamicalSystem**
  - Entity: `FiniteDynamicalSystem`
  - Module: `sage.dynamics.finite_dynamical_system`
  - Type: `class`

- **sage.dynamics.finite_dynamical_system.InvertibleDiscreteDynamicalSystem**
  - Entity: `InvertibleDiscreteDynamicalSystem`
  - Module: `sage.dynamics.finite_dynamical_system`
  - Type: `class`

- **sage.dynamics.finite_dynamical_system.InvertibleFiniteDynamicalSystem**
  - Entity: `InvertibleFiniteDynamicalSystem`
  - Module: `sage.dynamics.finite_dynamical_system`
  - Type: `class`

- **sage.functions.bessel.Function_Bessel_I**
  - Entity: `Function_Bessel_I`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.Function_Bessel_J**
  - Entity: `Function_Bessel_J`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.Function_Bessel_K**
  - Entity: `Function_Bessel_K`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.Function_Bessel_Y**
  - Entity: `Function_Bessel_Y`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.Function_Hankel1**
  - Entity: `Function_Hankel1`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.Function_Hankel2**
  - Entity: `Function_Hankel2`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.Function_Struve_H**
  - Entity: `Function_Struve_H`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.Function_Struve_L**
  - Entity: `Function_Struve_L`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.SphericalBesselJ**
  - Entity: `SphericalBesselJ`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.SphericalBesselY**
  - Entity: `SphericalBesselY`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.SphericalHankel1**
  - Entity: `SphericalHankel1`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.bessel.SphericalHankel2**
  - Entity: `SphericalHankel2`
  - Module: `sage.functions.bessel`
  - Type: `class`

- **sage.functions.error.Function_Fresnel_cos**
  - Entity: `Function_Fresnel_cos`
  - Module: `sage.functions.error`
  - Type: `class`

- **sage.functions.error.Function_Fresnel_sin**
  - Entity: `Function_Fresnel_sin`
  - Module: `sage.functions.error`
  - Type: `class`

- **sage.functions.error.Function_erf**
  - Entity: `Function_erf`
  - Module: `sage.functions.error`
  - Type: `class`

- **sage.functions.error.Function_erfc**
  - Entity: `Function_erfc`
  - Module: `sage.functions.error`
  - Type: `class`

- **sage.functions.error.Function_erfi**
  - Entity: `Function_erfi`
  - Module: `sage.functions.error`
  - Type: `class`

- **sage.functions.error.Function_erfinv**
  - Entity: `Function_erfinv`
  - Module: `sage.functions.error`
  - Type: `class`

- **sage.functions.exp_integral.Function_cos_integral**
  - Entity: `Function_cos_integral`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_cosh_integral**
  - Entity: `Function_cosh_integral`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_exp_integral**
  - Entity: `Function_exp_integral`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_exp_integral_e**
  - Entity: `Function_exp_integral_e`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_exp_integral_e1**
  - Entity: `Function_exp_integral_e1`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_log_integral**
  - Entity: `Function_log_integral`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_log_integral_offset**
  - Entity: `Function_log_integral_offset`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_sin_integral**
  - Entity: `Function_sin_integral`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.exp_integral.Function_sinh_integral**
  - Entity: `Function_sinh_integral`
  - Module: `sage.functions.exp_integral`
  - Type: `class`

- **sage.functions.hyperbolic.Function_arccosh**
  - Entity: `Function_arccosh`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_arccoth**
  - Entity: `Function_arccoth`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_arccsch**
  - Entity: `Function_arccsch`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_arcsech**
  - Entity: `Function_arcsech`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_arcsinh**
  - Entity: `Function_arcsinh`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_arctanh**
  - Entity: `Function_arctanh`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_cosh**
  - Entity: `Function_cosh`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_coth**
  - Entity: `Function_coth`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_csch**
  - Entity: `Function_csch`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_sech**
  - Entity: `Function_sech`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_sinh**
  - Entity: `Function_sinh`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.hyperbolic.Function_tanh**
  - Entity: `Function_tanh`
  - Module: `sage.functions.hyperbolic`
  - Type: `class`

- **sage.functions.log.Function_dilog**
  - Entity: `Function_dilog`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_exp**
  - Entity: `Function_exp`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_exp_polar**
  - Entity: `Function_exp_polar`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_harmonic_number**
  - Entity: `Function_harmonic_number`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_harmonic_number_generalized**
  - Entity: `Function_harmonic_number_generalized`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_lambert_w**
  - Entity: `Function_lambert_w`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_log1**
  - Entity: `Function_log1`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_log2**
  - Entity: `Function_log2`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.log.Function_polylog**
  - Entity: `Function_polylog`
  - Module: `sage.functions.log`
  - Type: `class`

- **sage.functions.trig.Function_arccos**
  - Entity: `Function_arccos`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_arccot**
  - Entity: `Function_arccot`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_arccsc**
  - Entity: `Function_arccsc`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_arcsec**
  - Entity: `Function_arcsec`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_arcsin**
  - Entity: `Function_arcsin`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_arctan**
  - Entity: `Function_arctan`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_arctan2**
  - Entity: `Function_arctan2`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_cos**
  - Entity: `Function_cos`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_cot**
  - Entity: `Function_cot`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_csc**
  - Entity: `Function_csc`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_sec**
  - Entity: `Function_sec`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_sin**
  - Entity: `Function_sin`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.functions.trig.Function_tan**
  - Entity: `Function_tan`
  - Module: `sage.functions.trig`
  - Type: `class`

- **sage.geometry.cone.ConvexRationalPolyhedralCone**
  - Entity: `ConvexRationalPolyhedralCone`
  - Module: `sage.geometry.cone`
  - Type: `class`

- **sage.geometry.cone.IntegralRayCollection**
  - Entity: `IntegralRayCollection`
  - Module: `sage.geometry.cone`
  - Type: `class`

- **sage.geometry.fan.Cone_of_fan**
  - Entity: `Cone_of_fan`
  - Module: `sage.geometry.fan`
  - Type: `class`

- **sage.geometry.fan.RationalPolyhedralFan**
  - Entity: `RationalPolyhedralFan`
  - Module: `sage.geometry.fan`
  - Type: `class`

- **sage.geometry.fan_morphism.FanMorphism**
  - Entity: `FanMorphism`
  - Module: `sage.geometry.fan_morphism`
  - Type: `class`

- **sage.geometry.polyhedron.backend_cdd.Polyhedron_QQ_cdd**
  - Entity: `Polyhedron_QQ_cdd`
  - Module: `sage.geometry.polyhedron.backend_cdd`
  - Type: `class`

- **sage.geometry.polyhedron.backend_cdd.Polyhedron_cdd**
  - Entity: `Polyhedron_cdd`
  - Module: `sage.geometry.polyhedron.backend_cdd`
  - Type: `class`

- **sage.geometry.polyhedron.backend_cdd_rdf.Polyhedron_RDF_cdd**
  - Entity: `Polyhedron_RDF_cdd`
  - Module: `sage.geometry.polyhedron.backend_cdd_rdf`
  - Type: `class`

- **sage.geometry.polyhedron.backend_field.Polyhedron_field**
  - Entity: `Polyhedron_field`
  - Module: `sage.geometry.polyhedron.backend_field`
  - Type: `class`

- **sage.geometry.polyhedron.backend_normaliz.Polyhedron_QQ_normaliz**
  - Entity: `Polyhedron_QQ_normaliz`
  - Module: `sage.geometry.polyhedron.backend_normaliz`
  - Type: `class`

- **sage.geometry.polyhedron.backend_normaliz.Polyhedron_ZZ_normaliz**
  - Entity: `Polyhedron_ZZ_normaliz`
  - Module: `sage.geometry.polyhedron.backend_normaliz`
  - Type: `class`

- **sage.geometry.polyhedron.backend_normaliz.Polyhedron_normaliz**
  - Entity: `Polyhedron_normaliz`
  - Module: `sage.geometry.polyhedron.backend_normaliz`
  - Type: `class`

- **sage.geometry.polyhedron.backend_number_field.Polyhedron_number_field**
  - Entity: `Polyhedron_number_field`
  - Module: `sage.geometry.polyhedron.backend_number_field`
  - Type: `class`

- **sage.geometry.polyhedron.backend_polymake.Polyhedron_QQ_polymake**
  - Entity: `Polyhedron_QQ_polymake`
  - Module: `sage.geometry.polyhedron.backend_polymake`
  - Type: `class`

- **sage.geometry.polyhedron.backend_polymake.Polyhedron_ZZ_polymake**
  - Entity: `Polyhedron_ZZ_polymake`
  - Module: `sage.geometry.polyhedron.backend_polymake`
  - Type: `class`

- **sage.geometry.polyhedron.backend_polymake.Polyhedron_polymake**
  - Entity: `Polyhedron_polymake`
  - Module: `sage.geometry.polyhedron.backend_polymake`
  - Type: `class`

- **sage.geometry.polyhedron.backend_ppl.Polyhedron_QQ_ppl**
  - Entity: `Polyhedron_QQ_ppl`
  - Module: `sage.geometry.polyhedron.backend_ppl`
  - Type: `class`

- **sage.geometry.polyhedron.backend_ppl.Polyhedron_ZZ_ppl**
  - Entity: `Polyhedron_ZZ_ppl`
  - Module: `sage.geometry.polyhedron.backend_ppl`
  - Type: `class`

- **sage.geometry.polyhedron.backend_ppl.Polyhedron_ppl**
  - Entity: `Polyhedron_ppl`
  - Module: `sage.geometry.polyhedron.backend_ppl`
  - Type: `class`

- **sage.geometry.polyhedron.base.Polyhedron_base**
  - Entity: `Polyhedron_base`
  - Module: `sage.geometry.polyhedron.base`
  - Type: `class`

- **sage.geometry.polyhedron.base0.Polyhedron_base0**
  - Entity: `Polyhedron_base0`
  - Module: `sage.geometry.polyhedron.base0`
  - Type: `class`

- **sage.geometry.polyhedron.base1.Polyhedron_base1**
  - Entity: `Polyhedron_base1`
  - Module: `sage.geometry.polyhedron.base1`
  - Type: `class`

- **sage.geometry.polyhedron.base2.Polyhedron_base2**
  - Entity: `Polyhedron_base2`
  - Module: `sage.geometry.polyhedron.base2`
  - Type: `class`

- **sage.geometry.polyhedron.base3.Polyhedron_base3**
  - Entity: `Polyhedron_base3`
  - Module: `sage.geometry.polyhedron.base3`
  - Type: `class`

- **sage.geometry.polyhedron.base4.Polyhedron_base4**
  - Entity: `Polyhedron_base4`
  - Module: `sage.geometry.polyhedron.base4`
  - Type: `class`

- **sage.geometry.polyhedron.base5.Polyhedron_base5**
  - Entity: `Polyhedron_base5`
  - Module: `sage.geometry.polyhedron.base5`
  - Type: `class`

- **sage.geometry.polyhedron.base6.Polyhedron_base6**
  - Entity: `Polyhedron_base6`
  - Module: `sage.geometry.polyhedron.base6`
  - Type: `class`

- **sage.geometry.polyhedron.base7.Polyhedron_base7**
  - Entity: `Polyhedron_base7`
  - Module: `sage.geometry.polyhedron.base7`
  - Type: `class`

- **sage.geometry.polyhedron.base_QQ.Polyhedron_QQ**
  - Entity: `Polyhedron_QQ`
  - Module: `sage.geometry.polyhedron.base_QQ`
  - Type: `class`

- **sage.geometry.polyhedron.base_RDF.Polyhedron_RDF**
  - Entity: `Polyhedron_RDF`
  - Module: `sage.geometry.polyhedron.base_RDF`
  - Type: `class`

- **sage.geometry.polyhedron.base_ZZ.Polyhedron_ZZ**
  - Entity: `Polyhedron_ZZ`
  - Module: `sage.geometry.polyhedron.base_ZZ`
  - Type: `class`

- **sage.geometry.polyhedron.combinatorial_polyhedron.base.CombinatorialPolyhedron**
  - Entity: `CombinatorialPolyhedron`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.base`
  - Type: `class`

- **sage.geometry.polyhedron.combinatorial_polyhedron.combinatorial_face.CombinatorialFace**
  - Entity: `CombinatorialFace`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.combinatorial_face`
  - Type: `class`

- **sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator.FaceIterator**
  - Entity: `FaceIterator`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator`
  - Type: `class`

- **sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator.FaceIterator_base**
  - Entity: `FaceIterator_base`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator`
  - Type: `class`

- **sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator.FaceIterator_geom**
  - Entity: `FaceIterator_geom`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator`
  - Type: `class`

- **sage.geometry.polyhedron.combinatorial_polyhedron.list_of_faces.ListOfFaces**
  - Entity: `ListOfFaces`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.list_of_faces`
  - Type: `class`

- **sage.geometry.polyhedron.combinatorial_polyhedron.polyhedron_face_lattice.PolyhedronFaceLattice**
  - Entity: `PolyhedronFaceLattice`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.polyhedron_face_lattice`
  - Type: `class`

- **sage.geometry.polyhedron.double_description.DoubleDescriptionPair**
  - Entity: `DoubleDescriptionPair`
  - Module: `sage.geometry.polyhedron.double_description`
  - Type: `class`

- **sage.geometry.polyhedron.double_description.Problem**
  - Entity: `Problem`
  - Module: `sage.geometry.polyhedron.double_description`
  - Type: `class`

- **sage.geometry.polyhedron.double_description.StandardAlgorithm**
  - Entity: `StandardAlgorithm`
  - Module: `sage.geometry.polyhedron.double_description`
  - Type: `class`

- **sage.geometry.polyhedron.double_description.StandardDoubleDescriptionPair**
  - Entity: `StandardDoubleDescriptionPair`
  - Module: `sage.geometry.polyhedron.double_description`
  - Type: `class`

- **sage.geometry.polyhedron.double_description_inhomogeneous.Hrep2Vrep**
  - Entity: `Hrep2Vrep`
  - Module: `sage.geometry.polyhedron.double_description_inhomogeneous`
  - Type: `class`

- **sage.geometry.polyhedron.double_description_inhomogeneous.PivotedInequalities**
  - Entity: `PivotedInequalities`
  - Module: `sage.geometry.polyhedron.double_description_inhomogeneous`
  - Type: `class`

- **sage.geometry.polyhedron.double_description_inhomogeneous.Vrep2Hrep**
  - Entity: `Vrep2Hrep`
  - Module: `sage.geometry.polyhedron.double_description_inhomogeneous`
  - Type: `class`

- **sage.geometry.polyhedron.face.PolyhedronFace**
  - Entity: `PolyhedronFace`
  - Module: `sage.geometry.polyhedron.face`
  - Type: `class`

- **sage.geometry.polyhedron.lattice_euclidean_group_element.LatticeEuclideanGroupElement**
  - Entity: `LatticeEuclideanGroupElement`
  - Module: `sage.geometry.polyhedron.lattice_euclidean_group_element`
  - Type: `class`

- **sage.geometry.polyhedron.library.Polytopes**
  - Entity: `Polytopes`
  - Module: `sage.geometry.polyhedron.library`
  - Type: `class`

- **sage.geometry.polyhedron.modules.formal_polyhedra_module.FormalPolyhedraModule**
  - Entity: `FormalPolyhedraModule`
  - Module: `sage.geometry.polyhedron.modules.formal_polyhedra_module`
  - Type: `class`

- **sage.geometry.polyhedron.palp_database.PALPreader**
  - Entity: `PALPreader`
  - Module: `sage.geometry.polyhedron.palp_database`
  - Type: `class`

- **sage.geometry.polyhedron.palp_database.Reflexive4dHodge**
  - Entity: `Reflexive4dHodge`
  - Module: `sage.geometry.polyhedron.palp_database`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_QQ_cdd**
  - Entity: `Polyhedra_QQ_cdd`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_QQ_normaliz**
  - Entity: `Polyhedra_QQ_normaliz`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_QQ_ppl**
  - Entity: `Polyhedra_QQ_ppl`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_RDF_cdd**
  - Entity: `Polyhedra_RDF_cdd`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_ZZ_normaliz**
  - Entity: `Polyhedra_ZZ_normaliz`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_ZZ_ppl**
  - Entity: `Polyhedra_ZZ_ppl`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_base**
  - Entity: `Polyhedra_base`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_field**
  - Entity: `Polyhedra_field`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_normaliz**
  - Entity: `Polyhedra_normaliz`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_number_field**
  - Entity: `Polyhedra_number_field`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.parent.Polyhedra_polymake**
  - Entity: `Polyhedra_polymake`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `class`

- **sage.geometry.polyhedron.plot.Projection**
  - Entity: `Projection`
  - Module: `sage.geometry.polyhedron.plot`
  - Type: `class`

- **sage.geometry.polyhedron.plot.ProjectionFuncSchlegel**
  - Entity: `ProjectionFuncSchlegel`
  - Module: `sage.geometry.polyhedron.plot`
  - Type: `class`

- **sage.geometry.polyhedron.plot.ProjectionFuncStereographic**
  - Entity: `ProjectionFuncStereographic`
  - Module: `sage.geometry.polyhedron.plot`
  - Type: `class`

- **sage.geometry.polyhedron.ppl_lattice_polygon.LatticePolygon_PPL_class**
  - Entity: `LatticePolygon_PPL_class`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `class`

- **sage.geometry.polyhedron.ppl_lattice_polytope.LatticePolytope_PPL_class**
  - Entity: `LatticePolytope_PPL_class`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polytope`
  - Type: `class`

- **sage.geometry.polyhedron.representation.Equation**
  - Entity: `Equation`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.polyhedron.representation.Hrepresentation**
  - Entity: `Hrepresentation`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.polyhedron.representation.Inequality**
  - Entity: `Inequality`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.polyhedron.representation.Line**
  - Entity: `Line`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.polyhedron.representation.PolyhedronRepresentation**
  - Entity: `PolyhedronRepresentation`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.polyhedron.representation.Ray**
  - Entity: `Ray`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.polyhedron.representation.Vertex**
  - Entity: `Vertex`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.polyhedron.representation.Vrepresentation**
  - Entity: `Vrepresentation`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `class`

- **sage.geometry.toric_lattice.ToricLatticeFactory**
  - Entity: `ToricLatticeFactory`
  - Module: `sage.geometry.toric_lattice`
  - Type: `class`

- **sage.geometry.toric_lattice.ToricLattice_ambient**
  - Entity: `ToricLattice_ambient`
  - Module: `sage.geometry.toric_lattice`
  - Type: `class`

- **sage.geometry.toric_lattice.ToricLattice_generic**
  - Entity: `ToricLattice_generic`
  - Module: `sage.geometry.toric_lattice`
  - Type: `class`

- **sage.geometry.toric_lattice.ToricLattice_quotient**
  - Entity: `ToricLattice_quotient`
  - Module: `sage.geometry.toric_lattice`
  - Type: `class`

- **sage.geometry.toric_lattice.ToricLattice_quotient_element**
  - Entity: `ToricLattice_quotient_element`
  - Module: `sage.geometry.toric_lattice`
  - Type: `class`

- **sage.geometry.toric_lattice.ToricLattice_sublattice**
  - Entity: `ToricLattice_sublattice`
  - Module: `sage.geometry.toric_lattice`
  - Type: `class`

- **sage.geometry.toric_lattice.ToricLattice_sublattice_with_basis**
  - Entity: `ToricLattice_sublattice_with_basis`
  - Module: `sage.geometry.toric_lattice`
  - Type: `class`

- **sage.geometry.toric_plotter.ToricPlotter**
  - Entity: `ToricPlotter`
  - Module: `sage.geometry.toric_plotter`
  - Type: `class`

- **sage.geometry.triangulation.base.ConnectedTriangulationsIterator**
  - Entity: `ConnectedTriangulationsIterator`
  - Module: `sage.geometry.triangulation.base`
  - Type: `class`

- **sage.geometry.triangulation.base.Point**
  - Entity: `Point`
  - Module: `sage.geometry.triangulation.base`
  - Type: `class`

- **sage.geometry.triangulation.base.PointConfiguration_base**
  - Entity: `PointConfiguration_base`
  - Module: `sage.geometry.triangulation.base`
  - Type: `class`

- **sage.geometry.triangulation.element.Triangulation**
  - Entity: `Triangulation`
  - Module: `sage.geometry.triangulation.element`
  - Type: `class`

- **sage.geometry.triangulation.point_configuration.PointConfiguration**
  - Entity: `PointConfiguration`
  - Module: `sage.geometry.triangulation.point_configuration`
  - Type: `class`

#### FUNCTION (76 entries)

- **sage.databases.oeis.to_tuple**
  - Entity: `to_tuple`
  - Module: `sage.databases.oeis`
  - Type: `function`

- **sage.dynamics.arithmetic_dynamics.wehlerK3.WehlerK3Surface**
  - Entity: `WehlerK3Surface`
  - Module: `sage.dynamics.arithmetic_dynamics.wehlerK3`
  - Type: `function`

- **sage.dynamics.arithmetic_dynamics.wehlerK3.random_WehlerK3Surface**
  - Entity: `random_WehlerK3Surface`
  - Module: `sage.dynamics.arithmetic_dynamics.wehlerK3`
  - Type: `function`

- **sage.dynamics.complex_dynamics.mandel_julia.external_ray**
  - Entity: `external_ray`
  - Module: `sage.dynamics.complex_dynamics.mandel_julia`
  - Type: `function`

- **sage.dynamics.complex_dynamics.mandel_julia.julia_plot**
  - Entity: `julia_plot`
  - Module: `sage.dynamics.complex_dynamics.mandel_julia`
  - Type: `function`

- **sage.dynamics.complex_dynamics.mandel_julia.kneading_sequence**
  - Entity: `kneading_sequence`
  - Module: `sage.dynamics.complex_dynamics.mandel_julia`
  - Type: `function`

- **sage.dynamics.complex_dynamics.mandel_julia.mandelbrot_plot**
  - Entity: `mandelbrot_plot`
  - Module: `sage.dynamics.complex_dynamics.mandel_julia`
  - Type: `function`

- **sage.functions.bessel.Bessel**
  - Entity: `Bessel`
  - Module: `sage.functions.bessel`
  - Type: `function`

- **sage.functions.bessel.spherical_bessel_f**
  - Entity: `spherical_bessel_f`
  - Module: `sage.functions.bessel`
  - Type: `function`

- **sage.functions.exp_integral.exponential_integral_1**
  - Entity: `exponential_integral_1`
  - Module: `sage.functions.exp_integral`
  - Type: `function`

- **sage.geometry.cone.Cone**
  - Entity: `Cone`
  - Module: `sage.geometry.cone`
  - Type: `function`

- **sage.geometry.cone.classify_cone_2d**
  - Entity: `classify_cone_2d`
  - Module: `sage.geometry.cone`
  - Type: `function`

- **sage.geometry.cone.integral_length**
  - Entity: `integral_length`
  - Module: `sage.geometry.cone`
  - Type: `function`

- **sage.geometry.cone.is_Cone**
  - Entity: `is_Cone`
  - Module: `sage.geometry.cone`
  - Type: `function`

- **sage.geometry.cone.normalize_rays**
  - Entity: `normalize_rays`
  - Module: `sage.geometry.cone`
  - Type: `function`

- **sage.geometry.cone.random_cone**
  - Entity: `random_cone`
  - Module: `sage.geometry.cone`
  - Type: `function`

- **sage.geometry.cone_catalog.downward_monotone**
  - Entity: `downward_monotone`
  - Module: `sage.geometry.cone_catalog`
  - Type: `function`

- **sage.geometry.cone_catalog.nonnegative_orthant**
  - Entity: `nonnegative_orthant`
  - Module: `sage.geometry.cone_catalog`
  - Type: `function`

- **sage.geometry.cone_catalog.rearrangement**
  - Entity: `rearrangement`
  - Module: `sage.geometry.cone_catalog`
  - Type: `function`

- **sage.geometry.cone_catalog.schur**
  - Entity: `schur`
  - Module: `sage.geometry.cone_catalog`
  - Type: `function`

- **sage.geometry.cone_catalog.trivial**
  - Entity: `trivial`
  - Module: `sage.geometry.cone_catalog`
  - Type: `function`

- **sage.geometry.cone_critical_angles.check_gevp_feasibility**
  - Entity: `check_gevp_feasibility`
  - Module: `sage.geometry.cone_critical_angles`
  - Type: `function`

- **sage.geometry.cone_critical_angles.compute_gevp_M**
  - Entity: `compute_gevp_M`
  - Module: `sage.geometry.cone_critical_angles`
  - Type: `function`

- **sage.geometry.cone_critical_angles.gevp_licis**
  - Entity: `gevp_licis`
  - Module: `sage.geometry.cone_critical_angles`
  - Type: `function`

- **sage.geometry.cone_critical_angles.max_angle**
  - Entity: `max_angle`
  - Module: `sage.geometry.cone_critical_angles`
  - Type: `function`

- **sage.geometry.cone_critical_angles.solve_gevp_nonzero**
  - Entity: `solve_gevp_nonzero`
  - Module: `sage.geometry.cone_critical_angles`
  - Type: `function`

- **sage.geometry.cone_critical_angles.solve_gevp_zero**
  - Entity: `solve_gevp_zero`
  - Module: `sage.geometry.cone_critical_angles`
  - Type: `function`

- **sage.geometry.fan.FaceFan**
  - Entity: `FaceFan`
  - Module: `sage.geometry.fan`
  - Type: `function`

- **sage.geometry.fan.Fan**
  - Entity: `Fan`
  - Module: `sage.geometry.fan`
  - Type: `function`

- **sage.geometry.fan.Fan2d**
  - Entity: `Fan2d`
  - Module: `sage.geometry.fan`
  - Type: `function`

- **sage.geometry.fan.NormalFan**
  - Entity: `NormalFan`
  - Module: `sage.geometry.fan`
  - Type: `function`

- **sage.geometry.fan.discard_faces**
  - Entity: `discard_faces`
  - Module: `sage.geometry.fan`
  - Type: `function`

- **sage.geometry.fan.is_Fan**
  - Entity: `is_Fan`
  - Module: `sage.geometry.fan`
  - Type: `function`

- **sage.geometry.fan_isomorphism.fan_2d_cyclically_ordered_rays**
  - Entity: `fan_2d_cyclically_ordered_rays`
  - Module: `sage.geometry.fan_isomorphism`
  - Type: `function`

- **sage.geometry.fan_isomorphism.fan_2d_echelon_form**
  - Entity: `fan_2d_echelon_form`
  - Module: `sage.geometry.fan_isomorphism`
  - Type: `function`

- **sage.geometry.fan_isomorphism.fan_2d_echelon_forms**
  - Entity: `fan_2d_echelon_forms`
  - Module: `sage.geometry.fan_isomorphism`
  - Type: `function`

- **sage.geometry.fan_isomorphism.fan_isomorphic_necessary_conditions**
  - Entity: `fan_isomorphic_necessary_conditions`
  - Module: `sage.geometry.fan_isomorphism`
  - Type: `function`

- **sage.geometry.fan_isomorphism.fan_isomorphism_generator**
  - Entity: `fan_isomorphism_generator`
  - Module: `sage.geometry.fan_isomorphism`
  - Type: `function`

- **sage.geometry.fan_isomorphism.find_isomorphism**
  - Entity: `find_isomorphism`
  - Module: `sage.geometry.fan_isomorphism`
  - Type: `function`

- **sage.geometry.polyhedron.base.is_Polyhedron**
  - Entity: `is_Polyhedron`
  - Module: `sage.geometry.polyhedron.base`
  - Type: `function`

- **sage.geometry.polyhedron.cdd_file_format.cdd_Hrepresentation**
  - Entity: `cdd_Hrepresentation`
  - Module: `sage.geometry.polyhedron.cdd_file_format`
  - Type: `function`

- **sage.geometry.polyhedron.cdd_file_format.cdd_Vrepresentation**
  - Entity: `cdd_Vrepresentation`
  - Module: `sage.geometry.polyhedron.cdd_file_format`
  - Type: `function`

- **sage.geometry.polyhedron.combinatorial_polyhedron.conversions.facets_tuple_to_bit_rep_of_Vrep**
  - Entity: `facets_tuple_to_bit_rep_of_Vrep`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.conversions`
  - Type: `function`

- **sage.geometry.polyhedron.combinatorial_polyhedron.conversions.facets_tuple_to_bit_rep_of_facets**
  - Entity: `facets_tuple_to_bit_rep_of_facets`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.conversions`
  - Type: `function`

- **sage.geometry.polyhedron.combinatorial_polyhedron.conversions.incidence_matrix_to_bit_rep_of_Vrep**
  - Entity: `incidence_matrix_to_bit_rep_of_Vrep`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.conversions`
  - Type: `function`

- **sage.geometry.polyhedron.combinatorial_polyhedron.conversions.incidence_matrix_to_bit_rep_of_facets**
  - Entity: `incidence_matrix_to_bit_rep_of_facets`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.conversions`
  - Type: `function`

- **sage.geometry.polyhedron.constructor.Polyhedron**
  - Entity: `Polyhedron`
  - Module: `sage.geometry.polyhedron.constructor`
  - Type: `function`

- **sage.geometry.polyhedron.double_description.random_inequalities**
  - Entity: `random_inequalities`
  - Module: `sage.geometry.polyhedron.double_description`
  - Type: `function`

- **sage.geometry.polyhedron.face.combinatorial_face_to_polyhedral_face**
  - Entity: `combinatorial_face_to_polyhedral_face`
  - Module: `sage.geometry.polyhedron.face`
  - Type: `function`

- **sage.geometry.polyhedron.generating_function.generating_function_of_integral_points**
  - Entity: `generating_function_of_integral_points`
  - Module: `sage.geometry.polyhedron.generating_function`
  - Type: `function`

- **sage.geometry.polyhedron.library.gale_transform_to_polytope**
  - Entity: `gale_transform_to_polytope`
  - Module: `sage.geometry.polyhedron.library`
  - Type: `function`

- **sage.geometry.polyhedron.library.gale_transform_to_primal**
  - Entity: `gale_transform_to_primal`
  - Module: `sage.geometry.polyhedron.library`
  - Type: `function`

- **sage.geometry.polyhedron.library.project_points**
  - Entity: `project_points`
  - Module: `sage.geometry.polyhedron.library`
  - Type: `function`

- **sage.geometry.polyhedron.library.zero_sum_projection**
  - Entity: `zero_sum_projection`
  - Module: `sage.geometry.polyhedron.library`
  - Type: `function`

- **sage.geometry.polyhedron.parent.Polyhedra**
  - Entity: `Polyhedra`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `function`

- **sage.geometry.polyhedron.parent.does_backend_handle_base_ring**
  - Entity: `does_backend_handle_base_ring`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `function`

- **sage.geometry.polyhedron.plot.cyclic_sort_vertices_2d**
  - Entity: `cyclic_sort_vertices_2d`
  - Module: `sage.geometry.polyhedron.plot`
  - Type: `function`

- **sage.geometry.polyhedron.plot.projection_func_identity**
  - Entity: `projection_func_identity`
  - Module: `sage.geometry.polyhedron.plot`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polygon.polar_P1xP1_polytope**
  - Entity: `polar_P1xP1_polytope`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polygon.polar_P2_112_polytope**
  - Entity: `polar_P2_112_polytope`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polygon.polar_P2_polytope**
  - Entity: `polar_P2_polytope`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polygon.sub_reflexive_polygons**
  - Entity: `sub_reflexive_polygons`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polygon.subpolygons_of_polar_P1xP1**
  - Entity: `subpolygons_of_polar_P1xP1`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polygon.subpolygons_of_polar_P2**
  - Entity: `subpolygons_of_polar_P2`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polygon.subpolygons_of_polar_P2_112**
  - Entity: `subpolygons_of_polar_P2_112`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `function`

- **sage.geometry.polyhedron.ppl_lattice_polytope.LatticePolytope_PPL**
  - Entity: `LatticePolytope_PPL`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polytope`
  - Type: `function`

- **sage.geometry.polyhedron.representation.repr_pretty**
  - Entity: `repr_pretty`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `function`

- **sage.geometry.toric_lattice.is_ToricLattice**
  - Entity: `is_ToricLattice`
  - Module: `sage.geometry.toric_lattice`
  - Type: `function`

- **sage.geometry.toric_lattice.is_ToricLatticeQuotient**
  - Entity: `is_ToricLatticeQuotient`
  - Module: `sage.geometry.toric_lattice`
  - Type: `function`

- **sage.geometry.toric_plotter.color_list**
  - Entity: `color_list`
  - Module: `sage.geometry.toric_plotter`
  - Type: `function`

- **sage.geometry.toric_plotter.label_list**
  - Entity: `label_list`
  - Module: `sage.geometry.toric_plotter`
  - Type: `function`

- **sage.geometry.toric_plotter.options**
  - Entity: `options`
  - Module: `sage.geometry.toric_plotter`
  - Type: `function`

- **sage.geometry.toric_plotter.reset_options**
  - Entity: `reset_options`
  - Module: `sage.geometry.toric_plotter`
  - Type: `function`

- **sage.geometry.toric_plotter.sector**
  - Entity: `sector`
  - Module: `sage.geometry.toric_plotter`
  - Type: `function`

- **sage.geometry.triangulation.element.triangulation_render_2d**
  - Entity: `triangulation_render_2d`
  - Module: `sage.geometry.triangulation.element`
  - Type: `function`

- **sage.geometry.triangulation.element.triangulation_render_3d**
  - Entity: `triangulation_render_3d`
  - Module: `sage.geometry.triangulation.element`
  - Type: `function`

#### MODULE (72 entries)

- **sage.databases.oeis**
  - Entity: `oeis`
  - Module: `sage.databases.oeis`
  - Type: `module`

- **sage.dynamics**
  - Entity: `dynamics`
  - Module: `sage.dynamics`
  - Type: `module`

- **sage.dynamics.arithmetic_dynamics.affine_ds**
  - Entity: `affine_ds`
  - Module: `sage.dynamics.arithmetic_dynamics.affine_ds`
  - Type: `module`

- **sage.dynamics.arithmetic_dynamics.berkovich_ds**
  - Entity: `berkovich_ds`
  - Module: `sage.dynamics.arithmetic_dynamics.berkovich_ds`
  - Type: `module`

- **sage.dynamics.arithmetic_dynamics.dynamical_semigroup**
  - Entity: `dynamical_semigroup`
  - Module: `sage.dynamics.arithmetic_dynamics.dynamical_semigroup`
  - Type: `module`

- **sage.dynamics.arithmetic_dynamics.generic_ds**
  - Entity: `generic_ds`
  - Module: `sage.dynamics.arithmetic_dynamics.generic_ds`
  - Type: `module`

- **sage.dynamics.arithmetic_dynamics.product_projective_ds**
  - Entity: `product_projective_ds`
  - Module: `sage.dynamics.arithmetic_dynamics.product_projective_ds`
  - Type: `module`

- **sage.dynamics.arithmetic_dynamics.projective_ds**
  - Entity: `projective_ds`
  - Module: `sage.dynamics.arithmetic_dynamics.projective_ds`
  - Type: `module`

- **sage.dynamics.arithmetic_dynamics.wehlerK3**
  - Entity: `wehlerK3`
  - Module: `sage.dynamics.arithmetic_dynamics.wehlerK3`
  - Type: `module`

- **sage.dynamics.cellular_automata.catalog**
  - Entity: `catalog`
  - Module: `sage.dynamics.cellular_automata.catalog`
  - Type: `module`

- **sage.dynamics.cellular_automata.elementary**
  - Entity: `elementary`
  - Module: `sage.dynamics.cellular_automata.elementary`
  - Type: `module`

- **sage.dynamics.cellular_automata.glca**
  - Entity: `glca`
  - Module: `sage.dynamics.cellular_automata.glca`
  - Type: `module`

- **sage.dynamics.cellular_automata.solitons**
  - Entity: `solitons`
  - Module: `sage.dynamics.cellular_automata.solitons`
  - Type: `module`

- **sage.dynamics.complex_dynamics.mandel_julia**
  - Entity: `mandel_julia`
  - Module: `sage.dynamics.complex_dynamics.mandel_julia`
  - Type: `module`

- **sage.dynamics.finite_dynamical_system**
  - Entity: `finite_dynamical_system`
  - Module: `sage.dynamics.finite_dynamical_system`
  - Type: `module`

- **sage.functions.bessel**
  - Entity: `bessel`
  - Module: `sage.functions.bessel`
  - Type: `module`

- **sage.functions.error**
  - Entity: `error`
  - Module: `sage.functions.error`
  - Type: `module`

- **sage.functions.exp_integral**
  - Entity: `exp_integral`
  - Module: `sage.functions.exp_integral`
  - Type: `module`

- **sage.functions.hyperbolic**
  - Entity: `hyperbolic`
  - Module: `sage.functions.hyperbolic`
  - Type: `module`

- **sage.functions.log**
  - Entity: `log`
  - Module: `sage.functions.log`
  - Type: `module`

- **sage.functions.trig**
  - Entity: `trig`
  - Module: `sage.functions.trig`
  - Type: `module`

- **sage.geometry.cone**
  - Entity: `cone`
  - Module: `sage.geometry.cone`
  - Type: `module`

- **sage.geometry.cone_catalog**
  - Entity: `cone_catalog`
  - Module: `sage.geometry.cone_catalog`
  - Type: `module`

- **sage.geometry.cone_critical_angles**
  - Entity: `cone_critical_angles`
  - Module: `sage.geometry.cone_critical_angles`
  - Type: `module`

- **sage.geometry.fan**
  - Entity: `fan`
  - Module: `sage.geometry.fan`
  - Type: `module`

- **sage.geometry.fan_isomorphism**
  - Entity: `fan_isomorphism`
  - Module: `sage.geometry.fan_isomorphism`
  - Type: `module`

- **sage.geometry.fan_morphism**
  - Entity: `fan_morphism`
  - Module: `sage.geometry.fan_morphism`
  - Type: `module`

- **sage.geometry.polyhedron.backend_cdd**
  - Entity: `backend_cdd`
  - Module: `sage.geometry.polyhedron.backend_cdd`
  - Type: `module`

- **sage.geometry.polyhedron.backend_cdd_rdf**
  - Entity: `backend_cdd_rdf`
  - Module: `sage.geometry.polyhedron.backend_cdd_rdf`
  - Type: `module`

- **sage.geometry.polyhedron.backend_field**
  - Entity: `backend_field`
  - Module: `sage.geometry.polyhedron.backend_field`
  - Type: `module`

- **sage.geometry.polyhedron.backend_normaliz**
  - Entity: `backend_normaliz`
  - Module: `sage.geometry.polyhedron.backend_normaliz`
  - Type: `module`

- **sage.geometry.polyhedron.backend_number_field**
  - Entity: `backend_number_field`
  - Module: `sage.geometry.polyhedron.backend_number_field`
  - Type: `module`

- **sage.geometry.polyhedron.backend_polymake**
  - Entity: `backend_polymake`
  - Module: `sage.geometry.polyhedron.backend_polymake`
  - Type: `module`

- **sage.geometry.polyhedron.backend_ppl**
  - Entity: `backend_ppl`
  - Module: `sage.geometry.polyhedron.backend_ppl`
  - Type: `module`

- **sage.geometry.polyhedron.base**
  - Entity: `base`
  - Module: `sage.geometry.polyhedron.base`
  - Type: `module`

- **sage.geometry.polyhedron.base0**
  - Entity: `base0`
  - Module: `sage.geometry.polyhedron.base0`
  - Type: `module`

- **sage.geometry.polyhedron.base1**
  - Entity: `base1`
  - Module: `sage.geometry.polyhedron.base1`
  - Type: `module`

- **sage.geometry.polyhedron.base2**
  - Entity: `base2`
  - Module: `sage.geometry.polyhedron.base2`
  - Type: `module`

- **sage.geometry.polyhedron.base3**
  - Entity: `base3`
  - Module: `sage.geometry.polyhedron.base3`
  - Type: `module`

- **sage.geometry.polyhedron.base4**
  - Entity: `base4`
  - Module: `sage.geometry.polyhedron.base4`
  - Type: `module`

- **sage.geometry.polyhedron.base5**
  - Entity: `base5`
  - Module: `sage.geometry.polyhedron.base5`
  - Type: `module`

- **sage.geometry.polyhedron.base6**
  - Entity: `base6`
  - Module: `sage.geometry.polyhedron.base6`
  - Type: `module`

- **sage.geometry.polyhedron.base7**
  - Entity: `base7`
  - Module: `sage.geometry.polyhedron.base7`
  - Type: `module`

- **sage.geometry.polyhedron.base_QQ**
  - Entity: `base_QQ`
  - Module: `sage.geometry.polyhedron.base_QQ`
  - Type: `module`

- **sage.geometry.polyhedron.base_RDF**
  - Entity: `base_RDF`
  - Module: `sage.geometry.polyhedron.base_RDF`
  - Type: `module`

- **sage.geometry.polyhedron.base_ZZ**
  - Entity: `base_ZZ`
  - Module: `sage.geometry.polyhedron.base_ZZ`
  - Type: `module`

- **sage.geometry.polyhedron.cdd_file_format**
  - Entity: `cdd_file_format`
  - Module: `sage.geometry.polyhedron.cdd_file_format`
  - Type: `module`

- **sage.geometry.polyhedron.combinatorial_polyhedron.base**
  - Entity: `base`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.base`
  - Type: `module`

- **sage.geometry.polyhedron.combinatorial_polyhedron.combinatorial_face**
  - Entity: `combinatorial_face`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.combinatorial_face`
  - Type: `module`

- **sage.geometry.polyhedron.combinatorial_polyhedron.conversions**
  - Entity: `conversions`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.conversions`
  - Type: `module`

- **sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator**
  - Entity: `face_iterator`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.face_iterator`
  - Type: `module`

- **sage.geometry.polyhedron.combinatorial_polyhedron.list_of_faces**
  - Entity: `list_of_faces`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.list_of_faces`
  - Type: `module`

- **sage.geometry.polyhedron.combinatorial_polyhedron.polyhedron_face_lattice**
  - Entity: `polyhedron_face_lattice`
  - Module: `sage.geometry.polyhedron.combinatorial_polyhedron.polyhedron_face_lattice`
  - Type: `module`

- **sage.geometry.polyhedron.constructor**
  - Entity: `constructor`
  - Module: `sage.geometry.polyhedron.constructor`
  - Type: `module`

- **sage.geometry.polyhedron.double_description**
  - Entity: `double_description`
  - Module: `sage.geometry.polyhedron.double_description`
  - Type: `module`

- **sage.geometry.polyhedron.double_description_inhomogeneous**
  - Entity: `double_description_inhomogeneous`
  - Module: `sage.geometry.polyhedron.double_description_inhomogeneous`
  - Type: `module`

- **sage.geometry.polyhedron.face**
  - Entity: `face`
  - Module: `sage.geometry.polyhedron.face`
  - Type: `module`

- **sage.geometry.polyhedron.generating_function**
  - Entity: `generating_function`
  - Module: `sage.geometry.polyhedron.generating_function`
  - Type: `module`

- **sage.geometry.polyhedron.lattice_euclidean_group_element**
  - Entity: `lattice_euclidean_group_element`
  - Module: `sage.geometry.polyhedron.lattice_euclidean_group_element`
  - Type: `module`

- **sage.geometry.polyhedron.library**
  - Entity: `library`
  - Module: `sage.geometry.polyhedron.library`
  - Type: `module`

- **sage.geometry.polyhedron.modules.formal_polyhedra_module**
  - Entity: `formal_polyhedra_module`
  - Module: `sage.geometry.polyhedron.modules.formal_polyhedra_module`
  - Type: `module`

- **sage.geometry.polyhedron.palp_database**
  - Entity: `palp_database`
  - Module: `sage.geometry.polyhedron.palp_database`
  - Type: `module`

- **sage.geometry.polyhedron.parent**
  - Entity: `parent`
  - Module: `sage.geometry.polyhedron.parent`
  - Type: `module`

- **sage.geometry.polyhedron.plot**
  - Entity: `plot`
  - Module: `sage.geometry.polyhedron.plot`
  - Type: `module`

- **sage.geometry.polyhedron.ppl_lattice_polygon**
  - Entity: `ppl_lattice_polygon`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polygon`
  - Type: `module`

- **sage.geometry.polyhedron.ppl_lattice_polytope**
  - Entity: `ppl_lattice_polytope`
  - Module: `sage.geometry.polyhedron.ppl_lattice_polytope`
  - Type: `module`

- **sage.geometry.polyhedron.representation**
  - Entity: `representation`
  - Module: `sage.geometry.polyhedron.representation`
  - Type: `module`

- **sage.geometry.toric_lattice**
  - Entity: `toric_lattice`
  - Module: `sage.geometry.toric_lattice`
  - Type: `module`

- **sage.geometry.toric_plotter**
  - Entity: `toric_plotter`
  - Module: `sage.geometry.toric_plotter`
  - Type: `module`

- **sage.geometry.triangulation.base**
  - Entity: `base`
  - Module: `sage.geometry.triangulation.base`
  - Type: `module`

- **sage.geometry.triangulation.element**
  - Entity: `element`
  - Module: `sage.geometry.triangulation.element`
  - Type: `module`

- **sage.geometry.triangulation.point_configuration**
  - Entity: `point_configuration`
  - Module: `sage.geometry.triangulation.point_configuration`
  - Type: `module`


### Part 07 (145 entries)

#### CLASS (52 entries)

- **sage.graphs.bipartite_graph.BipartiteGraph**
  - Entity: `BipartiteGraph`
  - Module: `sage.graphs.bipartite_graph`
  - Type: `class`

- **sage.graphs.digraph.DiGraph**
  - Entity: `DiGraph`
  - Module: `sage.graphs.digraph`
  - Type: `class`

- **sage.graphs.digraph_generators.DiGraphGenerators**
  - Entity: `DiGraphGenerators`
  - Module: `sage.graphs.digraph_generators`
  - Type: `class`

- **sage.graphs.graph.Graph**
  - Entity: `Graph`
  - Module: `sage.graphs.graph`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_aut.AbelianGroupAutomorphism**
  - Entity: `AbelianGroupAutomorphism`
  - Module: `sage.groups.abelian_gps.abelian_aut`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_aut.AbelianGroupAutomorphismGroup**
  - Entity: `AbelianGroupAutomorphismGroup`
  - Module: `sage.groups.abelian_gps.abelian_aut`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_aut.AbelianGroupAutomorphismGroup_gap**
  - Entity: `AbelianGroupAutomorphismGroup_gap`
  - Module: `sage.groups.abelian_gps.abelian_aut`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_aut.AbelianGroupAutomorphismGroup_subgroup**
  - Entity: `AbelianGroupAutomorphismGroup_subgroup`
  - Module: `sage.groups.abelian_gps.abelian_aut`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group.AbelianGroup_class**
  - Entity: `AbelianGroup_class`
  - Module: `sage.groups.abelian_gps.abelian_group`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group.AbelianGroup_subgroup**
  - Entity: `AbelianGroup_subgroup`
  - Module: `sage.groups.abelian_gps.abelian_group`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_element.AbelianGroupElement**
  - Entity: `AbelianGroupElement`
  - Module: `sage.groups.abelian_gps.abelian_group_element`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_gap.AbelianGroupElement_gap**
  - Entity: `AbelianGroupElement_gap`
  - Module: `sage.groups.abelian_gps.abelian_group_gap`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_gap.AbelianGroupElement_polycyclic**
  - Entity: `AbelianGroupElement_polycyclic`
  - Module: `sage.groups.abelian_gps.abelian_group_gap`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_gap.AbelianGroupGap**
  - Entity: `AbelianGroupGap`
  - Module: `sage.groups.abelian_gps.abelian_group_gap`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_gap.AbelianGroupQuotient_gap**
  - Entity: `AbelianGroupQuotient_gap`
  - Module: `sage.groups.abelian_gps.abelian_group_gap`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_gap.AbelianGroupSubgroup_gap**
  - Entity: `AbelianGroupSubgroup_gap`
  - Module: `sage.groups.abelian_gps.abelian_group_gap`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_gap.AbelianGroup_gap**
  - Entity: `AbelianGroup_gap`
  - Module: `sage.groups.abelian_gps.abelian_group_gap`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_morphism.AbelianGroupMap**
  - Entity: `AbelianGroupMap`
  - Module: `sage.groups.abelian_gps.abelian_group_morphism`
  - Type: `class`

- **sage.groups.abelian_gps.abelian_group_morphism.AbelianGroupMorphism**
  - Entity: `AbelianGroupMorphism`
  - Module: `sage.groups.abelian_gps.abelian_group_morphism`
  - Type: `class`

- **sage.groups.abelian_gps.dual_abelian_group.DualAbelianGroup_class**
  - Entity: `DualAbelianGroup_class`
  - Module: `sage.groups.abelian_gps.dual_abelian_group`
  - Type: `class`

- **sage.groups.abelian_gps.dual_abelian_group_element.DualAbelianGroupElement**
  - Entity: `DualAbelianGroupElement`
  - Module: `sage.groups.abelian_gps.dual_abelian_group_element`
  - Type: `class`

- **sage.groups.abelian_gps.element_base.AbelianGroupElementBase**
  - Entity: `AbelianGroupElementBase`
  - Module: `sage.groups.abelian_gps.element_base`
  - Type: `class`

- **sage.groups.abelian_gps.values.AbelianGroupWithValuesElement**
  - Entity: `AbelianGroupWithValuesElement`
  - Module: `sage.groups.abelian_gps.values`
  - Type: `class`

- **sage.groups.abelian_gps.values.AbelianGroupWithValuesEmbedding**
  - Entity: `AbelianGroupWithValuesEmbedding`
  - Module: `sage.groups.abelian_gps.values`
  - Type: `class`

- **sage.groups.abelian_gps.values.AbelianGroupWithValues_class**
  - Entity: `AbelianGroupWithValues_class`
  - Module: `sage.groups.abelian_gps.values`
  - Type: `class`

- **sage.groups.matrix_gps.binary_dihedral.BinaryDihedralGroup**
  - Entity: `BinaryDihedralGroup`
  - Module: `sage.groups.matrix_gps.binary_dihedral`
  - Type: `class`

- **sage.groups.matrix_gps.coxeter_group.CoxeterMatrixGroup**
  - Entity: `CoxeterMatrixGroup`
  - Module: `sage.groups.matrix_gps.coxeter_group`
  - Type: `class`

- **sage.groups.matrix_gps.coxeter_group.Element**
  - Entity: `Element`
  - Module: `sage.groups.matrix_gps.coxeter_group`
  - Type: `class`

- **sage.groups.matrix_gps.finitely_generated.FinitelyGeneratedMatrixGroup_generic**
  - Entity: `FinitelyGeneratedMatrixGroup_generic`
  - Module: `sage.groups.matrix_gps.finitely_generated`
  - Type: `class`

- **sage.groups.matrix_gps.finitely_generated_gap.FinitelyGeneratedMatrixGroup_gap**
  - Entity: `FinitelyGeneratedMatrixGroup_gap`
  - Module: `sage.groups.matrix_gps.finitely_generated_gap`
  - Type: `class`

- **sage.groups.matrix_gps.group_element.MatrixGroupElement_generic**
  - Entity: `MatrixGroupElement_generic`
  - Module: `sage.groups.matrix_gps.group_element`
  - Type: `class`

- **sage.groups.matrix_gps.group_element_gap.MatrixGroupElement_gap**
  - Entity: `MatrixGroupElement_gap`
  - Module: `sage.groups.matrix_gps.group_element_gap`
  - Type: `class`

- **sage.groups.matrix_gps.heisenberg.HeisenbergGroup**
  - Entity: `HeisenbergGroup`
  - Module: `sage.groups.matrix_gps.heisenberg`
  - Type: `class`

- **sage.groups.matrix_gps.isometries.GroupActionOnQuotientModule**
  - Entity: `GroupActionOnQuotientModule`
  - Module: `sage.groups.matrix_gps.isometries`
  - Type: `class`

- **sage.groups.matrix_gps.isometries.GroupActionOnSubmodule**
  - Entity: `GroupActionOnSubmodule`
  - Module: `sage.groups.matrix_gps.isometries`
  - Type: `class`

- **sage.groups.matrix_gps.isometries.GroupOfIsometries**
  - Entity: `GroupOfIsometries`
  - Module: `sage.groups.matrix_gps.isometries`
  - Type: `class`

- **sage.groups.matrix_gps.linear.LinearMatrixGroup_generic**
  - Entity: `LinearMatrixGroup_generic`
  - Module: `sage.groups.matrix_gps.linear`
  - Type: `class`

- **sage.groups.matrix_gps.linear_gap.LinearMatrixGroup_gap**
  - Entity: `LinearMatrixGroup_gap`
  - Module: `sage.groups.matrix_gps.linear_gap`
  - Type: `class`

- **sage.groups.matrix_gps.matrix_group.MatrixGroup_base**
  - Entity: `MatrixGroup_base`
  - Module: `sage.groups.matrix_gps.matrix_group`
  - Type: `class`

- **sage.groups.matrix_gps.matrix_group.MatrixGroup_generic**
  - Entity: `MatrixGroup_generic`
  - Module: `sage.groups.matrix_gps.matrix_group`
  - Type: `class`

- **sage.groups.matrix_gps.matrix_group_gap.MatrixGroup_gap**
  - Entity: `MatrixGroup_gap`
  - Module: `sage.groups.matrix_gps.matrix_group_gap`
  - Type: `class`

- **sage.groups.matrix_gps.named_group.NamedMatrixGroup_generic**
  - Entity: `NamedMatrixGroup_generic`
  - Module: `sage.groups.matrix_gps.named_group`
  - Type: `class`

- **sage.groups.matrix_gps.named_group_gap.NamedMatrixGroup_gap**
  - Entity: `NamedMatrixGroup_gap`
  - Module: `sage.groups.matrix_gps.named_group_gap`
  - Type: `class`

- **sage.groups.matrix_gps.orthogonal.OrthogonalMatrixGroup_generic**
  - Entity: `OrthogonalMatrixGroup_generic`
  - Module: `sage.groups.matrix_gps.orthogonal`
  - Type: `class`

- **sage.groups.matrix_gps.orthogonal_gap.OrthogonalMatrixGroup_gap**
  - Entity: `OrthogonalMatrixGroup_gap`
  - Module: `sage.groups.matrix_gps.orthogonal_gap`
  - Type: `class`

- **sage.groups.matrix_gps.symplectic.SymplecticMatrixGroup_generic**
  - Entity: `SymplecticMatrixGroup_generic`
  - Module: `sage.groups.matrix_gps.symplectic`
  - Type: `class`

- **sage.groups.matrix_gps.symplectic_gap.SymplecticMatrixGroup_gap**
  - Entity: `SymplecticMatrixGroup_gap`
  - Module: `sage.groups.matrix_gps.symplectic_gap`
  - Type: `class`

- **sage.groups.matrix_gps.unitary.UnitaryMatrixGroup_generic**
  - Entity: `UnitaryMatrixGroup_generic`
  - Module: `sage.groups.matrix_gps.unitary`
  - Type: `class`

- **sage.groups.matrix_gps.unitary_gap.UnitaryMatrixGroup_gap**
  - Entity: `UnitaryMatrixGroup_gap`
  - Module: `sage.groups.matrix_gps.unitary_gap`
  - Type: `class`

- **sage.groups.perm_gps.cubegroup.CubeGroup**
  - Entity: `CubeGroup`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `class`

- **sage.groups.perm_gps.cubegroup.RubiksCube**
  - Entity: `RubiksCube`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `class`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.GraphStruct**
  - Entity: `GraphStruct`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `class`

#### FUNCTION (52 entries)

- **sage.groups.abelian_gps.abelian_group.AbelianGroup**
  - Entity: `AbelianGroup`
  - Module: `sage.groups.abelian_gps.abelian_group`
  - Type: `function`

- **sage.groups.abelian_gps.abelian_group.is_AbelianGroup**
  - Entity: `is_AbelianGroup`
  - Module: `sage.groups.abelian_gps.abelian_group`
  - Type: `function`

- **sage.groups.abelian_gps.abelian_group.word_problem**
  - Entity: `word_problem`
  - Module: `sage.groups.abelian_gps.abelian_group`
  - Type: `function`

- **sage.groups.abelian_gps.abelian_group_element.is_AbelianGroupElement**
  - Entity: `is_AbelianGroupElement`
  - Module: `sage.groups.abelian_gps.abelian_group_element`
  - Type: `function`

- **sage.groups.abelian_gps.abelian_group_morphism.is_AbelianGroupMorphism**
  - Entity: `is_AbelianGroupMorphism`
  - Module: `sage.groups.abelian_gps.abelian_group_morphism`
  - Type: `function`

- **sage.groups.abelian_gps.dual_abelian_group.is_DualAbelianGroup**
  - Entity: `is_DualAbelianGroup`
  - Module: `sage.groups.abelian_gps.dual_abelian_group`
  - Type: `function`

- **sage.groups.abelian_gps.dual_abelian_group_element.is_DualAbelianGroupElement**
  - Entity: `is_DualAbelianGroupElement`
  - Module: `sage.groups.abelian_gps.dual_abelian_group_element`
  - Type: `function`

- **sage.groups.abelian_gps.values.AbelianGroupWithValues**
  - Entity: `AbelianGroupWithValues`
  - Module: `sage.groups.abelian_gps.values`
  - Type: `function`

- **sage.groups.matrix_gps.finitely_generated.MatrixGroup**
  - Entity: `MatrixGroup`
  - Module: `sage.groups.matrix_gps.finitely_generated`
  - Type: `function`

- **sage.groups.matrix_gps.finitely_generated.QuaternionMatrixGroupGF3**
  - Entity: `QuaternionMatrixGroupGF3`
  - Module: `sage.groups.matrix_gps.finitely_generated`
  - Type: `function`

- **sage.groups.matrix_gps.finitely_generated.normalize_square_matrices**
  - Entity: `normalize_square_matrices`
  - Module: `sage.groups.matrix_gps.finitely_generated`
  - Type: `function`

- **sage.groups.matrix_gps.group_element.is_MatrixGroupElement**
  - Entity: `is_MatrixGroupElement`
  - Module: `sage.groups.matrix_gps.group_element`
  - Type: `function`

- **sage.groups.matrix_gps.linear.GL**
  - Entity: `GL`
  - Module: `sage.groups.matrix_gps.linear`
  - Type: `function`

- **sage.groups.matrix_gps.linear.SL**
  - Entity: `SL`
  - Module: `sage.groups.matrix_gps.linear`
  - Type: `function`

- **sage.groups.matrix_gps.matrix_group.is_MatrixGroup**
  - Entity: `is_MatrixGroup`
  - Module: `sage.groups.matrix_gps.matrix_group`
  - Type: `function`

- **sage.groups.matrix_gps.named_group.normalize_args_invariant_form**
  - Entity: `normalize_args_invariant_form`
  - Module: `sage.groups.matrix_gps.named_group`
  - Type: `function`

- **sage.groups.matrix_gps.named_group.normalize_args_vectorspace**
  - Entity: `normalize_args_vectorspace`
  - Module: `sage.groups.matrix_gps.named_group`
  - Type: `function`

- **sage.groups.matrix_gps.orthogonal.GO**
  - Entity: `GO`
  - Module: `sage.groups.matrix_gps.orthogonal`
  - Type: `function`

- **sage.groups.matrix_gps.orthogonal.SO**
  - Entity: `SO`
  - Module: `sage.groups.matrix_gps.orthogonal`
  - Type: `function`

- **sage.groups.matrix_gps.orthogonal.normalize_args_e**
  - Entity: `normalize_args_e`
  - Module: `sage.groups.matrix_gps.orthogonal`
  - Type: `function`

- **sage.groups.matrix_gps.symplectic.Sp**
  - Entity: `Sp`
  - Module: `sage.groups.matrix_gps.symplectic`
  - Type: `function`

- **sage.groups.matrix_gps.unitary.GU**
  - Entity: `GU`
  - Module: `sage.groups.matrix_gps.unitary`
  - Type: `function`

- **sage.groups.matrix_gps.unitary.SU**
  - Entity: `SU`
  - Module: `sage.groups.matrix_gps.unitary`
  - Type: `function`

- **sage.groups.matrix_gps.unitary.finite_field_sqrt**
  - Entity: `finite_field_sqrt`
  - Module: `sage.groups.matrix_gps.unitary`
  - Type: `function`

- **sage.groups.perm_gps.constructor.PermutationGroupElement**
  - Entity: `PermutationGroupElement`
  - Module: `sage.groups.perm_gps.constructor`
  - Type: `function`

- **sage.groups.perm_gps.constructor.standardize_generator**
  - Entity: `standardize_generator`
  - Module: `sage.groups.perm_gps.constructor`
  - Type: `function`

- **sage.groups.perm_gps.constructor.string_to_tuples**
  - Entity: `string_to_tuples`
  - Module: `sage.groups.perm_gps.constructor`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.color_of_square**
  - Entity: `color_of_square`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.create_poly**
  - Entity: `create_poly`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.cubie_centers**
  - Entity: `cubie_centers`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.cubie_colors**
  - Entity: `cubie_colors`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.cubie_faces**
  - Entity: `cubie_faces`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.index2singmaster**
  - Entity: `index2singmaster`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.inv_list**
  - Entity: `inv_list`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.plot3d_cubie**
  - Entity: `plot3d_cubie`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.polygon_plot3d**
  - Entity: `polygon_plot3d`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.rotation_list**
  - Entity: `rotation_list`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.xproj**
  - Entity: `xproj`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.cubegroup.yproj**
  - Entity: `yproj`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.data_structures.OP_represent**
  - Entity: `OP_represent`
  - Module: `sage.groups.perm_gps.partn_ref.data_structures`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.data_structures.PS_represent**
  - Entity: `PS_represent`
  - Module: `sage.groups.perm_gps.partn_ref.data_structures`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.data_structures.SC_test_list_perms**
  - Entity: `SC_test_list_perms`
  - Module: `sage.groups.perm_gps.partn_ref.data_structures`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.all_labeled_graphs**
  - Entity: `all_labeled_graphs`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.coarsest_equitable_refinement**
  - Entity: `coarsest_equitable_refinement`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.generate_dense_graphs_edge_addition**
  - Entity: `generate_dense_graphs_edge_addition`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.generate_dense_graphs_vert_addition**
  - Entity: `generate_dense_graphs_vert_addition`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.get_orbits**
  - Entity: `get_orbits`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.isomorphic**
  - Entity: `isomorphic`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.orbit_partition**
  - Entity: `orbit_partition`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.random_tests**
  - Entity: `random_tests`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_graphs.search_tree**
  - Entity: `search_tree`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `function`

- **sage.groups.perm_gps.partn_ref.refinement_lists.is_isomorphic**
  - Entity: `is_isomorphic`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_lists`
  - Type: `function`

#### MODULE (41 entries)

- **sage.graphs.bipartite_graph**
  - Entity: `bipartite_graph`
  - Module: `sage.graphs.bipartite_graph`
  - Type: `module`

- **sage.graphs.digraph**
  - Entity: `digraph`
  - Module: `sage.graphs.digraph`
  - Type: `module`

- **sage.graphs.digraph_generators**
  - Entity: `digraph_generators`
  - Module: `sage.graphs.digraph_generators`
  - Type: `module`

- **sage.graphs.graph**
  - Entity: `graph`
  - Module: `sage.graphs.graph`
  - Type: `module`

- **sage.groups.abelian_gps.abelian_aut**
  - Entity: `abelian_aut`
  - Module: `sage.groups.abelian_gps.abelian_aut`
  - Type: `module`

- **sage.groups.abelian_gps.abelian_group**
  - Entity: `abelian_group`
  - Module: `sage.groups.abelian_gps.abelian_group`
  - Type: `module`

- **sage.groups.abelian_gps.abelian_group_element**
  - Entity: `abelian_group_element`
  - Module: `sage.groups.abelian_gps.abelian_group_element`
  - Type: `module`

- **sage.groups.abelian_gps.abelian_group_gap**
  - Entity: `abelian_group_gap`
  - Module: `sage.groups.abelian_gps.abelian_group_gap`
  - Type: `module`

- **sage.groups.abelian_gps.abelian_group_morphism**
  - Entity: `abelian_group_morphism`
  - Module: `sage.groups.abelian_gps.abelian_group_morphism`
  - Type: `module`

- **sage.groups.abelian_gps.dual_abelian_group**
  - Entity: `dual_abelian_group`
  - Module: `sage.groups.abelian_gps.dual_abelian_group`
  - Type: `module`

- **sage.groups.abelian_gps.dual_abelian_group_element**
  - Entity: `dual_abelian_group_element`
  - Module: `sage.groups.abelian_gps.dual_abelian_group_element`
  - Type: `module`

- **sage.groups.abelian_gps.element_base**
  - Entity: `element_base`
  - Module: `sage.groups.abelian_gps.element_base`
  - Type: `module`

- **sage.groups.abelian_gps.values**
  - Entity: `values`
  - Module: `sage.groups.abelian_gps.values`
  - Type: `module`

- **sage.groups.matrix_gps.binary_dihedral**
  - Entity: `binary_dihedral`
  - Module: `sage.groups.matrix_gps.binary_dihedral`
  - Type: `module`

- **sage.groups.matrix_gps.catalog**
  - Entity: `catalog`
  - Module: `sage.groups.matrix_gps.catalog`
  - Type: `module`

- **sage.groups.matrix_gps.coxeter_group**
  - Entity: `coxeter_group`
  - Module: `sage.groups.matrix_gps.coxeter_group`
  - Type: `module`

- **sage.groups.matrix_gps.finitely_generated**
  - Entity: `finitely_generated`
  - Module: `sage.groups.matrix_gps.finitely_generated`
  - Type: `module`

- **sage.groups.matrix_gps.finitely_generated_gap**
  - Entity: `finitely_generated_gap`
  - Module: `sage.groups.matrix_gps.finitely_generated_gap`
  - Type: `module`

- **sage.groups.matrix_gps.group_element**
  - Entity: `group_element`
  - Module: `sage.groups.matrix_gps.group_element`
  - Type: `module`

- **sage.groups.matrix_gps.group_element_gap**
  - Entity: `group_element_gap`
  - Module: `sage.groups.matrix_gps.group_element_gap`
  - Type: `module`

- **sage.groups.matrix_gps.heisenberg**
  - Entity: `heisenberg`
  - Module: `sage.groups.matrix_gps.heisenberg`
  - Type: `module`

- **sage.groups.matrix_gps.isometries**
  - Entity: `isometries`
  - Module: `sage.groups.matrix_gps.isometries`
  - Type: `module`

- **sage.groups.matrix_gps.linear**
  - Entity: `linear`
  - Module: `sage.groups.matrix_gps.linear`
  - Type: `module`

- **sage.groups.matrix_gps.linear_gap**
  - Entity: `linear_gap`
  - Module: `sage.groups.matrix_gps.linear_gap`
  - Type: `module`

- **sage.groups.matrix_gps.matrix_group**
  - Entity: `matrix_group`
  - Module: `sage.groups.matrix_gps.matrix_group`
  - Type: `module`

- **sage.groups.matrix_gps.matrix_group_gap**
  - Entity: `matrix_group_gap`
  - Module: `sage.groups.matrix_gps.matrix_group_gap`
  - Type: `module`

- **sage.groups.matrix_gps.named_group**
  - Entity: `named_group`
  - Module: `sage.groups.matrix_gps.named_group`
  - Type: `module`

- **sage.groups.matrix_gps.named_group_gap**
  - Entity: `named_group_gap`
  - Module: `sage.groups.matrix_gps.named_group_gap`
  - Type: `module`

- **sage.groups.matrix_gps.orthogonal**
  - Entity: `orthogonal`
  - Module: `sage.groups.matrix_gps.orthogonal`
  - Type: `module`

- **sage.groups.matrix_gps.orthogonal_gap**
  - Entity: `orthogonal_gap`
  - Module: `sage.groups.matrix_gps.orthogonal_gap`
  - Type: `module`

- **sage.groups.matrix_gps.symplectic**
  - Entity: `symplectic`
  - Module: `sage.groups.matrix_gps.symplectic`
  - Type: `module`

- **sage.groups.matrix_gps.symplectic_gap**
  - Entity: `symplectic_gap`
  - Module: `sage.groups.matrix_gps.symplectic_gap`
  - Type: `module`

- **sage.groups.matrix_gps.unitary**
  - Entity: `unitary`
  - Module: `sage.groups.matrix_gps.unitary`
  - Type: `module`

- **sage.groups.matrix_gps.unitary_gap**
  - Entity: `unitary_gap`
  - Module: `sage.groups.matrix_gps.unitary_gap`
  - Type: `module`

- **sage.groups.perm_gps.constructor**
  - Entity: `constructor`
  - Module: `sage.groups.perm_gps.constructor`
  - Type: `module`

- **sage.groups.perm_gps.cubegroup**
  - Entity: `cubegroup`
  - Module: `sage.groups.perm_gps.cubegroup`
  - Type: `module`

- **sage.groups.perm_gps.partn_ref.canonical_augmentation**
  - Entity: `canonical_augmentation`
  - Module: `sage.groups.perm_gps.partn_ref.canonical_augmentation`
  - Type: `module`

- **sage.groups.perm_gps.partn_ref.data_structures**
  - Entity: `data_structures`
  - Module: `sage.groups.perm_gps.partn_ref.data_structures`
  - Type: `module`

- **sage.groups.perm_gps.partn_ref.refinement_graphs**
  - Entity: `refinement_graphs`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_graphs`
  - Type: `module`

- **sage.groups.perm_gps.partn_ref.refinement_lists**
  - Entity: `refinement_lists`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_lists`
  - Type: `module`

- **sage.groups.perm_gps.partn_ref.refinement_matrices**
  - Entity: `refinement_matrices`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_matrices`
  - Type: `module`


### Part 08 (358 entries)

#### ATTRIBUTE (5 entries)

- **sage.manifolds.manifold.options**
  - Entity: `options`
  - Module: `sage.manifolds.manifold`
  - Type: `attribute`

- **sage.manifolds.structure.chart**
  - Entity: `chart`
  - Module: `sage.manifolds.structure`
  - Type: `attribute`

- **sage.manifolds.structure.homset**
  - Entity: `homset`
  - Module: `sage.manifolds.structure`
  - Type: `attribute`

- **sage.manifolds.structure.name**
  - Entity: `name`
  - Module: `sage.manifolds.structure`
  - Type: `attribute`

- **sage.manifolds.structure.scalar_field_algebra**
  - Entity: `scalar_field_algebra`
  - Module: `sage.manifolds.structure`
  - Type: `attribute`

#### CLASS (193 entries)

- **sage.groups.perm_gps.partn_ref.refinement_matrices.MatrixStruct**
  - Entity: `MatrixStruct`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_matrices`
  - Type: `class`

- **sage.groups.perm_gps.permgroup.PermutationGroup_action**
  - Entity: `PermutationGroup_action`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `class`

- **sage.groups.perm_gps.permgroup.PermutationGroup_generic**
  - Entity: `PermutationGroup_generic`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `class`

- **sage.groups.perm_gps.permgroup.PermutationGroup_subgroup**
  - Entity: `PermutationGroup_subgroup`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_element.PermutationGroupElement**
  - Entity: `PermutationGroupElement`
  - Module: `sage.groups.perm_gps.permgroup_element`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_element.SymmetricGroupElement**
  - Entity: `SymmetricGroupElement`
  - Module: `sage.groups.perm_gps.permgroup_element`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_morphism.PermutationGroupMorphism**
  - Entity: `PermutationGroupMorphism`
  - Module: `sage.groups.perm_gps.permgroup_morphism`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_morphism.PermutationGroupMorphism_from_gap**
  - Entity: `PermutationGroupMorphism_from_gap`
  - Module: `sage.groups.perm_gps.permgroup_morphism`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_morphism.PermutationGroupMorphism_id**
  - Entity: `PermutationGroupMorphism_id`
  - Module: `sage.groups.perm_gps.permgroup_morphism`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_morphism.PermutationGroupMorphism_im_gens**
  - Entity: `PermutationGroupMorphism_im_gens`
  - Module: `sage.groups.perm_gps.permgroup_morphism`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.AlternatingGroup**
  - Entity: `AlternatingGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.ComplexReflectionGroup**
  - Entity: `ComplexReflectionGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.CyclicPermutationGroup**
  - Entity: `CyclicPermutationGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.DiCyclicGroup**
  - Entity: `DiCyclicGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.DihedralGroup**
  - Entity: `DihedralGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.GeneralDihedralGroup**
  - Entity: `GeneralDihedralGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.JankoGroup**
  - Entity: `JankoGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.KleinFourGroup**
  - Entity: `KleinFourGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.MathieuGroup**
  - Entity: `MathieuGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PGL**
  - Entity: `PGL`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PGU**
  - Entity: `PGU`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PSL**
  - Entity: `PSL`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PSU**
  - Entity: `PSU`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PSp**
  - Entity: `PSp`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PermutationGroup_plg**
  - Entity: `PermutationGroup_plg`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PermutationGroup_pug**
  - Entity: `PermutationGroup_pug`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PermutationGroup_symalt**
  - Entity: `PermutationGroup_symalt`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PermutationGroup_unique**
  - Entity: `PermutationGroup_unique`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PrimitiveGroup**
  - Entity: `PrimitiveGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PrimitiveGroupsAll**
  - Entity: `PrimitiveGroupsAll`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.PrimitiveGroupsOfDegree**
  - Entity: `PrimitiveGroupsOfDegree`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.QuaternionGroup**
  - Entity: `QuaternionGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.SemidihedralGroup**
  - Entity: `SemidihedralGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.SmallPermutationGroup**
  - Entity: `SmallPermutationGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.SplitMetacyclicGroup**
  - Entity: `SplitMetacyclicGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.SuzukiGroup**
  - Entity: `SuzukiGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.SuzukiSporadicGroup**
  - Entity: `SuzukiSporadicGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.SymmetricGroup**
  - Entity: `SymmetricGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.TransitiveGroup**
  - Entity: `TransitiveGroup`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.TransitiveGroupsAll**
  - Entity: `TransitiveGroupsAll`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.permgroup_named.TransitiveGroupsOfDegree**
  - Entity: `TransitiveGroupsOfDegree`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `class`

- **sage.groups.perm_gps.symgp_conjugacy_class.PermutationsConjugacyClass**
  - Entity: `PermutationsConjugacyClass`
  - Module: `sage.groups.perm_gps.symgp_conjugacy_class`
  - Type: `class`

- **sage.groups.perm_gps.symgp_conjugacy_class.SymmetricGroupConjugacyClass**
  - Entity: `SymmetricGroupConjugacyClass`
  - Module: `sage.groups.perm_gps.symgp_conjugacy_class`
  - Type: `class`

- **sage.groups.perm_gps.symgp_conjugacy_class.SymmetricGroupConjugacyClassMixin**
  - Entity: `SymmetricGroupConjugacyClassMixin`
  - Module: `sage.groups.perm_gps.symgp_conjugacy_class`
  - Type: `class`

- **sage.homology.chain_complex.ChainComplex_class**
  - Entity: `ChainComplex_class`
  - Module: `sage.homology.chain_complex`
  - Type: `class`

- **sage.homology.chain_complex.Chain_class**
  - Entity: `Chain_class`
  - Module: `sage.homology.chain_complex`
  - Type: `class`

- **sage.homology.chain_complex_homspace.ChainComplexHomspace**
  - Entity: `ChainComplexHomspace`
  - Module: `sage.homology.chain_complex_homspace`
  - Type: `class`

- **sage.homology.chain_complex_morphism.ChainComplexMorphism**
  - Entity: `ChainComplexMorphism`
  - Module: `sage.homology.chain_complex_morphism`
  - Type: `class`

- **sage.homology.chain_homotopy.ChainContraction**
  - Entity: `ChainContraction`
  - Module: `sage.homology.chain_homotopy`
  - Type: `class`

- **sage.homology.chain_homotopy.ChainHomotopy**
  - Entity: `ChainHomotopy`
  - Module: `sage.homology.chain_homotopy`
  - Type: `class`

- **sage.homology.chains.CellComplexReference**
  - Entity: `CellComplexReference`
  - Module: `sage.homology.chains`
  - Type: `class`

- **sage.homology.chains.Chains**
  - Entity: `Chains`
  - Module: `sage.homology.chains`
  - Type: `class`

- **sage.homology.chains.Cochains**
  - Entity: `Cochains`
  - Module: `sage.homology.chains`
  - Type: `class`

- **sage.homology.chains.Element**
  - Entity: `Element`
  - Module: `sage.homology.chains`
  - Type: `class`

- **sage.homology.free_resolution.FiniteFreeResolution**
  - Entity: `FiniteFreeResolution`
  - Module: `sage.homology.free_resolution`
  - Type: `class`

- **sage.homology.free_resolution.FiniteFreeResolution_free_module**
  - Entity: `FiniteFreeResolution_free_module`
  - Module: `sage.homology.free_resolution`
  - Type: `class`

- **sage.homology.free_resolution.FiniteFreeResolution_singular**
  - Entity: `FiniteFreeResolution_singular`
  - Module: `sage.homology.free_resolution`
  - Type: `class`

- **sage.homology.free_resolution.FreeResolution**
  - Entity: `FreeResolution`
  - Module: `sage.homology.free_resolution`
  - Type: `class`

- **sage.homology.graded_resolution.GradedFiniteFreeResolution**
  - Entity: `GradedFiniteFreeResolution`
  - Module: `sage.homology.graded_resolution`
  - Type: `class`

- **sage.homology.graded_resolution.GradedFiniteFreeResolution_free_module**
  - Entity: `GradedFiniteFreeResolution_free_module`
  - Module: `sage.homology.graded_resolution`
  - Type: `class`

- **sage.homology.graded_resolution.GradedFiniteFreeResolution_singular**
  - Entity: `GradedFiniteFreeResolution_singular`
  - Module: `sage.homology.graded_resolution`
  - Type: `class`

- **sage.homology.hochschild_complex.Element**
  - Entity: `Element`
  - Module: `sage.homology.hochschild_complex`
  - Type: `class`

- **sage.homology.hochschild_complex.HochschildComplex**
  - Entity: `HochschildComplex`
  - Module: `sage.homology.hochschild_complex`
  - Type: `class`

- **sage.homology.homology_group.HomologyGroup_class**
  - Entity: `HomologyGroup_class`
  - Module: `sage.homology.homology_group`
  - Type: `class`

- **sage.homology.homology_morphism.InducedHomologyMorphism**
  - Entity: `InducedHomologyMorphism`
  - Module: `sage.homology.homology_morphism`
  - Type: `class`

- **sage.homology.homology_vector_space_with_basis.CohomologyRing**
  - Entity: `CohomologyRing`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `class`

- **sage.homology.homology_vector_space_with_basis.CohomologyRing_mod2**
  - Entity: `CohomologyRing_mod2`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `class`

- **sage.homology.homology_vector_space_with_basis.Element**
  - Entity: `Element`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `class`

- **sage.homology.homology_vector_space_with_basis.HomologyVectorSpaceWithBasis**
  - Entity: `HomologyVectorSpaceWithBasis`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `class`

- **sage.homology.homology_vector_space_with_basis.HomologyVectorSpaceWithBasis_mod2**
  - Entity: `HomologyVectorSpaceWithBasis_mod2`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `class`

- **sage.homology.koszul_complex.KoszulComplex**
  - Entity: `KoszulComplex`
  - Module: `sage.homology.koszul_complex`
  - Type: `class`

- **sage.logic.boolformula.BooleanFormula**
  - Entity: `BooleanFormula`
  - Module: `sage.logic.boolformula`
  - Type: `class`

- **sage.logic.logic.SymbolicLogic**
  - Entity: `SymbolicLogic`
  - Module: `sage.logic.logic`
  - Type: `class`

- **sage.logic.logictable.Truthtable**
  - Entity: `Truthtable`
  - Module: `sage.logic.logictable`
  - Type: `class`

- **sage.manifolds.calculus_method.CalculusMethod**
  - Entity: `CalculusMethod`
  - Module: `sage.manifolds.calculus_method`
  - Type: `class`

- **sage.manifolds.chart.Chart**
  - Entity: `Chart`
  - Module: `sage.manifolds.chart`
  - Type: `class`

- **sage.manifolds.chart.CoordChange**
  - Entity: `CoordChange`
  - Module: `sage.manifolds.chart`
  - Type: `class`

- **sage.manifolds.chart.RealChart**
  - Entity: `RealChart`
  - Module: `sage.manifolds.chart`
  - Type: `class`

- **sage.manifolds.chart_func.ChartFunction**
  - Entity: `ChartFunction`
  - Module: `sage.manifolds.chart_func`
  - Type: `class`

- **sage.manifolds.chart_func.ChartFunctionRing**
  - Entity: `ChartFunctionRing`
  - Module: `sage.manifolds.chart_func`
  - Type: `class`

- **sage.manifolds.chart_func.MultiCoordFunction**
  - Entity: `MultiCoordFunction`
  - Module: `sage.manifolds.chart_func`
  - Type: `class`

- **sage.manifolds.continuous_map.ContinuousMap**
  - Entity: `ContinuousMap`
  - Module: `sage.manifolds.continuous_map`
  - Type: `class`

- **sage.manifolds.continuous_map_image.ImageManifoldSubset**
  - Entity: `ImageManifoldSubset`
  - Module: `sage.manifolds.continuous_map_image`
  - Type: `class`

- **sage.manifolds.differentiable.affine_connection.AffineConnection**
  - Entity: `AffineConnection`
  - Module: `sage.manifolds.differentiable.affine_connection`
  - Type: `class`

- **sage.manifolds.differentiable.automorphismfield.AutomorphismField**
  - Entity: `AutomorphismField`
  - Module: `sage.manifolds.differentiable.automorphismfield`
  - Type: `class`

- **sage.manifolds.differentiable.automorphismfield.AutomorphismFieldParal**
  - Entity: `AutomorphismFieldParal`
  - Module: `sage.manifolds.differentiable.automorphismfield`
  - Type: `class`

- **sage.manifolds.differentiable.automorphismfield_group.AutomorphismFieldGroup**
  - Entity: `AutomorphismFieldGroup`
  - Module: `sage.manifolds.differentiable.automorphismfield_group`
  - Type: `class`

- **sage.manifolds.differentiable.automorphismfield_group.AutomorphismFieldParalGroup**
  - Entity: `AutomorphismFieldParalGroup`
  - Module: `sage.manifolds.differentiable.automorphismfield_group`
  - Type: `class`

- **sage.manifolds.differentiable.bundle_connection.BundleConnection**
  - Entity: `BundleConnection`
  - Module: `sage.manifolds.differentiable.bundle_connection`
  - Type: `class`

- **sage.manifolds.differentiable.characteristic_cohomology_class.Algorithm_generic**
  - Entity: `Algorithm_generic`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `class`

- **sage.manifolds.differentiable.characteristic_cohomology_class.CharacteristicCohomologyClassRing**
  - Entity: `CharacteristicCohomologyClassRing`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `class`

- **sage.manifolds.differentiable.characteristic_cohomology_class.CharacteristicCohomologyClassRingElement**
  - Entity: `CharacteristicCohomologyClassRingElement`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `class`

- **sage.manifolds.differentiable.characteristic_cohomology_class.ChernAlgorithm**
  - Entity: `ChernAlgorithm`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `class`

- **sage.manifolds.differentiable.characteristic_cohomology_class.EulerAlgorithm**
  - Entity: `EulerAlgorithm`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `class`

- **sage.manifolds.differentiable.characteristic_cohomology_class.PontryaginAlgorithm**
  - Entity: `PontryaginAlgorithm`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `class`

- **sage.manifolds.differentiable.characteristic_cohomology_class.PontryaginEulerAlgorithm**
  - Entity: `PontryaginEulerAlgorithm`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `class`

- **sage.manifolds.differentiable.chart.DiffChart**
  - Entity: `DiffChart`
  - Module: `sage.manifolds.differentiable.chart`
  - Type: `class`

- **sage.manifolds.differentiable.chart.DiffCoordChange**
  - Entity: `DiffCoordChange`
  - Module: `sage.manifolds.differentiable.chart`
  - Type: `class`

- **sage.manifolds.differentiable.chart.RealDiffChart**
  - Entity: `RealDiffChart`
  - Module: `sage.manifolds.differentiable.chart`
  - Type: `class`

- **sage.manifolds.differentiable.curve.DifferentiableCurve**
  - Entity: `DifferentiableCurve`
  - Module: `sage.manifolds.differentiable.curve`
  - Type: `class`

- **sage.manifolds.differentiable.de_rham_cohomology.DeRhamCohomologyClass**
  - Entity: `DeRhamCohomologyClass`
  - Module: `sage.manifolds.differentiable.de_rham_cohomology`
  - Type: `class`

- **sage.manifolds.differentiable.de_rham_cohomology.DeRhamCohomologyRing**
  - Entity: `DeRhamCohomologyRing`
  - Module: `sage.manifolds.differentiable.de_rham_cohomology`
  - Type: `class`

- **sage.manifolds.differentiable.degenerate.DegenerateManifold**
  - Entity: `DegenerateManifold`
  - Module: `sage.manifolds.differentiable.degenerate`
  - Type: `class`

- **sage.manifolds.differentiable.degenerate.TangentTensor**
  - Entity: `TangentTensor`
  - Module: `sage.manifolds.differentiable.degenerate`
  - Type: `class`

- **sage.manifolds.differentiable.degenerate_submanifold.DegenerateSubmanifold**
  - Entity: `DegenerateSubmanifold`
  - Module: `sage.manifolds.differentiable.degenerate_submanifold`
  - Type: `class`

- **sage.manifolds.differentiable.degenerate_submanifold.Screen**
  - Entity: `Screen`
  - Module: `sage.manifolds.differentiable.degenerate_submanifold`
  - Type: `class`

- **sage.manifolds.differentiable.diff_form.DiffForm**
  - Entity: `DiffForm`
  - Module: `sage.manifolds.differentiable.diff_form`
  - Type: `class`

- **sage.manifolds.differentiable.diff_form.DiffFormParal**
  - Entity: `DiffFormParal`
  - Module: `sage.manifolds.differentiable.diff_form`
  - Type: `class`

- **sage.manifolds.differentiable.diff_form_module.DiffFormFreeModule**
  - Entity: `DiffFormFreeModule`
  - Module: `sage.manifolds.differentiable.diff_form_module`
  - Type: `class`

- **sage.manifolds.differentiable.diff_form_module.DiffFormModule**
  - Entity: `DiffFormModule`
  - Module: `sage.manifolds.differentiable.diff_form_module`
  - Type: `class`

- **sage.manifolds.differentiable.diff_form_module.VectorFieldDualFreeModule**
  - Entity: `VectorFieldDualFreeModule`
  - Module: `sage.manifolds.differentiable.diff_form_module`
  - Type: `class`

- **sage.manifolds.differentiable.diff_map.DiffMap**
  - Entity: `DiffMap`
  - Module: `sage.manifolds.differentiable.diff_map`
  - Type: `class`

- **sage.manifolds.differentiable.differentiable_submanifold.DifferentiableSubmanifold**
  - Entity: `DifferentiableSubmanifold`
  - Module: `sage.manifolds.differentiable.differentiable_submanifold`
  - Type: `class`

- **sage.manifolds.differentiable.examples.euclidean.Euclidean3dimSpace**
  - Entity: `Euclidean3dimSpace`
  - Module: `sage.manifolds.differentiable.examples.euclidean`
  - Type: `class`

- **sage.manifolds.differentiable.examples.euclidean.EuclideanPlane**
  - Entity: `EuclideanPlane`
  - Module: `sage.manifolds.differentiable.examples.euclidean`
  - Type: `class`

- **sage.manifolds.differentiable.examples.euclidean.EuclideanSpace**
  - Entity: `EuclideanSpace`
  - Module: `sage.manifolds.differentiable.examples.euclidean`
  - Type: `class`

- **sage.manifolds.differentiable.examples.real_line.OpenInterval**
  - Entity: `OpenInterval`
  - Module: `sage.manifolds.differentiable.examples.real_line`
  - Type: `class`

- **sage.manifolds.differentiable.examples.real_line.RealLine**
  - Entity: `RealLine`
  - Module: `sage.manifolds.differentiable.examples.real_line`
  - Type: `class`

- **sage.manifolds.differentiable.examples.sphere.Sphere**
  - Entity: `Sphere`
  - Module: `sage.manifolds.differentiable.examples.sphere`
  - Type: `class`

- **sage.manifolds.differentiable.examples.symplectic_space.StandardSymplecticSpace**
  - Entity: `StandardSymplecticSpace`
  - Module: `sage.manifolds.differentiable.examples.symplectic_space`
  - Type: `class`

- **sage.manifolds.differentiable.integrated_curve.IntegratedAutoparallelCurve**
  - Entity: `IntegratedAutoparallelCurve`
  - Module: `sage.manifolds.differentiable.integrated_curve`
  - Type: `class`

- **sage.manifolds.differentiable.integrated_curve.IntegratedCurve**
  - Entity: `IntegratedCurve`
  - Module: `sage.manifolds.differentiable.integrated_curve`
  - Type: `class`

- **sage.manifolds.differentiable.integrated_curve.IntegratedGeodesic**
  - Entity: `IntegratedGeodesic`
  - Module: `sage.manifolds.differentiable.integrated_curve`
  - Type: `class`

- **sage.manifolds.differentiable.levi_civita_connection.LeviCivitaConnection**
  - Entity: `LeviCivitaConnection`
  - Module: `sage.manifolds.differentiable.levi_civita_connection`
  - Type: `class`

- **sage.manifolds.differentiable.manifold.DifferentiableManifold**
  - Entity: `DifferentiableManifold`
  - Module: `sage.manifolds.differentiable.manifold`
  - Type: `class`

- **sage.manifolds.differentiable.manifold_homset.DifferentiableCurveSet**
  - Entity: `DifferentiableCurveSet`
  - Module: `sage.manifolds.differentiable.manifold_homset`
  - Type: `class`

- **sage.manifolds.differentiable.manifold_homset.DifferentiableManifoldHomset**
  - Entity: `DifferentiableManifoldHomset`
  - Module: `sage.manifolds.differentiable.manifold_homset`
  - Type: `class`

- **sage.manifolds.differentiable.manifold_homset.IntegratedAutoparallelCurveSet**
  - Entity: `IntegratedAutoparallelCurveSet`
  - Module: `sage.manifolds.differentiable.manifold_homset`
  - Type: `class`

- **sage.manifolds.differentiable.manifold_homset.IntegratedCurveSet**
  - Entity: `IntegratedCurveSet`
  - Module: `sage.manifolds.differentiable.manifold_homset`
  - Type: `class`

- **sage.manifolds.differentiable.manifold_homset.IntegratedGeodesicSet**
  - Entity: `IntegratedGeodesicSet`
  - Module: `sage.manifolds.differentiable.manifold_homset`
  - Type: `class`

- **sage.manifolds.differentiable.metric.DegenerateMetric**
  - Entity: `DegenerateMetric`
  - Module: `sage.manifolds.differentiable.metric`
  - Type: `class`

- **sage.manifolds.differentiable.metric.DegenerateMetricParal**
  - Entity: `DegenerateMetricParal`
  - Module: `sage.manifolds.differentiable.metric`
  - Type: `class`

- **sage.manifolds.differentiable.metric.PseudoRiemannianMetric**
  - Entity: `PseudoRiemannianMetric`
  - Module: `sage.manifolds.differentiable.metric`
  - Type: `class`

- **sage.manifolds.differentiable.metric.PseudoRiemannianMetricParal**
  - Entity: `PseudoRiemannianMetricParal`
  - Module: `sage.manifolds.differentiable.metric`
  - Type: `class`

- **sage.manifolds.differentiable.mixed_form.MixedForm**
  - Entity: `MixedForm`
  - Module: `sage.manifolds.differentiable.mixed_form`
  - Type: `class`

- **sage.manifolds.differentiable.mixed_form_algebra.MixedFormAlgebra**
  - Entity: `MixedFormAlgebra`
  - Module: `sage.manifolds.differentiable.mixed_form_algebra`
  - Type: `class`

- **sage.manifolds.differentiable.multivector_module.MultivectorFreeModule**
  - Entity: `MultivectorFreeModule`
  - Module: `sage.manifolds.differentiable.multivector_module`
  - Type: `class`

- **sage.manifolds.differentiable.multivector_module.MultivectorModule**
  - Entity: `MultivectorModule`
  - Module: `sage.manifolds.differentiable.multivector_module`
  - Type: `class`

- **sage.manifolds.differentiable.multivectorfield.MultivectorField**
  - Entity: `MultivectorField`
  - Module: `sage.manifolds.differentiable.multivectorfield`
  - Type: `class`

- **sage.manifolds.differentiable.multivectorfield.MultivectorFieldParal**
  - Entity: `MultivectorFieldParal`
  - Module: `sage.manifolds.differentiable.multivectorfield`
  - Type: `class`

- **sage.manifolds.differentiable.poisson_tensor.PoissonTensorField**
  - Entity: `PoissonTensorField`
  - Module: `sage.manifolds.differentiable.poisson_tensor`
  - Type: `class`

- **sage.manifolds.differentiable.poisson_tensor.PoissonTensorFieldParal**
  - Entity: `PoissonTensorFieldParal`
  - Module: `sage.manifolds.differentiable.poisson_tensor`
  - Type: `class`

- **sage.manifolds.differentiable.pseudo_riemannian.PseudoRiemannianManifold**
  - Entity: `PseudoRiemannianManifold`
  - Module: `sage.manifolds.differentiable.pseudo_riemannian`
  - Type: `class`

- **sage.manifolds.differentiable.pseudo_riemannian_submanifold.PseudoRiemannianSubmanifold**
  - Entity: `PseudoRiemannianSubmanifold`
  - Module: `sage.manifolds.differentiable.pseudo_riemannian_submanifold`
  - Type: `class`

- **sage.manifolds.differentiable.scalarfield.DiffScalarField**
  - Entity: `DiffScalarField`
  - Module: `sage.manifolds.differentiable.scalarfield`
  - Type: `class`

- **sage.manifolds.differentiable.scalarfield_algebra.DiffScalarFieldAlgebra**
  - Entity: `DiffScalarFieldAlgebra`
  - Module: `sage.manifolds.differentiable.scalarfield_algebra`
  - Type: `class`

- **sage.manifolds.differentiable.symplectic_form.SymplecticForm**
  - Entity: `SymplecticForm`
  - Module: `sage.manifolds.differentiable.symplectic_form`
  - Type: `class`

- **sage.manifolds.differentiable.symplectic_form.SymplecticFormParal**
  - Entity: `SymplecticFormParal`
  - Module: `sage.manifolds.differentiable.symplectic_form`
  - Type: `class`

- **sage.manifolds.differentiable.tangent_space.TangentSpace**
  - Entity: `TangentSpace`
  - Module: `sage.manifolds.differentiable.tangent_space`
  - Type: `class`

- **sage.manifolds.differentiable.tangent_vector.TangentVector**
  - Entity: `TangentVector`
  - Module: `sage.manifolds.differentiable.tangent_vector`
  - Type: `class`

- **sage.manifolds.differentiable.tensorfield.TensorField**
  - Entity: `TensorField`
  - Module: `sage.manifolds.differentiable.tensorfield`
  - Type: `class`

- **sage.manifolds.differentiable.tensorfield_module.TensorFieldFreeModule**
  - Entity: `TensorFieldFreeModule`
  - Module: `sage.manifolds.differentiable.tensorfield_module`
  - Type: `class`

- **sage.manifolds.differentiable.tensorfield_module.TensorFieldModule**
  - Entity: `TensorFieldModule`
  - Module: `sage.manifolds.differentiable.tensorfield_module`
  - Type: `class`

- **sage.manifolds.differentiable.tensorfield_paral.TensorFieldParal**
  - Entity: `TensorFieldParal`
  - Module: `sage.manifolds.differentiable.tensorfield_paral`
  - Type: `class`

- **sage.manifolds.differentiable.vector_bundle.DifferentiableVectorBundle**
  - Entity: `DifferentiableVectorBundle`
  - Module: `sage.manifolds.differentiable.vector_bundle`
  - Type: `class`

- **sage.manifolds.differentiable.vector_bundle.TensorBundle**
  - Entity: `TensorBundle`
  - Module: `sage.manifolds.differentiable.vector_bundle`
  - Type: `class`

- **sage.manifolds.differentiable.vectorfield.VectorField**
  - Entity: `VectorField`
  - Module: `sage.manifolds.differentiable.vectorfield`
  - Type: `class`

- **sage.manifolds.differentiable.vectorfield.VectorFieldParal**
  - Entity: `VectorFieldParal`
  - Module: `sage.manifolds.differentiable.vectorfield`
  - Type: `class`

- **sage.manifolds.differentiable.vectorfield_module.VectorFieldFreeModule**
  - Entity: `VectorFieldFreeModule`
  - Module: `sage.manifolds.differentiable.vectorfield_module`
  - Type: `class`

- **sage.manifolds.differentiable.vectorfield_module.VectorFieldModule**
  - Entity: `VectorFieldModule`
  - Module: `sage.manifolds.differentiable.vectorfield_module`
  - Type: `class`

- **sage.manifolds.differentiable.vectorframe.CoFrame**
  - Entity: `CoFrame`
  - Module: `sage.manifolds.differentiable.vectorframe`
  - Type: `class`

- **sage.manifolds.differentiable.vectorframe.CoordCoFrame**
  - Entity: `CoordCoFrame`
  - Module: `sage.manifolds.differentiable.vectorframe`
  - Type: `class`

- **sage.manifolds.differentiable.vectorframe.CoordFrame**
  - Entity: `CoordFrame`
  - Module: `sage.manifolds.differentiable.vectorframe`
  - Type: `class`

- **sage.manifolds.differentiable.vectorframe.VectorFrame**
  - Entity: `VectorFrame`
  - Module: `sage.manifolds.differentiable.vectorframe`
  - Type: `class`

- **sage.manifolds.family.ManifoldObjectFiniteFamily**
  - Entity: `ManifoldObjectFiniteFamily`
  - Module: `sage.manifolds.family`
  - Type: `class`

- **sage.manifolds.family.ManifoldSubsetFiniteFamily**
  - Entity: `ManifoldSubsetFiniteFamily`
  - Module: `sage.manifolds.family`
  - Type: `class`

- **sage.manifolds.local_frame.LocalCoFrame**
  - Entity: `LocalCoFrame`
  - Module: `sage.manifolds.local_frame`
  - Type: `class`

- **sage.manifolds.local_frame.LocalFrame**
  - Entity: `LocalFrame`
  - Module: `sage.manifolds.local_frame`
  - Type: `class`

- **sage.manifolds.local_frame.TrivializationCoFrame**
  - Entity: `TrivializationCoFrame`
  - Module: `sage.manifolds.local_frame`
  - Type: `class`

- **sage.manifolds.local_frame.TrivializationFrame**
  - Entity: `TrivializationFrame`
  - Module: `sage.manifolds.local_frame`
  - Type: `class`

- **sage.manifolds.manifold.TopologicalManifold**
  - Entity: `TopologicalManifold`
  - Module: `sage.manifolds.manifold`
  - Type: `class`

- **sage.manifolds.manifold_homset.TopologicalManifoldHomset**
  - Entity: `TopologicalManifoldHomset`
  - Module: `sage.manifolds.manifold_homset`
  - Type: `class`

- **sage.manifolds.point.ManifoldPoint**
  - Entity: `ManifoldPoint`
  - Module: `sage.manifolds.point`
  - Type: `class`

- **sage.manifolds.scalarfield.ScalarField**
  - Entity: `ScalarField`
  - Module: `sage.manifolds.scalarfield`
  - Type: `class`

- **sage.manifolds.scalarfield_algebra.ScalarFieldAlgebra**
  - Entity: `ScalarFieldAlgebra`
  - Module: `sage.manifolds.scalarfield_algebra`
  - Type: `class`

- **sage.manifolds.section.Section**
  - Entity: `Section`
  - Module: `sage.manifolds.section`
  - Type: `class`

- **sage.manifolds.section.TrivialSection**
  - Entity: `TrivialSection`
  - Module: `sage.manifolds.section`
  - Type: `class`

- **sage.manifolds.section_module.SectionFreeModule**
  - Entity: `SectionFreeModule`
  - Module: `sage.manifolds.section_module`
  - Type: `class`

- **sage.manifolds.section_module.SectionModule**
  - Entity: `SectionModule`
  - Module: `sage.manifolds.section_module`
  - Type: `class`

- **sage.manifolds.structure.DegenerateStructure**
  - Entity: `DegenerateStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.structure.DifferentialStructure**
  - Entity: `DifferentialStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.structure.LorentzianStructure**
  - Entity: `LorentzianStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.structure.PseudoRiemannianStructure**
  - Entity: `PseudoRiemannianStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.structure.RealDifferentialStructure**
  - Entity: `RealDifferentialStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.structure.RealTopologicalStructure**
  - Entity: `RealTopologicalStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.structure.RiemannianStructure**
  - Entity: `RiemannianStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.structure.TopologicalStructure**
  - Entity: `TopologicalStructure`
  - Module: `sage.manifolds.structure`
  - Type: `class`

- **sage.manifolds.subset.ManifoldSubset**
  - Entity: `ManifoldSubset`
  - Module: `sage.manifolds.subset`
  - Type: `class`

- **sage.manifolds.subsets.closure.ManifoldSubsetClosure**
  - Entity: `ManifoldSubsetClosure`
  - Module: `sage.manifolds.subsets.closure`
  - Type: `class`

- **sage.manifolds.subsets.pullback.ManifoldSubsetPullback**
  - Entity: `ManifoldSubsetPullback`
  - Module: `sage.manifolds.subsets.pullback`
  - Type: `class`

- **sage.manifolds.topological_submanifold.TopologicalSubmanifold**
  - Entity: `TopologicalSubmanifold`
  - Module: `sage.manifolds.topological_submanifold`
  - Type: `class`

- **sage.manifolds.trivialization.TransitionMap**
  - Entity: `TransitionMap`
  - Module: `sage.manifolds.trivialization`
  - Type: `class`

- **sage.manifolds.trivialization.Trivialization**
  - Entity: `Trivialization`
  - Module: `sage.manifolds.trivialization`
  - Type: `class`

#### FUNCTION (66 entries)

- **sage.groups.perm_gps.partn_ref.refinement_matrices.random_tests**
  - Entity: `random_tests`
  - Module: `sage.groups.perm_gps.partn_ref.refinement_matrices`
  - Type: `function`

- **sage.groups.perm_gps.permgroup.PermutationGroup**
  - Entity: `PermutationGroup`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `function`

- **sage.groups.perm_gps.permgroup.direct_product_permgroups**
  - Entity: `direct_product_permgroups`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `function`

- **sage.groups.perm_gps.permgroup.from_gap_list**
  - Entity: `from_gap_list`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `function`

- **sage.groups.perm_gps.permgroup.hap_decorator**
  - Entity: `hap_decorator`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `function`

- **sage.groups.perm_gps.permgroup.load_hap**
  - Entity: `load_hap`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `function`

- **sage.groups.perm_gps.permgroup_element.is_PermutationGroupElement**
  - Entity: `is_PermutationGroupElement`
  - Module: `sage.groups.perm_gps.permgroup_element`
  - Type: `function`

- **sage.groups.perm_gps.permgroup_element.make_permgroup_element**
  - Entity: `make_permgroup_element`
  - Module: `sage.groups.perm_gps.permgroup_element`
  - Type: `function`

- **sage.groups.perm_gps.permgroup_element.make_permgroup_element_v2**
  - Entity: `make_permgroup_element_v2`
  - Module: `sage.groups.perm_gps.permgroup_element`
  - Type: `function`

- **sage.groups.perm_gps.permgroup_morphism.is_PermutationGroupMorphism**
  - Entity: `is_PermutationGroupMorphism`
  - Module: `sage.groups.perm_gps.permgroup_morphism`
  - Type: `function`

- **sage.groups.perm_gps.permgroup_named.PrimitiveGroups**
  - Entity: `PrimitiveGroups`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `function`

- **sage.groups.perm_gps.permgroup_named.TransitiveGroups**
  - Entity: `TransitiveGroups`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `function`

- **sage.groups.perm_gps.symgp_conjugacy_class.conjugacy_class_iterator**
  - Entity: `conjugacy_class_iterator`
  - Module: `sage.groups.perm_gps.symgp_conjugacy_class`
  - Type: `function`

- **sage.groups.perm_gps.symgp_conjugacy_class.default_representative**
  - Entity: `default_representative`
  - Module: `sage.groups.perm_gps.symgp_conjugacy_class`
  - Type: `function`

- **sage.homology.algebraic_topological_model.algebraic_topological_model**
  - Entity: `algebraic_topological_model`
  - Module: `sage.homology.algebraic_topological_model`
  - Type: `function`

- **sage.homology.algebraic_topological_model.algebraic_topological_model_delta_complex**
  - Entity: `algebraic_topological_model_delta_complex`
  - Module: `sage.homology.algebraic_topological_model`
  - Type: `function`

- **sage.homology.chain_complex.ChainComplex**
  - Entity: `ChainComplex`
  - Module: `sage.homology.chain_complex`
  - Type: `function`

- **sage.homology.chain_complex_homspace.is_ChainComplexHomspace**
  - Entity: `is_ChainComplexHomspace`
  - Module: `sage.homology.chain_complex_homspace`
  - Type: `function`

- **sage.homology.chain_complex_morphism.is_ChainComplexMorphism**
  - Entity: `is_ChainComplexMorphism`
  - Module: `sage.homology.chain_complex_morphism`
  - Type: `function`

- **sage.homology.homology_group.HomologyGroup**
  - Entity: `HomologyGroup`
  - Module: `sage.homology.homology_group`
  - Type: `function`

- **sage.homology.homology_vector_space_with_basis.is_GF2**
  - Entity: `is_GF2`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `function`

- **sage.homology.homology_vector_space_with_basis.sum_indices**
  - Entity: `sum_indices`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `function`

- **sage.homology.matrix_utils.dhsw_snf**
  - Entity: `dhsw_snf`
  - Module: `sage.homology.matrix_utils`
  - Type: `function`

- **sage.logic.booleval.eval_f**
  - Entity: `eval_f`
  - Module: `sage.logic.booleval`
  - Type: `function`

- **sage.logic.booleval.eval_formula**
  - Entity: `eval_formula`
  - Module: `sage.logic.booleval`
  - Type: `function`

- **sage.logic.booleval.eval_op**
  - Entity: `eval_op`
  - Module: `sage.logic.booleval`
  - Type: `function`

- **sage.logic.boolformula.is_consequence**
  - Entity: `is_consequence`
  - Module: `sage.logic.boolformula`
  - Type: `function`

- **sage.logic.logic.eval**
  - Entity: `eval`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.eval_and_op**
  - Entity: `eval_and_op`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.eval_bin_op**
  - Entity: `eval_bin_op`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.eval_iff_op**
  - Entity: `eval_iff_op`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.eval_ifthen_op**
  - Entity: `eval_ifthen_op`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.eval_ltor_toks**
  - Entity: `eval_ltor_toks`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.eval_mon_op**
  - Entity: `eval_mon_op`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.eval_or_op**
  - Entity: `eval_or_op`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.get_bit**
  - Entity: `get_bit`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.reduce_bins**
  - Entity: `reduce_bins`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.reduce_monos**
  - Entity: `reduce_monos`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logic.tokenize**
  - Entity: `tokenize`
  - Module: `sage.logic.logic`
  - Type: `function`

- **sage.logic.logicparser.apply_func**
  - Entity: `apply_func`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.get_trees**
  - Entity: `get_trees`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.parse**
  - Entity: `parse`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.parse_ltor**
  - Entity: `parse_ltor`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.polish_parse**
  - Entity: `polish_parse`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.prefix_to_infix**
  - Entity: `prefix_to_infix`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.recover_formula**
  - Entity: `recover_formula`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.recover_formula_internal**
  - Entity: `recover_formula_internal`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.to_infix_internal**
  - Entity: `to_infix_internal`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.tokenize**
  - Entity: `tokenize`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.logicparser.tree_parse**
  - Entity: `tree_parse`
  - Module: `sage.logic.logicparser`
  - Type: `function`

- **sage.logic.propcalc.consistent**
  - Entity: `consistent`
  - Module: `sage.logic.propcalc`
  - Type: `function`

- **sage.logic.propcalc.formula**
  - Entity: `formula`
  - Module: `sage.logic.propcalc`
  - Type: `function`

- **sage.logic.propcalc.get_formulas**
  - Entity: `get_formulas`
  - Module: `sage.logic.propcalc`
  - Type: `function`

- **sage.manifolds.catalog.Kerr**
  - Entity: `Kerr`
  - Module: `sage.manifolds.catalog`
  - Type: `function`

- **sage.manifolds.catalog.Minkowski**
  - Entity: `Minkowski`
  - Module: `sage.manifolds.catalog`
  - Type: `function`

- **sage.manifolds.catalog.RealProjectiveSpace**
  - Entity: `RealProjectiveSpace`
  - Module: `sage.manifolds.catalog`
  - Type: `function`

- **sage.manifolds.catalog.Torus**
  - Entity: `Torus`
  - Module: `sage.manifolds.catalog`
  - Type: `function`

- **sage.manifolds.differentiable.characteristic_cohomology_class.additive_sequence**
  - Entity: `additive_sequence`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `function`

- **sage.manifolds.differentiable.characteristic_cohomology_class.fast_wedge_power**
  - Entity: `fast_wedge_power`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `function`

- **sage.manifolds.differentiable.characteristic_cohomology_class.multiplicative_sequence**
  - Entity: `multiplicative_sequence`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `function`

- **sage.manifolds.manifold.Manifold**
  - Entity: `Manifold`
  - Module: `sage.manifolds.manifold`
  - Type: `function`

- **sage.manifolds.operators.curl**
  - Entity: `curl`
  - Module: `sage.manifolds.operators`
  - Type: `function`

- **sage.manifolds.operators.dalembertian**
  - Entity: `dalembertian`
  - Module: `sage.manifolds.operators`
  - Type: `function`

- **sage.manifolds.operators.div**
  - Entity: `div`
  - Module: `sage.manifolds.operators`
  - Type: `function`

- **sage.manifolds.operators.grad**
  - Entity: `grad`
  - Module: `sage.manifolds.operators`
  - Type: `function`

- **sage.manifolds.operators.laplacian**
  - Entity: `laplacian`
  - Module: `sage.manifolds.operators`
  - Type: `function`

#### MODULE (94 entries)

- **sage.groups.perm_gps.permgroup**
  - Entity: `permgroup`
  - Module: `sage.groups.perm_gps.permgroup`
  - Type: `module`

- **sage.groups.perm_gps.permgroup_element**
  - Entity: `permgroup_element`
  - Module: `sage.groups.perm_gps.permgroup_element`
  - Type: `module`

- **sage.groups.perm_gps.permgroup_morphism**
  - Entity: `permgroup_morphism`
  - Module: `sage.groups.perm_gps.permgroup_morphism`
  - Type: `module`

- **sage.groups.perm_gps.permgroup_named**
  - Entity: `permgroup_named`
  - Module: `sage.groups.perm_gps.permgroup_named`
  - Type: `module`

- **sage.groups.perm_gps.permutation_groups_catalog**
  - Entity: `permutation_groups_catalog`
  - Module: `sage.groups.perm_gps.permutation_groups_catalog`
  - Type: `module`

- **sage.groups.perm_gps.symgp_conjugacy_class**
  - Entity: `symgp_conjugacy_class`
  - Module: `sage.groups.perm_gps.symgp_conjugacy_class`
  - Type: `module`

- **sage.homology**
  - Entity: `homology`
  - Module: `sage.homology`
  - Type: `module`

- **sage.homology.algebraic_topological_model**
  - Entity: `algebraic_topological_model`
  - Module: `sage.homology.algebraic_topological_model`
  - Type: `module`

- **sage.homology.chain_complex**
  - Entity: `chain_complex`
  - Module: `sage.homology.chain_complex`
  - Type: `module`

- **sage.homology.chain_complex_homspace**
  - Entity: `chain_complex_homspace`
  - Module: `sage.homology.chain_complex_homspace`
  - Type: `module`

- **sage.homology.chain_complex_morphism**
  - Entity: `chain_complex_morphism`
  - Module: `sage.homology.chain_complex_morphism`
  - Type: `module`

- **sage.homology.chain_homotopy**
  - Entity: `chain_homotopy`
  - Module: `sage.homology.chain_homotopy`
  - Type: `module`

- **sage.homology.chains**
  - Entity: `chains`
  - Module: `sage.homology.chains`
  - Type: `module`

- **sage.homology.free_resolution**
  - Entity: `free_resolution`
  - Module: `sage.homology.free_resolution`
  - Type: `module`

- **sage.homology.graded_resolution**
  - Entity: `graded_resolution`
  - Module: `sage.homology.graded_resolution`
  - Type: `module`

- **sage.homology.hochschild_complex**
  - Entity: `hochschild_complex`
  - Module: `sage.homology.hochschild_complex`
  - Type: `module`

- **sage.homology.homology_group**
  - Entity: `homology_group`
  - Module: `sage.homology.homology_group`
  - Type: `module`

- **sage.homology.homology_morphism**
  - Entity: `homology_morphism`
  - Module: `sage.homology.homology_morphism`
  - Type: `module`

- **sage.homology.homology_vector_space_with_basis**
  - Entity: `homology_vector_space_with_basis`
  - Module: `sage.homology.homology_vector_space_with_basis`
  - Type: `module`

- **sage.homology.koszul_complex**
  - Entity: `koszul_complex`
  - Module: `sage.homology.koszul_complex`
  - Type: `module`

- **sage.homology.matrix_utils**
  - Entity: `matrix_utils`
  - Module: `sage.homology.matrix_utils`
  - Type: `module`

- **sage.libs.flint.arith_sage**
  - Entity: `arith_sage`
  - Module: `sage.libs.flint.arith_sage`
  - Type: `module`

- **sage.logic**
  - Entity: `logic`
  - Module: `sage.logic`
  - Type: `module`

- **sage.logic.booleval**
  - Entity: `booleval`
  - Module: `sage.logic.booleval`
  - Type: `module`

- **sage.logic.boolformula**
  - Entity: `boolformula`
  - Module: `sage.logic.boolformula`
  - Type: `module`

- **sage.logic.logic**
  - Entity: `logic`
  - Module: `sage.logic.logic`
  - Type: `module`

- **sage.logic.logicparser**
  - Entity: `logicparser`
  - Module: `sage.logic.logicparser`
  - Type: `module`

- **sage.logic.logictable**
  - Entity: `logictable`
  - Module: `sage.logic.logictable`
  - Type: `module`

- **sage.logic.propcalc**
  - Entity: `propcalc`
  - Module: `sage.logic.propcalc`
  - Type: `module`

- **sage.manifolds**
  - Entity: `manifolds`
  - Module: `sage.manifolds`
  - Type: `module`

- **sage.manifolds.calculus_method**
  - Entity: `calculus_method`
  - Module: `sage.manifolds.calculus_method`
  - Type: `module`

- **sage.manifolds.catalog**
  - Entity: `catalog`
  - Module: `sage.manifolds.catalog`
  - Type: `module`

- **sage.manifolds.chart**
  - Entity: `chart`
  - Module: `sage.manifolds.chart`
  - Type: `module`

- **sage.manifolds.chart_func**
  - Entity: `chart_func`
  - Module: `sage.manifolds.chart_func`
  - Type: `module`

- **sage.manifolds.continuous_map**
  - Entity: `continuous_map`
  - Module: `sage.manifolds.continuous_map`
  - Type: `module`

- **sage.manifolds.continuous_map_image**
  - Entity: `continuous_map_image`
  - Module: `sage.manifolds.continuous_map_image`
  - Type: `module`

- **sage.manifolds.differentiable.affine_connection**
  - Entity: `affine_connection`
  - Module: `sage.manifolds.differentiable.affine_connection`
  - Type: `module`

- **sage.manifolds.differentiable.automorphismfield**
  - Entity: `automorphismfield`
  - Module: `sage.manifolds.differentiable.automorphismfield`
  - Type: `module`

- **sage.manifolds.differentiable.automorphismfield_group**
  - Entity: `automorphismfield_group`
  - Module: `sage.manifolds.differentiable.automorphismfield_group`
  - Type: `module`

- **sage.manifolds.differentiable.bundle_connection**
  - Entity: `bundle_connection`
  - Module: `sage.manifolds.differentiable.bundle_connection`
  - Type: `module`

- **sage.manifolds.differentiable.characteristic_cohomology_class**
  - Entity: `characteristic_cohomology_class`
  - Module: `sage.manifolds.differentiable.characteristic_cohomology_class`
  - Type: `module`

- **sage.manifolds.differentiable.chart**
  - Entity: `chart`
  - Module: `sage.manifolds.differentiable.chart`
  - Type: `module`

- **sage.manifolds.differentiable.curve**
  - Entity: `curve`
  - Module: `sage.manifolds.differentiable.curve`
  - Type: `module`

- **sage.manifolds.differentiable.de_rham_cohomology**
  - Entity: `de_rham_cohomology`
  - Module: `sage.manifolds.differentiable.de_rham_cohomology`
  - Type: `module`

- **sage.manifolds.differentiable.degenerate**
  - Entity: `degenerate`
  - Module: `sage.manifolds.differentiable.degenerate`
  - Type: `module`

- **sage.manifolds.differentiable.degenerate_submanifold**
  - Entity: `degenerate_submanifold`
  - Module: `sage.manifolds.differentiable.degenerate_submanifold`
  - Type: `module`

- **sage.manifolds.differentiable.diff_form**
  - Entity: `diff_form`
  - Module: `sage.manifolds.differentiable.diff_form`
  - Type: `module`

- **sage.manifolds.differentiable.diff_form_module**
  - Entity: `diff_form_module`
  - Module: `sage.manifolds.differentiable.diff_form_module`
  - Type: `module`

- **sage.manifolds.differentiable.diff_map**
  - Entity: `diff_map`
  - Module: `sage.manifolds.differentiable.diff_map`
  - Type: `module`

- **sage.manifolds.differentiable.differentiable_submanifold**
  - Entity: `differentiable_submanifold`
  - Module: `sage.manifolds.differentiable.differentiable_submanifold`
  - Type: `module`

- **sage.manifolds.differentiable.examples.euclidean**
  - Entity: `euclidean`
  - Module: `sage.manifolds.differentiable.examples.euclidean`
  - Type: `module`

- **sage.manifolds.differentiable.examples.real_line**
  - Entity: `real_line`
  - Module: `sage.manifolds.differentiable.examples.real_line`
  - Type: `module`

- **sage.manifolds.differentiable.examples.sphere**
  - Entity: `sphere`
  - Module: `sage.manifolds.differentiable.examples.sphere`
  - Type: `module`

- **sage.manifolds.differentiable.examples.symplectic_space**
  - Entity: `symplectic_space`
  - Module: `sage.manifolds.differentiable.examples.symplectic_space`
  - Type: `module`

- **sage.manifolds.differentiable.integrated_curve**
  - Entity: `integrated_curve`
  - Module: `sage.manifolds.differentiable.integrated_curve`
  - Type: `module`

- **sage.manifolds.differentiable.levi_civita_connection**
  - Entity: `levi_civita_connection`
  - Module: `sage.manifolds.differentiable.levi_civita_connection`
  - Type: `module`

- **sage.manifolds.differentiable.manifold**
  - Entity: `manifold`
  - Module: `sage.manifolds.differentiable.manifold`
  - Type: `module`

- **sage.manifolds.differentiable.manifold_homset**
  - Entity: `manifold_homset`
  - Module: `sage.manifolds.differentiable.manifold_homset`
  - Type: `module`

- **sage.manifolds.differentiable.metric**
  - Entity: `metric`
  - Module: `sage.manifolds.differentiable.metric`
  - Type: `module`

- **sage.manifolds.differentiable.mixed_form**
  - Entity: `mixed_form`
  - Module: `sage.manifolds.differentiable.mixed_form`
  - Type: `module`

- **sage.manifolds.differentiable.mixed_form_algebra**
  - Entity: `mixed_form_algebra`
  - Module: `sage.manifolds.differentiable.mixed_form_algebra`
  - Type: `module`

- **sage.manifolds.differentiable.multivector_module**
  - Entity: `multivector_module`
  - Module: `sage.manifolds.differentiable.multivector_module`
  - Type: `module`

- **sage.manifolds.differentiable.multivectorfield**
  - Entity: `multivectorfield`
  - Module: `sage.manifolds.differentiable.multivectorfield`
  - Type: `module`

- **sage.manifolds.differentiable.poisson_tensor**
  - Entity: `poisson_tensor`
  - Module: `sage.manifolds.differentiable.poisson_tensor`
  - Type: `module`

- **sage.manifolds.differentiable.pseudo_riemannian**
  - Entity: `pseudo_riemannian`
  - Module: `sage.manifolds.differentiable.pseudo_riemannian`
  - Type: `module`

- **sage.manifolds.differentiable.pseudo_riemannian_submanifold**
  - Entity: `pseudo_riemannian_submanifold`
  - Module: `sage.manifolds.differentiable.pseudo_riemannian_submanifold`
  - Type: `module`

- **sage.manifolds.differentiable.scalarfield**
  - Entity: `scalarfield`
  - Module: `sage.manifolds.differentiable.scalarfield`
  - Type: `module`

- **sage.manifolds.differentiable.scalarfield_algebra**
  - Entity: `scalarfield_algebra`
  - Module: `sage.manifolds.differentiable.scalarfield_algebra`
  - Type: `module`

- **sage.manifolds.differentiable.symplectic_form**
  - Entity: `symplectic_form`
  - Module: `sage.manifolds.differentiable.symplectic_form`
  - Type: `module`

- **sage.manifolds.differentiable.tangent_space**
  - Entity: `tangent_space`
  - Module: `sage.manifolds.differentiable.tangent_space`
  - Type: `module`

- **sage.manifolds.differentiable.tangent_vector**
  - Entity: `tangent_vector`
  - Module: `sage.manifolds.differentiable.tangent_vector`
  - Type: `module`

- **sage.manifolds.differentiable.tensorfield**
  - Entity: `tensorfield`
  - Module: `sage.manifolds.differentiable.tensorfield`
  - Type: `module`

- **sage.manifolds.differentiable.tensorfield_module**
  - Entity: `tensorfield_module`
  - Module: `sage.manifolds.differentiable.tensorfield_module`
  - Type: `module`

- **sage.manifolds.differentiable.tensorfield_paral**
  - Entity: `tensorfield_paral`
  - Module: `sage.manifolds.differentiable.tensorfield_paral`
  - Type: `module`

- **sage.manifolds.differentiable.vector_bundle**
  - Entity: `vector_bundle`
  - Module: `sage.manifolds.differentiable.vector_bundle`
  - Type: `module`

- **sage.manifolds.differentiable.vectorfield**
  - Entity: `vectorfield`
  - Module: `sage.manifolds.differentiable.vectorfield`
  - Type: `module`

- **sage.manifolds.differentiable.vectorfield_module**
  - Entity: `vectorfield_module`
  - Module: `sage.manifolds.differentiable.vectorfield_module`
  - Type: `module`

- **sage.manifolds.differentiable.vectorframe**
  - Entity: `vectorframe`
  - Module: `sage.manifolds.differentiable.vectorframe`
  - Type: `module`

- **sage.manifolds.family**
  - Entity: `family`
  - Module: `sage.manifolds.family`
  - Type: `module`

- **sage.manifolds.local_frame**
  - Entity: `local_frame`
  - Module: `sage.manifolds.local_frame`
  - Type: `module`

- **sage.manifolds.manifold**
  - Entity: `manifold`
  - Module: `sage.manifolds.manifold`
  - Type: `module`

- **sage.manifolds.manifold_homset**
  - Entity: `manifold_homset`
  - Module: `sage.manifolds.manifold_homset`
  - Type: `module`

- **sage.manifolds.operators**
  - Entity: `operators`
  - Module: `sage.manifolds.operators`
  - Type: `module`

- **sage.manifolds.point**
  - Entity: `point`
  - Module: `sage.manifolds.point`
  - Type: `module`

- **sage.manifolds.scalarfield**
  - Entity: `scalarfield`
  - Module: `sage.manifolds.scalarfield`
  - Type: `module`

- **sage.manifolds.scalarfield_algebra**
  - Entity: `scalarfield_algebra`
  - Module: `sage.manifolds.scalarfield_algebra`
  - Type: `module`

- **sage.manifolds.section**
  - Entity: `section`
  - Module: `sage.manifolds.section`
  - Type: `module`

- **sage.manifolds.section_module**
  - Entity: `section_module`
  - Module: `sage.manifolds.section_module`
  - Type: `module`

- **sage.manifolds.structure**
  - Entity: `structure`
  - Module: `sage.manifolds.structure`
  - Type: `module`

- **sage.manifolds.subset**
  - Entity: `subset`
  - Module: `sage.manifolds.subset`
  - Type: `module`

- **sage.manifolds.subsets.closure**
  - Entity: `closure`
  - Module: `sage.manifolds.subsets.closure`
  - Type: `module`

- **sage.manifolds.subsets.pullback**
  - Entity: `pullback`
  - Module: `sage.manifolds.subsets.pullback`
  - Type: `module`

- **sage.manifolds.topological_submanifold**
  - Entity: `topological_submanifold`
  - Module: `sage.manifolds.topological_submanifold`
  - Type: `module`

- **sage.manifolds.trivialization**
  - Entity: `trivialization`
  - Module: `sage.manifolds.trivialization`
  - Type: `module`


### Part 09 (116 entries)

#### ATTRIBUTE (1 entries)

- **sage.matrix.matrix_modn_sparse.p**
  - Entity: `p`
  - Module: `sage.matrix.matrix_modn_sparse`
  - Type: `attribute`

#### CLASS (31 entries)

- **sage.matrix.matrix0.Matrix**
  - Entity: `Matrix`
  - Module: `sage.matrix.matrix0`
  - Type: `class`

- **sage.matrix.matrix1.Matrix**
  - Entity: `Matrix`
  - Module: `sage.matrix.matrix1`
  - Type: `class`

- **sage.matrix.matrix2.Matrix**
  - Entity: `Matrix`
  - Module: `sage.matrix.matrix2`
  - Type: `class`

- **sage.matrix.matrix_complex_ball_dense.Matrix_complex_ball_dense**
  - Entity: `Matrix_complex_ball_dense`
  - Module: `sage.matrix.matrix_complex_ball_dense`
  - Type: `class`

- **sage.matrix.matrix_complex_double_dense.Matrix_complex_double_dense**
  - Entity: `Matrix_complex_double_dense`
  - Module: `sage.matrix.matrix_complex_double_dense`
  - Type: `class`

- **sage.matrix.matrix_cyclo_dense.Matrix_cyclo_dense**
  - Entity: `Matrix_cyclo_dense`
  - Module: `sage.matrix.matrix_cyclo_dense`
  - Type: `class`

- **sage.matrix.matrix_dense.Matrix_dense**
  - Entity: `Matrix_dense`
  - Module: `sage.matrix.matrix_dense`
  - Type: `class`

- **sage.matrix.matrix_double_dense.Matrix_double_dense**
  - Entity: `Matrix_double_dense`
  - Module: `sage.matrix.matrix_double_dense`
  - Type: `class`

- **sage.matrix.matrix_generic_dense.Matrix_generic_dense**
  - Entity: `Matrix_generic_dense`
  - Module: `sage.matrix.matrix_generic_dense`
  - Type: `class`

- **sage.matrix.matrix_generic_sparse.Matrix_generic_sparse**
  - Entity: `Matrix_generic_sparse`
  - Module: `sage.matrix.matrix_generic_sparse`
  - Type: `class`

- **sage.matrix.matrix_gf2e_dense.M4RIE_finite_field**
  - Entity: `M4RIE_finite_field`
  - Module: `sage.matrix.matrix_gf2e_dense`
  - Type: `class`

- **sage.matrix.matrix_gf2e_dense.Matrix_gf2e_dense**
  - Entity: `Matrix_gf2e_dense`
  - Module: `sage.matrix.matrix_gf2e_dense`
  - Type: `class`

- **sage.matrix.matrix_integer_dense.Matrix_integer_dense**
  - Entity: `Matrix_integer_dense`
  - Module: `sage.matrix.matrix_integer_dense`
  - Type: `class`

- **sage.matrix.matrix_integer_sparse.Matrix_integer_sparse**
  - Entity: `Matrix_integer_sparse`
  - Module: `sage.matrix.matrix_integer_sparse`
  - Type: `class`

- **sage.matrix.matrix_mod2_dense.Matrix_mod2_dense**
  - Entity: `Matrix_mod2_dense`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `class`

- **sage.matrix.matrix_modn_dense_double.Matrix_modn_dense_double**
  - Entity: `Matrix_modn_dense_double`
  - Module: `sage.matrix.matrix_modn_dense_double`
  - Type: `class`

- **sage.matrix.matrix_modn_dense_double.Matrix_modn_dense_template**
  - Entity: `Matrix_modn_dense_template`
  - Module: `sage.matrix.matrix_modn_dense_double`
  - Type: `class`

- **sage.matrix.matrix_modn_dense_float.Matrix_modn_dense_float**
  - Entity: `Matrix_modn_dense_float`
  - Module: `sage.matrix.matrix_modn_dense_float`
  - Type: `class`

- **sage.matrix.matrix_modn_dense_float.Matrix_modn_dense_template**
  - Entity: `Matrix_modn_dense_template`
  - Module: `sage.matrix.matrix_modn_dense_float`
  - Type: `class`

- **sage.matrix.matrix_modn_sparse.Matrix_modn_sparse**
  - Entity: `Matrix_modn_sparse`
  - Module: `sage.matrix.matrix_modn_sparse`
  - Type: `class`

- **sage.matrix.matrix_mpolynomial_dense.Matrix_mpolynomial_dense**
  - Entity: `Matrix_mpolynomial_dense`
  - Module: `sage.matrix.matrix_mpolynomial_dense`
  - Type: `class`

- **sage.matrix.matrix_polynomial_dense.Matrix_polynomial_dense**
  - Entity: `Matrix_polynomial_dense`
  - Module: `sage.matrix.matrix_polynomial_dense`
  - Type: `class`

- **sage.matrix.matrix_rational_dense.MatrixWindow**
  - Entity: `MatrixWindow`
  - Module: `sage.matrix.matrix_rational_dense`
  - Type: `class`

- **sage.matrix.matrix_rational_dense.Matrix_rational_dense**
  - Entity: `Matrix_rational_dense`
  - Module: `sage.matrix.matrix_rational_dense`
  - Type: `class`

- **sage.matrix.matrix_rational_sparse.Matrix_rational_sparse**
  - Entity: `Matrix_rational_sparse`
  - Module: `sage.matrix.matrix_rational_sparse`
  - Type: `class`

- **sage.matrix.matrix_real_double_dense.Matrix_real_double_dense**
  - Entity: `Matrix_real_double_dense`
  - Module: `sage.matrix.matrix_real_double_dense`
  - Type: `class`

- **sage.matrix.matrix_space.MatrixSpace**
  - Entity: `MatrixSpace`
  - Module: `sage.matrix.matrix_space`
  - Type: `class`

- **sage.matrix.matrix_sparse.Matrix_sparse**
  - Entity: `Matrix_sparse`
  - Module: `sage.matrix.matrix_sparse`
  - Type: `class`

- **sage.matrix.matrix_symbolic_dense.Matrix_symbolic_dense**
  - Entity: `Matrix_symbolic_dense`
  - Module: `sage.matrix.matrix_symbolic_dense`
  - Type: `class`

- **sage.matrix.matrix_symbolic_sparse.Matrix_symbolic_sparse**
  - Entity: `Matrix_symbolic_sparse`
  - Module: `sage.matrix.matrix_symbolic_sparse`
  - Type: `class`

- **sage.matrix.matrix_window.MatrixWindow**
  - Entity: `MatrixWindow`
  - Module: `sage.matrix.matrix_window`
  - Type: `class`

#### FUNCTION (52 entries)

- **sage.matrix.constructor.Matrix**
  - Entity: `Matrix`
  - Module: `sage.matrix.constructor`
  - Type: `function`

- **sage.matrix.constructor.matrix**
  - Entity: `matrix`
  - Module: `sage.matrix.constructor`
  - Type: `function`

- **sage.matrix.constructor.options**
  - Entity: `options`
  - Module: `sage.matrix.constructor`
  - Type: `function`

- **sage.matrix.matrix0.set_max_cols**
  - Entity: `set_max_cols`
  - Module: `sage.matrix.matrix0`
  - Type: `function`

- **sage.matrix.matrix0.set_max_rows**
  - Entity: `set_max_rows`
  - Module: `sage.matrix.matrix0`
  - Type: `function`

- **sage.matrix.matrix0.unpickle**
  - Entity: `unpickle`
  - Module: `sage.matrix.matrix0`
  - Type: `function`

- **sage.matrix.matrix2.decomp_seq**
  - Entity: `decomp_seq`
  - Module: `sage.matrix.matrix2`
  - Type: `function`

- **sage.matrix.matrix2.ideal_or_fractional**
  - Entity: `ideal_or_fractional`
  - Module: `sage.matrix.matrix2`
  - Type: `function`

- **sage.matrix.matrix_generic_sparse.Matrix_sparse_from_rows**
  - Entity: `Matrix_sparse_from_rows`
  - Module: `sage.matrix.matrix_generic_sparse`
  - Type: `function`

- **sage.matrix.matrix_gf2e_dense.unpickle_matrix_gf2e_dense_v0**
  - Entity: `unpickle_matrix_gf2e_dense_v0`
  - Module: `sage.matrix.matrix_gf2e_dense`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.add_column**
  - Entity: `add_column`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.add_column_fallback**
  - Entity: `add_column_fallback`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.add_row**
  - Entity: `add_row`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.benchmark_hnf**
  - Entity: `benchmark_hnf`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.benchmark_magma_hnf**
  - Entity: `benchmark_magma_hnf`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.det_from_modp_and_divisor**
  - Entity: `det_from_modp_and_divisor`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.det_given_divisor**
  - Entity: `det_given_divisor`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.det_padic**
  - Entity: `det_padic`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.double_det**
  - Entity: `double_det`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.extract_ones_data**
  - Entity: `extract_ones_data`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.hnf**
  - Entity: `hnf`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.hnf_square**
  - Entity: `hnf_square`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.hnf_with_transformation**
  - Entity: `hnf_with_transformation`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.hnf_with_transformation_tests**
  - Entity: `hnf_with_transformation_tests`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.interleave_matrices**
  - Entity: `interleave_matrices`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.is_in_hnf_form**
  - Entity: `is_in_hnf_form`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.max_det_prime**
  - Entity: `max_det_prime`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.ones**
  - Entity: `ones`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.pad_zeros**
  - Entity: `pad_zeros`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.pivots_of_hnf_matrix**
  - Entity: `pivots_of_hnf_matrix`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.probable_hnf**
  - Entity: `probable_hnf`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.probable_pivot_columns**
  - Entity: `probable_pivot_columns`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.probable_pivot_rows**
  - Entity: `probable_pivot_rows`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.sanity_checks**
  - Entity: `sanity_checks`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_hnf.solve_system_with_difficult_last_row**
  - Entity: `solve_system_with_difficult_last_row`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_saturation.index_in_saturation**
  - Entity: `index_in_saturation`
  - Module: `sage.matrix.matrix_integer_dense_saturation`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_saturation.p_saturation**
  - Entity: `p_saturation`
  - Module: `sage.matrix.matrix_integer_dense_saturation`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_saturation.random_sublist_of_size**
  - Entity: `random_sublist_of_size`
  - Module: `sage.matrix.matrix_integer_dense_saturation`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_saturation.saturation**
  - Entity: `saturation`
  - Module: `sage.matrix.matrix_integer_dense_saturation`
  - Type: `function`

- **sage.matrix.matrix_integer_dense_saturation.solve_system_with_difficult_last_row**
  - Entity: `solve_system_with_difficult_last_row`
  - Module: `sage.matrix.matrix_integer_dense_saturation`
  - Type: `function`

- **sage.matrix.matrix_misc.permanental_minor_polynomial**
  - Entity: `permanental_minor_polynomial`
  - Module: `sage.matrix.matrix_misc`
  - Type: `function`

- **sage.matrix.matrix_misc.prm_mul**
  - Entity: `prm_mul`
  - Module: `sage.matrix.matrix_misc`
  - Type: `function`

- **sage.matrix.matrix_misc.row_iterator**
  - Entity: `row_iterator`
  - Module: `sage.matrix.matrix_misc`
  - Type: `function`

- **sage.matrix.matrix_mod2_dense.from_png**
  - Entity: `from_png`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `function`

- **sage.matrix.matrix_mod2_dense.parity**
  - Entity: `parity`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `function`

- **sage.matrix.matrix_mod2_dense.ple**
  - Entity: `ple`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `function`

- **sage.matrix.matrix_mod2_dense.pluq**
  - Entity: `pluq`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `function`

- **sage.matrix.matrix_mod2_dense.to_png**
  - Entity: `to_png`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `function`

- **sage.matrix.matrix_mod2_dense.unpickle_matrix_mod2_dense_v2**
  - Entity: `unpickle_matrix_mod2_dense_v2`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `function`

- **sage.matrix.matrix_space.dict_to_list**
  - Entity: `dict_to_list`
  - Module: `sage.matrix.matrix_space`
  - Type: `function`

- **sage.matrix.matrix_space.get_matrix_class**
  - Entity: `get_matrix_class`
  - Module: `sage.matrix.matrix_space`
  - Type: `function`

- **sage.matrix.matrix_space.is_MatrixSpace**
  - Entity: `is_MatrixSpace`
  - Module: `sage.matrix.matrix_space`
  - Type: `function`

#### MODULE (32 entries)

- **sage.matrix**
  - Entity: `matrix`
  - Module: `sage.matrix`
  - Type: `module`

- **sage.matrix.constructor**
  - Entity: `constructor`
  - Module: `sage.matrix.constructor`
  - Type: `module`

- **sage.matrix.matrix0**
  - Entity: `matrix0`
  - Module: `sage.matrix.matrix0`
  - Type: `module`

- **sage.matrix.matrix1**
  - Entity: `matrix1`
  - Module: `sage.matrix.matrix1`
  - Type: `module`

- **sage.matrix.matrix2**
  - Entity: `matrix2`
  - Module: `sage.matrix.matrix2`
  - Type: `module`

- **sage.matrix.matrix_complex_ball_dense**
  - Entity: `matrix_complex_ball_dense`
  - Module: `sage.matrix.matrix_complex_ball_dense`
  - Type: `module`

- **sage.matrix.matrix_complex_double_dense**
  - Entity: `matrix_complex_double_dense`
  - Module: `sage.matrix.matrix_complex_double_dense`
  - Type: `module`

- **sage.matrix.matrix_cyclo_dense**
  - Entity: `matrix_cyclo_dense`
  - Module: `sage.matrix.matrix_cyclo_dense`
  - Type: `module`

- **sage.matrix.matrix_dense**
  - Entity: `matrix_dense`
  - Module: `sage.matrix.matrix_dense`
  - Type: `module`

- **sage.matrix.matrix_double_dense**
  - Entity: `matrix_double_dense`
  - Module: `sage.matrix.matrix_double_dense`
  - Type: `module`

- **sage.matrix.matrix_generic_dense**
  - Entity: `matrix_generic_dense`
  - Module: `sage.matrix.matrix_generic_dense`
  - Type: `module`

- **sage.matrix.matrix_generic_sparse**
  - Entity: `matrix_generic_sparse`
  - Module: `sage.matrix.matrix_generic_sparse`
  - Type: `module`

- **sage.matrix.matrix_gf2e_dense**
  - Entity: `matrix_gf2e_dense`
  - Module: `sage.matrix.matrix_gf2e_dense`
  - Type: `module`

- **sage.matrix.matrix_integer_dense**
  - Entity: `matrix_integer_dense`
  - Module: `sage.matrix.matrix_integer_dense`
  - Type: `module`

- **sage.matrix.matrix_integer_dense_hnf**
  - Entity: `matrix_integer_dense_hnf`
  - Module: `sage.matrix.matrix_integer_dense_hnf`
  - Type: `module`

- **sage.matrix.matrix_integer_dense_saturation**
  - Entity: `matrix_integer_dense_saturation`
  - Module: `sage.matrix.matrix_integer_dense_saturation`
  - Type: `module`

- **sage.matrix.matrix_integer_sparse**
  - Entity: `matrix_integer_sparse`
  - Module: `sage.matrix.matrix_integer_sparse`
  - Type: `module`

- **sage.matrix.matrix_misc**
  - Entity: `matrix_misc`
  - Module: `sage.matrix.matrix_misc`
  - Type: `module`

- **sage.matrix.matrix_mod2_dense**
  - Entity: `matrix_mod2_dense`
  - Module: `sage.matrix.matrix_mod2_dense`
  - Type: `module`

- **sage.matrix.matrix_modn_dense_double**
  - Entity: `matrix_modn_dense_double`
  - Module: `sage.matrix.matrix_modn_dense_double`
  - Type: `module`

- **sage.matrix.matrix_modn_dense_float**
  - Entity: `matrix_modn_dense_float`
  - Module: `sage.matrix.matrix_modn_dense_float`
  - Type: `module`

- **sage.matrix.matrix_modn_sparse**
  - Entity: `matrix_modn_sparse`
  - Module: `sage.matrix.matrix_modn_sparse`
  - Type: `module`

- **sage.matrix.matrix_mpolynomial_dense**
  - Entity: `matrix_mpolynomial_dense`
  - Module: `sage.matrix.matrix_mpolynomial_dense`
  - Type: `module`

- **sage.matrix.matrix_polynomial_dense**
  - Entity: `matrix_polynomial_dense`
  - Module: `sage.matrix.matrix_polynomial_dense`
  - Type: `module`

- **sage.matrix.matrix_rational_dense**
  - Entity: `matrix_rational_dense`
  - Module: `sage.matrix.matrix_rational_dense`
  - Type: `module`

- **sage.matrix.matrix_rational_sparse**
  - Entity: `matrix_rational_sparse`
  - Module: `sage.matrix.matrix_rational_sparse`
  - Type: `module`

- **sage.matrix.matrix_real_double_dense**
  - Entity: `matrix_real_double_dense`
  - Module: `sage.matrix.matrix_real_double_dense`
  - Type: `module`

- **sage.matrix.matrix_space**
  - Entity: `matrix_space`
  - Module: `sage.matrix.matrix_space`
  - Type: `module`

- **sage.matrix.matrix_sparse**
  - Entity: `matrix_sparse`
  - Module: `sage.matrix.matrix_sparse`
  - Type: `module`

- **sage.matrix.matrix_symbolic_dense**
  - Entity: `matrix_symbolic_dense`
  - Module: `sage.matrix.matrix_symbolic_dense`
  - Type: `module`

- **sage.matrix.matrix_symbolic_sparse**
  - Entity: `matrix_symbolic_sparse`
  - Module: `sage.matrix.matrix_symbolic_sparse`
  - Type: `module`

- **sage.matrix.matrix_window**
  - Entity: `matrix_window`
  - Module: `sage.matrix.matrix_window`
  - Type: `module`


### Part 10 (999 entries)

#### ATTRIBUTE (11 entries)

- **sage.misc.sage_unittest.longMessage**
  - Entity: `longMessage`
  - Module: `sage.misc.sage_unittest`
  - Type: `attribute`

- **sage.misc.sagedoc_conf.default_priority**
  - Entity: `default_priority`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `attribute`

- **sage.modular.modform_hecketriangle.functors.rank**
  - Entity: `rank`
  - Module: `sage.modular.modform_hecketriangle.functors`
  - Type: `attribute`

- **sage.modular.modsym.heilbronn.n**
  - Entity: `n`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `attribute`

- **sage.modular.modsym.heilbronn.p**
  - Entity: `p`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `attribute`

- **sage.modular.modsym.manin_symbol.i**
  - Entity: `i`
  - Module: `sage.modular.modsym.manin_symbol`
  - Type: `attribute`

- **sage.modular.modsym.manin_symbol.u**
  - Entity: `u`
  - Module: `sage.modular.modsym.manin_symbol`
  - Type: `attribute`

- **sage.modular.modsym.manin_symbol.v**
  - Entity: `v`
  - Module: `sage.modular.modsym.manin_symbol`
  - Type: `attribute`

- **sage.modular.modsym.p1list_nf.c**
  - Entity: `c`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `attribute`

- **sage.modular.modsym.p1list_nf.d**
  - Entity: `d`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `attribute`

- **sage.modules.free_module_integer.reduced_basis**
  - Entity: `reduced_basis`
  - Module: `sage.modules.free_module_integer`
  - Type: `attribute`

#### CLASS (403 entries)

- **sage.misc.sagedoc_conf.SagemathTransform**
  - Entity: `SagemathTransform`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `class`

- **sage.misc.sageinspect.BlockFinder**
  - Entity: `BlockFinder`
  - Module: `sage.misc.sageinspect`
  - Type: `class`

- **sage.misc.sageinspect.SageArgSpecVisitor**
  - Entity: `SageArgSpecVisitor`
  - Module: `sage.misc.sageinspect`
  - Type: `class`

- **sage.misc.sh.Sh**
  - Entity: `Sh`
  - Module: `sage.misc.sh`
  - Type: `class`

- **sage.misc.superseded.DeprecatedFunctionAlias**
  - Entity: `DeprecatedFunctionAlias`
  - Module: `sage.misc.superseded`
  - Type: `class`

- **sage.misc.superseded.experimental**
  - Entity: `experimental`
  - Module: `sage.misc.superseded`
  - Type: `class`

- **sage.misc.table.table**
  - Entity: `table`
  - Module: `sage.misc.table`
  - Type: `class`

- **sage.misc.temporary_file.atomic_dir**
  - Entity: `atomic_dir`
  - Module: `sage.misc.temporary_file`
  - Type: `class`

- **sage.misc.temporary_file.atomic_write**
  - Entity: `atomic_write`
  - Module: `sage.misc.temporary_file`
  - Type: `class`

- **sage.misc.unknown.UnknownClass**
  - Entity: `UnknownClass`
  - Module: `sage.misc.unknown`
  - Type: `class`

- **sage.misc.viewer.Viewer**
  - Entity: `Viewer`
  - Module: `sage.misc.viewer`
  - Type: `class`

- **sage.misc.weak_dict.CachedWeakValueDictionary**
  - Entity: `CachedWeakValueDictionary`
  - Module: `sage.misc.weak_dict`
  - Type: `class`

- **sage.misc.weak_dict.WeakValueDictEraser**
  - Entity: `WeakValueDictEraser`
  - Module: `sage.misc.weak_dict`
  - Type: `class`

- **sage.misc.weak_dict.WeakValueDictionary**
  - Entity: `WeakValueDictionary`
  - Module: `sage.misc.weak_dict`
  - Type: `class`

- **sage.modular.abvar.abvar.ModularAbelianVariety**
  - Entity: `ModularAbelianVariety`
  - Module: `sage.modular.abvar.abvar`
  - Type: `class`

- **sage.modular.abvar.abvar.ModularAbelianVariety_abstract**
  - Entity: `ModularAbelianVariety_abstract`
  - Module: `sage.modular.abvar.abvar`
  - Type: `class`

- **sage.modular.abvar.abvar.ModularAbelianVariety_modsym**
  - Entity: `ModularAbelianVariety_modsym`
  - Module: `sage.modular.abvar.abvar`
  - Type: `class`

- **sage.modular.abvar.abvar.ModularAbelianVariety_modsym_abstract**
  - Entity: `ModularAbelianVariety_modsym_abstract`
  - Module: `sage.modular.abvar.abvar`
  - Type: `class`

- **sage.modular.abvar.abvar_ambient_jacobian.ModAbVar_ambient_jacobian_class**
  - Entity: `ModAbVar_ambient_jacobian_class`
  - Module: `sage.modular.abvar.abvar_ambient_jacobian`
  - Type: `class`

- **sage.modular.abvar.abvar_newform.ModularAbelianVariety_newform**
  - Entity: `ModularAbelianVariety_newform`
  - Module: `sage.modular.abvar.abvar_newform`
  - Type: `class`

- **sage.modular.abvar.cuspidal_subgroup.CuspidalSubgroup**
  - Entity: `CuspidalSubgroup`
  - Module: `sage.modular.abvar.cuspidal_subgroup`
  - Type: `class`

- **sage.modular.abvar.cuspidal_subgroup.CuspidalSubgroup_generic**
  - Entity: `CuspidalSubgroup_generic`
  - Module: `sage.modular.abvar.cuspidal_subgroup`
  - Type: `class`

- **sage.modular.abvar.cuspidal_subgroup.RationalCuspSubgroup**
  - Entity: `RationalCuspSubgroup`
  - Module: `sage.modular.abvar.cuspidal_subgroup`
  - Type: `class`

- **sage.modular.abvar.cuspidal_subgroup.RationalCuspidalSubgroup**
  - Entity: `RationalCuspidalSubgroup`
  - Module: `sage.modular.abvar.cuspidal_subgroup`
  - Type: `class`

- **sage.modular.abvar.finite_subgroup.FiniteSubgroup**
  - Entity: `FiniteSubgroup`
  - Module: `sage.modular.abvar.finite_subgroup`
  - Type: `class`

- **sage.modular.abvar.finite_subgroup.FiniteSubgroup_lattice**
  - Entity: `FiniteSubgroup_lattice`
  - Module: `sage.modular.abvar.finite_subgroup`
  - Type: `class`

- **sage.modular.abvar.homology.Homology**
  - Entity: `Homology`
  - Module: `sage.modular.abvar.homology`
  - Type: `class`

- **sage.modular.abvar.homology.Homology_abvar**
  - Entity: `Homology_abvar`
  - Module: `sage.modular.abvar.homology`
  - Type: `class`

- **sage.modular.abvar.homology.Homology_over_base**
  - Entity: `Homology_over_base`
  - Module: `sage.modular.abvar.homology`
  - Type: `class`

- **sage.modular.abvar.homology.Homology_submodule**
  - Entity: `Homology_submodule`
  - Module: `sage.modular.abvar.homology`
  - Type: `class`

- **sage.modular.abvar.homology.IntegralHomology**
  - Entity: `IntegralHomology`
  - Module: `sage.modular.abvar.homology`
  - Type: `class`

- **sage.modular.abvar.homology.RationalHomology**
  - Entity: `RationalHomology`
  - Module: `sage.modular.abvar.homology`
  - Type: `class`

- **sage.modular.abvar.homspace.EndomorphismSubring**
  - Entity: `EndomorphismSubring`
  - Module: `sage.modular.abvar.homspace`
  - Type: `class`

- **sage.modular.abvar.homspace.Homspace**
  - Entity: `Homspace`
  - Module: `sage.modular.abvar.homspace`
  - Type: `class`

- **sage.modular.abvar.lseries.Lseries**
  - Entity: `Lseries`
  - Module: `sage.modular.abvar.lseries`
  - Type: `class`

- **sage.modular.abvar.lseries.Lseries_complex**
  - Entity: `Lseries_complex`
  - Module: `sage.modular.abvar.lseries`
  - Type: `class`

- **sage.modular.abvar.lseries.Lseries_padic**
  - Entity: `Lseries_padic`
  - Module: `sage.modular.abvar.lseries`
  - Type: `class`

- **sage.modular.abvar.morphism.DegeneracyMap**
  - Entity: `DegeneracyMap`
  - Module: `sage.modular.abvar.morphism`
  - Type: `class`

- **sage.modular.abvar.morphism.HeckeOperator**
  - Entity: `HeckeOperator`
  - Module: `sage.modular.abvar.morphism`
  - Type: `class`

- **sage.modular.abvar.morphism.Morphism**
  - Entity: `Morphism`
  - Module: `sage.modular.abvar.morphism`
  - Type: `class`

- **sage.modular.abvar.morphism.Morphism_abstract**
  - Entity: `Morphism_abstract`
  - Module: `sage.modular.abvar.morphism`
  - Type: `class`

- **sage.modular.abvar.torsion_subgroup.QQbarTorsionSubgroup**
  - Entity: `QQbarTorsionSubgroup`
  - Module: `sage.modular.abvar.torsion_subgroup`
  - Type: `class`

- **sage.modular.abvar.torsion_subgroup.RationalTorsionSubgroup**
  - Entity: `RationalTorsionSubgroup`
  - Module: `sage.modular.abvar.torsion_subgroup`
  - Type: `class`

- **sage.modular.arithgroup.arithgroup_element.ArithmeticSubgroupElement**
  - Entity: `ArithmeticSubgroupElement`
  - Module: `sage.modular.arithgroup.arithgroup_element`
  - Type: `class`

- **sage.modular.arithgroup.arithgroup_generic.ArithmeticSubgroup**
  - Entity: `ArithmeticSubgroup`
  - Module: `sage.modular.arithgroup.arithgroup_generic`
  - Type: `class`

- **sage.modular.arithgroup.arithgroup_perm.ArithmeticSubgroup_Permutation_class**
  - Entity: `ArithmeticSubgroup_Permutation_class`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `class`

- **sage.modular.arithgroup.arithgroup_perm.EvenArithmeticSubgroup_Permutation**
  - Entity: `EvenArithmeticSubgroup_Permutation`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `class`

- **sage.modular.arithgroup.arithgroup_perm.OddArithmeticSubgroup_Permutation**
  - Entity: `OddArithmeticSubgroup_Permutation`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `class`

- **sage.modular.arithgroup.congroup_gamma.Gamma_class**
  - Entity: `Gamma_class`
  - Module: `sage.modular.arithgroup.congroup_gamma`
  - Type: `class`

- **sage.modular.arithgroup.congroup_gamma0.Gamma0_class**
  - Entity: `Gamma0_class`
  - Module: `sage.modular.arithgroup.congroup_gamma0`
  - Type: `class`

- **sage.modular.arithgroup.congroup_gamma1.Gamma1_class**
  - Entity: `Gamma1_class`
  - Module: `sage.modular.arithgroup.congroup_gamma1`
  - Type: `class`

- **sage.modular.arithgroup.congroup_gammaH.GammaH_class**
  - Entity: `GammaH_class`
  - Module: `sage.modular.arithgroup.congroup_gammaH`
  - Type: `class`

- **sage.modular.arithgroup.congroup_generic.CongruenceSubgroup**
  - Entity: `CongruenceSubgroup`
  - Module: `sage.modular.arithgroup.congroup_generic`
  - Type: `class`

- **sage.modular.arithgroup.congroup_generic.CongruenceSubgroupBase**
  - Entity: `CongruenceSubgroupBase`
  - Module: `sage.modular.arithgroup.congroup_generic`
  - Type: `class`

- **sage.modular.arithgroup.congroup_generic.CongruenceSubgroupFromGroup**
  - Entity: `CongruenceSubgroupFromGroup`
  - Module: `sage.modular.arithgroup.congroup_generic`
  - Type: `class`

- **sage.modular.arithgroup.congroup_sl2z.SL2Z_class**
  - Entity: `SL2Z_class`
  - Module: `sage.modular.arithgroup.congroup_sl2z`
  - Type: `class`

- **sage.modular.arithgroup.farey_symbol.Farey**
  - Entity: `Farey`
  - Module: `sage.modular.arithgroup.farey_symbol`
  - Type: `class`

- **sage.modular.btquotients.btquotient.BruhatTitsQuotient**
  - Entity: `BruhatTitsQuotient`
  - Module: `sage.modular.btquotients.btquotient`
  - Type: `class`

- **sage.modular.btquotients.btquotient.BruhatTitsTree**
  - Entity: `BruhatTitsTree`
  - Module: `sage.modular.btquotients.btquotient`
  - Type: `class`

- **sage.modular.btquotients.btquotient.DoubleCosetReduction**
  - Entity: `DoubleCosetReduction`
  - Module: `sage.modular.btquotients.btquotient`
  - Type: `class`

- **sage.modular.btquotients.btquotient.Edge**
  - Entity: `Edge`
  - Module: `sage.modular.btquotients.btquotient`
  - Type: `class`

- **sage.modular.btquotients.btquotient.Vertex**
  - Entity: `Vertex`
  - Module: `sage.modular.btquotients.btquotient`
  - Type: `class`

- **sage.modular.btquotients.pautomorphicform.BruhatTitsHarmonicCocycleElement**
  - Entity: `BruhatTitsHarmonicCocycleElement`
  - Module: `sage.modular.btquotients.pautomorphicform`
  - Type: `class`

- **sage.modular.btquotients.pautomorphicform.BruhatTitsHarmonicCocycles**
  - Entity: `BruhatTitsHarmonicCocycles`
  - Module: `sage.modular.btquotients.pautomorphicform`
  - Type: `class`

- **sage.modular.btquotients.pautomorphicform.pAdicAutomorphicFormElement**
  - Entity: `pAdicAutomorphicFormElement`
  - Module: `sage.modular.btquotients.pautomorphicform`
  - Type: `class`

- **sage.modular.btquotients.pautomorphicform.pAdicAutomorphicForms**
  - Entity: `pAdicAutomorphicForms`
  - Module: `sage.modular.btquotients.pautomorphicform`
  - Type: `class`

- **sage.modular.cusps.Cusp**
  - Entity: `Cusp`
  - Module: `sage.modular.cusps`
  - Type: `class`

- **sage.modular.cusps.Cusps_class**
  - Entity: `Cusps_class`
  - Module: `sage.modular.cusps`
  - Type: `class`

- **sage.modular.cusps_nf.NFCusp**
  - Entity: `NFCusp`
  - Module: `sage.modular.cusps_nf`
  - Type: `class`

- **sage.modular.cusps_nf.NFCuspsSpace**
  - Entity: `NFCuspsSpace`
  - Module: `sage.modular.cusps_nf`
  - Type: `class`

- **sage.modular.dirichlet.DirichletCharacter**
  - Entity: `DirichletCharacter`
  - Module: `sage.modular.dirichlet`
  - Type: `class`

- **sage.modular.dirichlet.DirichletGroupFactory**
  - Entity: `DirichletGroupFactory`
  - Module: `sage.modular.dirichlet`
  - Type: `class`

- **sage.modular.dirichlet.DirichletGroup_class**
  - Entity: `DirichletGroup_class`
  - Module: `sage.modular.dirichlet`
  - Type: `class`

- **sage.modular.drinfeld_modform.element.DrinfeldModularFormsElement**
  - Entity: `DrinfeldModularFormsElement`
  - Module: `sage.modular.drinfeld_modform.element`
  - Type: `class`

- **sage.modular.drinfeld_modform.ring.DrinfeldModularForms**
  - Entity: `DrinfeldModularForms`
  - Module: `sage.modular.drinfeld_modform.ring`
  - Type: `class`

- **sage.modular.etaproducts.CuspFamily**
  - Entity: `CuspFamily`
  - Module: `sage.modular.etaproducts`
  - Type: `class`

- **sage.modular.etaproducts.EtaGroupElement**
  - Entity: `EtaGroupElement`
  - Module: `sage.modular.etaproducts`
  - Type: `class`

- **sage.modular.etaproducts.EtaGroup_class**
  - Entity: `EtaGroup_class`
  - Module: `sage.modular.etaproducts`
  - Type: `class`

- **sage.modular.hecke.algebra.HeckeAlgebra_anemic**
  - Entity: `HeckeAlgebra_anemic`
  - Module: `sage.modular.hecke.algebra`
  - Type: `class`

- **sage.modular.hecke.algebra.HeckeAlgebra_base**
  - Entity: `HeckeAlgebra_base`
  - Module: `sage.modular.hecke.algebra`
  - Type: `class`

- **sage.modular.hecke.algebra.HeckeAlgebra_full**
  - Entity: `HeckeAlgebra_full`
  - Module: `sage.modular.hecke.algebra`
  - Type: `class`

- **sage.modular.hecke.ambient_module.AmbientHeckeModule**
  - Entity: `AmbientHeckeModule`
  - Module: `sage.modular.hecke.ambient_module`
  - Type: `class`

- **sage.modular.hecke.degenmap.DegeneracyMap**
  - Entity: `DegeneracyMap`
  - Module: `sage.modular.hecke.degenmap`
  - Type: `class`

- **sage.modular.hecke.element.HeckeModuleElement**
  - Entity: `HeckeModuleElement`
  - Module: `sage.modular.hecke.element`
  - Type: `class`

- **sage.modular.hecke.hecke_operator.DiamondBracketOperator**
  - Entity: `DiamondBracketOperator`
  - Module: `sage.modular.hecke.hecke_operator`
  - Type: `class`

- **sage.modular.hecke.hecke_operator.HeckeAlgebraElement**
  - Entity: `HeckeAlgebraElement`
  - Module: `sage.modular.hecke.hecke_operator`
  - Type: `class`

- **sage.modular.hecke.hecke_operator.HeckeAlgebraElement_matrix**
  - Entity: `HeckeAlgebraElement_matrix`
  - Module: `sage.modular.hecke.hecke_operator`
  - Type: `class`

- **sage.modular.hecke.hecke_operator.HeckeOperator**
  - Entity: `HeckeOperator`
  - Module: `sage.modular.hecke.hecke_operator`
  - Type: `class`

- **sage.modular.hecke.homspace.HeckeModuleHomspace**
  - Entity: `HeckeModuleHomspace`
  - Module: `sage.modular.hecke.homspace`
  - Type: `class`

- **sage.modular.hecke.module.HeckeModule_free_module**
  - Entity: `HeckeModule_free_module`
  - Module: `sage.modular.hecke.module`
  - Type: `class`

- **sage.modular.hecke.module.HeckeModule_generic**
  - Entity: `HeckeModule_generic`
  - Module: `sage.modular.hecke.module`
  - Type: `class`

- **sage.modular.hecke.morphism.HeckeModuleMorphism**
  - Entity: `HeckeModuleMorphism`
  - Module: `sage.modular.hecke.morphism`
  - Type: `class`

- **sage.modular.hecke.morphism.HeckeModuleMorphism_matrix**
  - Entity: `HeckeModuleMorphism_matrix`
  - Module: `sage.modular.hecke.morphism`
  - Type: `class`

- **sage.modular.hecke.submodule.HeckeSubmodule**
  - Entity: `HeckeSubmodule`
  - Module: `sage.modular.hecke.submodule`
  - Type: `class`

- **sage.modular.hypergeometric_motive.HypergeometricData**
  - Entity: `HypergeometricData`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `class`

- **sage.modular.local_comp.local_comp.ImprimitiveLocalComponent**
  - Entity: `ImprimitiveLocalComponent`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.local_comp.LocalComponentBase**
  - Entity: `LocalComponentBase`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.local_comp.PrimitiveLocalComponent**
  - Entity: `PrimitiveLocalComponent`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.local_comp.PrimitivePrincipalSeries**
  - Entity: `PrimitivePrincipalSeries`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.local_comp.PrimitiveSpecial**
  - Entity: `PrimitiveSpecial`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.local_comp.PrimitiveSupercuspidal**
  - Entity: `PrimitiveSupercuspidal`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.local_comp.PrincipalSeries**
  - Entity: `PrincipalSeries`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.local_comp.UnramifiedPrincipalSeries**
  - Entity: `UnramifiedPrincipalSeries`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `class`

- **sage.modular.local_comp.smoothchar.SmoothCharacterGeneric**
  - Entity: `SmoothCharacterGeneric`
  - Module: `sage.modular.local_comp.smoothchar`
  - Type: `class`

- **sage.modular.local_comp.smoothchar.SmoothCharacterGroupGeneric**
  - Entity: `SmoothCharacterGroupGeneric`
  - Module: `sage.modular.local_comp.smoothchar`
  - Type: `class`

- **sage.modular.local_comp.smoothchar.SmoothCharacterGroupQp**
  - Entity: `SmoothCharacterGroupQp`
  - Module: `sage.modular.local_comp.smoothchar`
  - Type: `class`

- **sage.modular.local_comp.smoothchar.SmoothCharacterGroupQuadratic**
  - Entity: `SmoothCharacterGroupQuadratic`
  - Module: `sage.modular.local_comp.smoothchar`
  - Type: `class`

- **sage.modular.local_comp.smoothchar.SmoothCharacterGroupRamifiedQuadratic**
  - Entity: `SmoothCharacterGroupRamifiedQuadratic`
  - Module: `sage.modular.local_comp.smoothchar`
  - Type: `class`

- **sage.modular.local_comp.smoothchar.SmoothCharacterGroupUnramifiedQuadratic**
  - Entity: `SmoothCharacterGroupUnramifiedQuadratic`
  - Module: `sage.modular.local_comp.smoothchar`
  - Type: `class`

- **sage.modular.local_comp.type_space.TypeSpace**
  - Entity: `TypeSpace`
  - Module: `sage.modular.local_comp.type_space`
  - Type: `class`

- **sage.modular.modform.ambient.ModularFormsAmbient**
  - Entity: `ModularFormsAmbient`
  - Module: `sage.modular.modform.ambient`
  - Type: `class`

- **sage.modular.modform.ambient_R.ModularFormsAmbient_R**
  - Entity: `ModularFormsAmbient_R`
  - Module: `sage.modular.modform.ambient_R`
  - Type: `class`

- **sage.modular.modform.ambient_eps.ModularFormsAmbient_eps**
  - Entity: `ModularFormsAmbient_eps`
  - Module: `sage.modular.modform.ambient_eps`
  - Type: `class`

- **sage.modular.modform.ambient_g0.ModularFormsAmbient_g0_Q**
  - Entity: `ModularFormsAmbient_g0_Q`
  - Module: `sage.modular.modform.ambient_g0`
  - Type: `class`

- **sage.modular.modform.ambient_g1.ModularFormsAmbient_g1_Q**
  - Entity: `ModularFormsAmbient_g1_Q`
  - Module: `sage.modular.modform.ambient_g1`
  - Type: `class`

- **sage.modular.modform.ambient_g1.ModularFormsAmbient_gH_Q**
  - Entity: `ModularFormsAmbient_gH_Q`
  - Module: `sage.modular.modform.ambient_g1`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule**
  - Entity: `CuspidalSubmodule`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_R**
  - Entity: `CuspidalSubmodule_R`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_eps**
  - Entity: `CuspidalSubmodule_eps`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_g0_Q**
  - Entity: `CuspidalSubmodule_g0_Q`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_g1_Q**
  - Entity: `CuspidalSubmodule_g1_Q`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_gH_Q**
  - Entity: `CuspidalSubmodule_gH_Q`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_level1_Q**
  - Entity: `CuspidalSubmodule_level1_Q`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_modsym_qexp**
  - Entity: `CuspidalSubmodule_modsym_qexp`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_wt1_eps**
  - Entity: `CuspidalSubmodule_wt1_eps`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.cuspidal_submodule.CuspidalSubmodule_wt1_gH**
  - Entity: `CuspidalSubmodule_wt1_gH`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `class`

- **sage.modular.modform.eisenstein_submodule.EisensteinSubmodule**
  - Entity: `EisensteinSubmodule`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `class`

- **sage.modular.modform.eisenstein_submodule.EisensteinSubmodule_eps**
  - Entity: `EisensteinSubmodule_eps`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `class`

- **sage.modular.modform.eisenstein_submodule.EisensteinSubmodule_g0_Q**
  - Entity: `EisensteinSubmodule_g0_Q`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `class`

- **sage.modular.modform.eisenstein_submodule.EisensteinSubmodule_g1_Q**
  - Entity: `EisensteinSubmodule_g1_Q`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `class`

- **sage.modular.modform.eisenstein_submodule.EisensteinSubmodule_gH_Q**
  - Entity: `EisensteinSubmodule_gH_Q`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `class`

- **sage.modular.modform.eisenstein_submodule.EisensteinSubmodule_params**
  - Entity: `EisensteinSubmodule_params`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `class`

- **sage.modular.modform.element.EisensteinSeries**
  - Entity: `EisensteinSeries`
  - Module: `sage.modular.modform.element`
  - Type: `class`

- **sage.modular.modform.element.GradedModularFormElement**
  - Entity: `GradedModularFormElement`
  - Module: `sage.modular.modform.element`
  - Type: `class`

- **sage.modular.modform.element.ModularFormElement**
  - Entity: `ModularFormElement`
  - Module: `sage.modular.modform.element`
  - Type: `class`

- **sage.modular.modform.element.ModularFormElement_elliptic_curve**
  - Entity: `ModularFormElement_elliptic_curve`
  - Module: `sage.modular.modform.element`
  - Type: `class`

- **sage.modular.modform.element.ModularForm_abstract**
  - Entity: `ModularForm_abstract`
  - Module: `sage.modular.modform.element`
  - Type: `class`

- **sage.modular.modform.element.Newform**
  - Entity: `Newform`
  - Module: `sage.modular.modform.element`
  - Type: `class`

- **sage.modular.modform.numerical.NumericalEigenforms**
  - Entity: `NumericalEigenforms`
  - Module: `sage.modular.modform.numerical`
  - Type: `class`

- **sage.modular.modform.ring.ModularFormsRing**
  - Entity: `ModularFormsRing`
  - Module: `sage.modular.modform.ring`
  - Type: `class`

- **sage.modular.modform.space.ModularFormsSpace**
  - Entity: `ModularFormsSpace`
  - Module: `sage.modular.modform.space`
  - Type: `class`

- **sage.modular.modform.submodule.ModularFormsSubmodule**
  - Entity: `ModularFormsSubmodule`
  - Module: `sage.modular.modform.submodule`
  - Type: `class`

- **sage.modular.modform.submodule.ModularFormsSubmoduleWithBasis**
  - Entity: `ModularFormsSubmoduleWithBasis`
  - Module: `sage.modular.modform.submodule`
  - Type: `class`

- **sage.modular.modform_hecketriangle.abstract_ring.FormsRing_abstract**
  - Entity: `FormsRing_abstract`
  - Module: `sage.modular.modform_hecketriangle.abstract_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.abstract_space.FormsSpace_abstract**
  - Entity: `FormsSpace_abstract`
  - Module: `sage.modular.modform_hecketriangle.abstract_space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.analytic_type.AnalyticType**
  - Entity: `AnalyticType`
  - Module: `sage.modular.modform_hecketriangle.analytic_type`
  - Type: `class`

- **sage.modular.modform_hecketriangle.analytic_type.AnalyticTypeElement**
  - Entity: `AnalyticTypeElement`
  - Module: `sage.modular.modform_hecketriangle.analytic_type`
  - Type: `class`

- **sage.modular.modform_hecketriangle.element.FormsElement**
  - Entity: `FormsElement`
  - Module: `sage.modular.modform_hecketriangle.element`
  - Type: `class`

- **sage.modular.modform_hecketriangle.functors.BaseFacade**
  - Entity: `BaseFacade`
  - Module: `sage.modular.modform_hecketriangle.functors`
  - Type: `class`

- **sage.modular.modform_hecketriangle.functors.FormsRingFunctor**
  - Entity: `FormsRingFunctor`
  - Module: `sage.modular.modform_hecketriangle.functors`
  - Type: `class`

- **sage.modular.modform_hecketriangle.functors.FormsSpaceFunctor**
  - Entity: `FormsSpaceFunctor`
  - Module: `sage.modular.modform_hecketriangle.functors`
  - Type: `class`

- **sage.modular.modform_hecketriangle.functors.FormsSubSpaceFunctor**
  - Entity: `FormsSubSpaceFunctor`
  - Module: `sage.modular.modform_hecketriangle.functors`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.CuspFormsRing**
  - Entity: `CuspFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.MeromorphicModularFormsRing**
  - Entity: `MeromorphicModularFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.ModularFormsRing**
  - Entity: `ModularFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.QuasiCuspFormsRing**
  - Entity: `QuasiCuspFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.QuasiMeromorphicModularFormsRing**
  - Entity: `QuasiMeromorphicModularFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.QuasiModularFormsRing**
  - Entity: `QuasiModularFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.QuasiWeakModularFormsRing**
  - Entity: `QuasiWeakModularFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring.WeakModularFormsRing**
  - Entity: `WeakModularFormsRing`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `class`

- **sage.modular.modform_hecketriangle.graded_ring_element.FormsRingElement**
  - Entity: `FormsRingElement`
  - Module: `sage.modular.modform_hecketriangle.graded_ring_element`
  - Type: `class`

- **sage.modular.modform_hecketriangle.hecke_triangle_group_element.HeckeTriangleGroupElement**
  - Entity: `HeckeTriangleGroupElement`
  - Module: `sage.modular.modform_hecketriangle.hecke_triangle_group_element`
  - Type: `class`

- **sage.modular.modform_hecketriangle.hecke_triangle_groups.HeckeTriangleGroup**
  - Entity: `HeckeTriangleGroup`
  - Module: `sage.modular.modform_hecketriangle.hecke_triangle_groups`
  - Type: `class`

- **sage.modular.modform_hecketriangle.series_constructor.MFSeriesConstructor**
  - Entity: `MFSeriesConstructor`
  - Module: `sage.modular.modform_hecketriangle.series_constructor`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.CuspForms**
  - Entity: `CuspForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.MeromorphicModularForms**
  - Entity: `MeromorphicModularForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.ModularForms**
  - Entity: `ModularForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.QuasiCuspForms**
  - Entity: `QuasiCuspForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.QuasiMeromorphicModularForms**
  - Entity: `QuasiMeromorphicModularForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.QuasiModularForms**
  - Entity: `QuasiModularForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.QuasiWeakModularForms**
  - Entity: `QuasiWeakModularForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.WeakModularForms**
  - Entity: `WeakModularForms`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.space.ZeroForm**
  - Entity: `ZeroForm`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `class`

- **sage.modular.modform_hecketriangle.subspace.SubSpaceForms**
  - Entity: `SubSpaceForms`
  - Module: `sage.modular.modform_hecketriangle.subspace`
  - Type: `class`

- **sage.modular.modsym.ambient.ModularSymbolsAmbient**
  - Entity: `ModularSymbolsAmbient`
  - Module: `sage.modular.modsym.ambient`
  - Type: `class`

- **sage.modular.modsym.ambient.ModularSymbolsAmbient_wt2_g0**
  - Entity: `ModularSymbolsAmbient_wt2_g0`
  - Module: `sage.modular.modsym.ambient`
  - Type: `class`

- **sage.modular.modsym.ambient.ModularSymbolsAmbient_wtk_eps**
  - Entity: `ModularSymbolsAmbient_wtk_eps`
  - Module: `sage.modular.modsym.ambient`
  - Type: `class`

- **sage.modular.modsym.ambient.ModularSymbolsAmbient_wtk_g0**
  - Entity: `ModularSymbolsAmbient_wtk_g0`
  - Module: `sage.modular.modsym.ambient`
  - Type: `class`

- **sage.modular.modsym.ambient.ModularSymbolsAmbient_wtk_g1**
  - Entity: `ModularSymbolsAmbient_wtk_g1`
  - Module: `sage.modular.modsym.ambient`
  - Type: `class`

- **sage.modular.modsym.ambient.ModularSymbolsAmbient_wtk_gamma_h**
  - Entity: `ModularSymbolsAmbient_wtk_gamma_h`
  - Module: `sage.modular.modsym.ambient`
  - Type: `class`

- **sage.modular.modsym.apply.Apply**
  - Entity: `Apply`
  - Module: `sage.modular.modsym.apply`
  - Type: `class`

- **sage.modular.modsym.boundary.BoundarySpace**
  - Entity: `BoundarySpace`
  - Module: `sage.modular.modsym.boundary`
  - Type: `class`

- **sage.modular.modsym.boundary.BoundarySpaceElement**
  - Entity: `BoundarySpaceElement`
  - Module: `sage.modular.modsym.boundary`
  - Type: `class`

- **sage.modular.modsym.boundary.BoundarySpace_wtk_eps**
  - Entity: `BoundarySpace_wtk_eps`
  - Module: `sage.modular.modsym.boundary`
  - Type: `class`

- **sage.modular.modsym.boundary.BoundarySpace_wtk_g0**
  - Entity: `BoundarySpace_wtk_g0`
  - Module: `sage.modular.modsym.boundary`
  - Type: `class`

- **sage.modular.modsym.boundary.BoundarySpace_wtk_g1**
  - Entity: `BoundarySpace_wtk_g1`
  - Module: `sage.modular.modsym.boundary`
  - Type: `class`

- **sage.modular.modsym.boundary.BoundarySpace_wtk_gamma_h**
  - Entity: `BoundarySpace_wtk_gamma_h`
  - Module: `sage.modular.modsym.boundary`
  - Type: `class`

- **sage.modular.modsym.element.ModularSymbolsElement**
  - Entity: `ModularSymbolsElement`
  - Module: `sage.modular.modsym.element`
  - Type: `class`

- **sage.modular.modsym.g1list.G1list**
  - Entity: `G1list`
  - Module: `sage.modular.modsym.g1list`
  - Type: `class`

- **sage.modular.modsym.ghlist.GHlist**
  - Entity: `GHlist`
  - Module: `sage.modular.modsym.ghlist`
  - Type: `class`

- **sage.modular.modsym.hecke_operator.HeckeOperator**
  - Entity: `HeckeOperator`
  - Module: `sage.modular.modsym.hecke_operator`
  - Type: `class`

- **sage.modular.modsym.heilbronn.Heilbronn**
  - Entity: `Heilbronn`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `class`

- **sage.modular.modsym.heilbronn.HeilbronnCremona**
  - Entity: `HeilbronnCremona`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `class`

- **sage.modular.modsym.heilbronn.HeilbronnMerel**
  - Entity: `HeilbronnMerel`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `class`

- **sage.modular.modsym.manin_symbol.ManinSymbol**
  - Entity: `ManinSymbol`
  - Module: `sage.modular.modsym.manin_symbol`
  - Type: `class`

- **sage.modular.modsym.manin_symbol_list.ManinSymbolList**
  - Entity: `ManinSymbolList`
  - Module: `sage.modular.modsym.manin_symbol_list`
  - Type: `class`

- **sage.modular.modsym.manin_symbol_list.ManinSymbolList_character**
  - Entity: `ManinSymbolList_character`
  - Module: `sage.modular.modsym.manin_symbol_list`
  - Type: `class`

- **sage.modular.modsym.manin_symbol_list.ManinSymbolList_gamma0**
  - Entity: `ManinSymbolList_gamma0`
  - Module: `sage.modular.modsym.manin_symbol_list`
  - Type: `class`

- **sage.modular.modsym.manin_symbol_list.ManinSymbolList_gamma1**
  - Entity: `ManinSymbolList_gamma1`
  - Module: `sage.modular.modsym.manin_symbol_list`
  - Type: `class`

- **sage.modular.modsym.manin_symbol_list.ManinSymbolList_gamma_h**
  - Entity: `ManinSymbolList_gamma_h`
  - Module: `sage.modular.modsym.manin_symbol_list`
  - Type: `class`

- **sage.modular.modsym.manin_symbol_list.ManinSymbolList_group**
  - Entity: `ManinSymbolList_group`
  - Module: `sage.modular.modsym.manin_symbol_list`
  - Type: `class`

- **sage.modular.modsym.modular_symbols.ModularSymbol**
  - Entity: `ModularSymbol`
  - Module: `sage.modular.modsym.modular_symbols`
  - Type: `class`

- **sage.modular.modsym.p1list.P1List**
  - Entity: `P1List`
  - Module: `sage.modular.modsym.p1list`
  - Type: `class`

- **sage.modular.modsym.p1list.export**
  - Entity: `export`
  - Module: `sage.modular.modsym.p1list`
  - Type: `class`

- **sage.modular.modsym.p1list_nf.MSymbol**
  - Entity: `MSymbol`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `class`

- **sage.modular.modsym.p1list_nf.P1NFList**
  - Entity: `P1NFList`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `class`

- **sage.modular.modsym.space.IntegralPeriodMapping**
  - Entity: `IntegralPeriodMapping`
  - Module: `sage.modular.modsym.space`
  - Type: `class`

- **sage.modular.modsym.space.ModularSymbolsSpace**
  - Entity: `ModularSymbolsSpace`
  - Module: `sage.modular.modsym.space`
  - Type: `class`

- **sage.modular.modsym.space.PeriodMapping**
  - Entity: `PeriodMapping`
  - Module: `sage.modular.modsym.space`
  - Type: `class`

- **sage.modular.modsym.space.RationalPeriodMapping**
  - Entity: `RationalPeriodMapping`
  - Module: `sage.modular.modsym.space`
  - Type: `class`

- **sage.modular.modsym.subspace.ModularSymbolsSubspace**
  - Entity: `ModularSymbolsSubspace`
  - Module: `sage.modular.modsym.subspace`
  - Type: `class`

- **sage.modular.multiple_zeta.All_iterated**
  - Entity: `All_iterated`
  - Module: `sage.modular.multiple_zeta`
  - Type: `class`

- **sage.modular.multiple_zeta.Element**
  - Entity: `Element`
  - Module: `sage.modular.multiple_zeta`
  - Type: `class`

- **sage.modular.multiple_zeta.MultizetaValues**
  - Entity: `MultizetaValues`
  - Module: `sage.modular.multiple_zeta`
  - Type: `class`

- **sage.modular.multiple_zeta.Multizetas**
  - Entity: `Multizetas`
  - Module: `sage.modular.multiple_zeta`
  - Type: `class`

- **sage.modular.multiple_zeta.Multizetas_iterated**
  - Entity: `Multizetas_iterated`
  - Module: `sage.modular.multiple_zeta`
  - Type: `class`

- **sage.modular.overconvergent.genus0.OverconvergentModularFormElement**
  - Entity: `OverconvergentModularFormElement`
  - Module: `sage.modular.overconvergent.genus0`
  - Type: `class`

- **sage.modular.overconvergent.genus0.OverconvergentModularFormsSpace**
  - Entity: `OverconvergentModularFormsSpace`
  - Module: `sage.modular.overconvergent.genus0`
  - Type: `class`

- **sage.modular.overconvergent.weightspace.AlgebraicWeight**
  - Entity: `AlgebraicWeight`
  - Module: `sage.modular.overconvergent.weightspace`
  - Type: `class`

- **sage.modular.overconvergent.weightspace.ArbitraryWeight**
  - Entity: `ArbitraryWeight`
  - Module: `sage.modular.overconvergent.weightspace`
  - Type: `class`

- **sage.modular.overconvergent.weightspace.WeightCharacter**
  - Entity: `WeightCharacter`
  - Module: `sage.modular.overconvergent.weightspace`
  - Type: `class`

- **sage.modular.overconvergent.weightspace.WeightSpace_class**
  - Entity: `WeightSpace_class`
  - Module: `sage.modular.overconvergent.weightspace`
  - Type: `class`

- **sage.modular.pollack_stevens.distributions.OverconvergentDistributions_abstract**
  - Entity: `OverconvergentDistributions_abstract`
  - Module: `sage.modular.pollack_stevens.distributions`
  - Type: `class`

- **sage.modular.pollack_stevens.distributions.OverconvergentDistributions_class**
  - Entity: `OverconvergentDistributions_class`
  - Module: `sage.modular.pollack_stevens.distributions`
  - Type: `class`

- **sage.modular.pollack_stevens.distributions.OverconvergentDistributions_factory**
  - Entity: `OverconvergentDistributions_factory`
  - Module: `sage.modular.pollack_stevens.distributions`
  - Type: `class`

- **sage.modular.pollack_stevens.distributions.Symk_class**
  - Entity: `Symk_class`
  - Module: `sage.modular.pollack_stevens.distributions`
  - Type: `class`

- **sage.modular.pollack_stevens.distributions.Symk_factory**
  - Entity: `Symk_factory`
  - Module: `sage.modular.pollack_stevens.distributions`
  - Type: `class`

- **sage.modular.pollack_stevens.fund_domain.ManinRelations**
  - Entity: `ManinRelations`
  - Module: `sage.modular.pollack_stevens.fund_domain`
  - Type: `class`

- **sage.modular.pollack_stevens.fund_domain.PollackStevensModularDomain**
  - Entity: `PollackStevensModularDomain`
  - Module: `sage.modular.pollack_stevens.fund_domain`
  - Type: `class`

- **sage.modular.pollack_stevens.manin_map.ManinMap**
  - Entity: `ManinMap`
  - Module: `sage.modular.pollack_stevens.manin_map`
  - Type: `class`

- **sage.modular.pollack_stevens.modsym.PSModSymAction**
  - Entity: `PSModSymAction`
  - Module: `sage.modular.pollack_stevens.modsym`
  - Type: `class`

- **sage.modular.pollack_stevens.modsym.PSModularSymbolElement**
  - Entity: `PSModularSymbolElement`
  - Module: `sage.modular.pollack_stevens.modsym`
  - Type: `class`

- **sage.modular.pollack_stevens.modsym.PSModularSymbolElement_dist**
  - Entity: `PSModularSymbolElement_dist`
  - Module: `sage.modular.pollack_stevens.modsym`
  - Type: `class`

- **sage.modular.pollack_stevens.modsym.PSModularSymbolElement_symk**
  - Entity: `PSModularSymbolElement_symk`
  - Module: `sage.modular.pollack_stevens.modsym`
  - Type: `class`

- **sage.modular.pollack_stevens.padic_lseries.pAdicLseries**
  - Entity: `pAdicLseries`
  - Module: `sage.modular.pollack_stevens.padic_lseries`
  - Type: `class`

- **sage.modular.pollack_stevens.space.PollackStevensModularSymbols_factory**
  - Entity: `PollackStevensModularSymbols_factory`
  - Module: `sage.modular.pollack_stevens.space`
  - Type: `class`

- **sage.modular.pollack_stevens.space.PollackStevensModularSymbolspace**
  - Entity: `PollackStevensModularSymbolspace`
  - Module: `sage.modular.pollack_stevens.space`
  - Type: `class`

- **sage.modular.quasimodform.element.QuasiModularFormsElement**
  - Entity: `QuasiModularFormsElement`
  - Module: `sage.modular.quasimodform.element`
  - Type: `class`

- **sage.modular.quasimodform.ring.QuasiModularForms**
  - Entity: `QuasiModularForms`
  - Module: `sage.modular.quasimodform.ring`
  - Type: `class`

- **sage.modular.quatalg.brandt.BrandtModuleElement**
  - Entity: `BrandtModuleElement`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `class`

- **sage.modular.quatalg.brandt.BrandtModule_class**
  - Entity: `BrandtModule_class`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `class`

- **sage.modular.quatalg.brandt.BrandtSubmodule**
  - Entity: `BrandtSubmodule`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `class`

- **sage.modular.ssmod.ssmod.SupersingularModule**
  - Entity: `SupersingularModule`
  - Module: `sage.modular.ssmod.ssmod`
  - Type: `class`

- **sage.modules.fg_pid.fgp_element.FGP_Element**
  - Entity: `FGP_Element`
  - Module: `sage.modules.fg_pid.fgp_element`
  - Type: `class`

- **sage.modules.fg_pid.fgp_module.FGP_Module_class**
  - Entity: `FGP_Module_class`
  - Module: `sage.modules.fg_pid.fgp_module`
  - Type: `class`

- **sage.modules.fg_pid.fgp_morphism.FGP_Homset_class**
  - Entity: `FGP_Homset_class`
  - Module: `sage.modules.fg_pid.fgp_morphism`
  - Type: `class`

- **sage.modules.fg_pid.fgp_morphism.FGP_Morphism**
  - Entity: `FGP_Morphism`
  - Module: `sage.modules.fg_pid.fgp_morphism`
  - Type: `class`

- **sage.modules.filtered_vector_space.FilteredVectorSpace_class**
  - Entity: `FilteredVectorSpace_class`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `class`

- **sage.modules.finite_submodule_iter.FiniteFieldsubspace_iterator**
  - Entity: `FiniteFieldsubspace_iterator`
  - Module: `sage.modules.finite_submodule_iter`
  - Type: `class`

- **sage.modules.finite_submodule_iter.FiniteFieldsubspace_projPoint_iterator**
  - Entity: `FiniteFieldsubspace_projPoint_iterator`
  - Module: `sage.modules.finite_submodule_iter`
  - Type: `class`

- **sage.modules.finite_submodule_iter.FiniteZZsubmodule_iterator**
  - Entity: `FiniteZZsubmodule_iterator`
  - Module: `sage.modules.finite_submodule_iter`
  - Type: `class`

- **sage.modules.fp_graded.element.FPElement**
  - Entity: `FPElement`
  - Module: `sage.modules.fp_graded.element`
  - Type: `class`

- **sage.modules.fp_graded.free_element.FreeGradedModuleElement**
  - Entity: `FreeGradedModuleElement`
  - Module: `sage.modules.fp_graded.free_element`
  - Type: `class`

- **sage.modules.fp_graded.free_homspace.FreeGradedModuleHomspace**
  - Entity: `FreeGradedModuleHomspace`
  - Module: `sage.modules.fp_graded.free_homspace`
  - Type: `class`

- **sage.modules.fp_graded.free_module.FreeGradedModule**
  - Entity: `FreeGradedModule`
  - Module: `sage.modules.fp_graded.free_module`
  - Type: `class`

- **sage.modules.fp_graded.free_morphism.FreeGradedModuleMorphism**
  - Entity: `FreeGradedModuleMorphism`
  - Module: `sage.modules.fp_graded.free_morphism`
  - Type: `class`

- **sage.modules.fp_graded.homspace.FPModuleHomspace**
  - Entity: `FPModuleHomspace`
  - Module: `sage.modules.fp_graded.homspace`
  - Type: `class`

- **sage.modules.fp_graded.module.FPModule**
  - Entity: `FPModule`
  - Module: `sage.modules.fp_graded.module`
  - Type: `class`

- **sage.modules.fp_graded.morphism.FPModuleMorphism**
  - Entity: `FPModuleMorphism`
  - Module: `sage.modules.fp_graded.morphism`
  - Type: `class`

- **sage.modules.fp_graded.steenrod.module.SteenrodFPModule**
  - Entity: `SteenrodFPModule`
  - Module: `sage.modules.fp_graded.steenrod.module`
  - Type: `class`

- **sage.modules.fp_graded.steenrod.module.SteenrodFreeModule**
  - Entity: `SteenrodFreeModule`
  - Module: `sage.modules.fp_graded.steenrod.module`
  - Type: `class`

- **sage.modules.fp_graded.steenrod.module.SteenrodModuleMixin**
  - Entity: `SteenrodModuleMixin`
  - Module: `sage.modules.fp_graded.steenrod.module`
  - Type: `class`

- **sage.modules.fp_graded.steenrod.morphism.SteenrodFPModuleMorphism**
  - Entity: `SteenrodFPModuleMorphism`
  - Module: `sage.modules.fp_graded.steenrod.morphism`
  - Type: `class`

- **sage.modules.fp_graded.steenrod.morphism.SteenrodFreeModuleMorphism**
  - Entity: `SteenrodFreeModuleMorphism`
  - Module: `sage.modules.fp_graded.steenrod.morphism`
  - Type: `class`

- **sage.modules.free_module.ComplexDoubleVectorSpace_class**
  - Entity: `ComplexDoubleVectorSpace_class`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.EchelonMatrixKey**
  - Entity: `EchelonMatrixKey`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModuleFactory**
  - Entity: `FreeModuleFactory`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_ambient**
  - Entity: `FreeModule_ambient`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_ambient_domain**
  - Entity: `FreeModule_ambient_domain`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_ambient_field**
  - Entity: `FreeModule_ambient_field`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_ambient_pid**
  - Entity: `FreeModule_ambient_pid`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_generic**
  - Entity: `FreeModule_generic`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_generic_domain**
  - Entity: `FreeModule_generic_domain`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_generic_field**
  - Entity: `FreeModule_generic_field`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_generic_pid**
  - Entity: `FreeModule_generic_pid`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_submodule_field**
  - Entity: `FreeModule_submodule_field`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_submodule_pid**
  - Entity: `FreeModule_submodule_pid`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_submodule_with_basis_field**
  - Entity: `FreeModule_submodule_with_basis_field`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.FreeModule_submodule_with_basis_pid**
  - Entity: `FreeModule_submodule_with_basis_pid`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.Module_free_ambient**
  - Entity: `Module_free_ambient`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module.RealDoubleVectorSpace_class**
  - Entity: `RealDoubleVectorSpace_class`
  - Module: `sage.modules.free_module`
  - Type: `class`

- **sage.modules.free_module_element.FreeModuleElement**
  - Entity: `FreeModuleElement`
  - Module: `sage.modules.free_module_element`
  - Type: `class`

- **sage.modules.free_module_element.FreeModuleElement_generic_dense**
  - Entity: `FreeModuleElement_generic_dense`
  - Module: `sage.modules.free_module_element`
  - Type: `class`

- **sage.modules.free_module_element.FreeModuleElement_generic_sparse**
  - Entity: `FreeModuleElement_generic_sparse`
  - Module: `sage.modules.free_module_element`
  - Type: `class`

- **sage.modules.free_module_homspace.FreeModuleHomspace**
  - Entity: `FreeModuleHomspace`
  - Module: `sage.modules.free_module_homspace`
  - Type: `class`

- **sage.modules.free_module_integer.FreeModule_submodule_with_basis_integer**
  - Entity: `FreeModule_submodule_with_basis_integer`
  - Module: `sage.modules.free_module_integer`
  - Type: `class`

- **sage.modules.free_module_morphism.BaseIsomorphism1D**
  - Entity: `BaseIsomorphism1D`
  - Module: `sage.modules.free_module_morphism`
  - Type: `class`

- **sage.modules.free_module_morphism.BaseIsomorphism1D_from_FM**
  - Entity: `BaseIsomorphism1D_from_FM`
  - Module: `sage.modules.free_module_morphism`
  - Type: `class`

- **sage.modules.free_module_morphism.BaseIsomorphism1D_to_FM**
  - Entity: `BaseIsomorphism1D_to_FM`
  - Module: `sage.modules.free_module_morphism`
  - Type: `class`

- **sage.modules.free_module_morphism.FreeModuleMorphism**
  - Entity: `FreeModuleMorphism`
  - Module: `sage.modules.free_module_morphism`
  - Type: `class`

- **sage.modules.free_module_pseudohomspace.FreeModulePseudoHomspace**
  - Entity: `FreeModulePseudoHomspace`
  - Module: `sage.modules.free_module_pseudohomspace`
  - Type: `class`

- **sage.modules.free_module_pseudomorphism.FreeModulePseudoMorphism**
  - Entity: `FreeModulePseudoMorphism`
  - Module: `sage.modules.free_module_pseudomorphism`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_ambient**
  - Entity: `FreeQuadraticModule_ambient`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_ambient_domain**
  - Entity: `FreeQuadraticModule_ambient_domain`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_ambient_field**
  - Entity: `FreeQuadraticModule_ambient_field`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_ambient_pid**
  - Entity: `FreeQuadraticModule_ambient_pid`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_generic**
  - Entity: `FreeQuadraticModule_generic`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_generic_field**
  - Entity: `FreeQuadraticModule_generic_field`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_generic_pid**
  - Entity: `FreeQuadraticModule_generic_pid`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_submodule_field**
  - Entity: `FreeQuadraticModule_submodule_field`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_submodule_pid**
  - Entity: `FreeQuadraticModule_submodule_pid`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_submodule_with_basis_field**
  - Entity: `FreeQuadraticModule_submodule_with_basis_field`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module.FreeQuadraticModule_submodule_with_basis_pid**
  - Entity: `FreeQuadraticModule_submodule_with_basis_pid`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `class`

- **sage.modules.free_quadratic_module_integer_symmetric.FreeQuadraticModule_integer_symmetric**
  - Entity: `FreeQuadraticModule_integer_symmetric`
  - Module: `sage.modules.free_quadratic_module_integer_symmetric`
  - Type: `class`

- **sage.modules.matrix_morphism.MatrixMorphism**
  - Entity: `MatrixMorphism`
  - Module: `sage.modules.matrix_morphism`
  - Type: `class`

- **sage.modules.matrix_morphism.MatrixMorphism_abstract**
  - Entity: `MatrixMorphism_abstract`
  - Module: `sage.modules.matrix_morphism`
  - Type: `class`

- **sage.modules.module.Module**
  - Entity: `Module`
  - Module: `sage.modules.module`
  - Type: `class`

- **sage.modules.multi_filtered_vector_space.MultiFilteredVectorSpace_class**
  - Entity: `MultiFilteredVectorSpace_class`
  - Module: `sage.modules.multi_filtered_vector_space`
  - Type: `class`

- **sage.modules.ore_module.OreAction**
  - Entity: `OreAction`
  - Module: `sage.modules.ore_module`
  - Type: `class`

- **sage.modules.ore_module.OreModule**
  - Entity: `OreModule`
  - Module: `sage.modules.ore_module`
  - Type: `class`

- **sage.modules.ore_module.OreQuotientModule**
  - Entity: `OreQuotientModule`
  - Module: `sage.modules.ore_module`
  - Type: `class`

- **sage.modules.ore_module.OreSubmodule**
  - Entity: `OreSubmodule`
  - Module: `sage.modules.ore_module`
  - Type: `class`

- **sage.modules.ore_module.ScalarAction**
  - Entity: `ScalarAction`
  - Module: `sage.modules.ore_module`
  - Type: `class`

- **sage.modules.ore_module_element.OreModuleElement**
  - Entity: `OreModuleElement`
  - Module: `sage.modules.ore_module_element`
  - Type: `class`

- **sage.modules.ore_module_homspace.OreModule_homspace**
  - Entity: `OreModule_homspace`
  - Module: `sage.modules.ore_module_homspace`
  - Type: `class`

- **sage.modules.ore_module_morphism.OreModuleMorphism**
  - Entity: `OreModuleMorphism`
  - Module: `sage.modules.ore_module_morphism`
  - Type: `class`

- **sage.modules.ore_module_morphism.OreModuleRetraction**
  - Entity: `OreModuleRetraction`
  - Module: `sage.modules.ore_module_morphism`
  - Type: `class`

- **sage.modules.ore_module_morphism.OreModuleSection**
  - Entity: `OreModuleSection`
  - Module: `sage.modules.ore_module_morphism`
  - Type: `class`

- **sage.modules.quotient_module.FreeModule_ambient_field_quotient**
  - Entity: `FreeModule_ambient_field_quotient`
  - Module: `sage.modules.quotient_module`
  - Type: `class`

- **sage.modules.quotient_module.QuotientModule_free_ambient**
  - Entity: `QuotientModule_free_ambient`
  - Module: `sage.modules.quotient_module`
  - Type: `class`

- **sage.modules.submodule.Submodule_free_ambient**
  - Entity: `Submodule_free_ambient`
  - Module: `sage.modules.submodule`
  - Type: `class`

- **sage.modules.tensor_operations.TensorOperation**
  - Entity: `TensorOperation`
  - Module: `sage.modules.tensor_operations`
  - Type: `class`

- **sage.modules.tensor_operations.VectorCollection**
  - Entity: `VectorCollection`
  - Module: `sage.modules.tensor_operations`
  - Type: `class`

- **sage.modules.torsion_quadratic_module.TorsionQuadraticModule**
  - Entity: `TorsionQuadraticModule`
  - Module: `sage.modules.torsion_quadratic_module`
  - Type: `class`

- **sage.modules.torsion_quadratic_module.TorsionQuadraticModuleElement**
  - Entity: `TorsionQuadraticModuleElement`
  - Module: `sage.modules.torsion_quadratic_module`
  - Type: `class`

- **sage.modules.vector_callable_symbolic_dense.Vector_callable_symbolic_dense**
  - Entity: `Vector_callable_symbolic_dense`
  - Module: `sage.modules.vector_callable_symbolic_dense`
  - Type: `class`

- **sage.modules.vector_complex_double_dense.Vector_complex_double_dense**
  - Entity: `Vector_complex_double_dense`
  - Module: `sage.modules.vector_complex_double_dense`
  - Type: `class`

- **sage.modules.vector_double_dense.Vector_double_dense**
  - Entity: `Vector_double_dense`
  - Module: `sage.modules.vector_double_dense`
  - Type: `class`

- **sage.modules.vector_integer_dense.Vector_integer_dense**
  - Entity: `Vector_integer_dense`
  - Module: `sage.modules.vector_integer_dense`
  - Type: `class`

- **sage.modules.vector_mod2_dense.Vector_mod2_dense**
  - Entity: `Vector_mod2_dense`
  - Module: `sage.modules.vector_mod2_dense`
  - Type: `class`

- **sage.modules.vector_modn_dense.Vector_modn_dense**
  - Entity: `Vector_modn_dense`
  - Module: `sage.modules.vector_modn_dense`
  - Type: `class`

- **sage.modules.vector_numpy_dense.Vector_numpy_dense**
  - Entity: `Vector_numpy_dense`
  - Module: `sage.modules.vector_numpy_dense`
  - Type: `class`

- **sage.modules.vector_numpy_integer_dense.Vector_numpy_integer_dense**
  - Entity: `Vector_numpy_integer_dense`
  - Module: `sage.modules.vector_numpy_integer_dense`
  - Type: `class`

- **sage.modules.vector_rational_dense.Vector_rational_dense**
  - Entity: `Vector_rational_dense`
  - Module: `sage.modules.vector_rational_dense`
  - Type: `class`

- **sage.modules.vector_real_double_dense.Vector_real_double_dense**
  - Entity: `Vector_real_double_dense`
  - Module: `sage.modules.vector_real_double_dense`
  - Type: `class`

- **sage.modules.vector_space_homspace.VectorSpaceHomspace**
  - Entity: `VectorSpaceHomspace`
  - Module: `sage.modules.vector_space_homspace`
  - Type: `class`

- **sage.modules.vector_space_morphism.VectorSpaceMorphism**
  - Entity: `VectorSpaceMorphism`
  - Module: `sage.modules.vector_space_morphism`
  - Type: `class`

- **sage.modules.vector_symbolic_dense.Vector_symbolic_dense**
  - Entity: `Vector_symbolic_dense`
  - Module: `sage.modules.vector_symbolic_dense`
  - Type: `class`

- **sage.modules.vector_symbolic_sparse.Vector_symbolic_sparse**
  - Entity: `Vector_symbolic_sparse`
  - Module: `sage.modules.vector_symbolic_sparse`
  - Type: `class`

- **sage.modules.with_basis.cell_module.CellModule**
  - Entity: `CellModule`
  - Module: `sage.modules.with_basis.cell_module`
  - Type: `class`

- **sage.modules.with_basis.cell_module.Element**
  - Entity: `Element`
  - Module: `sage.modules.with_basis.cell_module`
  - Type: `class`

- **sage.modules.with_basis.cell_module.SimpleModule**
  - Entity: `SimpleModule`
  - Module: `sage.modules.with_basis.cell_module`
  - Type: `class`

- **sage.modules.with_basis.indexed_element.IndexedFreeModuleElement**
  - Entity: `IndexedFreeModuleElement`
  - Module: `sage.modules.with_basis.indexed_element`
  - Type: `class`

- **sage.modules.with_basis.invariant.Element**
  - Entity: `Element`
  - Module: `sage.modules.with_basis.invariant`
  - Type: `class`

- **sage.modules.with_basis.invariant.FiniteDimensionalInvariantModule**
  - Entity: `FiniteDimensionalInvariantModule`
  - Module: `sage.modules.with_basis.invariant`
  - Type: `class`

- **sage.modules.with_basis.invariant.FiniteDimensionalTwistedInvariantModule**
  - Entity: `FiniteDimensionalTwistedInvariantModule`
  - Module: `sage.modules.with_basis.invariant`
  - Type: `class`

- **sage.modules.with_basis.morphism.DiagonalModuleMorphism**
  - Entity: `DiagonalModuleMorphism`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.ModuleMorphism**
  - Entity: `ModuleMorphism`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.ModuleMorphismByLinearity**
  - Entity: `ModuleMorphismByLinearity`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.ModuleMorphismFromFunction**
  - Entity: `ModuleMorphismFromFunction`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.ModuleMorphismFromMatrix**
  - Entity: `ModuleMorphismFromMatrix`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.PointwiseInverseFunction**
  - Entity: `PointwiseInverseFunction`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.TriangularModuleMorphism**
  - Entity: `TriangularModuleMorphism`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.TriangularModuleMorphismByLinearity**
  - Entity: `TriangularModuleMorphismByLinearity`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.morphism.TriangularModuleMorphismFromFunction**
  - Entity: `TriangularModuleMorphismFromFunction`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `class`

- **sage.modules.with_basis.representation.Element**
  - Entity: `Element`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.NaturalMatrixRepresentation**
  - Entity: `NaturalMatrixRepresentation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.QuotientRepresentation**
  - Entity: `QuotientRepresentation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.ReflectionRepresentation**
  - Entity: `ReflectionRepresentation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.RegularRepresentation**
  - Entity: `RegularRepresentation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.Representation**
  - Entity: `Representation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.Representation_Exterior**
  - Entity: `Representation_Exterior`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.Representation_ExteriorAlgebra**
  - Entity: `Representation_ExteriorAlgebra`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.Representation_Symmetric**
  - Entity: `Representation_Symmetric`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.Representation_Tensor**
  - Entity: `Representation_Tensor`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.Representation_abstract**
  - Entity: `Representation_abstract`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.SchurFunctorRepresentation**
  - Entity: `SchurFunctorRepresentation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.SignRepresentationCoxeterGroup**
  - Entity: `SignRepresentationCoxeterGroup`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.SignRepresentationMatrixGroup**
  - Entity: `SignRepresentationMatrixGroup`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.SignRepresentationPermgroup**
  - Entity: `SignRepresentationPermgroup`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.SignRepresentation_abstract**
  - Entity: `SignRepresentation_abstract`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.Subrepresentation**
  - Entity: `Subrepresentation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.representation.TrivialRepresentation**
  - Entity: `TrivialRepresentation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `class`

- **sage.modules.with_basis.subquotient.QuotientModuleWithBasis**
  - Entity: `QuotientModuleWithBasis`
  - Module: `sage.modules.with_basis.subquotient`
  - Type: `class`

- **sage.modules.with_basis.subquotient.SubmoduleWithBasis**
  - Entity: `SubmoduleWithBasis`
  - Module: `sage.modules.with_basis.subquotient`
  - Type: `class`

- **sage.monoids.automatic_semigroup.AutomaticMonoid**
  - Entity: `AutomaticMonoid`
  - Module: `sage.monoids.automatic_semigroup`
  - Type: `class`

- **sage.monoids.automatic_semigroup.AutomaticSemigroup**
  - Entity: `AutomaticSemigroup`
  - Module: `sage.monoids.automatic_semigroup`
  - Type: `class`

- **sage.monoids.automatic_semigroup.Element**
  - Entity: `Element`
  - Module: `sage.monoids.automatic_semigroup`
  - Type: `class`

- **sage.monoids.free_abelian_monoid.FreeAbelianMonoidFactory**
  - Entity: `FreeAbelianMonoidFactory`
  - Module: `sage.monoids.free_abelian_monoid`
  - Type: `class`

- **sage.monoids.free_abelian_monoid.FreeAbelianMonoid_class**
  - Entity: `FreeAbelianMonoid_class`
  - Module: `sage.monoids.free_abelian_monoid`
  - Type: `class`

- **sage.monoids.free_abelian_monoid_element.FreeAbelianMonoidElement**
  - Entity: `FreeAbelianMonoidElement`
  - Module: `sage.monoids.free_abelian_monoid_element`
  - Type: `class`

- **sage.monoids.free_monoid.FreeMonoid**
  - Entity: `FreeMonoid`
  - Module: `sage.monoids.free_monoid`
  - Type: `class`

- **sage.monoids.free_monoid_element.FreeMonoidElement**
  - Entity: `FreeMonoidElement`
  - Module: `sage.monoids.free_monoid_element`
  - Type: `class`

- **sage.monoids.indexed_free_monoid.IndexedFreeAbelianMonoid**
  - Entity: `IndexedFreeAbelianMonoid`
  - Module: `sage.monoids.indexed_free_monoid`
  - Type: `class`

- **sage.monoids.indexed_free_monoid.IndexedFreeAbelianMonoidElement**
  - Entity: `IndexedFreeAbelianMonoidElement`
  - Module: `sage.monoids.indexed_free_monoid`
  - Type: `class`

- **sage.monoids.indexed_free_monoid.IndexedFreeMonoid**
  - Entity: `IndexedFreeMonoid`
  - Module: `sage.monoids.indexed_free_monoid`
  - Type: `class`

- **sage.monoids.indexed_free_monoid.IndexedFreeMonoidElement**
  - Entity: `IndexedFreeMonoidElement`
  - Module: `sage.monoids.indexed_free_monoid`
  - Type: `class`

- **sage.monoids.indexed_free_monoid.IndexedMonoid**
  - Entity: `IndexedMonoid`
  - Module: `sage.monoids.indexed_free_monoid`
  - Type: `class`

- **sage.monoids.indexed_free_monoid.IndexedMonoidElement**
  - Entity: `IndexedMonoidElement`
  - Module: `sage.monoids.indexed_free_monoid`
  - Type: `class`

- **sage.monoids.monoid.Monoid_class**
  - Entity: `Monoid_class`
  - Module: `sage.monoids.monoid`
  - Type: `class`

- **sage.monoids.string_monoid.AlphabeticStringMonoid**
  - Entity: `AlphabeticStringMonoid`
  - Module: `sage.monoids.string_monoid`
  - Type: `class`

- **sage.monoids.string_monoid.BinaryStringMonoid**
  - Entity: `BinaryStringMonoid`
  - Module: `sage.monoids.string_monoid`
  - Type: `class`

- **sage.monoids.string_monoid.HexadecimalStringMonoid**
  - Entity: `HexadecimalStringMonoid`
  - Module: `sage.monoids.string_monoid`
  - Type: `class`

- **sage.monoids.string_monoid.OctalStringMonoid**
  - Entity: `OctalStringMonoid`
  - Module: `sage.monoids.string_monoid`
  - Type: `class`

- **sage.monoids.string_monoid.Radix64StringMonoid**
  - Entity: `Radix64StringMonoid`
  - Module: `sage.monoids.string_monoid`
  - Type: `class`

- **sage.monoids.string_monoid.StringMonoid_class**
  - Entity: `StringMonoid_class`
  - Module: `sage.monoids.string_monoid`
  - Type: `class`

- **sage.monoids.string_monoid_element.StringMonoidElement**
  - Entity: `StringMonoidElement`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `class`

- **sage.monoids.trace_monoid.TraceMonoid**
  - Entity: `TraceMonoid`
  - Module: `sage.monoids.trace_monoid`
  - Type: `class`

- **sage.monoids.trace_monoid.TraceMonoidElement**
  - Entity: `TraceMonoidElement`
  - Module: `sage.monoids.trace_monoid`
  - Type: `class`

- **sage.numerical.backends.cvxopt_backend.CVXOPTBackend**
  - Entity: `CVXOPTBackend`
  - Module: `sage.numerical.backends.cvxopt_backend`
  - Type: `class`

- **sage.numerical.backends.cvxopt_sdp_backend.CVXOPTSDPBackend**
  - Entity: `CVXOPTSDPBackend`
  - Module: `sage.numerical.backends.cvxopt_sdp_backend`
  - Type: `class`

- **sage.numerical.backends.generic_backend.GenericBackend**
  - Entity: `GenericBackend`
  - Module: `sage.numerical.backends.generic_backend`
  - Type: `class`

- **sage.numerical.backends.generic_sdp_backend.GenericSDPBackend**
  - Entity: `GenericSDPBackend`
  - Module: `sage.numerical.backends.generic_sdp_backend`
  - Type: `class`

#### FUNCTION (370 entries)

- **sage.misc.sagedoc.detex**
  - Entity: `detex`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.format**
  - Entity: `format`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.format_search_as_html**
  - Entity: `format_search_as_html`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.format_src**
  - Entity: `format_src`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.help**
  - Entity: `help`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.my_getsource**
  - Entity: `my_getsource`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.process_dollars**
  - Entity: `process_dollars`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.process_extlinks**
  - Entity: `process_extlinks`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.process_mathtt**
  - Entity: `process_mathtt`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.process_optional_doctest_tags**
  - Entity: `process_optional_doctest_tags`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.search_def**
  - Entity: `search_def`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.search_doc**
  - Entity: `search_doc`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.search_src**
  - Entity: `search_src`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc.skip_TESTS_block**
  - Entity: `skip_TESTS_block`
  - Module: `sage.misc.sagedoc`
  - Type: `function`

- **sage.misc.sagedoc_conf.process_directives**
  - Entity: `process_directives`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sagedoc_conf.process_docstring_aliases**
  - Entity: `process_docstring_aliases`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sagedoc_conf.process_docstring_cython**
  - Entity: `process_docstring_cython`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sagedoc_conf.process_docstring_module_title**
  - Entity: `process_docstring_module_title`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sagedoc_conf.process_dollars**
  - Entity: `process_dollars`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sagedoc_conf.process_inherited**
  - Entity: `process_inherited`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sagedoc_conf.setup**
  - Entity: `setup`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sagedoc_conf.skip_TESTS_block**
  - Entity: `skip_TESTS_block`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `function`

- **sage.misc.sageinspect.find_object_modules**
  - Entity: `find_object_modules`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.formatannotation**
  - Entity: `formatannotation`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.is_function_or_cython_function**
  - Entity: `is_function_or_cython_function`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.isclassinstance**
  - Entity: `isclassinstance`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getargspec**
  - Entity: `sage_getargspec`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getdef**
  - Entity: `sage_getdef`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getdoc**
  - Entity: `sage_getdoc`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getdoc_original**
  - Entity: `sage_getdoc_original`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getfile**
  - Entity: `sage_getfile`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getfile_relative**
  - Entity: `sage_getfile_relative`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getsource**
  - Entity: `sage_getsource`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getsourcelines**
  - Entity: `sage_getsourcelines`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.sageinspect.sage_getvariablename**
  - Entity: `sage_getvariablename`
  - Module: `sage.misc.sageinspect`
  - Type: `function`

- **sage.misc.search.search**
  - Entity: `search`
  - Module: `sage.misc.search`
  - Type: `function`

- **sage.misc.session.init**
  - Entity: `init`
  - Module: `sage.misc.session`
  - Type: `function`

- **sage.misc.session.load_session**
  - Entity: `load_session`
  - Module: `sage.misc.session`
  - Type: `function`

- **sage.misc.session.save_session**
  - Entity: `save_session`
  - Module: `sage.misc.session`
  - Type: `function`

- **sage.misc.session.show_identifiers**
  - Entity: `show_identifiers`
  - Module: `sage.misc.session`
  - Type: `function`

- **sage.misc.sphinxify.sphinxify**
  - Entity: `sphinxify`
  - Module: `sage.misc.sphinxify`
  - Type: `function`

- **sage.misc.stopgap.set_state**
  - Entity: `set_state`
  - Module: `sage.misc.stopgap`
  - Type: `function`

- **sage.misc.stopgap.stopgap**
  - Entity: `stopgap`
  - Module: `sage.misc.stopgap`
  - Type: `function`

- **sage.misc.superseded.deprecated_function_alias**
  - Entity: `deprecated_function_alias`
  - Module: `sage.misc.superseded`
  - Type: `function`

- **sage.misc.superseded.deprecation**
  - Entity: `deprecation`
  - Module: `sage.misc.superseded`
  - Type: `function`

- **sage.misc.superseded.deprecation_cython**
  - Entity: `deprecation_cython`
  - Module: `sage.misc.superseded`
  - Type: `function`

- **sage.misc.superseded.experimental_warning**
  - Entity: `experimental_warning`
  - Module: `sage.misc.superseded`
  - Type: `function`

- **sage.misc.temporary_file.spyx_tmp**
  - Entity: `spyx_tmp`
  - Module: `sage.misc.temporary_file`
  - Type: `function`

- **sage.misc.temporary_file.tmp_dir**
  - Entity: `tmp_dir`
  - Module: `sage.misc.temporary_file`
  - Type: `function`

- **sage.misc.temporary_file.tmp_filename**
  - Entity: `tmp_filename`
  - Module: `sage.misc.temporary_file`
  - Type: `function`

- **sage.misc.trace.trace**
  - Entity: `trace`
  - Module: `sage.misc.trace`
  - Type: `function`

- **sage.misc.verbose.get_verbose**
  - Entity: `get_verbose`
  - Module: `sage.misc.verbose`
  - Type: `function`

- **sage.misc.verbose.get_verbose_files**
  - Entity: `get_verbose_files`
  - Module: `sage.misc.verbose`
  - Type: `function`

- **sage.misc.verbose.set_verbose**
  - Entity: `set_verbose`
  - Module: `sage.misc.verbose`
  - Type: `function`

- **sage.misc.verbose.set_verbose_files**
  - Entity: `set_verbose_files`
  - Module: `sage.misc.verbose`
  - Type: `function`

- **sage.misc.verbose.unset_verbose_files**
  - Entity: `unset_verbose_files`
  - Module: `sage.misc.verbose`
  - Type: `function`

- **sage.misc.verbose.verbose**
  - Entity: `verbose`
  - Module: `sage.misc.verbose`
  - Type: `function`

- **sage.misc.viewer.browser**
  - Entity: `browser`
  - Module: `sage.misc.viewer`
  - Type: `function`

- **sage.misc.viewer.default_viewer**
  - Entity: `default_viewer`
  - Module: `sage.misc.viewer`
  - Type: `function`

- **sage.misc.viewer.dvi_viewer**
  - Entity: `dvi_viewer`
  - Module: `sage.misc.viewer`
  - Type: `function`

- **sage.misc.viewer.pdf_viewer**
  - Entity: `pdf_viewer`
  - Module: `sage.misc.viewer`
  - Type: `function`

- **sage.misc.viewer.png_viewer**
  - Entity: `png_viewer`
  - Module: `sage.misc.viewer`
  - Type: `function`

- **sage.modular.abvar.abvar.factor_modsym_space_new_factors**
  - Entity: `factor_modsym_space_new_factors`
  - Module: `sage.modular.abvar.abvar`
  - Type: `function`

- **sage.modular.abvar.abvar.factor_new_space**
  - Entity: `factor_new_space`
  - Module: `sage.modular.abvar.abvar`
  - Type: `function`

- **sage.modular.abvar.abvar.is_ModularAbelianVariety**
  - Entity: `is_ModularAbelianVariety`
  - Module: `sage.modular.abvar.abvar`
  - Type: `function`

- **sage.modular.abvar.abvar.modsym_lattices**
  - Entity: `modsym_lattices`
  - Module: `sage.modular.abvar.abvar`
  - Type: `function`

- **sage.modular.abvar.abvar.random_hecke_operator**
  - Entity: `random_hecke_operator`
  - Module: `sage.modular.abvar.abvar`
  - Type: `function`

- **sage.modular.abvar.abvar.simple_factorization_of_modsym_space**
  - Entity: `simple_factorization_of_modsym_space`
  - Module: `sage.modular.abvar.abvar`
  - Type: `function`

- **sage.modular.abvar.abvar.sqrt_poly**
  - Entity: `sqrt_poly`
  - Module: `sage.modular.abvar.abvar`
  - Type: `function`

- **sage.modular.abvar.abvar_ambient_jacobian.ModAbVar_ambient_jacobian**
  - Entity: `ModAbVar_ambient_jacobian`
  - Module: `sage.modular.abvar.abvar_ambient_jacobian`
  - Type: `function`

- **sage.modular.abvar.constructor.AbelianVariety**
  - Entity: `AbelianVariety`
  - Module: `sage.modular.abvar.constructor`
  - Type: `function`

- **sage.modular.abvar.constructor.J0**
  - Entity: `J0`
  - Module: `sage.modular.abvar.constructor`
  - Type: `function`

- **sage.modular.abvar.constructor.J1**
  - Entity: `J1`
  - Module: `sage.modular.abvar.constructor`
  - Type: `function`

- **sage.modular.abvar.constructor.JH**
  - Entity: `JH`
  - Module: `sage.modular.abvar.constructor`
  - Type: `function`

- **sage.modular.abvar.cuspidal_subgroup.is_rational_cusp_gamma0**
  - Entity: `is_rational_cusp_gamma0`
  - Module: `sage.modular.abvar.cuspidal_subgroup`
  - Type: `function`

- **sage.modular.arithgroup.arithgroup_generic.is_ArithmeticSubgroup**
  - Entity: `is_ArithmeticSubgroup`
  - Module: `sage.modular.arithgroup.arithgroup_generic`
  - Type: `function`

- **sage.modular.arithgroup.arithgroup_perm.ArithmeticSubgroup_Permutation**
  - Entity: `ArithmeticSubgroup_Permutation`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `function`

- **sage.modular.arithgroup.arithgroup_perm.HsuExample10**
  - Entity: `HsuExample10`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `function`

- **sage.modular.arithgroup.arithgroup_perm.HsuExample18**
  - Entity: `HsuExample18`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `function`

- **sage.modular.arithgroup.arithgroup_perm.eval_sl2z_word**
  - Entity: `eval_sl2z_word`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `function`

- **sage.modular.arithgroup.arithgroup_perm.sl2z_word_problem**
  - Entity: `sl2z_word_problem`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `function`

- **sage.modular.arithgroup.arithgroup_perm.word_of_perms**
  - Entity: `word_of_perms`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `function`

- **sage.modular.arithgroup.congroup.degeneracy_coset_representatives_gamma0**
  - Entity: `degeneracy_coset_representatives_gamma0`
  - Module: `sage.modular.arithgroup.congroup`
  - Type: `function`

- **sage.modular.arithgroup.congroup.degeneracy_coset_representatives_gamma1**
  - Entity: `degeneracy_coset_representatives_gamma1`
  - Module: `sage.modular.arithgroup.congroup`
  - Type: `function`

- **sage.modular.arithgroup.congroup.generators_helper**
  - Entity: `generators_helper`
  - Module: `sage.modular.arithgroup.congroup`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gamma.Gamma_constructor**
  - Entity: `Gamma_constructor`
  - Module: `sage.modular.arithgroup.congroup_gamma`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gamma.is_Gamma**
  - Entity: `is_Gamma`
  - Module: `sage.modular.arithgroup.congroup_gamma`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gamma0.Gamma0_constructor**
  - Entity: `Gamma0_constructor`
  - Module: `sage.modular.arithgroup.congroup_gamma0`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gamma0.is_Gamma0**
  - Entity: `is_Gamma0`
  - Module: `sage.modular.arithgroup.congroup_gamma0`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gamma1.Gamma1_constructor**
  - Entity: `Gamma1_constructor`
  - Module: `sage.modular.arithgroup.congroup_gamma1`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gamma1.is_Gamma1**
  - Entity: `is_Gamma1`
  - Module: `sage.modular.arithgroup.congroup_gamma1`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gammaH.GammaH_constructor**
  - Entity: `GammaH_constructor`
  - Module: `sage.modular.arithgroup.congroup_gammaH`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gammaH.is_GammaH**
  - Entity: `is_GammaH`
  - Module: `sage.modular.arithgroup.congroup_gammaH`
  - Type: `function`

- **sage.modular.arithgroup.congroup_gammaH.mumu**
  - Entity: `mumu`
  - Module: `sage.modular.arithgroup.congroup_gammaH`
  - Type: `function`

- **sage.modular.arithgroup.congroup_generic.CongruenceSubgroup_constructor**
  - Entity: `CongruenceSubgroup_constructor`
  - Module: `sage.modular.arithgroup.congroup_generic`
  - Type: `function`

- **sage.modular.arithgroup.congroup_generic.is_CongruenceSubgroup**
  - Entity: `is_CongruenceSubgroup`
  - Module: `sage.modular.arithgroup.congroup_generic`
  - Type: `function`

- **sage.modular.arithgroup.congroup_sl2z.is_SL2Z**
  - Entity: `is_SL2Z`
  - Module: `sage.modular.arithgroup.congroup_sl2z`
  - Type: `function`

- **sage.modular.btquotients.pautomorphicform.eval_dist_at_powseries**
  - Entity: `eval_dist_at_powseries`
  - Module: `sage.modular.btquotients.pautomorphicform`
  - Type: `function`

- **sage.modular.buzzard.buzzard_tpslopes**
  - Entity: `buzzard_tpslopes`
  - Module: `sage.modular.buzzard`
  - Type: `function`

- **sage.modular.cusps_nf.Gamma0_NFCusps**
  - Entity: `Gamma0_NFCusps`
  - Module: `sage.modular.cusps_nf`
  - Type: `function`

- **sage.modular.cusps_nf.NFCusps**
  - Entity: `NFCusps`
  - Module: `sage.modular.cusps_nf`
  - Type: `function`

- **sage.modular.cusps_nf.NFCusps_ideal_reps_for_levelN**
  - Entity: `NFCusps_ideal_reps_for_levelN`
  - Module: `sage.modular.cusps_nf`
  - Type: `function`

- **sage.modular.cusps_nf.list_of_representatives**
  - Entity: `list_of_representatives`
  - Module: `sage.modular.cusps_nf`
  - Type: `function`

- **sage.modular.cusps_nf.number_of_Gamma0_NFCusps**
  - Entity: `number_of_Gamma0_NFCusps`
  - Module: `sage.modular.cusps_nf`
  - Type: `function`

- **sage.modular.cusps_nf.units_mod_ideal**
  - Entity: `units_mod_ideal`
  - Module: `sage.modular.cusps_nf`
  - Type: `function`

- **sage.modular.dims.CO_delta**
  - Entity: `CO_delta`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.CO_nu**
  - Entity: `CO_nu`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.CohenOesterle**
  - Entity: `CohenOesterle`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.dimension_cusp_forms**
  - Entity: `dimension_cusp_forms`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.dimension_eis**
  - Entity: `dimension_eis`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.dimension_modular_forms**
  - Entity: `dimension_modular_forms`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.dimension_new_cusp_forms**
  - Entity: `dimension_new_cusp_forms`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.eisen**
  - Entity: `eisen`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dims.sturm_bound**
  - Entity: `sturm_bound`
  - Module: `sage.modular.dims`
  - Type: `function`

- **sage.modular.dirichlet.TrivialCharacter**
  - Entity: `TrivialCharacter`
  - Module: `sage.modular.dirichlet`
  - Type: `function`

- **sage.modular.dirichlet.is_DirichletCharacter**
  - Entity: `is_DirichletCharacter`
  - Module: `sage.modular.dirichlet`
  - Type: `function`

- **sage.modular.dirichlet.is_DirichletGroup**
  - Entity: `is_DirichletGroup`
  - Module: `sage.modular.dirichlet`
  - Type: `function`

- **sage.modular.dirichlet.kronecker_character**
  - Entity: `kronecker_character`
  - Module: `sage.modular.dirichlet`
  - Type: `function`

- **sage.modular.dirichlet.kronecker_character_upside_down**
  - Entity: `kronecker_character_upside_down`
  - Module: `sage.modular.dirichlet`
  - Type: `function`

- **sage.modular.dirichlet.trivial_character**
  - Entity: `trivial_character`
  - Module: `sage.modular.dirichlet`
  - Type: `function`

- **sage.modular.etaproducts.AllCusps**
  - Entity: `AllCusps`
  - Module: `sage.modular.etaproducts`
  - Type: `function`

- **sage.modular.etaproducts.EtaGroup**
  - Entity: `EtaGroup`
  - Module: `sage.modular.etaproducts`
  - Type: `function`

- **sage.modular.etaproducts.EtaProduct**
  - Entity: `EtaProduct`
  - Module: `sage.modular.etaproducts`
  - Type: `function`

- **sage.modular.etaproducts.eta_poly_relations**
  - Entity: `eta_poly_relations`
  - Module: `sage.modular.etaproducts`
  - Type: `function`

- **sage.modular.etaproducts.num_cusps_of_width**
  - Entity: `num_cusps_of_width`
  - Module: `sage.modular.etaproducts`
  - Type: `function`

- **sage.modular.etaproducts.qexp_eta**
  - Entity: `qexp_eta`
  - Module: `sage.modular.etaproducts`
  - Type: `function`

- **sage.modular.hecke.algebra.is_HeckeAlgebra**
  - Entity: `is_HeckeAlgebra`
  - Module: `sage.modular.hecke.algebra`
  - Type: `function`

- **sage.modular.hecke.ambient_module.is_AmbientHeckeModule**
  - Entity: `is_AmbientHeckeModule`
  - Module: `sage.modular.hecke.ambient_module`
  - Type: `function`

- **sage.modular.hecke.element.is_HeckeModuleElement**
  - Entity: `is_HeckeModuleElement`
  - Module: `sage.modular.hecke.element`
  - Type: `function`

- **sage.modular.hecke.hecke_operator.is_HeckeAlgebraElement**
  - Entity: `is_HeckeAlgebraElement`
  - Module: `sage.modular.hecke.hecke_operator`
  - Type: `function`

- **sage.modular.hecke.hecke_operator.is_HeckeOperator**
  - Entity: `is_HeckeOperator`
  - Module: `sage.modular.hecke.hecke_operator`
  - Type: `function`

- **sage.modular.hecke.homspace.is_HeckeModuleHomspace**
  - Entity: `is_HeckeModuleHomspace`
  - Module: `sage.modular.hecke.homspace`
  - Type: `function`

- **sage.modular.hecke.module.is_HeckeModule**
  - Entity: `is_HeckeModule`
  - Module: `sage.modular.hecke.module`
  - Type: `function`

- **sage.modular.hecke.morphism.is_HeckeModuleMorphism**
  - Entity: `is_HeckeModuleMorphism`
  - Module: `sage.modular.hecke.morphism`
  - Type: `function`

- **sage.modular.hecke.morphism.is_HeckeModuleMorphism_matrix**
  - Entity: `is_HeckeModuleMorphism_matrix`
  - Module: `sage.modular.hecke.morphism`
  - Type: `function`

- **sage.modular.hecke.submodule.is_HeckeSubmodule**
  - Entity: `is_HeckeSubmodule`
  - Module: `sage.modular.hecke.submodule`
  - Type: `function`

- **sage.modular.hypergeometric_motive.alpha_to_cyclotomic**
  - Entity: `alpha_to_cyclotomic`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.hypergeometric_motive.capital_M**
  - Entity: `capital_M`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.hypergeometric_motive.characteristic_polynomial_from_traces**
  - Entity: `characteristic_polynomial_from_traces`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.hypergeometric_motive.cyclotomic_to_alpha**
  - Entity: `cyclotomic_to_alpha`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.hypergeometric_motive.cyclotomic_to_gamma**
  - Entity: `cyclotomic_to_gamma`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.hypergeometric_motive.enumerate_hypergeometric_data**
  - Entity: `enumerate_hypergeometric_data`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.hypergeometric_motive.gamma_list_to_cyclotomic**
  - Entity: `gamma_list_to_cyclotomic`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.hypergeometric_motive.possible_hypergeometric_data**
  - Entity: `possible_hypergeometric_data`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `function`

- **sage.modular.local_comp.liftings.lift_for_SL**
  - Entity: `lift_for_SL`
  - Module: `sage.modular.local_comp.liftings`
  - Type: `function`

- **sage.modular.local_comp.liftings.lift_gen_to_gamma1**
  - Entity: `lift_gen_to_gamma1`
  - Module: `sage.modular.local_comp.liftings`
  - Type: `function`

- **sage.modular.local_comp.liftings.lift_matrix_to_sl2z**
  - Entity: `lift_matrix_to_sl2z`
  - Module: `sage.modular.local_comp.liftings`
  - Type: `function`

- **sage.modular.local_comp.liftings.lift_ramified**
  - Entity: `lift_ramified`
  - Module: `sage.modular.local_comp.liftings`
  - Type: `function`

- **sage.modular.local_comp.liftings.lift_to_gamma1**
  - Entity: `lift_to_gamma1`
  - Module: `sage.modular.local_comp.liftings`
  - Type: `function`

- **sage.modular.local_comp.liftings.lift_uniformiser_odd**
  - Entity: `lift_uniformiser_odd`
  - Module: `sage.modular.local_comp.liftings`
  - Type: `function`

- **sage.modular.local_comp.local_comp.LocalComponent**
  - Entity: `LocalComponent`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `function`

- **sage.modular.local_comp.type_space.example_type_space**
  - Entity: `example_type_space`
  - Module: `sage.modular.local_comp.type_space`
  - Type: `function`

- **sage.modular.local_comp.type_space.find_in_space**
  - Entity: `find_in_space`
  - Module: `sage.modular.local_comp.type_space`
  - Type: `function`

- **sage.modular.modform.constructor.CuspForms**
  - Entity: `CuspForms`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.constructor.EisensteinForms**
  - Entity: `EisensteinForms`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.constructor.ModularForms**
  - Entity: `ModularForms`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.constructor.ModularForms_clear_cache**
  - Entity: `ModularForms_clear_cache`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.constructor.Newform**
  - Entity: `Newform`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.constructor.Newforms**
  - Entity: `Newforms`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.constructor.canonical_parameters**
  - Entity: `canonical_parameters`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.constructor.parse_label**
  - Entity: `parse_label`
  - Module: `sage.modular.modform.constructor`
  - Type: `function`

- **sage.modular.modform.eis_series.compute_eisenstein_params**
  - Entity: `compute_eisenstein_params`
  - Module: `sage.modular.modform.eis_series`
  - Type: `function`

- **sage.modular.modform.eis_series.eisenstein_series_lseries**
  - Entity: `eisenstein_series_lseries`
  - Module: `sage.modular.modform.eis_series`
  - Type: `function`

- **sage.modular.modform.eis_series.eisenstein_series_qexp**
  - Entity: `eisenstein_series_qexp`
  - Module: `sage.modular.modform.eis_series`
  - Type: `function`

- **sage.modular.modform.eis_series_cython.Ek_ZZ**
  - Entity: `Ek_ZZ`
  - Module: `sage.modular.modform.eis_series_cython`
  - Type: `function`

- **sage.modular.modform.eis_series_cython.eisenstein_series_poly**
  - Entity: `eisenstein_series_poly`
  - Module: `sage.modular.modform.eis_series_cython`
  - Type: `function`

- **sage.modular.modform.eisenstein_submodule.cyclotomic_restriction**
  - Entity: `cyclotomic_restriction`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `function`

- **sage.modular.modform.eisenstein_submodule.cyclotomic_restriction_tower**
  - Entity: `cyclotomic_restriction_tower`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `function`

- **sage.modular.modform.element.delta_lseries**
  - Entity: `delta_lseries`
  - Module: `sage.modular.modform.element`
  - Type: `function`

- **sage.modular.modform.element.is_ModularFormElement**
  - Entity: `is_ModularFormElement`
  - Module: `sage.modular.modform.element`
  - Type: `function`

- **sage.modular.modform.half_integral.half_integral_weight_modform_basis**
  - Entity: `half_integral_weight_modform_basis`
  - Module: `sage.modular.modform.half_integral`
  - Type: `function`

- **sage.modular.modform.hecke_operator_on_qexp.hecke_operator_on_basis**
  - Entity: `hecke_operator_on_basis`
  - Module: `sage.modular.modform.hecke_operator_on_qexp`
  - Type: `function`

- **sage.modular.modform.hecke_operator_on_qexp.hecke_operator_on_qexp**
  - Entity: `hecke_operator_on_qexp`
  - Module: `sage.modular.modform.hecke_operator_on_qexp`
  - Type: `function`

- **sage.modular.modform.j_invariant.j_invariant_qexp**
  - Entity: `j_invariant_qexp`
  - Module: `sage.modular.modform.j_invariant`
  - Type: `function`

- **sage.modular.modform.numerical.support**
  - Entity: `support`
  - Module: `sage.modular.modform.numerical`
  - Type: `function`

- **sage.modular.modform.space.contains_each**
  - Entity: `contains_each`
  - Module: `sage.modular.modform.space`
  - Type: `function`

- **sage.modular.modform.space.is_ModularFormsSpace**
  - Entity: `is_ModularFormsSpace`
  - Module: `sage.modular.modform.space`
  - Type: `function`

- **sage.modular.modform.theta.theta2_qexp**
  - Entity: `theta2_qexp`
  - Module: `sage.modular.modform.theta`
  - Type: `function`

- **sage.modular.modform.theta.theta_qexp**
  - Entity: `theta_qexp`
  - Module: `sage.modular.modform.theta`
  - Type: `function`

- **sage.modular.modform.vm_basis.delta_qexp**
  - Entity: `delta_qexp`
  - Module: `sage.modular.modform.vm_basis`
  - Type: `function`

- **sage.modular.modform.vm_basis.victor_miller_basis**
  - Entity: `victor_miller_basis`
  - Module: `sage.modular.modform.vm_basis`
  - Type: `function`

- **sage.modular.modform_hecketriangle.constructor.FormsRing**
  - Entity: `FormsRing`
  - Module: `sage.modular.modform_hecketriangle.constructor`
  - Type: `function`

- **sage.modular.modform_hecketriangle.constructor.FormsSpace**
  - Entity: `FormsSpace`
  - Module: `sage.modular.modform_hecketriangle.constructor`
  - Type: `function`

- **sage.modular.modform_hecketriangle.constructor.rational_type**
  - Entity: `rational_type`
  - Module: `sage.modular.modform_hecketriangle.constructor`
  - Type: `function`

- **sage.modular.modform_hecketriangle.functors.ConstantFormsSpaceFunctor**
  - Entity: `ConstantFormsSpaceFunctor`
  - Module: `sage.modular.modform_hecketriangle.functors`
  - Type: `function`

- **sage.modular.modform_hecketriangle.graded_ring.canonical_parameters**
  - Entity: `canonical_parameters`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `function`

- **sage.modular.modform_hecketriangle.hecke_triangle_group_element.coerce_AA**
  - Entity: `coerce_AA`
  - Module: `sage.modular.modform_hecketriangle.hecke_triangle_group_element`
  - Type: `function`

- **sage.modular.modform_hecketriangle.hecke_triangle_group_element.cyclic_representative**
  - Entity: `cyclic_representative`
  - Module: `sage.modular.modform_hecketriangle.hecke_triangle_group_element`
  - Type: `function`

- **sage.modular.modform_hecketriangle.space.canonical_parameters**
  - Entity: `canonical_parameters`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `function`

- **sage.modular.modform_hecketriangle.subspace.ModularFormsSubSpace**
  - Entity: `ModularFormsSubSpace`
  - Module: `sage.modular.modform_hecketriangle.subspace`
  - Type: `function`

- **sage.modular.modform_hecketriangle.subspace.canonical_parameters**
  - Entity: `canonical_parameters`
  - Module: `sage.modular.modform_hecketriangle.subspace`
  - Type: `function`

- **sage.modular.modsym.apply.apply_to_monomial**
  - Entity: `apply_to_monomial`
  - Module: `sage.modular.modsym.apply`
  - Type: `function`

- **sage.modular.modsym.element.is_ModularSymbolsElement**
  - Entity: `is_ModularSymbolsElement`
  - Module: `sage.modular.modsym.element`
  - Type: `function`

- **sage.modular.modsym.element.set_modsym_print_mode**
  - Entity: `set_modsym_print_mode`
  - Module: `sage.modular.modsym.element`
  - Type: `function`

- **sage.modular.modsym.heilbronn.hecke_images_gamma0_weight2**
  - Entity: `hecke_images_gamma0_weight2`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `function`

- **sage.modular.modsym.heilbronn.hecke_images_gamma0_weight_k**
  - Entity: `hecke_images_gamma0_weight_k`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `function`

- **sage.modular.modsym.heilbronn.hecke_images_nonquad_character_weight2**
  - Entity: `hecke_images_nonquad_character_weight2`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `function`

- **sage.modular.modsym.heilbronn.hecke_images_quad_character_weight2**
  - Entity: `hecke_images_quad_character_weight2`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `function`

- **sage.modular.modsym.manin_symbol.is_ManinSymbol**
  - Entity: `is_ManinSymbol`
  - Module: `sage.modular.modsym.manin_symbol`
  - Type: `function`

- **sage.modular.modsym.modsym.ModularSymbols**
  - Entity: `ModularSymbols`
  - Module: `sage.modular.modsym.modsym`
  - Type: `function`

- **sage.modular.modsym.modsym.ModularSymbols_clear_cache**
  - Entity: `ModularSymbols_clear_cache`
  - Module: `sage.modular.modsym.modsym`
  - Type: `function`

- **sage.modular.modsym.modsym.canonical_parameters**
  - Entity: `canonical_parameters`
  - Module: `sage.modular.modsym.modsym`
  - Type: `function`

- **sage.modular.modsym.p1list.lift_to_sl2z**
  - Entity: `lift_to_sl2z`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.lift_to_sl2z_int**
  - Entity: `lift_to_sl2z_int`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.lift_to_sl2z_llong**
  - Entity: `lift_to_sl2z_llong`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.p1_normalize**
  - Entity: `p1_normalize`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.p1_normalize_int**
  - Entity: `p1_normalize_int`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.p1_normalize_llong**
  - Entity: `p1_normalize_llong`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.p1list**
  - Entity: `p1list`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.p1list_int**
  - Entity: `p1list_int`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list.p1list_llong**
  - Entity: `p1list_llong`
  - Module: `sage.modular.modsym.p1list`
  - Type: `function`

- **sage.modular.modsym.p1list_nf.P1NFList_clear_level_cache**
  - Entity: `P1NFList_clear_level_cache`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `function`

- **sage.modular.modsym.p1list_nf.lift_to_sl2_Ok**
  - Entity: `lift_to_sl2_Ok`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `function`

- **sage.modular.modsym.p1list_nf.make_coprime**
  - Entity: `make_coprime`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `function`

- **sage.modular.modsym.p1list_nf.p1NFlist**
  - Entity: `p1NFlist`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `function`

- **sage.modular.modsym.p1list_nf.psi**
  - Entity: `psi`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `function`

- **sage.modular.modsym.relation_matrix.T_relation_matrix_wtk_g0**
  - Entity: `T_relation_matrix_wtk_g0`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `function`

- **sage.modular.modsym.relation_matrix.compute_presentation**
  - Entity: `compute_presentation`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `function`

- **sage.modular.modsym.relation_matrix.gens_to_basis_matrix**
  - Entity: `gens_to_basis_matrix`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `function`

- **sage.modular.modsym.relation_matrix.modI_relations**
  - Entity: `modI_relations`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `function`

- **sage.modular.modsym.relation_matrix.modS_relations**
  - Entity: `modS_relations`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `function`

- **sage.modular.modsym.relation_matrix.relation_matrix_wtk_g0**
  - Entity: `relation_matrix_wtk_g0`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `function`

- **sage.modular.modsym.relation_matrix.sparse_2term_quotient**
  - Entity: `sparse_2term_quotient`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `function`

- **sage.modular.modsym.relation_matrix_pyx.sparse_2term_quotient_only_pm1**
  - Entity: `sparse_2term_quotient_only_pm1`
  - Module: `sage.modular.modsym.relation_matrix_pyx`
  - Type: `function`

- **sage.modular.modsym.space.is_ModularSymbolsSpace**
  - Entity: `is_ModularSymbolsSpace`
  - Module: `sage.modular.modsym.space`
  - Type: `function`

- **sage.modular.multiple_zeta.D_on_compo**
  - Entity: `D_on_compo`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.Multizeta**
  - Entity: `Multizeta`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.coeff_phi**
  - Entity: `coeff_phi`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.composition_to_iterated**
  - Entity: `composition_to_iterated`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.compute_u_on_basis**
  - Entity: `compute_u_on_basis`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.compute_u_on_compo**
  - Entity: `compute_u_on_compo`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.coproduct_iterator**
  - Entity: `coproduct_iterator`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.dual_composition**
  - Entity: `dual_composition`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.extend_multiplicative_basis**
  - Entity: `extend_multiplicative_basis`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.iterated_to_composition**
  - Entity: `iterated_to_composition`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.minimize_term**
  - Entity: `minimize_term`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.phi_on_basis**
  - Entity: `phi_on_basis`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.phi_on_multiplicative_basis**
  - Entity: `phi_on_multiplicative_basis`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.rho_inverse**
  - Entity: `rho_inverse`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.multiple_zeta.rho_matrix_inverse**
  - Entity: `rho_matrix_inverse`
  - Module: `sage.modular.multiple_zeta`
  - Type: `function`

- **sage.modular.overconvergent.genus0.OverconvergentModularForms**
  - Entity: `OverconvergentModularForms`
  - Module: `sage.modular.overconvergent.genus0`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.complementary_spaces**
  - Entity: `complementary_spaces`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.complementary_spaces_modp**
  - Entity: `complementary_spaces_modp`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.compute_G**
  - Entity: `compute_G`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.compute_Wi**
  - Entity: `compute_Wi`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.compute_elldash**
  - Entity: `compute_elldash`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.ech_form**
  - Entity: `ech_form`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.hecke_series**
  - Entity: `hecke_series`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.hecke_series_degree_bound**
  - Entity: `hecke_series_degree_bound`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.higher_level_UpGj**
  - Entity: `higher_level_UpGj`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.higher_level_katz_exp**
  - Entity: `higher_level_katz_exp`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.is_valid_weight_list**
  - Entity: `is_valid_weight_list`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.katz_expansions**
  - Entity: `katz_expansions`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.level1_UpGj**
  - Entity: `level1_UpGj`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.low_weight_bases**
  - Entity: `low_weight_bases`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.low_weight_generators**
  - Entity: `low_weight_generators`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.random_low_weight_bases**
  - Entity: `random_low_weight_bases`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.random_new_basis_modp**
  - Entity: `random_new_basis_modp`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.hecke_series.random_solution**
  - Entity: `random_solution`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `function`

- **sage.modular.overconvergent.weightspace.WeightSpace_constructor**
  - Entity: `WeightSpace_constructor`
  - Module: `sage.modular.overconvergent.weightspace`
  - Type: `function`

- **sage.modular.pollack_stevens.fund_domain.M2Z**
  - Entity: `M2Z`
  - Module: `sage.modular.pollack_stevens.fund_domain`
  - Type: `function`

- **sage.modular.pollack_stevens.fund_domain.basic_hecke_matrix**
  - Entity: `basic_hecke_matrix`
  - Module: `sage.modular.pollack_stevens.fund_domain`
  - Type: `function`

- **sage.modular.pollack_stevens.manin_map.unimod_matrices_from_infty**
  - Entity: `unimod_matrices_from_infty`
  - Module: `sage.modular.pollack_stevens.manin_map`
  - Type: `function`

- **sage.modular.pollack_stevens.manin_map.unimod_matrices_to_infty**
  - Entity: `unimod_matrices_to_infty`
  - Module: `sage.modular.pollack_stevens.manin_map`
  - Type: `function`

- **sage.modular.pollack_stevens.padic_lseries.log_gamma_binomial**
  - Entity: `log_gamma_binomial`
  - Module: `sage.modular.pollack_stevens.padic_lseries`
  - Type: `function`

- **sage.modular.pollack_stevens.space.cusps_from_mat**
  - Entity: `cusps_from_mat`
  - Module: `sage.modular.pollack_stevens.space`
  - Type: `function`

- **sage.modular.pollack_stevens.space.ps_modsym_from_elliptic_curve**
  - Entity: `ps_modsym_from_elliptic_curve`
  - Module: `sage.modular.pollack_stevens.space`
  - Type: `function`

- **sage.modular.pollack_stevens.space.ps_modsym_from_simple_modsym_space**
  - Entity: `ps_modsym_from_simple_modsym_space`
  - Module: `sage.modular.pollack_stevens.space`
  - Type: `function`

- **sage.modular.quatalg.brandt.BrandtModule**
  - Entity: `BrandtModule`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.quatalg.brandt.basis_for_left_ideal**
  - Entity: `basis_for_left_ideal`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.quatalg.brandt.benchmark_magma**
  - Entity: `benchmark_magma`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.quatalg.brandt.benchmark_sage**
  - Entity: `benchmark_sage`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.quatalg.brandt.class_number**
  - Entity: `class_number`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.quatalg.brandt.maximal_order**
  - Entity: `maximal_order`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.quatalg.brandt.quaternion_order_with_given_level**
  - Entity: `quaternion_order_with_given_level`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.quatalg.brandt.right_order**
  - Entity: `right_order`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `function`

- **sage.modular.ssmod.ssmod.Phi2_quad**
  - Entity: `Phi2_quad`
  - Module: `sage.modular.ssmod.ssmod`
  - Type: `function`

- **sage.modular.ssmod.ssmod.Phi_polys**
  - Entity: `Phi_polys`
  - Module: `sage.modular.ssmod.ssmod`
  - Type: `function`

- **sage.modular.ssmod.ssmod.dimension_supersingular_module**
  - Entity: `dimension_supersingular_module`
  - Module: `sage.modular.ssmod.ssmod`
  - Type: `function`

- **sage.modular.ssmod.ssmod.supersingular_D**
  - Entity: `supersingular_D`
  - Module: `sage.modular.ssmod.ssmod`
  - Type: `function`

- **sage.modular.ssmod.ssmod.supersingular_j**
  - Entity: `supersingular_j`
  - Module: `sage.modular.ssmod.ssmod`
  - Type: `function`

- **sage.modules.diamond_cutting.calculate_voronoi_cell**
  - Entity: `calculate_voronoi_cell`
  - Module: `sage.modules.diamond_cutting`
  - Type: `function`

- **sage.modules.diamond_cutting.diamond_cut**
  - Entity: `diamond_cut`
  - Module: `sage.modules.diamond_cutting`
  - Type: `function`

- **sage.modules.diamond_cutting.jacobi**
  - Entity: `jacobi`
  - Module: `sage.modules.diamond_cutting`
  - Type: `function`

- **sage.modules.diamond_cutting.plane_inequality**
  - Entity: `plane_inequality`
  - Module: `sage.modules.diamond_cutting`
  - Type: `function`

- **sage.modules.fg_pid.fgp_module.FGP_Module**
  - Entity: `FGP_Module`
  - Module: `sage.modules.fg_pid.fgp_module`
  - Type: `function`

- **sage.modules.fg_pid.fgp_module.is_FGP_Module**
  - Entity: `is_FGP_Module`
  - Module: `sage.modules.fg_pid.fgp_module`
  - Type: `function`

- **sage.modules.fg_pid.fgp_module.random_fgp_module**
  - Entity: `random_fgp_module`
  - Module: `sage.modules.fg_pid.fgp_module`
  - Type: `function`

- **sage.modules.fg_pid.fgp_module.random_fgp_morphism_0**
  - Entity: `random_fgp_morphism_0`
  - Module: `sage.modules.fg_pid.fgp_module`
  - Type: `function`

- **sage.modules.fg_pid.fgp_morphism.FGP_Homset**
  - Entity: `FGP_Homset`
  - Module: `sage.modules.fg_pid.fgp_morphism`
  - Type: `function`

- **sage.modules.filtered_vector_space.FilteredVectorSpace**
  - Entity: `FilteredVectorSpace`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `function`

- **sage.modules.filtered_vector_space.construct_from_dim_degree**
  - Entity: `construct_from_dim_degree`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `function`

- **sage.modules.filtered_vector_space.construct_from_generators**
  - Entity: `construct_from_generators`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `function`

- **sage.modules.filtered_vector_space.construct_from_generators_indices**
  - Entity: `construct_from_generators_indices`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `function`

- **sage.modules.filtered_vector_space.is_FilteredVectorSpace**
  - Entity: `is_FilteredVectorSpace`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `function`

- **sage.modules.filtered_vector_space.normalize_degree**
  - Entity: `normalize_degree`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `function`

- **sage.modules.free_module.FreeModule**
  - Entity: `FreeModule`
  - Module: `sage.modules.free_module`
  - Type: `function`

- **sage.modules.free_module.VectorSpace**
  - Entity: `VectorSpace`
  - Module: `sage.modules.free_module`
  - Type: `function`

- **sage.modules.free_module.basis_seq**
  - Entity: `basis_seq`
  - Module: `sage.modules.free_module`
  - Type: `function`

- **sage.modules.free_module.element_class**
  - Entity: `element_class`
  - Module: `sage.modules.free_module`
  - Type: `function`

- **sage.modules.free_module.is_FreeModule**
  - Entity: `is_FreeModule`
  - Module: `sage.modules.free_module`
  - Type: `function`

- **sage.modules.free_module.span**
  - Entity: `span`
  - Module: `sage.modules.free_module`
  - Type: `function`

- **sage.modules.free_module_element.free_module_element**
  - Entity: `free_module_element`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.is_FreeModuleElement**
  - Entity: `is_FreeModuleElement`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.make_FreeModuleElement_generic_dense**
  - Entity: `make_FreeModuleElement_generic_dense`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.make_FreeModuleElement_generic_dense_v1**
  - Entity: `make_FreeModuleElement_generic_dense_v1`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.make_FreeModuleElement_generic_sparse**
  - Entity: `make_FreeModuleElement_generic_sparse`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.make_FreeModuleElement_generic_sparse_v1**
  - Entity: `make_FreeModuleElement_generic_sparse_v1`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.prepare**
  - Entity: `prepare`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.random_vector**
  - Entity: `random_vector`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.vector**
  - Entity: `vector`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_element.zero_vector**
  - Entity: `zero_vector`
  - Module: `sage.modules.free_module_element`
  - Type: `function`

- **sage.modules.free_module_homspace.is_FreeModuleHomspace**
  - Entity: `is_FreeModuleHomspace`
  - Module: `sage.modules.free_module_homspace`
  - Type: `function`

- **sage.modules.free_module_integer.IntegerLattice**
  - Entity: `IntegerLattice`
  - Module: `sage.modules.free_module_integer`
  - Type: `function`

- **sage.modules.free_module_morphism.is_FreeModuleMorphism**
  - Entity: `is_FreeModuleMorphism`
  - Module: `sage.modules.free_module_morphism`
  - Type: `function`

- **sage.modules.free_quadratic_module.FreeQuadraticModule**
  - Entity: `FreeQuadraticModule`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `function`

- **sage.modules.free_quadratic_module.InnerProductSpace**
  - Entity: `InnerProductSpace`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `function`

- **sage.modules.free_quadratic_module.QuadraticSpace**
  - Entity: `QuadraticSpace`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `function`

- **sage.modules.free_quadratic_module.is_FreeQuadraticModule**
  - Entity: `is_FreeQuadraticModule`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `function`

- **sage.modules.free_quadratic_module_integer_symmetric.IntegralLattice**
  - Entity: `IntegralLattice`
  - Module: `sage.modules.free_quadratic_module_integer_symmetric`
  - Type: `function`

- **sage.modules.free_quadratic_module_integer_symmetric.IntegralLatticeDirectSum**
  - Entity: `IntegralLatticeDirectSum`
  - Module: `sage.modules.free_quadratic_module_integer_symmetric`
  - Type: `function`

- **sage.modules.free_quadratic_module_integer_symmetric.IntegralLatticeGluing**
  - Entity: `IntegralLatticeGluing`
  - Module: `sage.modules.free_quadratic_module_integer_symmetric`
  - Type: `function`

- **sage.modules.free_quadratic_module_integer_symmetric.local_modification**
  - Entity: `local_modification`
  - Module: `sage.modules.free_quadratic_module_integer_symmetric`
  - Type: `function`

- **sage.modules.matrix_morphism.is_MatrixMorphism**
  - Entity: `is_MatrixMorphism`
  - Module: `sage.modules.matrix_morphism`
  - Type: `function`

- **sage.modules.misc.gram_schmidt**
  - Entity: `gram_schmidt`
  - Module: `sage.modules.misc`
  - Type: `function`

- **sage.modules.module.is_Module**
  - Entity: `is_Module`
  - Module: `sage.modules.module`
  - Type: `function`

- **sage.modules.module.is_VectorSpace**
  - Entity: `is_VectorSpace`
  - Module: `sage.modules.module`
  - Type: `function`

- **sage.modules.multi_filtered_vector_space.MultiFilteredVectorSpace**
  - Entity: `MultiFilteredVectorSpace`
  - Module: `sage.modules.multi_filtered_vector_space`
  - Type: `function`

- **sage.modules.ore_module.normalize_names**
  - Entity: `normalize_names`
  - Module: `sage.modules.ore_module`
  - Type: `function`

- **sage.modules.tensor_operations.antisymmetrized_coordinate_sums**
  - Entity: `antisymmetrized_coordinate_sums`
  - Module: `sage.modules.tensor_operations`
  - Type: `function`

- **sage.modules.tensor_operations.symmetrized_coordinate_sums**
  - Entity: `symmetrized_coordinate_sums`
  - Module: `sage.modules.tensor_operations`
  - Type: `function`

- **sage.modules.torsion_quadratic_module.TorsionQuadraticForm**
  - Entity: `TorsionQuadraticForm`
  - Module: `sage.modules.torsion_quadratic_module`
  - Type: `function`

- **sage.modules.vector_complex_double_dense.unpickle_v0**
  - Entity: `unpickle_v0`
  - Module: `sage.modules.vector_complex_double_dense`
  - Type: `function`

- **sage.modules.vector_complex_double_dense.unpickle_v1**
  - Entity: `unpickle_v1`
  - Module: `sage.modules.vector_complex_double_dense`
  - Type: `function`

- **sage.modules.vector_integer_dense.unpickle_v0**
  - Entity: `unpickle_v0`
  - Module: `sage.modules.vector_integer_dense`
  - Type: `function`

- **sage.modules.vector_integer_dense.unpickle_v1**
  - Entity: `unpickle_v1`
  - Module: `sage.modules.vector_integer_dense`
  - Type: `function`

- **sage.modules.vector_mod2_dense.unpickle_v0**
  - Entity: `unpickle_v0`
  - Module: `sage.modules.vector_mod2_dense`
  - Type: `function`

- **sage.modules.vector_modn_dense.unpickle_v0**
  - Entity: `unpickle_v0`
  - Module: `sage.modules.vector_modn_dense`
  - Type: `function`

- **sage.modules.vector_modn_dense.unpickle_v1**
  - Entity: `unpickle_v1`
  - Module: `sage.modules.vector_modn_dense`
  - Type: `function`

- **sage.modules.vector_rational_dense.unpickle_v0**
  - Entity: `unpickle_v0`
  - Module: `sage.modules.vector_rational_dense`
  - Type: `function`

- **sage.modules.vector_rational_dense.unpickle_v1**
  - Entity: `unpickle_v1`
  - Module: `sage.modules.vector_rational_dense`
  - Type: `function`

- **sage.modules.vector_real_double_dense.unpickle_v0**
  - Entity: `unpickle_v0`
  - Module: `sage.modules.vector_real_double_dense`
  - Type: `function`

- **sage.modules.vector_real_double_dense.unpickle_v1**
  - Entity: `unpickle_v1`
  - Module: `sage.modules.vector_real_double_dense`
  - Type: `function`

- **sage.modules.vector_space_homspace.is_VectorSpaceHomspace**
  - Entity: `is_VectorSpaceHomspace`
  - Module: `sage.modules.vector_space_homspace`
  - Type: `function`

- **sage.modules.vector_space_morphism.is_VectorSpaceMorphism**
  - Entity: `is_VectorSpaceMorphism`
  - Module: `sage.modules.vector_space_morphism`
  - Type: `function`

- **sage.modules.vector_space_morphism.linear_transformation**
  - Entity: `linear_transformation`
  - Module: `sage.modules.vector_space_morphism`
  - Type: `function`

- **sage.modules.vector_symbolic_dense.apply_map**
  - Entity: `apply_map`
  - Module: `sage.modules.vector_symbolic_dense`
  - Type: `function`

- **sage.modules.vector_symbolic_sparse.apply_map**
  - Entity: `apply_map`
  - Module: `sage.modules.vector_symbolic_sparse`
  - Type: `function`

- **sage.modules.with_basis.morphism.pointwise_inverse_function**
  - Entity: `pointwise_inverse_function`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `function`

- **sage.monoids.free_abelian_monoid.FreeAbelianMonoid**
  - Entity: `FreeAbelianMonoid`
  - Module: `sage.monoids.free_abelian_monoid`
  - Type: `function`

- **sage.monoids.free_abelian_monoid.is_FreeAbelianMonoid**
  - Entity: `is_FreeAbelianMonoid`
  - Module: `sage.monoids.free_abelian_monoid`
  - Type: `function`

- **sage.monoids.free_abelian_monoid_element.is_FreeAbelianMonoidElement**
  - Entity: `is_FreeAbelianMonoidElement`
  - Module: `sage.monoids.free_abelian_monoid_element`
  - Type: `function`

- **sage.monoids.free_monoid.is_FreeMonoid**
  - Entity: `is_FreeMonoid`
  - Module: `sage.monoids.free_monoid`
  - Type: `function`

- **sage.monoids.free_monoid_element.is_FreeMonoidElement**
  - Entity: `is_FreeMonoidElement`
  - Module: `sage.monoids.free_monoid_element`
  - Type: `function`

- **sage.monoids.hecke_monoid.HeckeMonoid**
  - Entity: `HeckeMonoid`
  - Module: `sage.monoids.hecke_monoid`
  - Type: `function`

- **sage.monoids.monoid.is_Monoid**
  - Entity: `is_Monoid`
  - Module: `sage.monoids.monoid`
  - Type: `function`

- **sage.monoids.string_monoid_element.is_AlphabeticStringMonoidElement**
  - Entity: `is_AlphabeticStringMonoidElement`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `function`

- **sage.monoids.string_monoid_element.is_BinaryStringMonoidElement**
  - Entity: `is_BinaryStringMonoidElement`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `function`

- **sage.monoids.string_monoid_element.is_HexadecimalStringMonoidElement**
  - Entity: `is_HexadecimalStringMonoidElement`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `function`

- **sage.monoids.string_monoid_element.is_OctalStringMonoidElement**
  - Entity: `is_OctalStringMonoidElement`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `function`

- **sage.monoids.string_monoid_element.is_Radix64StringMonoidElement**
  - Entity: `is_Radix64StringMonoidElement`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `function`

- **sage.monoids.string_monoid_element.is_StringMonoidElement**
  - Entity: `is_StringMonoidElement`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `function`

- **sage.monoids.string_ops.coincidence_discriminant**
  - Entity: `coincidence_discriminant`
  - Module: `sage.monoids.string_ops`
  - Type: `function`

- **sage.monoids.string_ops.coincidence_index**
  - Entity: `coincidence_index`
  - Module: `sage.monoids.string_ops`
  - Type: `function`

- **sage.monoids.string_ops.frequency_distribution**
  - Entity: `frequency_distribution`
  - Module: `sage.monoids.string_ops`
  - Type: `function`

- **sage.monoids.string_ops.strip_encoding**
  - Entity: `strip_encoding`
  - Module: `sage.monoids.string_ops`
  - Type: `function`

- **sage.numerical.backends.generic_backend.default_mip_solver**
  - Entity: `default_mip_solver`
  - Module: `sage.numerical.backends.generic_backend`
  - Type: `function`

- **sage.numerical.backends.generic_backend.get_solver**
  - Entity: `get_solver`
  - Module: `sage.numerical.backends.generic_backend`
  - Type: `function`

- **sage.numerical.backends.generic_sdp_backend.default_sdp_solver**
  - Entity: `default_sdp_solver`
  - Module: `sage.numerical.backends.generic_sdp_backend`
  - Type: `function`

- **sage.numerical.backends.generic_sdp_backend.get_solver**
  - Entity: `get_solver`
  - Module: `sage.numerical.backends.generic_sdp_backend`
  - Type: `function`

#### MODULE (215 entries)

- **sage.misc.sagedoc**
  - Entity: `sagedoc`
  - Module: `sage.misc.sagedoc`
  - Type: `module`

- **sage.misc.sagedoc_conf**
  - Entity: `sagedoc_conf`
  - Module: `sage.misc.sagedoc_conf`
  - Type: `module`

- **sage.misc.sageinspect**
  - Entity: `sageinspect`
  - Module: `sage.misc.sageinspect`
  - Type: `module`

- **sage.misc.search**
  - Entity: `search`
  - Module: `sage.misc.search`
  - Type: `module`

- **sage.misc.session**
  - Entity: `session`
  - Module: `sage.misc.session`
  - Type: `module`

- **sage.misc.sh**
  - Entity: `sh`
  - Module: `sage.misc.sh`
  - Type: `module`

- **sage.misc.sphinxify**
  - Entity: `sphinxify`
  - Module: `sage.misc.sphinxify`
  - Type: `module`

- **sage.misc.stopgap**
  - Entity: `stopgap`
  - Module: `sage.misc.stopgap`
  - Type: `module`

- **sage.misc.superseded**
  - Entity: `superseded`
  - Module: `sage.misc.superseded`
  - Type: `module`

- **sage.misc.table**
  - Entity: `table`
  - Module: `sage.misc.table`
  - Type: `module`

- **sage.misc.temporary_file**
  - Entity: `temporary_file`
  - Module: `sage.misc.temporary_file`
  - Type: `module`

- **sage.misc.test_nested_class**
  - Entity: `test_nested_class`
  - Module: `sage.misc.test_nested_class`
  - Type: `module`

- **sage.misc.trace**
  - Entity: `trace`
  - Module: `sage.misc.trace`
  - Type: `module`

- **sage.misc.unknown**
  - Entity: `unknown`
  - Module: `sage.misc.unknown`
  - Type: `module`

- **sage.misc.verbose**
  - Entity: `verbose`
  - Module: `sage.misc.verbose`
  - Type: `module`

- **sage.misc.viewer**
  - Entity: `viewer`
  - Module: `sage.misc.viewer`
  - Type: `module`

- **sage.misc.weak_dict**
  - Entity: `weak_dict`
  - Module: `sage.misc.weak_dict`
  - Type: `module`

- **sage.modular**
  - Entity: `modular`
  - Module: `sage.modular`
  - Type: `module`

- **sage.modular.abvar.abvar**
  - Entity: `abvar`
  - Module: `sage.modular.abvar.abvar`
  - Type: `module`

- **sage.modular.abvar.abvar_ambient_jacobian**
  - Entity: `abvar_ambient_jacobian`
  - Module: `sage.modular.abvar.abvar_ambient_jacobian`
  - Type: `module`

- **sage.modular.abvar.abvar_newform**
  - Entity: `abvar_newform`
  - Module: `sage.modular.abvar.abvar_newform`
  - Type: `module`

- **sage.modular.abvar.constructor**
  - Entity: `constructor`
  - Module: `sage.modular.abvar.constructor`
  - Type: `module`

- **sage.modular.abvar.cuspidal_subgroup**
  - Entity: `cuspidal_subgroup`
  - Module: `sage.modular.abvar.cuspidal_subgroup`
  - Type: `module`

- **sage.modular.abvar.finite_subgroup**
  - Entity: `finite_subgroup`
  - Module: `sage.modular.abvar.finite_subgroup`
  - Type: `module`

- **sage.modular.abvar.homology**
  - Entity: `homology`
  - Module: `sage.modular.abvar.homology`
  - Type: `module`

- **sage.modular.abvar.homspace**
  - Entity: `homspace`
  - Module: `sage.modular.abvar.homspace`
  - Type: `module`

- **sage.modular.abvar.lseries**
  - Entity: `lseries`
  - Module: `sage.modular.abvar.lseries`
  - Type: `module`

- **sage.modular.abvar.morphism**
  - Entity: `morphism`
  - Module: `sage.modular.abvar.morphism`
  - Type: `module`

- **sage.modular.abvar.torsion_subgroup**
  - Entity: `torsion_subgroup`
  - Module: `sage.modular.abvar.torsion_subgroup`
  - Type: `module`

- **sage.modular.arithgroup.arithgroup_element**
  - Entity: `arithgroup_element`
  - Module: `sage.modular.arithgroup.arithgroup_element`
  - Type: `module`

- **sage.modular.arithgroup.arithgroup_generic**
  - Entity: `arithgroup_generic`
  - Module: `sage.modular.arithgroup.arithgroup_generic`
  - Type: `module`

- **sage.modular.arithgroup.arithgroup_perm**
  - Entity: `arithgroup_perm`
  - Module: `sage.modular.arithgroup.arithgroup_perm`
  - Type: `module`

- **sage.modular.arithgroup.congroup**
  - Entity: `congroup`
  - Module: `sage.modular.arithgroup.congroup`
  - Type: `module`

- **sage.modular.arithgroup.congroup_gamma**
  - Entity: `congroup_gamma`
  - Module: `sage.modular.arithgroup.congroup_gamma`
  - Type: `module`

- **sage.modular.arithgroup.congroup_gamma0**
  - Entity: `congroup_gamma0`
  - Module: `sage.modular.arithgroup.congroup_gamma0`
  - Type: `module`

- **sage.modular.arithgroup.congroup_gamma1**
  - Entity: `congroup_gamma1`
  - Module: `sage.modular.arithgroup.congroup_gamma1`
  - Type: `module`

- **sage.modular.arithgroup.congroup_gammaH**
  - Entity: `congroup_gammaH`
  - Module: `sage.modular.arithgroup.congroup_gammaH`
  - Type: `module`

- **sage.modular.arithgroup.congroup_generic**
  - Entity: `congroup_generic`
  - Module: `sage.modular.arithgroup.congroup_generic`
  - Type: `module`

- **sage.modular.arithgroup.congroup_sl2z**
  - Entity: `congroup_sl2z`
  - Module: `sage.modular.arithgroup.congroup_sl2z`
  - Type: `module`

- **sage.modular.arithgroup.farey_symbol**
  - Entity: `farey_symbol`
  - Module: `sage.modular.arithgroup.farey_symbol`
  - Type: `module`

- **sage.modular.btquotients.btquotient**
  - Entity: `btquotient`
  - Module: `sage.modular.btquotients.btquotient`
  - Type: `module`

- **sage.modular.btquotients.pautomorphicform**
  - Entity: `pautomorphicform`
  - Module: `sage.modular.btquotients.pautomorphicform`
  - Type: `module`

- **sage.modular.buzzard**
  - Entity: `buzzard`
  - Module: `sage.modular.buzzard`
  - Type: `module`

- **sage.modular.cusps**
  - Entity: `cusps`
  - Module: `sage.modular.cusps`
  - Type: `module`

- **sage.modular.cusps_nf**
  - Entity: `cusps_nf`
  - Module: `sage.modular.cusps_nf`
  - Type: `module`

- **sage.modular.dims**
  - Entity: `dims`
  - Module: `sage.modular.dims`
  - Type: `module`

- **sage.modular.dirichlet**
  - Entity: `dirichlet`
  - Module: `sage.modular.dirichlet`
  - Type: `module`

- **sage.modular.drinfeld_modform.element**
  - Entity: `element`
  - Module: `sage.modular.drinfeld_modform.element`
  - Type: `module`

- **sage.modular.drinfeld_modform.ring**
  - Entity: `ring`
  - Module: `sage.modular.drinfeld_modform.ring`
  - Type: `module`

- **sage.modular.drinfeld_modform.tutorial**
  - Entity: `tutorial`
  - Module: `sage.modular.drinfeld_modform.tutorial`
  - Type: `module`

- **sage.modular.etaproducts**
  - Entity: `etaproducts`
  - Module: `sage.modular.etaproducts`
  - Type: `module`

- **sage.modular.hecke.algebra**
  - Entity: `algebra`
  - Module: `sage.modular.hecke.algebra`
  - Type: `module`

- **sage.modular.hecke.ambient_module**
  - Entity: `ambient_module`
  - Module: `sage.modular.hecke.ambient_module`
  - Type: `module`

- **sage.modular.hecke.degenmap**
  - Entity: `degenmap`
  - Module: `sage.modular.hecke.degenmap`
  - Type: `module`

- **sage.modular.hecke.element**
  - Entity: `element`
  - Module: `sage.modular.hecke.element`
  - Type: `module`

- **sage.modular.hecke.hecke_operator**
  - Entity: `hecke_operator`
  - Module: `sage.modular.hecke.hecke_operator`
  - Type: `module`

- **sage.modular.hecke.homspace**
  - Entity: `homspace`
  - Module: `sage.modular.hecke.homspace`
  - Type: `module`

- **sage.modular.hecke.module**
  - Entity: `module`
  - Module: `sage.modular.hecke.module`
  - Type: `module`

- **sage.modular.hecke.morphism**
  - Entity: `morphism`
  - Module: `sage.modular.hecke.morphism`
  - Type: `module`

- **sage.modular.hecke.submodule**
  - Entity: `submodule`
  - Module: `sage.modular.hecke.submodule`
  - Type: `module`

- **sage.modular.hypergeometric_motive**
  - Entity: `hypergeometric_motive`
  - Module: `sage.modular.hypergeometric_motive`
  - Type: `module`

- **sage.modular.local_comp.liftings**
  - Entity: `liftings`
  - Module: `sage.modular.local_comp.liftings`
  - Type: `module`

- **sage.modular.local_comp.local_comp**
  - Entity: `local_comp`
  - Module: `sage.modular.local_comp.local_comp`
  - Type: `module`

- **sage.modular.local_comp.smoothchar**
  - Entity: `smoothchar`
  - Module: `sage.modular.local_comp.smoothchar`
  - Type: `module`

- **sage.modular.local_comp.type_space**
  - Entity: `type_space`
  - Module: `sage.modular.local_comp.type_space`
  - Type: `module`

- **sage.modular.modform.ambient**
  - Entity: `ambient`
  - Module: `sage.modular.modform.ambient`
  - Type: `module`

- **sage.modular.modform.ambient_R**
  - Entity: `ambient_R`
  - Module: `sage.modular.modform.ambient_R`
  - Type: `module`

- **sage.modular.modform.ambient_eps**
  - Entity: `ambient_eps`
  - Module: `sage.modular.modform.ambient_eps`
  - Type: `module`

- **sage.modular.modform.ambient_g0**
  - Entity: `ambient_g0`
  - Module: `sage.modular.modform.ambient_g0`
  - Type: `module`

- **sage.modular.modform.ambient_g1**
  - Entity: `ambient_g1`
  - Module: `sage.modular.modform.ambient_g1`
  - Type: `module`

- **sage.modular.modform.constructor**
  - Entity: `constructor`
  - Module: `sage.modular.modform.constructor`
  - Type: `module`

- **sage.modular.modform.cuspidal_submodule**
  - Entity: `cuspidal_submodule`
  - Module: `sage.modular.modform.cuspidal_submodule`
  - Type: `module`

- **sage.modular.modform.eis_series**
  - Entity: `eis_series`
  - Module: `sage.modular.modform.eis_series`
  - Type: `module`

- **sage.modular.modform.eis_series_cython**
  - Entity: `eis_series_cython`
  - Module: `sage.modular.modform.eis_series_cython`
  - Type: `module`

- **sage.modular.modform.eisenstein_submodule**
  - Entity: `eisenstein_submodule`
  - Module: `sage.modular.modform.eisenstein_submodule`
  - Type: `module`

- **sage.modular.modform.element**
  - Entity: `element`
  - Module: `sage.modular.modform.element`
  - Type: `module`

- **sage.modular.modform.half_integral**
  - Entity: `half_integral`
  - Module: `sage.modular.modform.half_integral`
  - Type: `module`

- **sage.modular.modform.hecke_operator_on_qexp**
  - Entity: `hecke_operator_on_qexp`
  - Module: `sage.modular.modform.hecke_operator_on_qexp`
  - Type: `module`

- **sage.modular.modform.j_invariant**
  - Entity: `j_invariant`
  - Module: `sage.modular.modform.j_invariant`
  - Type: `module`

- **sage.modular.modform.notes**
  - Entity: `notes`
  - Module: `sage.modular.modform.notes`
  - Type: `module`

- **sage.modular.modform.numerical**
  - Entity: `numerical`
  - Module: `sage.modular.modform.numerical`
  - Type: `module`

- **sage.modular.modform.ring**
  - Entity: `ring`
  - Module: `sage.modular.modform.ring`
  - Type: `module`

- **sage.modular.modform.space**
  - Entity: `space`
  - Module: `sage.modular.modform.space`
  - Type: `module`

- **sage.modular.modform.submodule**
  - Entity: `submodule`
  - Module: `sage.modular.modform.submodule`
  - Type: `module`

- **sage.modular.modform.theta**
  - Entity: `theta`
  - Module: `sage.modular.modform.theta`
  - Type: `module`

- **sage.modular.modform.vm_basis**
  - Entity: `vm_basis`
  - Module: `sage.modular.modform.vm_basis`
  - Type: `module`

- **sage.modular.modform_hecketriangle.abstract_ring**
  - Entity: `abstract_ring`
  - Module: `sage.modular.modform_hecketriangle.abstract_ring`
  - Type: `module`

- **sage.modular.modform_hecketriangle.abstract_space**
  - Entity: `abstract_space`
  - Module: `sage.modular.modform_hecketriangle.abstract_space`
  - Type: `module`

- **sage.modular.modform_hecketriangle.analytic_type**
  - Entity: `analytic_type`
  - Module: `sage.modular.modform_hecketriangle.analytic_type`
  - Type: `module`

- **sage.modular.modform_hecketriangle.constructor**
  - Entity: `constructor`
  - Module: `sage.modular.modform_hecketriangle.constructor`
  - Type: `module`

- **sage.modular.modform_hecketriangle.element**
  - Entity: `element`
  - Module: `sage.modular.modform_hecketriangle.element`
  - Type: `module`

- **sage.modular.modform_hecketriangle.functors**
  - Entity: `functors`
  - Module: `sage.modular.modform_hecketriangle.functors`
  - Type: `module`

- **sage.modular.modform_hecketriangle.graded_ring**
  - Entity: `graded_ring`
  - Module: `sage.modular.modform_hecketriangle.graded_ring`
  - Type: `module`

- **sage.modular.modform_hecketriangle.graded_ring_element**
  - Entity: `graded_ring_element`
  - Module: `sage.modular.modform_hecketriangle.graded_ring_element`
  - Type: `module`

- **sage.modular.modform_hecketriangle.hecke_triangle_group_element**
  - Entity: `hecke_triangle_group_element`
  - Module: `sage.modular.modform_hecketriangle.hecke_triangle_group_element`
  - Type: `module`

- **sage.modular.modform_hecketriangle.hecke_triangle_groups**
  - Entity: `hecke_triangle_groups`
  - Module: `sage.modular.modform_hecketriangle.hecke_triangle_groups`
  - Type: `module`

- **sage.modular.modform_hecketriangle.readme**
  - Entity: `readme`
  - Module: `sage.modular.modform_hecketriangle.readme`
  - Type: `module`

- **sage.modular.modform_hecketriangle.series_constructor**
  - Entity: `series_constructor`
  - Module: `sage.modular.modform_hecketriangle.series_constructor`
  - Type: `module`

- **sage.modular.modform_hecketriangle.space**
  - Entity: `space`
  - Module: `sage.modular.modform_hecketriangle.space`
  - Type: `module`

- **sage.modular.modform_hecketriangle.subspace**
  - Entity: `subspace`
  - Module: `sage.modular.modform_hecketriangle.subspace`
  - Type: `module`

- **sage.modular.modsym.ambient**
  - Entity: `ambient`
  - Module: `sage.modular.modsym.ambient`
  - Type: `module`

- **sage.modular.modsym.apply**
  - Entity: `apply`
  - Module: `sage.modular.modsym.apply`
  - Type: `module`

- **sage.modular.modsym.boundary**
  - Entity: `boundary`
  - Module: `sage.modular.modsym.boundary`
  - Type: `module`

- **sage.modular.modsym.element**
  - Entity: `element`
  - Module: `sage.modular.modsym.element`
  - Type: `module`

- **sage.modular.modsym.g1list**
  - Entity: `g1list`
  - Module: `sage.modular.modsym.g1list`
  - Type: `module`

- **sage.modular.modsym.ghlist**
  - Entity: `ghlist`
  - Module: `sage.modular.modsym.ghlist`
  - Type: `module`

- **sage.modular.modsym.hecke_operator**
  - Entity: `hecke_operator`
  - Module: `sage.modular.modsym.hecke_operator`
  - Type: `module`

- **sage.modular.modsym.heilbronn**
  - Entity: `heilbronn`
  - Module: `sage.modular.modsym.heilbronn`
  - Type: `module`

- **sage.modular.modsym.manin_symbol**
  - Entity: `manin_symbol`
  - Module: `sage.modular.modsym.manin_symbol`
  - Type: `module`

- **sage.modular.modsym.manin_symbol_list**
  - Entity: `manin_symbol_list`
  - Module: `sage.modular.modsym.manin_symbol_list`
  - Type: `module`

- **sage.modular.modsym.modsym**
  - Entity: `modsym`
  - Module: `sage.modular.modsym.modsym`
  - Type: `module`

- **sage.modular.modsym.modular_symbols**
  - Entity: `modular_symbols`
  - Module: `sage.modular.modsym.modular_symbols`
  - Type: `module`

- **sage.modular.modsym.p1list**
  - Entity: `p1list`
  - Module: `sage.modular.modsym.p1list`
  - Type: `module`

- **sage.modular.modsym.p1list_nf**
  - Entity: `p1list_nf`
  - Module: `sage.modular.modsym.p1list_nf`
  - Type: `module`

- **sage.modular.modsym.relation_matrix**
  - Entity: `relation_matrix`
  - Module: `sage.modular.modsym.relation_matrix`
  - Type: `module`

- **sage.modular.modsym.relation_matrix_pyx**
  - Entity: `relation_matrix_pyx`
  - Module: `sage.modular.modsym.relation_matrix_pyx`
  - Type: `module`

- **sage.modular.modsym.space**
  - Entity: `space`
  - Module: `sage.modular.modsym.space`
  - Type: `module`

- **sage.modular.modsym.subspace**
  - Entity: `subspace`
  - Module: `sage.modular.modsym.subspace`
  - Type: `module`

- **sage.modular.multiple_zeta**
  - Entity: `multiple_zeta`
  - Module: `sage.modular.multiple_zeta`
  - Type: `module`

- **sage.modular.overconvergent.genus0**
  - Entity: `genus0`
  - Module: `sage.modular.overconvergent.genus0`
  - Type: `module`

- **sage.modular.overconvergent.hecke_series**
  - Entity: `hecke_series`
  - Module: `sage.modular.overconvergent.hecke_series`
  - Type: `module`

- **sage.modular.overconvergent.weightspace**
  - Entity: `weightspace`
  - Module: `sage.modular.overconvergent.weightspace`
  - Type: `module`

- **sage.modular.pollack_stevens.distributions**
  - Entity: `distributions`
  - Module: `sage.modular.pollack_stevens.distributions`
  - Type: `module`

- **sage.modular.pollack_stevens.fund_domain**
  - Entity: `fund_domain`
  - Module: `sage.modular.pollack_stevens.fund_domain`
  - Type: `module`

- **sage.modular.pollack_stevens.manin_map**
  - Entity: `manin_map`
  - Module: `sage.modular.pollack_stevens.manin_map`
  - Type: `module`

- **sage.modular.pollack_stevens.modsym**
  - Entity: `modsym`
  - Module: `sage.modular.pollack_stevens.modsym`
  - Type: `module`

- **sage.modular.pollack_stevens.padic_lseries**
  - Entity: `padic_lseries`
  - Module: `sage.modular.pollack_stevens.padic_lseries`
  - Type: `module`

- **sage.modular.pollack_stevens.space**
  - Entity: `space`
  - Module: `sage.modular.pollack_stevens.space`
  - Type: `module`

- **sage.modular.quasimodform.element**
  - Entity: `element`
  - Module: `sage.modular.quasimodform.element`
  - Type: `module`

- **sage.modular.quasimodform.ring**
  - Entity: `ring`
  - Module: `sage.modular.quasimodform.ring`
  - Type: `module`

- **sage.modular.quatalg.brandt**
  - Entity: `brandt`
  - Module: `sage.modular.quatalg.brandt`
  - Type: `module`

- **sage.modular.ssmod.ssmod**
  - Entity: `ssmod`
  - Module: `sage.modular.ssmod.ssmod`
  - Type: `module`

- **sage.modules.complex_double_vector**
  - Entity: `complex_double_vector`
  - Module: `sage.modules.complex_double_vector`
  - Type: `module`

- **sage.modules.diamond_cutting**
  - Entity: `diamond_cutting`
  - Module: `sage.modules.diamond_cutting`
  - Type: `module`

- **sage.modules.fg_pid.fgp_element**
  - Entity: `fgp_element`
  - Module: `sage.modules.fg_pid.fgp_element`
  - Type: `module`

- **sage.modules.fg_pid.fgp_module**
  - Entity: `fgp_module`
  - Module: `sage.modules.fg_pid.fgp_module`
  - Type: `module`

- **sage.modules.fg_pid.fgp_morphism**
  - Entity: `fgp_morphism`
  - Module: `sage.modules.fg_pid.fgp_morphism`
  - Type: `module`

- **sage.modules.filtered_vector_space**
  - Entity: `filtered_vector_space`
  - Module: `sage.modules.filtered_vector_space`
  - Type: `module`

- **sage.modules.finite_submodule_iter**
  - Entity: `finite_submodule_iter`
  - Module: `sage.modules.finite_submodule_iter`
  - Type: `module`

- **sage.modules.fp_graded.element**
  - Entity: `element`
  - Module: `sage.modules.fp_graded.element`
  - Type: `module`

- **sage.modules.fp_graded.free_element**
  - Entity: `free_element`
  - Module: `sage.modules.fp_graded.free_element`
  - Type: `module`

- **sage.modules.fp_graded.free_homspace**
  - Entity: `free_homspace`
  - Module: `sage.modules.fp_graded.free_homspace`
  - Type: `module`

- **sage.modules.fp_graded.free_module**
  - Entity: `free_module`
  - Module: `sage.modules.fp_graded.free_module`
  - Type: `module`

- **sage.modules.fp_graded.free_morphism**
  - Entity: `free_morphism`
  - Module: `sage.modules.fp_graded.free_morphism`
  - Type: `module`

- **sage.modules.fp_graded.homspace**
  - Entity: `homspace`
  - Module: `sage.modules.fp_graded.homspace`
  - Type: `module`

- **sage.modules.fp_graded.module**
  - Entity: `module`
  - Module: `sage.modules.fp_graded.module`
  - Type: `module`

- **sage.modules.fp_graded.morphism**
  - Entity: `morphism`
  - Module: `sage.modules.fp_graded.morphism`
  - Type: `module`

- **sage.modules.fp_graded.steenrod.module**
  - Entity: `module`
  - Module: `sage.modules.fp_graded.steenrod.module`
  - Type: `module`

- **sage.modules.fp_graded.steenrod.morphism**
  - Entity: `morphism`
  - Module: `sage.modules.fp_graded.steenrod.morphism`
  - Type: `module`

- **sage.modules.free_module**
  - Entity: `free_module`
  - Module: `sage.modules.free_module`
  - Type: `module`

- **sage.modules.free_module_element**
  - Entity: `free_module_element`
  - Module: `sage.modules.free_module_element`
  - Type: `module`

- **sage.modules.free_module_homspace**
  - Entity: `free_module_homspace`
  - Module: `sage.modules.free_module_homspace`
  - Type: `module`

- **sage.modules.free_module_integer**
  - Entity: `free_module_integer`
  - Module: `sage.modules.free_module_integer`
  - Type: `module`

- **sage.modules.free_module_morphism**
  - Entity: `free_module_morphism`
  - Module: `sage.modules.free_module_morphism`
  - Type: `module`

- **sage.modules.free_module_pseudohomspace**
  - Entity: `free_module_pseudohomspace`
  - Module: `sage.modules.free_module_pseudohomspace`
  - Type: `module`

- **sage.modules.free_module_pseudomorphism**
  - Entity: `free_module_pseudomorphism`
  - Module: `sage.modules.free_module_pseudomorphism`
  - Type: `module`

- **sage.modules.free_quadratic_module**
  - Entity: `free_quadratic_module`
  - Module: `sage.modules.free_quadratic_module`
  - Type: `module`

- **sage.modules.free_quadratic_module_integer_symmetric**
  - Entity: `free_quadratic_module_integer_symmetric`
  - Module: `sage.modules.free_quadratic_module_integer_symmetric`
  - Type: `module`

- **sage.modules.matrix_morphism**
  - Entity: `matrix_morphism`
  - Module: `sage.modules.matrix_morphism`
  - Type: `module`

- **sage.modules.misc**
  - Entity: `misc`
  - Module: `sage.modules.misc`
  - Type: `module`

- **sage.modules.module**
  - Entity: `module`
  - Module: `sage.modules.module`
  - Type: `module`

- **sage.modules.multi_filtered_vector_space**
  - Entity: `multi_filtered_vector_space`
  - Module: `sage.modules.multi_filtered_vector_space`
  - Type: `module`

- **sage.modules.ore_module**
  - Entity: `ore_module`
  - Module: `sage.modules.ore_module`
  - Type: `module`

- **sage.modules.ore_module_element**
  - Entity: `ore_module_element`
  - Module: `sage.modules.ore_module_element`
  - Type: `module`

- **sage.modules.ore_module_homspace**
  - Entity: `ore_module_homspace`
  - Module: `sage.modules.ore_module_homspace`
  - Type: `module`

- **sage.modules.ore_module_morphism**
  - Entity: `ore_module_morphism`
  - Module: `sage.modules.ore_module_morphism`
  - Type: `module`

- **sage.modules.quotient_module**
  - Entity: `quotient_module`
  - Module: `sage.modules.quotient_module`
  - Type: `module`

- **sage.modules.real_double_vector**
  - Entity: `real_double_vector`
  - Module: `sage.modules.real_double_vector`
  - Type: `module`

- **sage.modules.submodule**
  - Entity: `submodule`
  - Module: `sage.modules.submodule`
  - Type: `module`

- **sage.modules.tensor_operations**
  - Entity: `tensor_operations`
  - Module: `sage.modules.tensor_operations`
  - Type: `module`

- **sage.modules.torsion_quadratic_module**
  - Entity: `torsion_quadratic_module`
  - Module: `sage.modules.torsion_quadratic_module`
  - Type: `module`

- **sage.modules.tutorial_free_modules**
  - Entity: `tutorial_free_modules`
  - Module: `sage.modules.tutorial_free_modules`
  - Type: `module`

- **sage.modules.vector_callable_symbolic_dense**
  - Entity: `vector_callable_symbolic_dense`
  - Module: `sage.modules.vector_callable_symbolic_dense`
  - Type: `module`

- **sage.modules.vector_complex_double_dense**
  - Entity: `vector_complex_double_dense`
  - Module: `sage.modules.vector_complex_double_dense`
  - Type: `module`

- **sage.modules.vector_double_dense**
  - Entity: `vector_double_dense`
  - Module: `sage.modules.vector_double_dense`
  - Type: `module`

- **sage.modules.vector_integer_dense**
  - Entity: `vector_integer_dense`
  - Module: `sage.modules.vector_integer_dense`
  - Type: `module`

- **sage.modules.vector_integer_sparse**
  - Entity: `vector_integer_sparse`
  - Module: `sage.modules.vector_integer_sparse`
  - Type: `module`

- **sage.modules.vector_mod2_dense**
  - Entity: `vector_mod2_dense`
  - Module: `sage.modules.vector_mod2_dense`
  - Type: `module`

- **sage.modules.vector_modn_dense**
  - Entity: `vector_modn_dense`
  - Module: `sage.modules.vector_modn_dense`
  - Type: `module`

- **sage.modules.vector_modn_sparse**
  - Entity: `vector_modn_sparse`
  - Module: `sage.modules.vector_modn_sparse`
  - Type: `module`

- **sage.modules.vector_numpy_dense**
  - Entity: `vector_numpy_dense`
  - Module: `sage.modules.vector_numpy_dense`
  - Type: `module`

- **sage.modules.vector_numpy_integer_dense**
  - Entity: `vector_numpy_integer_dense`
  - Module: `sage.modules.vector_numpy_integer_dense`
  - Type: `module`

- **sage.modules.vector_rational_dense**
  - Entity: `vector_rational_dense`
  - Module: `sage.modules.vector_rational_dense`
  - Type: `module`

- **sage.modules.vector_rational_sparse**
  - Entity: `vector_rational_sparse`
  - Module: `sage.modules.vector_rational_sparse`
  - Type: `module`

- **sage.modules.vector_real_double_dense**
  - Entity: `vector_real_double_dense`
  - Module: `sage.modules.vector_real_double_dense`
  - Type: `module`

- **sage.modules.vector_space_homspace**
  - Entity: `vector_space_homspace`
  - Module: `sage.modules.vector_space_homspace`
  - Type: `module`

- **sage.modules.vector_space_morphism**
  - Entity: `vector_space_morphism`
  - Module: `sage.modules.vector_space_morphism`
  - Type: `module`

- **sage.modules.vector_symbolic_dense**
  - Entity: `vector_symbolic_dense`
  - Module: `sage.modules.vector_symbolic_dense`
  - Type: `module`

- **sage.modules.vector_symbolic_sparse**
  - Entity: `vector_symbolic_sparse`
  - Module: `sage.modules.vector_symbolic_sparse`
  - Type: `module`

- **sage.modules.with_basis.all**
  - Entity: `all`
  - Module: `sage.modules.with_basis.all`
  - Type: `module`

- **sage.modules.with_basis.cell_module**
  - Entity: `cell_module`
  - Module: `sage.modules.with_basis.cell_module`
  - Type: `module`

- **sage.modules.with_basis.indexed_element**
  - Entity: `indexed_element`
  - Module: `sage.modules.with_basis.indexed_element`
  - Type: `module`

- **sage.modules.with_basis.invariant**
  - Entity: `invariant`
  - Module: `sage.modules.with_basis.invariant`
  - Type: `module`

- **sage.modules.with_basis.morphism**
  - Entity: `morphism`
  - Module: `sage.modules.with_basis.morphism`
  - Type: `module`

- **sage.modules.with_basis.representation**
  - Entity: `representation`
  - Module: `sage.modules.with_basis.representation`
  - Type: `module`

- **sage.modules.with_basis.subquotient**
  - Entity: `subquotient`
  - Module: `sage.modules.with_basis.subquotient`
  - Type: `module`

- **sage.monoids**
  - Entity: `monoids`
  - Module: `sage.monoids`
  - Type: `module`

- **sage.monoids.automatic_semigroup**
  - Entity: `automatic_semigroup`
  - Module: `sage.monoids.automatic_semigroup`
  - Type: `module`

- **sage.monoids.free_abelian_monoid**
  - Entity: `free_abelian_monoid`
  - Module: `sage.monoids.free_abelian_monoid`
  - Type: `module`

- **sage.monoids.free_abelian_monoid_element**
  - Entity: `free_abelian_monoid_element`
  - Module: `sage.monoids.free_abelian_monoid_element`
  - Type: `module`

- **sage.monoids.free_monoid**
  - Entity: `free_monoid`
  - Module: `sage.monoids.free_monoid`
  - Type: `module`

- **sage.monoids.free_monoid_element**
  - Entity: `free_monoid_element`
  - Module: `sage.monoids.free_monoid_element`
  - Type: `module`

- **sage.monoids.hecke_monoid**
  - Entity: `hecke_monoid`
  - Module: `sage.monoids.hecke_monoid`
  - Type: `module`

- **sage.monoids.indexed_free_monoid**
  - Entity: `indexed_free_monoid`
  - Module: `sage.monoids.indexed_free_monoid`
  - Type: `module`

- **sage.monoids.monoid**
  - Entity: `monoid`
  - Module: `sage.monoids.monoid`
  - Type: `module`

- **sage.monoids.string_monoid**
  - Entity: `string_monoid`
  - Module: `sage.monoids.string_monoid`
  - Type: `module`

- **sage.monoids.string_monoid_element**
  - Entity: `string_monoid_element`
  - Module: `sage.monoids.string_monoid_element`
  - Type: `module`

- **sage.monoids.string_ops**
  - Entity: `string_ops`
  - Module: `sage.monoids.string_ops`
  - Type: `module`

- **sage.monoids.trace_monoid**
  - Entity: `trace_monoid`
  - Module: `sage.monoids.trace_monoid`
  - Type: `module`

- **sage.numerical**
  - Entity: `numerical`
  - Module: `sage.numerical`
  - Type: `module`

- **sage.numerical.backends.cvxopt_backend**
  - Entity: `cvxopt_backend`
  - Module: `sage.numerical.backends.cvxopt_backend`
  - Type: `module`

- **sage.numerical.backends.cvxopt_sdp_backend**
  - Entity: `cvxopt_sdp_backend`
  - Module: `sage.numerical.backends.cvxopt_sdp_backend`
  - Type: `module`

- **sage.numerical.backends.generic_backend**
  - Entity: `generic_backend`
  - Module: `sage.numerical.backends.generic_backend`
  - Type: `module`

- **sage.numerical.backends.generic_sdp_backend**
  - Entity: `generic_sdp_backend`
  - Module: `sage.numerical.backends.generic_sdp_backend`
  - Type: `module`

- **sage.numerical.backends.glpk_backend**
  - Entity: `glpk_backend`
  - Module: `sage.numerical.backends.glpk_backend`
  - Type: `module`


### Part 11 (196 entries)

#### ATTRIBUTE (1 entries)

- **sage.quadratic_forms.ternary_qf.possible_automorphisms**
  - Entity: `possible_automorphisms`
  - Module: `sage.quadratic_forms.ternary_qf`
  - Type: `attribute`

#### CLASS (57 entries)

- **sage.numerical.backends.glpk_backend.GLPKBackend**
  - Entity: `GLPKBackend`
  - Module: `sage.numerical.backends.glpk_backend`
  - Type: `class`

- **sage.numerical.backends.glpk_exact_backend.GLPKExactBackend**
  - Entity: `GLPKExactBackend`
  - Module: `sage.numerical.backends.glpk_exact_backend`
  - Type: `class`

- **sage.numerical.backends.glpk_graph_backend.GLPKGraphBackend**
  - Entity: `GLPKGraphBackend`
  - Module: `sage.numerical.backends.glpk_graph_backend`
  - Type: `class`

- **sage.numerical.backends.interactivelp_backend.InteractiveLPBackend**
  - Entity: `InteractiveLPBackend`
  - Module: `sage.numerical.backends.interactivelp_backend`
  - Type: `class`

- **sage.numerical.backends.logging_backend.LoggingBackend**
  - Entity: `LoggingBackend`
  - Module: `sage.numerical.backends.logging_backend`
  - Type: `class`

- **sage.numerical.backends.ppl_backend.PPLBackend**
  - Entity: `PPLBackend`
  - Module: `sage.numerical.backends.ppl_backend`
  - Type: `class`

- **sage.numerical.interactive_simplex_method.InteractiveLPProblem**
  - Entity: `InteractiveLPProblem`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `class`

- **sage.numerical.interactive_simplex_method.InteractiveLPProblemStandardForm**
  - Entity: `InteractiveLPProblemStandardForm`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `class`

- **sage.numerical.interactive_simplex_method.LPAbstractDictionary**
  - Entity: `LPAbstractDictionary`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `class`

- **sage.numerical.interactive_simplex_method.LPDictionary**
  - Entity: `LPDictionary`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `class`

- **sage.numerical.interactive_simplex_method.LPRevisedDictionary**
  - Entity: `LPRevisedDictionary`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `class`

- **sage.numerical.knapsack.Superincreasing**
  - Entity: `Superincreasing`
  - Module: `sage.numerical.knapsack`
  - Type: `class`

- **sage.numerical.linear_functions.LinearConstraint**
  - Entity: `LinearConstraint`
  - Module: `sage.numerical.linear_functions`
  - Type: `class`

- **sage.numerical.linear_functions.LinearConstraintsParent_class**
  - Entity: `LinearConstraintsParent_class`
  - Module: `sage.numerical.linear_functions`
  - Type: `class`

- **sage.numerical.linear_functions.LinearFunction**
  - Entity: `LinearFunction`
  - Module: `sage.numerical.linear_functions`
  - Type: `class`

- **sage.numerical.linear_functions.LinearFunctionOrConstraint**
  - Entity: `LinearFunctionOrConstraint`
  - Module: `sage.numerical.linear_functions`
  - Type: `class`

- **sage.numerical.linear_functions.LinearFunctionsParent_class**
  - Entity: `LinearFunctionsParent_class`
  - Module: `sage.numerical.linear_functions`
  - Type: `class`

- **sage.numerical.linear_tensor.LinearTensorParent_class**
  - Entity: `LinearTensorParent_class`
  - Module: `sage.numerical.linear_tensor`
  - Type: `class`

- **sage.numerical.linear_tensor_constraints.LinearTensorConstraint**
  - Entity: `LinearTensorConstraint`
  - Module: `sage.numerical.linear_tensor_constraints`
  - Type: `class`

- **sage.numerical.linear_tensor_constraints.LinearTensorConstraintsParent_class**
  - Entity: `LinearTensorConstraintsParent_class`
  - Module: `sage.numerical.linear_tensor_constraints`
  - Type: `class`

- **sage.numerical.linear_tensor_element.LinearTensor**
  - Entity: `LinearTensor`
  - Module: `sage.numerical.linear_tensor_element`
  - Type: `class`

- **sage.numerical.mip.MIPVariable**
  - Entity: `MIPVariable`
  - Module: `sage.numerical.mip`
  - Type: `class`

- **sage.numerical.mip.MixedIntegerLinearProgram**
  - Entity: `MixedIntegerLinearProgram`
  - Module: `sage.numerical.mip`
  - Type: `class`

- **sage.numerical.sdp.SDPVariable**
  - Entity: `SDPVariable`
  - Module: `sage.numerical.sdp`
  - Type: `class`

- **sage.numerical.sdp.SDPVariableParent**
  - Entity: `SDPVariableParent`
  - Module: `sage.numerical.sdp`
  - Type: `class`

- **sage.numerical.sdp.SemidefiniteProgram**
  - Entity: `SemidefiniteProgram`
  - Module: `sage.numerical.sdp`
  - Type: `class`

- **sage.probability.probability_distribution.GeneralDiscreteDistribution**
  - Entity: `GeneralDiscreteDistribution`
  - Module: `sage.probability.probability_distribution`
  - Type: `class`

- **sage.probability.probability_distribution.ProbabilityDistribution**
  - Entity: `ProbabilityDistribution`
  - Module: `sage.probability.probability_distribution`
  - Type: `class`

- **sage.probability.probability_distribution.RealDistribution**
  - Entity: `RealDistribution`
  - Module: `sage.probability.probability_distribution`
  - Type: `class`

- **sage.probability.probability_distribution.SphericalDistribution**
  - Entity: `SphericalDistribution`
  - Module: `sage.probability.probability_distribution`
  - Type: `class`

- **sage.probability.random_variable.DiscreteProbabilitySpace**
  - Entity: `DiscreteProbabilitySpace`
  - Module: `sage.probability.random_variable`
  - Type: `class`

- **sage.probability.random_variable.DiscreteRandomVariable**
  - Entity: `DiscreteRandomVariable`
  - Module: `sage.probability.random_variable`
  - Type: `class`

- **sage.probability.random_variable.ProbabilitySpace_generic**
  - Entity: `ProbabilitySpace_generic`
  - Module: `sage.probability.random_variable`
  - Type: `class`

- **sage.probability.random_variable.RandomVariable_generic**
  - Entity: `RandomVariable_generic`
  - Module: `sage.probability.random_variable`
  - Type: `class`

- **sage.quadratic_forms.binary_qf.BinaryQF**
  - Entity: `BinaryQF`
  - Module: `sage.quadratic_forms.binary_qf`
  - Type: `class`

- **sage.quadratic_forms.bqf_class_group.BQFClassGroup**
  - Entity: `BQFClassGroup`
  - Module: `sage.quadratic_forms.bqf_class_group`
  - Type: `class`

- **sage.quadratic_forms.bqf_class_group.BQFClassGroupQuotientMorphism**
  - Entity: `BQFClassGroupQuotientMorphism`
  - Module: `sage.quadratic_forms.bqf_class_group`
  - Type: `class`

- **sage.quadratic_forms.bqf_class_group.BQFClassGroup_element**
  - Entity: `BQFClassGroup_element`
  - Module: `sage.quadratic_forms.bqf_class_group`
  - Type: `class`

- **sage.quadratic_forms.genera.genus.GenusSymbol_global_ring**
  - Entity: `GenusSymbol_global_ring`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `class`

- **sage.quadratic_forms.genera.genus.Genus_Symbol_p_adic_ring**
  - Entity: `Genus_Symbol_p_adic_ring`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `class`

- **sage.quadratic_forms.quadratic_form.QuadraticForm**
  - Entity: `QuadraticForm`
  - Module: `sage.quadratic_forms.quadratic_form`
  - Type: `class`

- **sage.quadratic_forms.ternary_qf.TernaryQF**
  - Entity: `TernaryQF`
  - Module: `sage.quadratic_forms.ternary_qf`
  - Type: `class`

- **sage.rings.complex_double.ComplexDoubleElement**
  - Entity: `ComplexDoubleElement`
  - Module: `sage.rings.complex_double`
  - Type: `class`

- **sage.rings.complex_double.ComplexDoubleField_class**
  - Entity: `ComplexDoubleField_class`
  - Module: `sage.rings.complex_double`
  - Type: `class`

- **sage.rings.complex_double.ComplexToCDF**
  - Entity: `ComplexToCDF`
  - Module: `sage.rings.complex_double`
  - Type: `class`

- **sage.rings.complex_double.FloatToCDF**
  - Entity: `FloatToCDF`
  - Module: `sage.rings.complex_double`
  - Type: `class`

- **sage.rings.complex_mpfr.ComplexField_class**
  - Entity: `ComplexField_class`
  - Module: `sage.rings.complex_mpfr`
  - Type: `class`

- **sage.rings.complex_mpfr.ComplexNumber**
  - Entity: `ComplexNumber`
  - Module: `sage.rings.complex_mpfr`
  - Type: `class`

- **sage.rings.complex_mpfr.RRtoCC**
  - Entity: `RRtoCC`
  - Module: `sage.rings.complex_mpfr`
  - Type: `class`

- **sage.rings.finite_rings.finite_field_base.FiniteField**
  - Entity: `FiniteField`
  - Module: `sage.rings.finite_rings.finite_field_base`
  - Type: `class`

- **sage.rings.finite_rings.finite_field_constructor.FiniteFieldFactory**
  - Entity: `FiniteFieldFactory`
  - Module: `sage.rings.finite_rings.finite_field_constructor`
  - Type: `class`

- **sage.rings.finite_rings.finite_field_givaro.FiniteField_givaro**
  - Entity: `FiniteField_givaro`
  - Module: `sage.rings.finite_rings.finite_field_givaro`
  - Type: `class`

- **sage.rings.finite_rings.finite_field_ntl_gf2e.FiniteField_ntl_gf2e**
  - Entity: `FiniteField_ntl_gf2e`
  - Module: `sage.rings.finite_rings.finite_field_ntl_gf2e`
  - Type: `class`

- **sage.rings.finite_rings.finite_field_pari_ffelt.FiniteField_pari_ffelt**
  - Entity: `FiniteField_pari_ffelt`
  - Module: `sage.rings.finite_rings.finite_field_pari_ffelt`
  - Type: `class`

- **sage.rings.finite_rings.finite_field_prime_modn.FiniteField_prime_modn**
  - Entity: `FiniteField_prime_modn`
  - Module: `sage.rings.finite_rings.finite_field_prime_modn`
  - Type: `class`

- **sage.rings.finite_rings.integer_mod_ring.IntegerModFactory**
  - Entity: `IntegerModFactory`
  - Module: `sage.rings.finite_rings.integer_mod_ring`
  - Type: `class`

- **sage.rings.finite_rings.integer_mod_ring.IntegerModRing_generic**
  - Entity: `IntegerModRing_generic`
  - Module: `sage.rings.finite_rings.integer_mod_ring`
  - Type: `class`

#### FUNCTION (96 entries)

- **sage.numerical.backends.logging_backend.LoggingBackendFactory**
  - Entity: `LoggingBackendFactory`
  - Module: `sage.numerical.backends.logging_backend`
  - Type: `function`

- **sage.numerical.gauss_legendre.estimate_error**
  - Entity: `estimate_error`
  - Module: `sage.numerical.gauss_legendre`
  - Type: `function`

- **sage.numerical.gauss_legendre.integrate_vector**
  - Entity: `integrate_vector`
  - Module: `sage.numerical.gauss_legendre`
  - Type: `function`

- **sage.numerical.gauss_legendre.integrate_vector_N**
  - Entity: `integrate_vector_N`
  - Module: `sage.numerical.gauss_legendre`
  - Type: `function`

- **sage.numerical.gauss_legendre.nodes**
  - Entity: `nodes`
  - Module: `sage.numerical.gauss_legendre`
  - Type: `function`

- **sage.numerical.gauss_legendre.nodes_uncached**
  - Entity: `nodes_uncached`
  - Module: `sage.numerical.gauss_legendre`
  - Type: `function`

- **sage.numerical.interactive_simplex_method.default_variable_name**
  - Entity: `default_variable_name`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `function`

- **sage.numerical.interactive_simplex_method.random_dictionary**
  - Entity: `random_dictionary`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `function`

- **sage.numerical.interactive_simplex_method.style**
  - Entity: `style`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `function`

- **sage.numerical.interactive_simplex_method.variable**
  - Entity: `variable`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `function`

- **sage.numerical.knapsack.knapsack**
  - Entity: `knapsack`
  - Module: `sage.numerical.knapsack`
  - Type: `function`

- **sage.numerical.linear_functions.LinearConstraintsParent**
  - Entity: `LinearConstraintsParent`
  - Module: `sage.numerical.linear_functions`
  - Type: `function`

- **sage.numerical.linear_functions.LinearFunctionsParent**
  - Entity: `LinearFunctionsParent`
  - Module: `sage.numerical.linear_functions`
  - Type: `function`

- **sage.numerical.linear_functions.is_LinearConstraint**
  - Entity: `is_LinearConstraint`
  - Module: `sage.numerical.linear_functions`
  - Type: `function`

- **sage.numerical.linear_functions.is_LinearFunction**
  - Entity: `is_LinearFunction`
  - Module: `sage.numerical.linear_functions`
  - Type: `function`

- **sage.numerical.linear_tensor.LinearTensorParent**
  - Entity: `LinearTensorParent`
  - Module: `sage.numerical.linear_tensor`
  - Type: `function`

- **sage.numerical.linear_tensor.is_LinearTensor**
  - Entity: `is_LinearTensor`
  - Module: `sage.numerical.linear_tensor`
  - Type: `function`

- **sage.numerical.linear_tensor_constraints.LinearTensorConstraintsParent**
  - Entity: `LinearTensorConstraintsParent`
  - Module: `sage.numerical.linear_tensor_constraints`
  - Type: `function`

- **sage.numerical.linear_tensor_constraints.is_LinearTensorConstraint**
  - Entity: `is_LinearTensorConstraint`
  - Module: `sage.numerical.linear_tensor_constraints`
  - Type: `function`

- **sage.numerical.optimize.binpacking**
  - Entity: `binpacking`
  - Module: `sage.numerical.optimize`
  - Type: `function`

- **sage.numerical.optimize.find_fit**
  - Entity: `find_fit`
  - Module: `sage.numerical.optimize`
  - Type: `function`

- **sage.numerical.optimize.find_local_maximum**
  - Entity: `find_local_maximum`
  - Module: `sage.numerical.optimize`
  - Type: `function`

- **sage.numerical.optimize.find_local_minimum**
  - Entity: `find_local_minimum`
  - Module: `sage.numerical.optimize`
  - Type: `function`

- **sage.numerical.optimize.find_root**
  - Entity: `find_root`
  - Module: `sage.numerical.optimize`
  - Type: `function`

- **sage.numerical.optimize.minimize**
  - Entity: `minimize`
  - Module: `sage.numerical.optimize`
  - Type: `function`

- **sage.numerical.optimize.minimize_constrained**
  - Entity: `minimize_constrained`
  - Module: `sage.numerical.optimize`
  - Type: `function`

- **sage.probability.random_variable.is_DiscreteProbabilitySpace**
  - Entity: `is_DiscreteProbabilitySpace`
  - Module: `sage.probability.random_variable`
  - Type: `function`

- **sage.probability.random_variable.is_DiscreteRandomVariable**
  - Entity: `is_DiscreteRandomVariable`
  - Module: `sage.probability.random_variable`
  - Type: `function`

- **sage.probability.random_variable.is_ProbabilitySpace**
  - Entity: `is_ProbabilitySpace`
  - Module: `sage.probability.random_variable`
  - Type: `function`

- **sage.probability.random_variable.is_RandomVariable**
  - Entity: `is_RandomVariable`
  - Module: `sage.probability.random_variable`
  - Type: `function`

- **sage.quadratic_forms.binary_qf.BinaryQF_reduced_representatives**
  - Entity: `BinaryQF_reduced_representatives`
  - Module: `sage.quadratic_forms.binary_qf`
  - Type: `function`

- **sage.quadratic_forms.constructions.BezoutianQuadraticForm**
  - Entity: `BezoutianQuadraticForm`
  - Module: `sage.quadratic_forms.constructions`
  - Type: `function`

- **sage.quadratic_forms.constructions.HyperbolicPlane_quadratic_form**
  - Entity: `HyperbolicPlane_quadratic_form`
  - Module: `sage.quadratic_forms.constructions`
  - Type: `function`

- **sage.quadratic_forms.count_local_2.CountAllLocalTypesNaive**
  - Entity: `CountAllLocalTypesNaive`
  - Module: `sage.quadratic_forms.count_local_2`
  - Type: `function`

- **sage.quadratic_forms.count_local_2.count_all_local_good_types_normal_form**
  - Entity: `count_all_local_good_types_normal_form`
  - Module: `sage.quadratic_forms.count_local_2`
  - Type: `function`

- **sage.quadratic_forms.count_local_2.count_modp__by_gauss_sum**
  - Entity: `count_modp__by_gauss_sum`
  - Module: `sage.quadratic_forms.count_local_2`
  - Type: `function`

- **sage.quadratic_forms.extras.extend_to_primitive**
  - Entity: `extend_to_primitive`
  - Module: `sage.quadratic_forms.extras`
  - Type: `function`

- **sage.quadratic_forms.extras.is_triangular_number**
  - Entity: `is_triangular_number`
  - Module: `sage.quadratic_forms.extras`
  - Type: `function`

- **sage.quadratic_forms.extras.least_quadratic_nonresidue**
  - Entity: `least_quadratic_nonresidue`
  - Module: `sage.quadratic_forms.extras`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.Genus**
  - Entity: `Genus`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.LocalGenusSymbol**
  - Entity: `LocalGenusSymbol`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.M_p**
  - Entity: `M_p`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.basis_complement**
  - Entity: `basis_complement`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.canonical_2_adic_compartments**
  - Entity: `canonical_2_adic_compartments`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.canonical_2_adic_reduction**
  - Entity: `canonical_2_adic_reduction`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.canonical_2_adic_trains**
  - Entity: `canonical_2_adic_trains`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.genera**
  - Entity: `genera`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.is_2_adic_genus**
  - Entity: `is_2_adic_genus`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.is_GlobalGenus**
  - Entity: `is_GlobalGenus`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.is_even_matrix**
  - Entity: `is_even_matrix`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.p_adic_symbol**
  - Entity: `p_adic_symbol`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.signature_pair_of_matrix**
  - Entity: `signature_pair_of_matrix`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.split_odd**
  - Entity: `split_odd`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.trace_diag_mod_8**
  - Entity: `trace_diag_mod_8`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.genus.two_adic_symbol**
  - Entity: `two_adic_symbol`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `function`

- **sage.quadratic_forms.genera.normal_form.collect_small_blocks**
  - Entity: `collect_small_blocks`
  - Module: `sage.quadratic_forms.genera.normal_form`
  - Type: `function`

- **sage.quadratic_forms.genera.normal_form.p_adic_normal_form**
  - Entity: `p_adic_normal_form`
  - Module: `sage.quadratic_forms.genera.normal_form`
  - Type: `function`

- **sage.quadratic_forms.qfsolve.qfparam**
  - Entity: `qfparam`
  - Module: `sage.quadratic_forms.qfsolve`
  - Type: `function`

- **sage.quadratic_forms.qfsolve.qfsolve**
  - Entity: `qfsolve`
  - Module: `sage.quadratic_forms.qfsolve`
  - Type: `function`

- **sage.quadratic_forms.qfsolve.solve**
  - Entity: `solve`
  - Module: `sage.quadratic_forms.qfsolve`
  - Type: `function`

- **sage.quadratic_forms.quadratic_form.DiagonalQuadraticForm**
  - Entity: `DiagonalQuadraticForm`
  - Module: `sage.quadratic_forms.quadratic_form`
  - Type: `function`

- **sage.quadratic_forms.quadratic_form.is_QuadraticForm**
  - Entity: `is_QuadraticForm`
  - Module: `sage.quadratic_forms.quadratic_form`
  - Type: `function`

- **sage.quadratic_forms.quadratic_form.quadratic_form_from_invariants**
  - Entity: `quadratic_form_from_invariants`
  - Module: `sage.quadratic_forms.quadratic_form`
  - Type: `function`

- **sage.quadratic_forms.quadratic_form__evaluate.QFEvaluateMatrix**
  - Entity: `QFEvaluateMatrix`
  - Module: `sage.quadratic_forms.quadratic_form__evaluate`
  - Type: `function`

- **sage.quadratic_forms.quadratic_form__evaluate.QFEvaluateVector**
  - Entity: `QFEvaluateVector`
  - Module: `sage.quadratic_forms.quadratic_form__evaluate`
  - Type: `function`

- **sage.quadratic_forms.random_quadraticform.random_quadraticform**
  - Entity: `random_quadraticform`
  - Module: `sage.quadratic_forms.random_quadraticform`
  - Type: `function`

- **sage.quadratic_forms.random_quadraticform.random_quadraticform_with_conditions**
  - Entity: `random_quadraticform_with_conditions`
  - Module: `sage.quadratic_forms.random_quadraticform`
  - Type: `function`

- **sage.quadratic_forms.random_quadraticform.random_ternaryqf**
  - Entity: `random_ternaryqf`
  - Module: `sage.quadratic_forms.random_quadraticform`
  - Type: `function`

- **sage.quadratic_forms.random_quadraticform.random_ternaryqf_with_conditions**
  - Entity: `random_ternaryqf_with_conditions`
  - Module: `sage.quadratic_forms.random_quadraticform`
  - Type: `function`

- **sage.quadratic_forms.special_values.QuadraticBernoulliNumber**
  - Entity: `QuadraticBernoulliNumber`
  - Module: `sage.quadratic_forms.special_values`
  - Type: `function`

- **sage.quadratic_forms.special_values.gamma__exact**
  - Entity: `gamma__exact`
  - Module: `sage.quadratic_forms.special_values`
  - Type: `function`

- **sage.quadratic_forms.special_values.quadratic_L_function__exact**
  - Entity: `quadratic_L_function__exact`
  - Module: `sage.quadratic_forms.special_values`
  - Type: `function`

- **sage.quadratic_forms.special_values.quadratic_L_function__numerical**
  - Entity: `quadratic_L_function__numerical`
  - Module: `sage.quadratic_forms.special_values`
  - Type: `function`

- **sage.quadratic_forms.special_values.zeta__exact**
  - Entity: `zeta__exact`
  - Module: `sage.quadratic_forms.special_values`
  - Type: `function`

- **sage.quadratic_forms.ternary.evaluate**
  - Entity: `evaluate`
  - Module: `sage.quadratic_forms.ternary`
  - Type: `function`

- **sage.quadratic_forms.ternary.extend**
  - Entity: `extend`
  - Module: `sage.quadratic_forms.ternary`
  - Type: `function`

- **sage.quadratic_forms.ternary.primitivize**
  - Entity: `primitivize`
  - Module: `sage.quadratic_forms.ternary`
  - Type: `function`

- **sage.quadratic_forms.ternary.pseudorandom_primitive_zero_mod_p**
  - Entity: `pseudorandom_primitive_zero_mod_p`
  - Module: `sage.quadratic_forms.ternary`
  - Type: `function`

- **sage.quadratic_forms.ternary.red_mfact**
  - Entity: `red_mfact`
  - Module: `sage.quadratic_forms.ternary`
  - Type: `function`

- **sage.quadratic_forms.ternary_qf.find_a_ternary_qf_by_level_disc**
  - Entity: `find_a_ternary_qf_by_level_disc`
  - Module: `sage.quadratic_forms.ternary_qf`
  - Type: `function`

- **sage.quadratic_forms.ternary_qf.find_all_ternary_qf_by_level_disc**
  - Entity: `find_all_ternary_qf_by_level_disc`
  - Module: `sage.quadratic_forms.ternary_qf`
  - Type: `function`

- **sage.rings.complex_double.ComplexDoubleField**
  - Entity: `ComplexDoubleField`
  - Module: `sage.rings.complex_double`
  - Type: `function`

- **sage.rings.complex_double.is_ComplexDoubleElement**
  - Entity: `is_ComplexDoubleElement`
  - Module: `sage.rings.complex_double`
  - Type: `function`

- **sage.rings.complex_mpfr.ComplexField**
  - Entity: `ComplexField`
  - Module: `sage.rings.complex_mpfr`
  - Type: `function`

- **sage.rings.complex_mpfr.cmp_abs**
  - Entity: `cmp_abs`
  - Module: `sage.rings.complex_mpfr`
  - Type: `function`

- **sage.rings.complex_mpfr.create_ComplexNumber**
  - Entity: `create_ComplexNumber`
  - Module: `sage.rings.complex_mpfr`
  - Type: `function`

- **sage.rings.complex_mpfr.is_ComplexNumber**
  - Entity: `is_ComplexNumber`
  - Module: `sage.rings.complex_mpfr`
  - Type: `function`

- **sage.rings.complex_mpfr.late_import**
  - Entity: `late_import`
  - Module: `sage.rings.complex_mpfr`
  - Type: `function`

- **sage.rings.complex_mpfr.make_ComplexNumber0**
  - Entity: `make_ComplexNumber0`
  - Module: `sage.rings.complex_mpfr`
  - Type: `function`

- **sage.rings.complex_mpfr.set_global_complex_round_mode**
  - Entity: `set_global_complex_round_mode`
  - Module: `sage.rings.complex_mpfr`
  - Type: `function`

- **sage.rings.finite_rings.finite_field_base.is_FiniteField**
  - Entity: `is_FiniteField`
  - Module: `sage.rings.finite_rings.finite_field_base`
  - Type: `function`

- **sage.rings.finite_rings.finite_field_base.unpickle_FiniteField_ext**
  - Entity: `unpickle_FiniteField_ext`
  - Module: `sage.rings.finite_rings.finite_field_base`
  - Type: `function`

- **sage.rings.finite_rings.finite_field_base.unpickle_FiniteField_prm**
  - Entity: `unpickle_FiniteField_prm`
  - Module: `sage.rings.finite_rings.finite_field_base`
  - Type: `function`

- **sage.rings.finite_rings.finite_field_constructor.is_PrimeFiniteField**
  - Entity: `is_PrimeFiniteField`
  - Module: `sage.rings.finite_rings.finite_field_constructor`
  - Type: `function`

- **sage.rings.finite_rings.finite_field_ntl_gf2e.late_import**
  - Entity: `late_import`
  - Module: `sage.rings.finite_rings.finite_field_ntl_gf2e`
  - Type: `function`

- **sage.rings.finite_rings.integer_mod_ring.crt**
  - Entity: `crt`
  - Module: `sage.rings.finite_rings.integer_mod_ring`
  - Type: `function`

#### MODULE (42 entries)

- **sage.numerical.backends.glpk_exact_backend**
  - Entity: `glpk_exact_backend`
  - Module: `sage.numerical.backends.glpk_exact_backend`
  - Type: `module`

- **sage.numerical.backends.glpk_graph_backend**
  - Entity: `glpk_graph_backend`
  - Module: `sage.numerical.backends.glpk_graph_backend`
  - Type: `module`

- **sage.numerical.backends.interactivelp_backend**
  - Entity: `interactivelp_backend`
  - Module: `sage.numerical.backends.interactivelp_backend`
  - Type: `module`

- **sage.numerical.backends.logging_backend**
  - Entity: `logging_backend`
  - Module: `sage.numerical.backends.logging_backend`
  - Type: `module`

- **sage.numerical.backends.ppl_backend**
  - Entity: `ppl_backend`
  - Module: `sage.numerical.backends.ppl_backend`
  - Type: `module`

- **sage.numerical.gauss_legendre**
  - Entity: `gauss_legendre`
  - Module: `sage.numerical.gauss_legendre`
  - Type: `module`

- **sage.numerical.interactive_simplex_method**
  - Entity: `interactive_simplex_method`
  - Module: `sage.numerical.interactive_simplex_method`
  - Type: `module`

- **sage.numerical.knapsack**
  - Entity: `knapsack`
  - Module: `sage.numerical.knapsack`
  - Type: `module`

- **sage.numerical.linear_functions**
  - Entity: `linear_functions`
  - Module: `sage.numerical.linear_functions`
  - Type: `module`

- **sage.numerical.linear_tensor**
  - Entity: `linear_tensor`
  - Module: `sage.numerical.linear_tensor`
  - Type: `module`

- **sage.numerical.linear_tensor_constraints**
  - Entity: `linear_tensor_constraints`
  - Module: `sage.numerical.linear_tensor_constraints`
  - Type: `module`

- **sage.numerical.linear_tensor_element**
  - Entity: `linear_tensor_element`
  - Module: `sage.numerical.linear_tensor_element`
  - Type: `module`

- **sage.numerical.mip**
  - Entity: `mip`
  - Module: `sage.numerical.mip`
  - Type: `module`

- **sage.numerical.optimize**
  - Entity: `optimize`
  - Module: `sage.numerical.optimize`
  - Type: `module`

- **sage.numerical.sdp**
  - Entity: `sdp`
  - Module: `sage.numerical.sdp`
  - Type: `module`

- **sage.probability**
  - Entity: `probability`
  - Module: `sage.probability`
  - Type: `module`

- **sage.probability.probability_distribution**
  - Entity: `probability_distribution`
  - Module: `sage.probability.probability_distribution`
  - Type: `module`

- **sage.probability.random_variable**
  - Entity: `random_variable`
  - Module: `sage.probability.random_variable`
  - Type: `module`

- **sage.quadratic_forms**
  - Entity: `quadratic_forms`
  - Module: `sage.quadratic_forms`
  - Type: `module`

- **sage.quadratic_forms.binary_qf**
  - Entity: `binary_qf`
  - Module: `sage.quadratic_forms.binary_qf`
  - Type: `module`

- **sage.quadratic_forms.bqf_class_group**
  - Entity: `bqf_class_group`
  - Module: `sage.quadratic_forms.bqf_class_group`
  - Type: `module`

- **sage.quadratic_forms.constructions**
  - Entity: `constructions`
  - Module: `sage.quadratic_forms.constructions`
  - Type: `module`

- **sage.quadratic_forms.count_local_2**
  - Entity: `count_local_2`
  - Module: `sage.quadratic_forms.count_local_2`
  - Type: `module`

- **sage.quadratic_forms.extras**
  - Entity: `extras`
  - Module: `sage.quadratic_forms.extras`
  - Type: `module`

- **sage.quadratic_forms.genera.genus**
  - Entity: `genus`
  - Module: `sage.quadratic_forms.genera.genus`
  - Type: `module`

- **sage.quadratic_forms.genera.normal_form**
  - Entity: `normal_form`
  - Module: `sage.quadratic_forms.genera.normal_form`
  - Type: `module`

- **sage.quadratic_forms.qfsolve**
  - Entity: `qfsolve`
  - Module: `sage.quadratic_forms.qfsolve`
  - Type: `module`

- **sage.quadratic_forms.quadratic_form**
  - Entity: `quadratic_form`
  - Module: `sage.quadratic_forms.quadratic_form`
  - Type: `module`

- **sage.quadratic_forms.quadratic_form__evaluate**
  - Entity: `quadratic_form__evaluate`
  - Module: `sage.quadratic_forms.quadratic_form__evaluate`
  - Type: `module`

- **sage.quadratic_forms.random_quadraticform**
  - Entity: `random_quadraticform`
  - Module: `sage.quadratic_forms.random_quadraticform`
  - Type: `module`

- **sage.quadratic_forms.special_values**
  - Entity: `special_values`
  - Module: `sage.quadratic_forms.special_values`
  - Type: `module`

- **sage.quadratic_forms.ternary**
  - Entity: `ternary`
  - Module: `sage.quadratic_forms.ternary`
  - Type: `module`

- **sage.quadratic_forms.ternary_qf**
  - Entity: `ternary_qf`
  - Module: `sage.quadratic_forms.ternary_qf`
  - Type: `module`

- **sage.rings.complex_double**
  - Entity: `complex_double`
  - Module: `sage.rings.complex_double`
  - Type: `module`

- **sage.rings.complex_mpfr**
  - Entity: `complex_mpfr`
  - Module: `sage.rings.complex_mpfr`
  - Type: `module`

- **sage.rings.finite_rings.finite_field_base**
  - Entity: `finite_field_base`
  - Module: `sage.rings.finite_rings.finite_field_base`
  - Type: `module`

- **sage.rings.finite_rings.finite_field_constructor**
  - Entity: `finite_field_constructor`
  - Module: `sage.rings.finite_rings.finite_field_constructor`
  - Type: `module`

- **sage.rings.finite_rings.finite_field_givaro**
  - Entity: `finite_field_givaro`
  - Module: `sage.rings.finite_rings.finite_field_givaro`
  - Type: `module`

- **sage.rings.finite_rings.finite_field_ntl_gf2e**
  - Entity: `finite_field_ntl_gf2e`
  - Module: `sage.rings.finite_rings.finite_field_ntl_gf2e`
  - Type: `module`

- **sage.rings.finite_rings.finite_field_pari_ffelt**
  - Entity: `finite_field_pari_ffelt`
  - Module: `sage.rings.finite_rings.finite_field_pari_ffelt`
  - Type: `module`

- **sage.rings.finite_rings.finite_field_prime_modn**
  - Entity: `finite_field_prime_modn`
  - Module: `sage.rings.finite_rings.finite_field_prime_modn`
  - Type: `module`

- **sage.rings.finite_rings.integer_mod_ring**
  - Entity: `integer_mod_ring`
  - Module: `sage.rings.finite_rings.integer_mod_ring`
  - Type: `module`


### Part 12 (741 entries)

#### ATTRIBUTE (5 entries)

- **sage.rings.number_field.number_field_morphisms.ambient_field**
  - Entity: `ambient_field`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `attribute`

- **sage.rings.polynomial.multi_polynomial_ideal.basis**
  - Entity: `basis`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `attribute`

- **sage.rings.polynomial.multi_polynomial_ideal.require_field**
  - Entity: `require_field`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `attribute`

- **sage.rings.polynomial.pbori.pbori.p**
  - Entity: `p`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `attribute`

- **sage.rings.polynomial.pbori.pbori.reduction_strategy**
  - Entity: `reduction_strategy`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `attribute`

#### CLASS (368 entries)

- **sage.rings.ideal.Ideal_fractional**
  - Entity: `Ideal_fractional`
  - Module: `sage.rings.ideal`
  - Type: `class`

- **sage.rings.ideal.Ideal_generic**
  - Entity: `Ideal_generic`
  - Module: `sage.rings.ideal`
  - Type: `class`

- **sage.rings.ideal.Ideal_pid**
  - Entity: `Ideal_pid`
  - Module: `sage.rings.ideal`
  - Type: `class`

- **sage.rings.ideal.Ideal_principal**
  - Entity: `Ideal_principal`
  - Module: `sage.rings.ideal`
  - Type: `class`

- **sage.rings.ideal_monoid.IdealMonoid_c**
  - Entity: `IdealMonoid_c`
  - Module: `sage.rings.ideal_monoid`
  - Type: `class`

- **sage.rings.integer.Integer**
  - Entity: `Integer`
  - Module: `sage.rings.integer`
  - Type: `class`

- **sage.rings.integer.IntegerWrapper**
  - Entity: `IntegerWrapper`
  - Module: `sage.rings.integer`
  - Type: `class`

- **sage.rings.integer.int_to_Z**
  - Entity: `int_to_Z`
  - Module: `sage.rings.integer`
  - Type: `class`

- **sage.rings.integer_ring.IntegerRing_class**
  - Entity: `IntegerRing_class`
  - Module: `sage.rings.integer_ring`
  - Type: `class`

- **sage.rings.number_field.class_group.ClassGroup**
  - Entity: `ClassGroup`
  - Module: `sage.rings.number_field.class_group`
  - Type: `class`

- **sage.rings.number_field.class_group.FractionalIdealClass**
  - Entity: `FractionalIdealClass`
  - Module: `sage.rings.number_field.class_group`
  - Type: `class`

- **sage.rings.number_field.class_group.SClassGroup**
  - Entity: `SClassGroup`
  - Module: `sage.rings.number_field.class_group`
  - Type: `class`

- **sage.rings.number_field.class_group.SFractionalIdealClass**
  - Entity: `SFractionalIdealClass`
  - Module: `sage.rings.number_field.class_group`
  - Type: `class`

- **sage.rings.number_field.galois_group.GaloisGroupElement**
  - Entity: `GaloisGroupElement`
  - Module: `sage.rings.number_field.galois_group`
  - Type: `class`

- **sage.rings.number_field.galois_group.GaloisGroup_subgroup**
  - Entity: `GaloisGroup_subgroup`
  - Module: `sage.rings.number_field.galois_group`
  - Type: `class`

- **sage.rings.number_field.galois_group.GaloisGroup_v1**
  - Entity: `GaloisGroup_v1`
  - Module: `sage.rings.number_field.galois_group`
  - Type: `class`

- **sage.rings.number_field.galois_group.GaloisGroup_v2**
  - Entity: `GaloisGroup_v2`
  - Module: `sage.rings.number_field.galois_group`
  - Type: `class`

- **sage.rings.number_field.homset.CyclotomicFieldHomset**
  - Entity: `CyclotomicFieldHomset`
  - Module: `sage.rings.number_field.homset`
  - Type: `class`

- **sage.rings.number_field.homset.NumberFieldHomset**
  - Entity: `NumberFieldHomset`
  - Module: `sage.rings.number_field.homset`
  - Type: `class`

- **sage.rings.number_field.homset.RelativeNumberFieldHomset**
  - Entity: `RelativeNumberFieldHomset`
  - Module: `sage.rings.number_field.homset`
  - Type: `class`

- **sage.rings.number_field.maps.MapAbsoluteToRelativeNumberField**
  - Entity: `MapAbsoluteToRelativeNumberField`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.MapNumberFieldToVectorSpace**
  - Entity: `MapNumberFieldToVectorSpace`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.MapRelativeNumberFieldToRelativeVectorSpace**
  - Entity: `MapRelativeNumberFieldToRelativeVectorSpace`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.MapRelativeNumberFieldToVectorSpace**
  - Entity: `MapRelativeNumberFieldToVectorSpace`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.MapRelativeToAbsoluteNumberField**
  - Entity: `MapRelativeToAbsoluteNumberField`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.MapRelativeVectorSpaceToRelativeNumberField**
  - Entity: `MapRelativeVectorSpaceToRelativeNumberField`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.MapVectorSpaceToNumberField**
  - Entity: `MapVectorSpaceToNumberField`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.MapVectorSpaceToRelativeNumberField**
  - Entity: `MapVectorSpaceToRelativeNumberField`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.NameChangeMap**
  - Entity: `NameChangeMap`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.maps.NumberFieldIsomorphism**
  - Entity: `NumberFieldIsomorphism`
  - Module: `sage.rings.number_field.maps`
  - Type: `class`

- **sage.rings.number_field.morphism.CyclotomicFieldHomomorphism_im_gens**
  - Entity: `CyclotomicFieldHomomorphism_im_gens`
  - Module: `sage.rings.number_field.morphism`
  - Type: `class`

- **sage.rings.number_field.morphism.NumberFieldHomomorphism_im_gens**
  - Entity: `NumberFieldHomomorphism_im_gens`
  - Module: `sage.rings.number_field.morphism`
  - Type: `class`

- **sage.rings.number_field.morphism.RelativeNumberFieldHomomorphism_from_abs**
  - Entity: `RelativeNumberFieldHomomorphism_from_abs`
  - Module: `sage.rings.number_field.morphism`
  - Type: `class`

- **sage.rings.number_field.number_field.CyclotomicFieldFactory**
  - Entity: `CyclotomicFieldFactory`
  - Module: `sage.rings.number_field.number_field`
  - Type: `class`

- **sage.rings.number_field.number_field.NumberFieldFactory**
  - Entity: `NumberFieldFactory`
  - Module: `sage.rings.number_field.number_field`
  - Type: `class`

- **sage.rings.number_field.number_field.NumberField_absolute**
  - Entity: `NumberField_absolute`
  - Module: `sage.rings.number_field.number_field`
  - Type: `class`

- **sage.rings.number_field.number_field.NumberField_cyclotomic**
  - Entity: `NumberField_cyclotomic`
  - Module: `sage.rings.number_field.number_field`
  - Type: `class`

- **sage.rings.number_field.number_field.NumberField_generic**
  - Entity: `NumberField_generic`
  - Module: `sage.rings.number_field.number_field`
  - Type: `class`

- **sage.rings.number_field.number_field.NumberField_quadratic**
  - Entity: `NumberField_quadratic`
  - Module: `sage.rings.number_field.number_field`
  - Type: `class`

- **sage.rings.number_field.number_field_base.NumberField**
  - Entity: `NumberField`
  - Module: `sage.rings.number_field.number_field_base`
  - Type: `class`

- **sage.rings.number_field.number_field_element.CoordinateFunction**
  - Entity: `CoordinateFunction`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `class`

- **sage.rings.number_field.number_field_element.NumberFieldElement**
  - Entity: `NumberFieldElement`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `class`

- **sage.rings.number_field.number_field_element.NumberFieldElement_absolute**
  - Entity: `NumberFieldElement_absolute`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `class`

- **sage.rings.number_field.number_field_element.NumberFieldElement_relative**
  - Entity: `NumberFieldElement_relative`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `class`

- **sage.rings.number_field.number_field_element.OrderElement_absolute**
  - Entity: `OrderElement_absolute`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `class`

- **sage.rings.number_field.number_field_element.OrderElement_relative**
  - Entity: `OrderElement_relative`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `class`

- **sage.rings.number_field.number_field_element_quadratic.NumberFieldElement_gaussian**
  - Entity: `NumberFieldElement_gaussian`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `class`

- **sage.rings.number_field.number_field_element_quadratic.NumberFieldElement_quadratic**
  - Entity: `NumberFieldElement_quadratic`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `class`

- **sage.rings.number_field.number_field_element_quadratic.NumberFieldElement_quadratic_sqrt**
  - Entity: `NumberFieldElement_quadratic_sqrt`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `class`

- **sage.rings.number_field.number_field_element_quadratic.OrderElement_quadratic**
  - Entity: `OrderElement_quadratic`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `class`

- **sage.rings.number_field.number_field_element_quadratic.Q_to_quadratic_field_element**
  - Entity: `Q_to_quadratic_field_element`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `class`

- **sage.rings.number_field.number_field_element_quadratic.Z_to_quadratic_field_element**
  - Entity: `Z_to_quadratic_field_element`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `class`

- **sage.rings.number_field.number_field_ideal.LiftMap**
  - Entity: `LiftMap`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `class`

- **sage.rings.number_field.number_field_ideal.NumberFieldFractionalIdeal**
  - Entity: `NumberFieldFractionalIdeal`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `class`

- **sage.rings.number_field.number_field_ideal.NumberFieldIdeal**
  - Entity: `NumberFieldIdeal`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `class`

- **sage.rings.number_field.number_field_ideal.QuotientMap**
  - Entity: `QuotientMap`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `class`

- **sage.rings.number_field.number_field_ideal_rel.NumberFieldFractionalIdeal_rel**
  - Entity: `NumberFieldFractionalIdeal_rel`
  - Module: `sage.rings.number_field.number_field_ideal_rel`
  - Type: `class`

- **sage.rings.number_field.number_field_morphisms.CyclotomicFieldConversion**
  - Entity: `CyclotomicFieldConversion`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `class`

- **sage.rings.number_field.number_field_morphisms.CyclotomicFieldEmbedding**
  - Entity: `CyclotomicFieldEmbedding`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `class`

- **sage.rings.number_field.number_field_morphisms.EmbeddedNumberFieldConversion**
  - Entity: `EmbeddedNumberFieldConversion`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `class`

- **sage.rings.number_field.number_field_morphisms.EmbeddedNumberFieldMorphism**
  - Entity: `EmbeddedNumberFieldMorphism`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `class`

- **sage.rings.number_field.number_field_morphisms.NumberFieldEmbedding**
  - Entity: `NumberFieldEmbedding`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `class`

- **sage.rings.number_field.number_field_rel.NumberField_relative**
  - Entity: `NumberField_relative`
  - Module: `sage.rings.number_field.number_field_rel`
  - Type: `class`

- **sage.rings.number_field.order.AbsoluteOrderFactory**
  - Entity: `AbsoluteOrderFactory`
  - Module: `sage.rings.number_field.order`
  - Type: `class`

- **sage.rings.number_field.order.Order**
  - Entity: `Order`
  - Module: `sage.rings.number_field.order`
  - Type: `class`

- **sage.rings.number_field.order.OrderFactory**
  - Entity: `OrderFactory`
  - Module: `sage.rings.number_field.order`
  - Type: `class`

- **sage.rings.number_field.order.Order_absolute**
  - Entity: `Order_absolute`
  - Module: `sage.rings.number_field.order`
  - Type: `class`

- **sage.rings.number_field.order.Order_relative**
  - Entity: `Order_relative`
  - Module: `sage.rings.number_field.order`
  - Type: `class`

- **sage.rings.number_field.order.RelativeOrderFactory**
  - Entity: `RelativeOrderFactory`
  - Module: `sage.rings.number_field.order`
  - Type: `class`

- **sage.rings.number_field.order_ideal.NumberFieldOrderIdeal_generic**
  - Entity: `NumberFieldOrderIdeal_generic`
  - Module: `sage.rings.number_field.order_ideal`
  - Type: `class`

- **sage.rings.number_field.order_ideal.NumberFieldOrderIdeal_quadratic**
  - Entity: `NumberFieldOrderIdeal_quadratic`
  - Module: `sage.rings.number_field.order_ideal`
  - Type: `class`

- **sage.rings.number_field.small_primes_of_degree_one.Small_primes_of_degree_one_iter**
  - Entity: `Small_primes_of_degree_one_iter`
  - Module: `sage.rings.number_field.small_primes_of_degree_one`
  - Type: `class`

- **sage.rings.number_field.splitting_field.SplittingData**
  - Entity: `SplittingData`
  - Module: `sage.rings.number_field.splitting_field`
  - Type: `class`

- **sage.rings.number_field.structure.AbsoluteFromRelative**
  - Entity: `AbsoluteFromRelative`
  - Module: `sage.rings.number_field.structure`
  - Type: `class`

- **sage.rings.number_field.structure.NameChange**
  - Entity: `NameChange`
  - Module: `sage.rings.number_field.structure`
  - Type: `class`

- **sage.rings.number_field.structure.NumberFieldStructure**
  - Entity: `NumberFieldStructure`
  - Module: `sage.rings.number_field.structure`
  - Type: `class`

- **sage.rings.number_field.structure.RelativeFromAbsolute**
  - Entity: `RelativeFromAbsolute`
  - Module: `sage.rings.number_field.structure`
  - Type: `class`

- **sage.rings.number_field.structure.RelativeFromRelative**
  - Entity: `RelativeFromRelative`
  - Module: `sage.rings.number_field.structure`
  - Type: `class`

- **sage.rings.number_field.totallyreal_data.tr_data**
  - Entity: `tr_data`
  - Module: `sage.rings.number_field.totallyreal_data`
  - Type: `class`

- **sage.rings.number_field.totallyreal_rel.tr_data_rel**
  - Entity: `tr_data_rel`
  - Module: `sage.rings.number_field.totallyreal_rel`
  - Type: `class`

- **sage.rings.number_field.unit_group.UnitGroup**
  - Entity: `UnitGroup`
  - Module: `sage.rings.number_field.unit_group`
  - Type: `class`

- **sage.rings.padics.eisenstein_extension_generic.EisensteinExtensionGeneric**
  - Entity: `EisensteinExtensionGeneric`
  - Module: `sage.rings.padics.eisenstein_extension_generic`
  - Type: `class`

- **sage.rings.padics.factory.Qp_class**
  - Entity: `Qp_class`
  - Module: `sage.rings.padics.factory`
  - Type: `class`

- **sage.rings.padics.factory.Zp_class**
  - Entity: `Zp_class`
  - Module: `sage.rings.padics.factory`
  - Type: `class`

- **sage.rings.padics.factory.pAdicExtension_class**
  - Entity: `pAdicExtension_class`
  - Module: `sage.rings.padics.factory`
  - Type: `class`

- **sage.rings.padics.generic_nodes.CappedAbsoluteGeneric**
  - Entity: `CappedAbsoluteGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.CappedRelativeFieldGeneric**
  - Entity: `CappedRelativeFieldGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.CappedRelativeGeneric**
  - Entity: `CappedRelativeGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.CappedRelativeRingGeneric**
  - Entity: `CappedRelativeRingGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.FixedModGeneric**
  - Entity: `FixedModGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.FloatingPointFieldGeneric**
  - Entity: `FloatingPointFieldGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.FloatingPointGeneric**
  - Entity: `FloatingPointGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.FloatingPointRingGeneric**
  - Entity: `FloatingPointRingGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicCappedAbsoluteRingGeneric**
  - Entity: `pAdicCappedAbsoluteRingGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicCappedRelativeFieldGeneric**
  - Entity: `pAdicCappedRelativeFieldGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicCappedRelativeRingGeneric**
  - Entity: `pAdicCappedRelativeRingGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicFieldBaseGeneric**
  - Entity: `pAdicFieldBaseGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicFieldGeneric**
  - Entity: `pAdicFieldGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicFixedModRingGeneric**
  - Entity: `pAdicFixedModRingGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicFloatingPointFieldGeneric**
  - Entity: `pAdicFloatingPointFieldGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicFloatingPointRingGeneric**
  - Entity: `pAdicFloatingPointRingGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicLatticeGeneric**
  - Entity: `pAdicLatticeGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicRelaxedGeneric**
  - Entity: `pAdicRelaxedGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicRingBaseGeneric**
  - Entity: `pAdicRingBaseGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.generic_nodes.pAdicRingGeneric**
  - Entity: `pAdicRingGeneric`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `class`

- **sage.rings.padics.local_generic.LocalGeneric**
  - Entity: `LocalGeneric`
  - Module: `sage.rings.padics.local_generic`
  - Type: `class`

- **sage.rings.padics.local_generic_element.LocalGenericElement**
  - Entity: `LocalGenericElement`
  - Module: `sage.rings.padics.local_generic_element`
  - Type: `class`

- **sage.rings.padics.morphism.FrobeniusEndomorphism_padics**
  - Entity: `FrobeniusEndomorphism_padics`
  - Module: `sage.rings.padics.morphism`
  - Type: `class`

- **sage.rings.padics.padic_ZZ_pX_CA_element.pAdicZZpXCAElement**
  - Entity: `pAdicZZpXCAElement`
  - Module: `sage.rings.padics.padic_ZZ_pX_CA_element`
  - Type: `class`

- **sage.rings.padics.padic_ZZ_pX_CR_element.pAdicZZpXCRElement**
  - Entity: `pAdicZZpXCRElement`
  - Module: `sage.rings.padics.padic_ZZ_pX_CR_element`
  - Type: `class`

- **sage.rings.padics.padic_ZZ_pX_FM_element.pAdicZZpXFMElement**
  - Entity: `pAdicZZpXFMElement`
  - Module: `sage.rings.padics.padic_ZZ_pX_FM_element`
  - Type: `class`

- **sage.rings.padics.padic_ZZ_pX_element.pAdicZZpXElement**
  - Entity: `pAdicZZpXElement`
  - Module: `sage.rings.padics.padic_ZZ_pX_element`
  - Type: `class`

- **sage.rings.padics.padic_base_generic.pAdicBaseGeneric**
  - Entity: `pAdicBaseGeneric`
  - Module: `sage.rings.padics.padic_base_generic`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicFieldCappedRelative**
  - Entity: `pAdicFieldCappedRelative`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicFieldFloatingPoint**
  - Entity: `pAdicFieldFloatingPoint`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicFieldLattice**
  - Entity: `pAdicFieldLattice`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicFieldRelaxed**
  - Entity: `pAdicFieldRelaxed`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicRingCappedAbsolute**
  - Entity: `pAdicRingCappedAbsolute`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicRingCappedRelative**
  - Entity: `pAdicRingCappedRelative`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicRingFixedMod**
  - Entity: `pAdicRingFixedMod`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicRingFloatingPoint**
  - Entity: `pAdicRingFloatingPoint`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicRingLattice**
  - Entity: `pAdicRingLattice`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_base_leaves.pAdicRingRelaxed**
  - Entity: `pAdicRingRelaxed`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.CAElement**
  - Entity: `CAElement`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.ExpansionIter**
  - Entity: `ExpansionIter`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.ExpansionIterable**
  - Entity: `ExpansionIterable`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.PowComputer_**
  - Entity: `PowComputer_`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.pAdicCappedAbsoluteElement**
  - Entity: `pAdicCappedAbsoluteElement`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.pAdicCoercion_CA_frac_field**
  - Entity: `pAdicCoercion_CA_frac_field`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.pAdicCoercion_ZZ_CA**
  - Entity: `pAdicCoercion_ZZ_CA`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.pAdicConvert_CA_ZZ**
  - Entity: `pAdicConvert_CA_ZZ`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.pAdicConvert_CA_frac_field**
  - Entity: `pAdicConvert_CA_frac_field`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.pAdicConvert_QQ_CA**
  - Entity: `pAdicConvert_QQ_CA`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_absolute_element.pAdicTemplateElement**
  - Entity: `pAdicTemplateElement`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.CRElement**
  - Entity: `CRElement`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.ExpansionIter**
  - Entity: `ExpansionIter`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.ExpansionIterable**
  - Entity: `ExpansionIterable`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.PowComputer_**
  - Entity: `PowComputer_`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicCappedRelativeElement**
  - Entity: `pAdicCappedRelativeElement`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicCoercion_CR_frac_field**
  - Entity: `pAdicCoercion_CR_frac_field`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicCoercion_QQ_CR**
  - Entity: `pAdicCoercion_QQ_CR`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicCoercion_ZZ_CR**
  - Entity: `pAdicCoercion_ZZ_CR`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicConvert_CR_QQ**
  - Entity: `pAdicConvert_CR_QQ`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicConvert_CR_ZZ**
  - Entity: `pAdicConvert_CR_ZZ`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicConvert_CR_frac_field**
  - Entity: `pAdicConvert_CR_frac_field`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicConvert_QQ_CR**
  - Entity: `pAdicConvert_QQ_CR`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_capped_relative_element.pAdicTemplateElement**
  - Entity: `pAdicTemplateElement`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `class`

- **sage.rings.padics.padic_ext_element.pAdicExtElement**
  - Entity: `pAdicExtElement`
  - Module: `sage.rings.padics.padic_ext_element`
  - Type: `class`

- **sage.rings.padics.padic_extension_generic.DefPolyConversion**
  - Entity: `DefPolyConversion`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `class`

- **sage.rings.padics.padic_extension_generic.MapFreeModuleToOneStep**
  - Entity: `MapFreeModuleToOneStep`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `class`

- **sage.rings.padics.padic_extension_generic.MapFreeModuleToTwoStep**
  - Entity: `MapFreeModuleToTwoStep`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `class`

- **sage.rings.padics.padic_extension_generic.MapOneStepToFreeModule**
  - Entity: `MapOneStepToFreeModule`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `class`

- **sage.rings.padics.padic_extension_generic.MapTwoStepToFreeModule**
  - Entity: `MapTwoStepToFreeModule`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `class`

- **sage.rings.padics.padic_extension_generic.pAdicExtensionGeneric**
  - Entity: `pAdicExtensionGeneric`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `class`

- **sage.rings.padics.padic_extension_generic.pAdicModuleIsomorphism**
  - Entity: `pAdicModuleIsomorphism`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.EisensteinExtensionFieldCappedRelative**
  - Entity: `EisensteinExtensionFieldCappedRelative`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.EisensteinExtensionRingCappedAbsolute**
  - Entity: `EisensteinExtensionRingCappedAbsolute`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.EisensteinExtensionRingCappedRelative**
  - Entity: `EisensteinExtensionRingCappedRelative`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.EisensteinExtensionRingFixedMod**
  - Entity: `EisensteinExtensionRingFixedMod`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.UnramifiedExtensionFieldCappedRelative**
  - Entity: `UnramifiedExtensionFieldCappedRelative`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.UnramifiedExtensionFieldFloatingPoint**
  - Entity: `UnramifiedExtensionFieldFloatingPoint`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.UnramifiedExtensionRingCappedAbsolute**
  - Entity: `UnramifiedExtensionRingCappedAbsolute`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.UnramifiedExtensionRingCappedRelative**
  - Entity: `UnramifiedExtensionRingCappedRelative`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.UnramifiedExtensionRingFixedMod**
  - Entity: `UnramifiedExtensionRingFixedMod`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_extension_leaves.UnramifiedExtensionRingFloatingPoint**
  - Entity: `UnramifiedExtensionRingFloatingPoint`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.ExpansionIter**
  - Entity: `ExpansionIter`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.ExpansionIterable**
  - Entity: `ExpansionIterable`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.FMElement**
  - Entity: `FMElement`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.PowComputer_**
  - Entity: `PowComputer_`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.pAdicCoercion_FM_frac_field**
  - Entity: `pAdicCoercion_FM_frac_field`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.pAdicCoercion_ZZ_FM**
  - Entity: `pAdicCoercion_ZZ_FM`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.pAdicConvert_FM_ZZ**
  - Entity: `pAdicConvert_FM_ZZ`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.pAdicConvert_FM_frac_field**
  - Entity: `pAdicConvert_FM_frac_field`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.pAdicConvert_QQ_FM**
  - Entity: `pAdicConvert_QQ_FM`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.pAdicFixedModElement**
  - Entity: `pAdicFixedModElement`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_fixed_mod_element.pAdicTemplateElement**
  - Entity: `pAdicTemplateElement`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `class`

- **sage.rings.padics.padic_generic.ResidueLiftingMap**
  - Entity: `ResidueLiftingMap`
  - Module: `sage.rings.padics.padic_generic`
  - Type: `class`

- **sage.rings.padics.padic_generic.ResidueReductionMap**
  - Entity: `ResidueReductionMap`
  - Module: `sage.rings.padics.padic_generic`
  - Type: `class`

- **sage.rings.padics.padic_generic.pAdicGeneric**
  - Entity: `pAdicGeneric`
  - Module: `sage.rings.padics.padic_generic`
  - Type: `class`

- **sage.rings.padics.padic_generic_element.pAdicGenericElement**
  - Entity: `pAdicGenericElement`
  - Module: `sage.rings.padics.padic_generic_element`
  - Type: `class`

- **sage.rings.padics.padic_printing.pAdicPrinterDefaults**
  - Entity: `pAdicPrinterDefaults`
  - Module: `sage.rings.padics.padic_printing`
  - Type: `class`

- **sage.rings.padics.padic_printing.pAdicPrinter_class**
  - Entity: `pAdicPrinter_class`
  - Module: `sage.rings.padics.padic_printing`
  - Type: `class`

- **sage.rings.padics.padic_valuation.PadicValuationFactory**
  - Entity: `PadicValuationFactory`
  - Module: `sage.rings.padics.padic_valuation`
  - Type: `class`

- **sage.rings.padics.padic_valuation.pAdicFromLimitValuation**
  - Entity: `pAdicFromLimitValuation`
  - Module: `sage.rings.padics.padic_valuation`
  - Type: `class`

- **sage.rings.padics.padic_valuation.pAdicValuation_base**
  - Entity: `pAdicValuation_base`
  - Module: `sage.rings.padics.padic_valuation`
  - Type: `class`

- **sage.rings.padics.padic_valuation.pAdicValuation_int**
  - Entity: `pAdicValuation_int`
  - Module: `sage.rings.padics.padic_valuation`
  - Type: `class`

- **sage.rings.padics.padic_valuation.pAdicValuation_padic**
  - Entity: `pAdicValuation_padic`
  - Module: `sage.rings.padics.padic_valuation`
  - Type: `class`

- **sage.rings.padics.pow_computer.PowComputer_base**
  - Entity: `PowComputer_base`
  - Module: `sage.rings.padics.pow_computer`
  - Type: `class`

- **sage.rings.padics.pow_computer.PowComputer_class**
  - Entity: `PowComputer_class`
  - Module: `sage.rings.padics.pow_computer`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ZZ_pX**
  - Entity: `PowComputer_ZZ_pX`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ZZ_pX_FM**
  - Entity: `PowComputer_ZZ_pX_FM`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ZZ_pX_FM_Eis**
  - Entity: `PowComputer_ZZ_pX_FM_Eis`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ZZ_pX_big**
  - Entity: `PowComputer_ZZ_pX_big`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ZZ_pX_big_Eis**
  - Entity: `PowComputer_ZZ_pX_big_Eis`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ZZ_pX_small**
  - Entity: `PowComputer_ZZ_pX_small`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ZZ_pX_small_Eis**
  - Entity: `PowComputer_ZZ_pX_small_Eis`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.pow_computer_ext.PowComputer_ext**
  - Entity: `PowComputer_ext`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `class`

- **sage.rings.padics.unramified_extension_generic.UnramifiedExtensionGeneric**
  - Entity: `UnramifiedExtensionGeneric`
  - Module: `sage.rings.padics.unramified_extension_generic`
  - Type: `class`

- **sage.rings.polynomial.flatten.FlatteningMorphism**
  - Entity: `FlatteningMorphism`
  - Module: `sage.rings.polynomial.flatten`
  - Type: `class`

- **sage.rings.polynomial.flatten.FractionSpecializationMorphism**
  - Entity: `FractionSpecializationMorphism`
  - Module: `sage.rings.polynomial.flatten`
  - Type: `class`

- **sage.rings.polynomial.flatten.SpecializationMorphism**
  - Entity: `SpecializationMorphism`
  - Module: `sage.rings.polynomial.flatten`
  - Type: `class`

- **sage.rings.polynomial.flatten.UnflatteningMorphism**
  - Entity: `UnflatteningMorphism`
  - Module: `sage.rings.polynomial.flatten`
  - Type: `class`

- **sage.rings.polynomial.groebner_fan.GroebnerFan**
  - Entity: `GroebnerFan`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `class`

- **sage.rings.polynomial.groebner_fan.InitialForm**
  - Entity: `InitialForm`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `class`

- **sage.rings.polynomial.groebner_fan.PolyhedralCone**
  - Entity: `PolyhedralCone`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `class`

- **sage.rings.polynomial.groebner_fan.PolyhedralFan**
  - Entity: `PolyhedralFan`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `class`

- **sage.rings.polynomial.groebner_fan.ReducedGroebnerBasis**
  - Entity: `ReducedGroebnerBasis`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `class`

- **sage.rings.polynomial.groebner_fan.TropicalPrevariety**
  - Entity: `TropicalPrevariety`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `class`

- **sage.rings.polynomial.hilbert.Node**
  - Entity: `Node`
  - Module: `sage.rings.polynomial.hilbert`
  - Type: `class`

- **sage.rings.polynomial.ideal.Ideal_1poly_field**
  - Entity: `Ideal_1poly_field`
  - Module: `sage.rings.polynomial.ideal`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial**
  - Entity: `InfinitePolynomial`
  - Module: `sage.rings.polynomial.infinite_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial_dense**
  - Entity: `InfinitePolynomial_dense`
  - Module: `sage.rings.polynomial.infinite_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial_sparse**
  - Entity: `InfinitePolynomial_sparse`
  - Module: `sage.rings.polynomial.infinite_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_ring.GenDictWithBasering**
  - Entity: `GenDictWithBasering`
  - Module: `sage.rings.polynomial.infinite_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_ring.InfiniteGenDict**
  - Entity: `InfiniteGenDict`
  - Module: `sage.rings.polynomial.infinite_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialGen**
  - Entity: `InfinitePolynomialGen`
  - Module: `sage.rings.polynomial.infinite_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRingFactory**
  - Entity: `InfinitePolynomialRingFactory`
  - Module: `sage.rings.polynomial.infinite_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRing_dense**
  - Entity: `InfinitePolynomialRing_dense`
  - Module: `sage.rings.polynomial.infinite_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRing_sparse**
  - Entity: `InfinitePolynomialRing_sparse`
  - Module: `sage.rings.polynomial.infinite_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.integer_valued_polynomials.Bases**
  - Entity: `Bases`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.integer_valued_polynomials.Binomial**
  - Entity: `Binomial`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.integer_valued_polynomials.Element**
  - Entity: `Element`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.integer_valued_polynomials.ElementMethods**
  - Entity: `ElementMethods`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.integer_valued_polynomials.IntegerValuedPolynomialRing**
  - Entity: `IntegerValuedPolynomialRing`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.integer_valued_polynomials.ParentMethods**
  - Entity: `ParentMethods`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.integer_valued_polynomials.Shifted**
  - Entity: `Shifted`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.laurent_polynomial.LaurentPolynomial**
  - Entity: `LaurentPolynomial`
  - Module: `sage.rings.polynomial.laurent_polynomial`
  - Type: `class`

- **sage.rings.polynomial.laurent_polynomial.LaurentPolynomial_univariate**
  - Entity: `LaurentPolynomial_univariate`
  - Module: `sage.rings.polynomial.laurent_polynomial`
  - Type: `class`

- **sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing_mpair**
  - Entity: `LaurentPolynomialRing_mpair`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing_univariate**
  - Entity: `LaurentPolynomialRing_univariate`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.laurent_polynomial_ring_base.LaurentPolynomialRing_generic**
  - Entity: `LaurentPolynomialRing_generic`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring_base`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial.MPolynomial**
  - Entity: `MPolynomial`
  - Module: `sage.rings.polynomial.multi_polynomial`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial.MPolynomial_libsingular**
  - Entity: `MPolynomial_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_element.MPolynomial_element**
  - Entity: `MPolynomial_element`
  - Module: `sage.rings.polynomial.multi_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_element.MPolynomial_polydict**
  - Entity: `MPolynomial_polydict`
  - Module: `sage.rings.polynomial.multi_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal**
  - Entity: `MPolynomialIdeal`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_macaulay2_repr**
  - Entity: `MPolynomialIdeal_macaulay2_repr`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_magma_repr**
  - Entity: `MPolynomialIdeal_magma_repr`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_quotient**
  - Entity: `MPolynomialIdeal_quotient`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_singular_base_repr**
  - Entity: `MPolynomialIdeal_singular_base_repr`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_singular_repr**
  - Entity: `MPolynomialIdeal_singular_repr`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.NCPolynomialIdeal**
  - Entity: `NCPolynomialIdeal`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ideal.RequireField**
  - Entity: `RequireField`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_libsingular.MPolynomialRing_libsingular**
  - Entity: `MPolynomialRing_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_libsingular`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular**
  - Entity: `MPolynomial_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_libsingular`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_macaulay2_repr**
  - Entity: `MPolynomialRing_macaulay2_repr`
  - Module: `sage.rings.polynomial.multi_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_polydict**
  - Entity: `MPolynomialRing_polydict`
  - Module: `sage.rings.polynomial.multi_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_polydict_domain**
  - Entity: `MPolynomialRing_polydict_domain`
  - Module: `sage.rings.polynomial.multi_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ring_base.BooleanPolynomialRing_base**
  - Entity: `BooleanPolynomialRing_base`
  - Module: `sage.rings.polynomial.multi_polynomial_ring_base`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_ring_base.MPolynomialRing_base**
  - Entity: `MPolynomialRing_base`
  - Module: `sage.rings.polynomial.multi_polynomial_ring_base`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_generic**
  - Entity: `PolynomialSequence_generic`
  - Module: `sage.rings.polynomial.multi_polynomial_sequence`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_gf2**
  - Entity: `PolynomialSequence_gf2`
  - Module: `sage.rings.polynomial.multi_polynomial_sequence`
  - Type: `class`

- **sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_gf2e**
  - Entity: `PolynomialSequence_gf2e`
  - Module: `sage.rings.polynomial.multi_polynomial_sequence`
  - Type: `class`

- **sage.rings.polynomial.ore_function_element.ConstantOreFunctionSection**
  - Entity: `ConstantOreFunctionSection`
  - Module: `sage.rings.polynomial.ore_function_element`
  - Type: `class`

- **sage.rings.polynomial.ore_function_element.OreFunction**
  - Entity: `OreFunction`
  - Module: `sage.rings.polynomial.ore_function_element`
  - Type: `class`

- **sage.rings.polynomial.ore_function_element.OreFunctionBaseringInjection**
  - Entity: `OreFunctionBaseringInjection`
  - Module: `sage.rings.polynomial.ore_function_element`
  - Type: `class`

- **sage.rings.polynomial.ore_function_element.OreFunction_with_large_center**
  - Entity: `OreFunction_with_large_center`
  - Module: `sage.rings.polynomial.ore_function_element`
  - Type: `class`

- **sage.rings.polynomial.ore_function_field.OreFunctionCenterInjection**
  - Entity: `OreFunctionCenterInjection`
  - Module: `sage.rings.polynomial.ore_function_field`
  - Type: `class`

- **sage.rings.polynomial.ore_function_field.OreFunctionField**
  - Entity: `OreFunctionField`
  - Module: `sage.rings.polynomial.ore_function_field`
  - Type: `class`

- **sage.rings.polynomial.ore_function_field.OreFunctionField_with_large_center**
  - Entity: `OreFunctionField_with_large_center`
  - Module: `sage.rings.polynomial.ore_function_field`
  - Type: `class`

- **sage.rings.polynomial.ore_function_field.SectionOreFunctionCenterInjection**
  - Entity: `SectionOreFunctionCenterInjection`
  - Module: `sage.rings.polynomial.ore_function_field`
  - Type: `class`

- **sage.rings.polynomial.ore_polynomial_element.ConstantOrePolynomialSection**
  - Entity: `ConstantOrePolynomialSection`
  - Module: `sage.rings.polynomial.ore_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.ore_polynomial_element.OrePolynomial**
  - Entity: `OrePolynomial`
  - Module: `sage.rings.polynomial.ore_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.ore_polynomial_element.OrePolynomialBaseringInjection**
  - Entity: `OrePolynomialBaseringInjection`
  - Module: `sage.rings.polynomial.ore_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.ore_polynomial_element.OrePolynomial_generic_dense**
  - Entity: `OrePolynomial_generic_dense`
  - Module: `sage.rings.polynomial.ore_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.ore_polynomial_ring.OrePolynomialRing**
  - Entity: `OrePolynomialRing`
  - Module: `sage.rings.polynomial.ore_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.padics.polynomial_padic.Polynomial_padic**
  - Entity: `Polynomial_padic`
  - Module: `sage.rings.polynomial.padics.polynomial_padic`
  - Type: `class`

- **sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense.Polynomial_padic_capped_relative_dense**
  - Entity: `Polynomial_padic_capped_relative_dense`
  - Module: `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense`
  - Type: `class`

- **sage.rings.polynomial.padics.polynomial_padic_flat.Polynomial_padic_flat**
  - Entity: `Polynomial_padic_flat`
  - Module: `sage.rings.polynomial.padics.polynomial_padic_flat`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleConstant**
  - Entity: `BooleConstant`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleSet**
  - Entity: `BooleSet`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleSetIterator**
  - Entity: `BooleSetIterator`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanMonomial**
  - Entity: `BooleanMonomial`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanMonomialIterator**
  - Entity: `BooleanMonomialIterator`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanMonomialMonoid**
  - Entity: `BooleanMonomialMonoid`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanMonomialVariableIterator**
  - Entity: `BooleanMonomialVariableIterator`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanMulAction**
  - Entity: `BooleanMulAction`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanPolynomial**
  - Entity: `BooleanPolynomial`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanPolynomialEntry**
  - Entity: `BooleanPolynomialEntry`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanPolynomialIdeal**
  - Entity: `BooleanPolynomialIdeal`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanPolynomialIterator**
  - Entity: `BooleanPolynomialIterator`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanPolynomialRing**
  - Entity: `BooleanPolynomialRing`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector**
  - Entity: `BooleanPolynomialVector`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.BooleanPolynomialVectorIterator**
  - Entity: `BooleanPolynomialVectorIterator`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.CCuddNavigator**
  - Entity: `CCuddNavigator`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.FGLMStrategy**
  - Entity: `FGLMStrategy`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.GroebnerStrategy**
  - Entity: `GroebnerStrategy`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.MonomialConstruct**
  - Entity: `MonomialConstruct`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.MonomialFactory**
  - Entity: `MonomialFactory`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.PolynomialConstruct**
  - Entity: `PolynomialConstruct`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.PolynomialFactory**
  - Entity: `PolynomialFactory`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.ReductionStrategy**
  - Entity: `ReductionStrategy`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.VariableBlock**
  - Entity: `VariableBlock`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.VariableConstruct**
  - Entity: `VariableConstruct`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.pbori.pbori.VariableFactory**
  - Entity: `VariableFactory`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `class`

- **sage.rings.polynomial.plural.ExteriorAlgebra_plural**
  - Entity: `ExteriorAlgebra_plural`
  - Module: `sage.rings.polynomial.plural`
  - Type: `class`

- **sage.rings.polynomial.plural.G_AlgFactory**
  - Entity: `G_AlgFactory`
  - Module: `sage.rings.polynomial.plural`
  - Type: `class`

- **sage.rings.polynomial.plural.NCPolynomialRing_plural**
  - Entity: `NCPolynomialRing_plural`
  - Module: `sage.rings.polynomial.plural`
  - Type: `class`

- **sage.rings.polynomial.plural.NCPolynomial_plural**
  - Entity: `NCPolynomial_plural`
  - Module: `sage.rings.polynomial.plural`
  - Type: `class`

- **sage.rings.polynomial.polydict.ETuple**
  - Entity: `ETuple`
  - Module: `sage.rings.polynomial.polydict`
  - Type: `class`

- **sage.rings.polynomial.polydict.PolyDict**
  - Entity: `PolyDict`
  - Module: `sage.rings.polynomial.polydict`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.CompiledPolynomialFunction**
  - Entity: `CompiledPolynomialFunction`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.abc_pd**
  - Entity: `abc_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.add_pd**
  - Entity: `add_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.binary_pd**
  - Entity: `binary_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.coeff_pd**
  - Entity: `coeff_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.dummy_pd**
  - Entity: `dummy_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.generic_pd**
  - Entity: `generic_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.mul_pd**
  - Entity: `mul_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.pow_pd**
  - Entity: `pow_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.sqr_pd**
  - Entity: `sqr_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.unary_pd**
  - Entity: `unary_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.univar_pd**
  - Entity: `univar_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_compiled.var_pd**
  - Entity: `var_pd`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element.ConstantPolynomialSection**
  - Entity: `ConstantPolynomialSection`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element.Polynomial**
  - Entity: `Polynomial`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element.PolynomialBaseringInjection**
  - Entity: `PolynomialBaseringInjection`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element.Polynomial_generic_dense**
  - Entity: `Polynomial_generic_dense`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element.Polynomial_generic_dense_inexact**
  - Entity: `Polynomial_generic_dense_inexact`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdv**
  - Entity: `Polynomial_generic_cdv`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdvf**
  - Entity: `Polynomial_generic_cdvf`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdvr**
  - Entity: `Polynomial_generic_cdvr`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdv**
  - Entity: `Polynomial_generic_dense_cdv`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdvf**
  - Entity: `Polynomial_generic_dense_cdvf`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdvr**
  - Entity: `Polynomial_generic_dense_cdvr`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_field**
  - Entity: `Polynomial_generic_dense_field`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_domain**
  - Entity: `Polynomial_generic_domain`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_field**
  - Entity: `Polynomial_generic_field`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse**
  - Entity: `Polynomial_generic_sparse`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdv**
  - Entity: `Polynomial_generic_sparse_cdv`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdvf**
  - Entity: `Polynomial_generic_sparse_cdvf`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdvr**
  - Entity: `Polynomial_generic_sparse_cdvr`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_field**
  - Entity: `Polynomial_generic_sparse_field`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `class`

- **sage.rings.polynomial.polynomial_gf2x.Polynomial_GF2X**
  - Entity: `Polynomial_GF2X`
  - Module: `sage.rings.polynomial.polynomial_gf2x`
  - Type: `class`

- **sage.rings.polynomial.polynomial_gf2x.Polynomial_template**
  - Entity: `Polynomial_template`
  - Module: `sage.rings.polynomial.polynomial_gf2x`
  - Type: `class`

- **sage.rings.polynomial.polynomial_integer_dense_flint.Polynomial_integer_dense_flint**
  - Entity: `Polynomial_integer_dense_flint`
  - Module: `sage.rings.polynomial.polynomial_integer_dense_flint`
  - Type: `class`

- **sage.rings.polynomial.polynomial_integer_dense_ntl.Polynomial_integer_dense_ntl**
  - Entity: `Polynomial_integer_dense_ntl`
  - Module: `sage.rings.polynomial.polynomial_integer_dense_ntl`
  - Type: `class`

- **sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_mod_n**
  - Entity: `Polynomial_dense_mod_n`
  - Module: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - Type: `class`

- **sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_mod_p**
  - Entity: `Polynomial_dense_mod_p`
  - Module: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - Type: `class`

- **sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_modn_ntl_ZZ**
  - Entity: `Polynomial_dense_modn_ntl_ZZ`
  - Module: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - Type: `class`

- **sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_modn_ntl_zz**
  - Entity: `Polynomial_dense_modn_ntl_zz`
  - Module: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - Type: `class`

- **sage.rings.polynomial.polynomial_number_field.Polynomial_absolute_number_field_dense**
  - Entity: `Polynomial_absolute_number_field_dense`
  - Module: `sage.rings.polynomial.polynomial_number_field`
  - Type: `class`

- **sage.rings.polynomial.polynomial_number_field.Polynomial_relative_number_field_dense**
  - Entity: `Polynomial_relative_number_field_dense`
  - Module: `sage.rings.polynomial.polynomial_number_field`
  - Type: `class`

- **sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRingFactory**
  - Entity: `PolynomialQuotientRingFactory`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_coercion**
  - Entity: `PolynomialQuotientRing_coercion`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_domain**
  - Entity: `PolynomialQuotientRing_domain`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_field**
  - Entity: `PolynomialQuotientRing_field`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_generic**
  - Entity: `PolynomialQuotientRing_generic`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_quotient_ring_element.PolynomialQuotientRingElement**
  - Entity: `PolynomialQuotientRingElement`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring_element`
  - Type: `class`

- **sage.rings.polynomial.polynomial_rational_flint.Polynomial_rational_flint**
  - Entity: `Polynomial_rational_flint`
  - Module: `sage.rings.polynomial.polynomial_rational_flint`
  - Type: `class`

- **sage.rings.polynomial.polynomial_real_mpfr_dense.PolynomialRealDense**
  - Entity: `PolynomialRealDense`
  - Module: `sage.rings.polynomial.polynomial_real_mpfr_dense`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_cdvf**
  - Entity: `PolynomialRing_cdvf`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_cdvr**
  - Entity: `PolynomialRing_cdvr`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_commutative**
  - Entity: `PolynomialRing_commutative`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_finite_field**
  - Entity: `PolynomialRing_dense_finite_field`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_mod_n**
  - Entity: `PolynomialRing_dense_mod_n`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_mod_p**
  - Entity: `PolynomialRing_dense_mod_p`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_field_capped_relative**
  - Entity: `PolynomialRing_dense_padic_field_capped_relative`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_field_generic**
  - Entity: `PolynomialRing_dense_padic_field_generic`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_capped_absolute**
  - Entity: `PolynomialRing_dense_padic_ring_capped_absolute`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_capped_relative**
  - Entity: `PolynomialRing_dense_padic_ring_capped_relative`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_fixed_mod**
  - Entity: `PolynomialRing_dense_padic_ring_fixed_mod`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_generic**
  - Entity: `PolynomialRing_dense_padic_ring_generic`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_field**
  - Entity: `PolynomialRing_field`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_generic**
  - Entity: `PolynomialRing_generic`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring.PolynomialRing_integral_domain**
  - Entity: `PolynomialRing_integral_domain`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.polynomial_ring_homomorphism.PolynomialRingHomomorphism_from_base**
  - Entity: `PolynomialRingHomomorphism_from_base`
  - Module: `sage.rings.polynomial.polynomial_ring_homomorphism`
  - Type: `class`

- **sage.rings.polynomial.polynomial_singular_interface.PolynomialRing_singular_repr**
  - Entity: `PolynomialRing_singular_repr`
  - Module: `sage.rings.polynomial.polynomial_singular_interface`
  - Type: `class`

#### FUNCTION (259 entries)

- **sage.rings.ideal.Cyclic**
  - Entity: `Cyclic`
  - Module: `sage.rings.ideal`
  - Type: `function`

- **sage.rings.ideal.FieldIdeal**
  - Entity: `FieldIdeal`
  - Module: `sage.rings.ideal`
  - Type: `function`

- **sage.rings.ideal.Ideal**
  - Entity: `Ideal`
  - Module: `sage.rings.ideal`
  - Type: `function`

- **sage.rings.ideal.Katsura**
  - Entity: `Katsura`
  - Module: `sage.rings.ideal`
  - Type: `function`

- **sage.rings.ideal.is_Ideal**
  - Entity: `is_Ideal`
  - Module: `sage.rings.ideal`
  - Type: `function`

- **sage.rings.ideal_monoid.IdealMonoid**
  - Entity: `IdealMonoid`
  - Module: `sage.rings.ideal_monoid`
  - Type: `function`

- **sage.rings.integer.GCD_list**
  - Entity: `GCD_list`
  - Module: `sage.rings.integer`
  - Type: `function`

- **sage.rings.integer.free_integer_pool**
  - Entity: `free_integer_pool`
  - Module: `sage.rings.integer`
  - Type: `function`

- **sage.rings.integer.is_Integer**
  - Entity: `is_Integer`
  - Module: `sage.rings.integer`
  - Type: `function`

- **sage.rings.integer.make_integer**
  - Entity: `make_integer`
  - Module: `sage.rings.integer`
  - Type: `function`

- **sage.rings.integer_ring.IntegerRing**
  - Entity: `IntegerRing`
  - Module: `sage.rings.integer_ring`
  - Type: `function`

- **sage.rings.integer_ring.crt_basis**
  - Entity: `crt_basis`
  - Module: `sage.rings.integer_ring`
  - Type: `function`

- **sage.rings.integer_ring.is_IntegerRing**
  - Entity: `is_IntegerRing`
  - Module: `sage.rings.integer_ring`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.K0_func**
  - Entity: `K0_func`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.K1_func**
  - Entity: `K1_func`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.Omega_prime**
  - Entity: `Omega_prime`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.Yu_C1_star**
  - Entity: `Yu_C1_star`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.Yu_a1_kappa1_c1**
  - Entity: `Yu_a1_kappa1_c1`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.Yu_bound**
  - Entity: `Yu_bound`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.Yu_condition_115**
  - Entity: `Yu_condition_115`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.Yu_modified_height**
  - Entity: `Yu_modified_height`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.beta_k**
  - Entity: `beta_k`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.c11_func**
  - Entity: `c11_func`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.c13_func**
  - Entity: `c13_func`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.c3_func**
  - Entity: `c3_func`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.c4_func**
  - Entity: `c4_func`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.clean_rfv_dict**
  - Entity: `clean_rfv_dict`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.clean_sfs**
  - Entity: `clean_sfs`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.column_Log**
  - Entity: `column_Log`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.compatible_system_lift**
  - Entity: `compatible_system_lift`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.compatible_systems**
  - Entity: `compatible_systems`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.compatible_vectors**
  - Entity: `compatible_vectors`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.compatible_vectors_check**
  - Entity: `compatible_vectors_check`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.construct_comp_exp_vec**
  - Entity: `construct_comp_exp_vec`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.construct_complement_dictionaries**
  - Entity: `construct_complement_dictionaries`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.construct_rfv_to_ev**
  - Entity: `construct_rfv_to_ev`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.cx_LLL_bound**
  - Entity: `cx_LLL_bound`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.defining_polynomial_for_Kp**
  - Entity: `defining_polynomial_for_Kp`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.drop_vector**
  - Entity: `drop_vector`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.embedding_to_Kp**
  - Entity: `embedding_to_Kp`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.eq_up_to_order**
  - Entity: `eq_up_to_order`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.log_p**
  - Entity: `log_p`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.log_p_series_part**
  - Entity: `log_p_series_part`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.minimal_vector**
  - Entity: `minimal_vector`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.mus**
  - Entity: `mus`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.p_adic_LLL_bound**
  - Entity: `p_adic_LLL_bound`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.p_adic_LLL_bound_one_prime**
  - Entity: `p_adic_LLL_bound_one_prime`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.possible_mu0s**
  - Entity: `possible_mu0s`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.reduction_step_complex_case**
  - Entity: `reduction_step_complex_case`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.sieve_below_bound**
  - Entity: `sieve_below_bound`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.sieve_ordering**
  - Entity: `sieve_ordering`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.solutions_from_systems**
  - Entity: `solutions_from_systems`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.solve_S_unit_equation**
  - Entity: `solve_S_unit_equation`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.S_unit_solver.split_primes_large_lcm**
  - Entity: `split_primes_large_lcm`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `function`

- **sage.rings.number_field.bdd_height.bdd_height**
  - Entity: `bdd_height`
  - Module: `sage.rings.number_field.bdd_height`
  - Type: `function`

- **sage.rings.number_field.bdd_height.bdd_height_iq**
  - Entity: `bdd_height_iq`
  - Module: `sage.rings.number_field.bdd_height`
  - Type: `function`

- **sage.rings.number_field.bdd_height.bdd_norm_pr_gens_iq**
  - Entity: `bdd_norm_pr_gens_iq`
  - Module: `sage.rings.number_field.bdd_height`
  - Type: `function`

- **sage.rings.number_field.bdd_height.bdd_norm_pr_ideal_gens**
  - Entity: `bdd_norm_pr_ideal_gens`
  - Module: `sage.rings.number_field.bdd_height`
  - Type: `function`

- **sage.rings.number_field.bdd_height.integer_points_in_polytope**
  - Entity: `integer_points_in_polytope`
  - Module: `sage.rings.number_field.bdd_height`
  - Type: `function`

- **sage.rings.number_field.number_field.GaussianField**
  - Entity: `GaussianField`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.NumberField**
  - Entity: `NumberField`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.NumberFieldTower**
  - Entity: `NumberFieldTower`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.NumberField_absolute_v1**
  - Entity: `NumberField_absolute_v1`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.NumberField_cyclotomic_v1**
  - Entity: `NumberField_cyclotomic_v1`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.NumberField_generic_v1**
  - Entity: `NumberField_generic_v1`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.NumberField_quadratic_v1**
  - Entity: `NumberField_quadratic_v1`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.QuadraticField**
  - Entity: `QuadraticField`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.is_AbsoluteNumberField**
  - Entity: `is_AbsoluteNumberField`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.is_NumberFieldHomsetCodomain**
  - Entity: `is_NumberFieldHomsetCodomain`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.is_fundamental_discriminant**
  - Entity: `is_fundamental_discriminant`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.is_real_place**
  - Entity: `is_real_place`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.proof_flag**
  - Entity: `proof_flag`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.put_natural_embedding_first**
  - Entity: `put_natural_embedding_first`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field.refine_embedding**
  - Entity: `refine_embedding`
  - Module: `sage.rings.number_field.number_field`
  - Type: `function`

- **sage.rings.number_field.number_field_base.is_NumberField**
  - Entity: `is_NumberField`
  - Module: `sage.rings.number_field.number_field_base`
  - Type: `function`

- **sage.rings.number_field.number_field_element.is_NumberFieldElement**
  - Entity: `is_NumberFieldElement`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `function`

- **sage.rings.number_field.number_field_element_quadratic.is_sqrt_disc**
  - Entity: `is_sqrt_disc`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `function`

- **sage.rings.number_field.number_field_ideal.basis_to_module**
  - Entity: `basis_to_module`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `function`

- **sage.rings.number_field.number_field_ideal.is_NumberFieldFractionalIdeal**
  - Entity: `is_NumberFieldFractionalIdeal`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `function`

- **sage.rings.number_field.number_field_ideal.is_NumberFieldIdeal**
  - Entity: `is_NumberFieldIdeal`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `function`

- **sage.rings.number_field.number_field_ideal.quotient_char_p**
  - Entity: `quotient_char_p`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `function`

- **sage.rings.number_field.number_field_ideal_rel.is_NumberFieldFractionalIdeal_rel**
  - Entity: `is_NumberFieldFractionalIdeal_rel`
  - Module: `sage.rings.number_field.number_field_ideal_rel`
  - Type: `function`

- **sage.rings.number_field.number_field_morphisms.closest**
  - Entity: `closest`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `function`

- **sage.rings.number_field.number_field_morphisms.create_embedding_from_approx**
  - Entity: `create_embedding_from_approx`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `function`

- **sage.rings.number_field.number_field_morphisms.matching_root**
  - Entity: `matching_root`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `function`

- **sage.rings.number_field.number_field_morphisms.root_from_approx**
  - Entity: `root_from_approx`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `function`

- **sage.rings.number_field.number_field_rel.NumberField_extension_v1**
  - Entity: `NumberField_extension_v1`
  - Module: `sage.rings.number_field.number_field_rel`
  - Type: `function`

- **sage.rings.number_field.number_field_rel.NumberField_relative_v1**
  - Entity: `NumberField_relative_v1`
  - Module: `sage.rings.number_field.number_field_rel`
  - Type: `function`

- **sage.rings.number_field.number_field_rel.is_RelativeNumberField**
  - Entity: `is_RelativeNumberField`
  - Module: `sage.rings.number_field.number_field_rel`
  - Type: `function`

- **sage.rings.number_field.order.EisensteinIntegers**
  - Entity: `EisensteinIntegers`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.EquationOrder**
  - Entity: `EquationOrder`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.GaussianIntegers**
  - Entity: `GaussianIntegers`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.absolute_order_from_module_generators**
  - Entity: `absolute_order_from_module_generators`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.absolute_order_from_ring_generators**
  - Entity: `absolute_order_from_ring_generators`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.each_is_integral**
  - Entity: `each_is_integral`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.is_NumberFieldOrder**
  - Entity: `is_NumberFieldOrder`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.quadratic_order_class_number**
  - Entity: `quadratic_order_class_number`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order.relative_order_from_ring_generators**
  - Entity: `relative_order_from_ring_generators`
  - Module: `sage.rings.number_field.order`
  - Type: `function`

- **sage.rings.number_field.order_ideal.NumberFieldOrderIdeal**
  - Entity: `NumberFieldOrderIdeal`
  - Module: `sage.rings.number_field.order_ideal`
  - Type: `function`

- **sage.rings.number_field.selmer_group.basis_for_p_cokernel**
  - Entity: `basis_for_p_cokernel`
  - Module: `sage.rings.number_field.selmer_group`
  - Type: `function`

- **sage.rings.number_field.selmer_group.coords_in_U_mod_p**
  - Entity: `coords_in_U_mod_p`
  - Module: `sage.rings.number_field.selmer_group`
  - Type: `function`

- **sage.rings.number_field.selmer_group.pSelmerGroup**
  - Entity: `pSelmerGroup`
  - Module: `sage.rings.number_field.selmer_group`
  - Type: `function`

- **sage.rings.number_field.splitting_field.SplittingFieldAbort**
  - Entity: `SplittingFieldAbort`
  - Module: `sage.rings.number_field.splitting_field`
  - Type: `function`

- **sage.rings.number_field.splitting_field.splitting_field**
  - Entity: `splitting_field`
  - Module: `sage.rings.number_field.splitting_field`
  - Type: `function`

- **sage.rings.number_field.totallyreal.enumerate_totallyreal_fields_prim**
  - Entity: `enumerate_totallyreal_fields_prim`
  - Module: `sage.rings.number_field.totallyreal`
  - Type: `function`

- **sage.rings.number_field.totallyreal.odlyzko_bound_totallyreal**
  - Entity: `odlyzko_bound_totallyreal`
  - Module: `sage.rings.number_field.totallyreal`
  - Type: `function`

- **sage.rings.number_field.totallyreal.weed_fields**
  - Entity: `weed_fields`
  - Module: `sage.rings.number_field.totallyreal`
  - Type: `function`

- **sage.rings.number_field.totallyreal_data.easy_is_irreducible_py**
  - Entity: `easy_is_irreducible_py`
  - Module: `sage.rings.number_field.totallyreal_data`
  - Type: `function`

- **sage.rings.number_field.totallyreal_data.hermite_constant**
  - Entity: `hermite_constant`
  - Module: `sage.rings.number_field.totallyreal_data`
  - Type: `function`

- **sage.rings.number_field.totallyreal_data.int_has_small_square_divisor**
  - Entity: `int_has_small_square_divisor`
  - Module: `sage.rings.number_field.totallyreal_data`
  - Type: `function`

- **sage.rings.number_field.totallyreal_data.lagrange_degree_3**
  - Entity: `lagrange_degree_3`
  - Module: `sage.rings.number_field.totallyreal_data`
  - Type: `function`

- **sage.rings.number_field.totallyreal_phc.coefficients_to_power_sums**
  - Entity: `coefficients_to_power_sums`
  - Module: `sage.rings.number_field.totallyreal_phc`
  - Type: `function`

- **sage.rings.number_field.totallyreal_rel.enumerate_totallyreal_fields_all**
  - Entity: `enumerate_totallyreal_fields_all`
  - Module: `sage.rings.number_field.totallyreal_rel`
  - Type: `function`

- **sage.rings.number_field.totallyreal_rel.enumerate_totallyreal_fields_rel**
  - Entity: `enumerate_totallyreal_fields_rel`
  - Module: `sage.rings.number_field.totallyreal_rel`
  - Type: `function`

- **sage.rings.number_field.totallyreal_rel.integral_elements_in_box**
  - Entity: `integral_elements_in_box`
  - Module: `sage.rings.number_field.totallyreal_rel`
  - Type: `function`

- **sage.rings.padics.factory.QpCR**
  - Entity: `QpCR`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.QpER**
  - Entity: `QpER`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.QpFP**
  - Entity: `QpFP`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.QpLC**
  - Entity: `QpLC`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.QpLF**
  - Entity: `QpLF`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.Qq**
  - Entity: `Qq`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.QqCR**
  - Entity: `QqCR`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.QqFP**
  - Entity: `QqFP`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZpCA**
  - Entity: `ZpCA`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZpCR**
  - Entity: `ZpCR`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZpER**
  - Entity: `ZpER`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZpFM**
  - Entity: `ZpFM`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZpFP**
  - Entity: `ZpFP`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZpLC**
  - Entity: `ZpLC`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZpLF**
  - Entity: `ZpLF`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.Zq**
  - Entity: `Zq`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZqCA**
  - Entity: `ZqCA`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZqCR**
  - Entity: `ZqCR`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZqFM**
  - Entity: `ZqFM`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.ZqFP**
  - Entity: `ZqFP`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.get_key_base**
  - Entity: `get_key_base`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.is_eisenstein**
  - Entity: `is_eisenstein`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.is_unramified**
  - Entity: `is_unramified`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.krasner_check**
  - Entity: `krasner_check`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.split**
  - Entity: `split`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.factory.truncate_to_prec**
  - Entity: `truncate_to_prec`
  - Module: `sage.rings.padics.factory`
  - Type: `function`

- **sage.rings.padics.misc.gauss_sum**
  - Entity: `gauss_sum`
  - Module: `sage.rings.padics.misc`
  - Type: `function`

- **sage.rings.padics.misc.max**
  - Entity: `max`
  - Module: `sage.rings.padics.misc`
  - Type: `function`

- **sage.rings.padics.misc.min**
  - Entity: `min`
  - Module: `sage.rings.padics.misc`
  - Type: `function`

- **sage.rings.padics.misc.precprint**
  - Entity: `precprint`
  - Module: `sage.rings.padics.misc`
  - Type: `function`

- **sage.rings.padics.misc.trim_zeros**
  - Entity: `trim_zeros`
  - Module: `sage.rings.padics.misc`
  - Type: `function`

- **sage.rings.padics.padic_ZZ_pX_CA_element.make_ZZpXCAElement**
  - Entity: `make_ZZpXCAElement`
  - Module: `sage.rings.padics.padic_ZZ_pX_CA_element`
  - Type: `function`

- **sage.rings.padics.padic_ZZ_pX_CR_element.make_ZZpXCRElement**
  - Entity: `make_ZZpXCRElement`
  - Module: `sage.rings.padics.padic_ZZ_pX_CR_element`
  - Type: `function`

- **sage.rings.padics.padic_ZZ_pX_FM_element.make_ZZpXFMElement**
  - Entity: `make_ZZpXFMElement`
  - Module: `sage.rings.padics.padic_ZZ_pX_FM_element`
  - Type: `function`

- **sage.rings.padics.padic_capped_absolute_element.make_pAdicCappedAbsoluteElement**
  - Entity: `make_pAdicCappedAbsoluteElement`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `function`

- **sage.rings.padics.padic_capped_absolute_element.unpickle_cae_v2**
  - Entity: `unpickle_cae_v2`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `function`

- **sage.rings.padics.padic_capped_relative_element.base_p_list**
  - Entity: `base_p_list`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `function`

- **sage.rings.padics.padic_capped_relative_element.unpickle_cre_v2**
  - Entity: `unpickle_cre_v2`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `function`

- **sage.rings.padics.padic_capped_relative_element.unpickle_pcre_v1**
  - Entity: `unpickle_pcre_v1`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `function`

- **sage.rings.padics.padic_fixed_mod_element.make_pAdicFixedModElement**
  - Entity: `make_pAdicFixedModElement`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `function`

- **sage.rings.padics.padic_fixed_mod_element.unpickle_fme_v2**
  - Entity: `unpickle_fme_v2`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `function`

- **sage.rings.padics.padic_generic.local_print_mode**
  - Entity: `local_print_mode`
  - Module: `sage.rings.padics.padic_generic`
  - Type: `function`

- **sage.rings.padics.padic_generic_element.dwork_mahler_coeffs**
  - Entity: `dwork_mahler_coeffs`
  - Module: `sage.rings.padics.padic_generic_element`
  - Type: `function`

- **sage.rings.padics.padic_generic_element.evaluate_dwork_mahler**
  - Entity: `evaluate_dwork_mahler`
  - Module: `sage.rings.padics.padic_generic_element`
  - Type: `function`

- **sage.rings.padics.padic_generic_element.gauss_table**
  - Entity: `gauss_table`
  - Module: `sage.rings.padics.padic_generic_element`
  - Type: `function`

- **sage.rings.padics.padic_printing.pAdicPrinter**
  - Entity: `pAdicPrinter`
  - Module: `sage.rings.padics.padic_printing`
  - Type: `function`

- **sage.rings.padics.pow_computer.PowComputer**
  - Entity: `PowComputer`
  - Module: `sage.rings.padics.pow_computer`
  - Type: `function`

- **sage.rings.padics.pow_computer_ext.PowComputer_ext_maker**
  - Entity: `PowComputer_ext_maker`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `function`

- **sage.rings.padics.pow_computer_ext.ZZ_pX_eis_shift_test**
  - Entity: `ZZ_pX_eis_shift_test`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `function`

- **sage.rings.polynomial.complex_roots.complex_roots**
  - Entity: `complex_roots`
  - Module: `sage.rings.polynomial.complex_roots`
  - Type: `function`

- **sage.rings.polynomial.complex_roots.interval_roots**
  - Entity: `interval_roots`
  - Module: `sage.rings.polynomial.complex_roots`
  - Type: `function`

- **sage.rings.polynomial.complex_roots.intervals_disjoint**
  - Entity: `intervals_disjoint`
  - Module: `sage.rings.polynomial.complex_roots`
  - Type: `function`

- **sage.rings.polynomial.convolution.convolution**
  - Entity: `convolution`
  - Module: `sage.rings.polynomial.convolution`
  - Type: `function`

- **sage.rings.polynomial.cyclotomic.bateman_bound**
  - Entity: `bateman_bound`
  - Module: `sage.rings.polynomial.cyclotomic`
  - Type: `function`

- **sage.rings.polynomial.cyclotomic.cyclotomic_coeffs**
  - Entity: `cyclotomic_coeffs`
  - Module: `sage.rings.polynomial.cyclotomic`
  - Type: `function`

- **sage.rings.polynomial.cyclotomic.cyclotomic_value**
  - Entity: `cyclotomic_value`
  - Module: `sage.rings.polynomial.cyclotomic`
  - Type: `function`

- **sage.rings.polynomial.groebner_fan.ideal_to_gfan_format**
  - Entity: `ideal_to_gfan_format`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `function`

- **sage.rings.polynomial.groebner_fan.max_degree**
  - Entity: `max_degree`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `function`

- **sage.rings.polynomial.groebner_fan.prefix_check**
  - Entity: `prefix_check`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `function`

- **sage.rings.polynomial.groebner_fan.ring_to_gfan_format**
  - Entity: `ring_to_gfan_format`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `function`

- **sage.rings.polynomial.groebner_fan.verts_for_normal**
  - Entity: `verts_for_normal`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `function`

- **sage.rings.polynomial.hilbert.first_hilbert_series**
  - Entity: `first_hilbert_series`
  - Module: `sage.rings.polynomial.hilbert`
  - Type: `function`

- **sage.rings.polynomial.hilbert.hilbert_poincare_series**
  - Entity: `hilbert_poincare_series`
  - Module: `sage.rings.polynomial.hilbert`
  - Type: `function`

- **sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing**
  - Entity: `LaurentPolynomialRing`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring`
  - Type: `function`

- **sage.rings.polynomial.laurent_polynomial_ring.from_fraction_field**
  - Entity: `from_fraction_field`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring`
  - Type: `function`

- **sage.rings.polynomial.laurent_polynomial_ring.is_LaurentPolynomialRing**
  - Entity: `is_LaurentPolynomialRing`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring`
  - Type: `function`

- **sage.rings.polynomial.msolve.groebner_basis_degrevlex**
  - Entity: `groebner_basis_degrevlex`
  - Module: `sage.rings.polynomial.msolve`
  - Type: `function`

- **sage.rings.polynomial.msolve.variety**
  - Entity: `variety`
  - Module: `sage.rings.polynomial.msolve`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial.is_MPolynomial**
  - Entity: `is_MPolynomial`
  - Module: `sage.rings.polynomial.multi_polynomial`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_element.degree_lowest_rational_function**
  - Entity: `degree_lowest_rational_function`
  - Module: `sage.rings.polynomial.multi_polynomial_element`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ideal.is_MPolynomialIdeal**
  - Entity: `is_MPolynomialIdeal`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ideal_libsingular.interred_libsingular**
  - Entity: `interred_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal_libsingular`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ideal_libsingular.kbase_libsingular**
  - Entity: `kbase_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal_libsingular`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ideal_libsingular.slimgb_libsingular**
  - Entity: `slimgb_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal_libsingular`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ideal_libsingular.std_libsingular**
  - Entity: `std_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal_libsingular`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_libsingular.unpickle_MPolynomialRing_libsingular**
  - Entity: `unpickle_MPolynomialRing_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_libsingular`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_libsingular.unpickle_MPolynomial_libsingular**
  - Entity: `unpickle_MPolynomial_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_libsingular`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ring_base.is_MPolynomialRing**
  - Entity: `is_MPolynomialRing`
  - Module: `sage.rings.polynomial.multi_polynomial_ring_base`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ring_base.unpickle_MPolynomialRing_generic**
  - Entity: `unpickle_MPolynomialRing_generic`
  - Module: `sage.rings.polynomial.multi_polynomial_ring_base`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_ring_base.unpickle_MPolynomialRing_generic_v1**
  - Entity: `unpickle_MPolynomialRing_generic_v1`
  - Module: `sage.rings.polynomial.multi_polynomial_ring_base`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence**
  - Entity: `PolynomialSequence`
  - Module: `sage.rings.polynomial.multi_polynomial_sequence`
  - Type: `function`

- **sage.rings.polynomial.multi_polynomial_sequence.is_PolynomialSequence**
  - Entity: `is_PolynomialSequence`
  - Module: `sage.rings.polynomial.multi_polynomial_sequence`
  - Type: `function`

- **sage.rings.polynomial.omega.MacMahonOmega**
  - Entity: `MacMahonOmega`
  - Module: `sage.rings.polynomial.omega`
  - Type: `function`

- **sage.rings.polynomial.omega.Omega_ge**
  - Entity: `Omega_ge`
  - Module: `sage.rings.polynomial.omega`
  - Type: `function`

- **sage.rings.polynomial.omega.homogeneous_symmetric_function**
  - Entity: `homogeneous_symmetric_function`
  - Module: `sage.rings.polynomial.omega`
  - Type: `function`

- **sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense.make_padic_poly**
  - Entity: `make_padic_poly`
  - Module: `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.TermOrder_from_pb_order**
  - Entity: `TermOrder_from_pb_order`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.add_up_polynomials**
  - Entity: `add_up_polynomials`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.contained_vars**
  - Entity: `contained_vars`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.easy_linear_factors**
  - Entity: `easy_linear_factors`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.gauss_on_polys**
  - Entity: `gauss_on_polys`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.get_var_mapping**
  - Entity: `get_var_mapping`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.if_then_else**
  - Entity: `if_then_else`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.interpolate**
  - Entity: `interpolate`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.interpolate_smallest_lex**
  - Entity: `interpolate_smallest_lex`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.ll_red_nf_noredsb**
  - Entity: `ll_red_nf_noredsb`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.ll_red_nf_noredsb_single_recursive_call**
  - Entity: `ll_red_nf_noredsb_single_recursive_call`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.ll_red_nf_redsb**
  - Entity: `ll_red_nf_redsb`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.map_every_x_to_x_plus_one**
  - Entity: `map_every_x_to_x_plus_one`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.mod_mon_set**
  - Entity: `mod_mon_set`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.mod_var_set**
  - Entity: `mod_var_set`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.mult_fact_sim_C**
  - Entity: `mult_fact_sim_C`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.nf3**
  - Entity: `nf3`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.parallel_reduce**
  - Entity: `parallel_reduce`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.random_set**
  - Entity: `random_set`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.recursively_insert**
  - Entity: `recursively_insert`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.red_tail**
  - Entity: `red_tail`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.set_random_seed**
  - Entity: `set_random_seed`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.substitute_variables**
  - Entity: `substitute_variables`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.top_index**
  - Entity: `top_index`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomial**
  - Entity: `unpickle_BooleanPolynomial`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomial0**
  - Entity: `unpickle_BooleanPolynomial0`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomialRing**
  - Entity: `unpickle_BooleanPolynomialRing`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.pbori.pbori.zeros**
  - Entity: `zeros`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `function`

- **sage.rings.polynomial.plural.ExteriorAlgebra**
  - Entity: `ExteriorAlgebra`
  - Module: `sage.rings.polynomial.plural`
  - Type: `function`

- **sage.rings.polynomial.plural.SCA**
  - Entity: `SCA`
  - Module: `sage.rings.polynomial.plural`
  - Type: `function`

- **sage.rings.polynomial.plural.new_CRing**
  - Entity: `new_CRing`
  - Module: `sage.rings.polynomial.plural`
  - Type: `function`

- **sage.rings.polynomial.plural.new_NRing**
  - Entity: `new_NRing`
  - Module: `sage.rings.polynomial.plural`
  - Type: `function`

- **sage.rings.polynomial.plural.new_Ring**
  - Entity: `new_Ring`
  - Module: `sage.rings.polynomial.plural`
  - Type: `function`

- **sage.rings.polynomial.plural.unpickle_NCPolynomial_plural**
  - Entity: `unpickle_NCPolynomial_plural`
  - Module: `sage.rings.polynomial.plural`
  - Type: `function`

- **sage.rings.polynomial.polydict.gen_index**
  - Entity: `gen_index`
  - Module: `sage.rings.polynomial.polydict`
  - Type: `function`

- **sage.rings.polynomial.polydict.make_ETuple**
  - Entity: `make_ETuple`
  - Module: `sage.rings.polynomial.polydict`
  - Type: `function`

- **sage.rings.polynomial.polydict.make_PolyDict**
  - Entity: `make_PolyDict`
  - Module: `sage.rings.polynomial.polydict`
  - Type: `function`

- **sage.rings.polynomial.polydict.monomial_exponent**
  - Entity: `monomial_exponent`
  - Module: `sage.rings.polynomial.polydict`
  - Type: `function`

- **sage.rings.polynomial.polynomial_element.generic_power_trunc**
  - Entity: `generic_power_trunc`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `function`

- **sage.rings.polynomial.polynomial_element.is_Polynomial**
  - Entity: `is_Polynomial`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `function`

- **sage.rings.polynomial.polynomial_element.make_generic_polynomial**
  - Entity: `make_generic_polynomial`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `function`

- **sage.rings.polynomial.polynomial_element.polynomial_is_variable**
  - Entity: `polynomial_is_variable`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `function`

- **sage.rings.polynomial.polynomial_element.universal_discriminant**
  - Entity: `universal_discriminant`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `function`

- **sage.rings.polynomial.polynomial_gf2x.GF2X_BuildIrred_list**
  - Entity: `GF2X_BuildIrred_list`
  - Module: `sage.rings.polynomial.polynomial_gf2x`
  - Type: `function`

- **sage.rings.polynomial.polynomial_gf2x.GF2X_BuildRandomIrred_list**
  - Entity: `GF2X_BuildRandomIrred_list`
  - Module: `sage.rings.polynomial.polynomial_gf2x`
  - Type: `function`

- **sage.rings.polynomial.polynomial_gf2x.GF2X_BuildSparseIrred_list**
  - Entity: `GF2X_BuildSparseIrred_list`
  - Module: `sage.rings.polynomial.polynomial_gf2x`
  - Type: `function`

- **sage.rings.polynomial.polynomial_gf2x.make_element**
  - Entity: `make_element`
  - Module: `sage.rings.polynomial.polynomial_gf2x`
  - Type: `function`

- **sage.rings.polynomial.polynomial_modn_dense_ntl.make_element**
  - Entity: `make_element`
  - Module: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - Type: `function`

- **sage.rings.polynomial.polynomial_modn_dense_ntl.small_roots**
  - Entity: `small_roots`
  - Module: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - Type: `function`

- **sage.rings.polynomial.polynomial_quotient_ring.is_PolynomialQuotientRing**
  - Entity: `is_PolynomialQuotientRing`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring`
  - Type: `function`

- **sage.rings.polynomial.polynomial_real_mpfr_dense.make_PolynomialRealDense**
  - Entity: `make_PolynomialRealDense`
  - Module: `sage.rings.polynomial.polynomial_real_mpfr_dense`
  - Type: `function`

- **sage.rings.polynomial.polynomial_ring.is_PolynomialRing**
  - Entity: `is_PolynomialRing`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `function`

- **sage.rings.polynomial.polynomial_ring.polygen**
  - Entity: `polygen`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `function`

- **sage.rings.polynomial.polynomial_ring.polygens**
  - Entity: `polygens`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `function`

- **sage.rings.polynomial.polynomial_ring_constructor.BooleanPolynomialRing_constructor**
  - Entity: `BooleanPolynomialRing_constructor`
  - Module: `sage.rings.polynomial.polynomial_ring_constructor`
  - Type: `function`

- **sage.rings.polynomial.polynomial_ring_constructor.PolynomialRing**
  - Entity: `PolynomialRing`
  - Module: `sage.rings.polynomial.polynomial_ring_constructor`
  - Type: `function`

- **sage.rings.polynomial.polynomial_ring_constructor.polynomial_default_category**
  - Entity: `polynomial_default_category`
  - Module: `sage.rings.polynomial.polynomial_ring_constructor`
  - Type: `function`

- **sage.rings.polynomial.polynomial_ring_constructor.unpickle_PolynomialRing**
  - Entity: `unpickle_PolynomialRing`
  - Module: `sage.rings.polynomial.polynomial_ring_constructor`
  - Type: `function`

#### MODULE (109 entries)

- **sage.rings.ideal**
  - Entity: `ideal`
  - Module: `sage.rings.ideal`
  - Type: `module`

- **sage.rings.ideal_monoid**
  - Entity: `ideal_monoid`
  - Module: `sage.rings.ideal_monoid`
  - Type: `module`

- **sage.rings.integer**
  - Entity: `integer`
  - Module: `sage.rings.integer`
  - Type: `module`

- **sage.rings.integer_ring**
  - Entity: `integer_ring`
  - Module: `sage.rings.integer_ring`
  - Type: `module`

- **sage.rings.number_field.S_unit_solver**
  - Entity: `S_unit_solver`
  - Module: `sage.rings.number_field.S_unit_solver`
  - Type: `module`

- **sage.rings.number_field.bdd_height**
  - Entity: `bdd_height`
  - Module: `sage.rings.number_field.bdd_height`
  - Type: `module`

- **sage.rings.number_field.class_group**
  - Entity: `class_group`
  - Module: `sage.rings.number_field.class_group`
  - Type: `module`

- **sage.rings.number_field.galois_group**
  - Entity: `galois_group`
  - Module: `sage.rings.number_field.galois_group`
  - Type: `module`

- **sage.rings.number_field.homset**
  - Entity: `homset`
  - Module: `sage.rings.number_field.homset`
  - Type: `module`

- **sage.rings.number_field.maps**
  - Entity: `maps`
  - Module: `sage.rings.number_field.maps`
  - Type: `module`

- **sage.rings.number_field.morphism**
  - Entity: `morphism`
  - Module: `sage.rings.number_field.morphism`
  - Type: `module`

- **sage.rings.number_field.number_field**
  - Entity: `number_field`
  - Module: `sage.rings.number_field.number_field`
  - Type: `module`

- **sage.rings.number_field.number_field_base**
  - Entity: `number_field_base`
  - Module: `sage.rings.number_field.number_field_base`
  - Type: `module`

- **sage.rings.number_field.number_field_element**
  - Entity: `number_field_element`
  - Module: `sage.rings.number_field.number_field_element`
  - Type: `module`

- **sage.rings.number_field.number_field_element_quadratic**
  - Entity: `number_field_element_quadratic`
  - Module: `sage.rings.number_field.number_field_element_quadratic`
  - Type: `module`

- **sage.rings.number_field.number_field_ideal**
  - Entity: `number_field_ideal`
  - Module: `sage.rings.number_field.number_field_ideal`
  - Type: `module`

- **sage.rings.number_field.number_field_ideal_rel**
  - Entity: `number_field_ideal_rel`
  - Module: `sage.rings.number_field.number_field_ideal_rel`
  - Type: `module`

- **sage.rings.number_field.number_field_morphisms**
  - Entity: `number_field_morphisms`
  - Module: `sage.rings.number_field.number_field_morphisms`
  - Type: `module`

- **sage.rings.number_field.number_field_rel**
  - Entity: `number_field_rel`
  - Module: `sage.rings.number_field.number_field_rel`
  - Type: `module`

- **sage.rings.number_field.order**
  - Entity: `order`
  - Module: `sage.rings.number_field.order`
  - Type: `module`

- **sage.rings.number_field.order_ideal**
  - Entity: `order_ideal`
  - Module: `sage.rings.number_field.order_ideal`
  - Type: `module`

- **sage.rings.number_field.selmer_group**
  - Entity: `selmer_group`
  - Module: `sage.rings.number_field.selmer_group`
  - Type: `module`

- **sage.rings.number_field.small_primes_of_degree_one**
  - Entity: `small_primes_of_degree_one`
  - Module: `sage.rings.number_field.small_primes_of_degree_one`
  - Type: `module`

- **sage.rings.number_field.splitting_field**
  - Entity: `splitting_field`
  - Module: `sage.rings.number_field.splitting_field`
  - Type: `module`

- **sage.rings.number_field.structure**
  - Entity: `structure`
  - Module: `sage.rings.number_field.structure`
  - Type: `module`

- **sage.rings.number_field.totallyreal**
  - Entity: `totallyreal`
  - Module: `sage.rings.number_field.totallyreal`
  - Type: `module`

- **sage.rings.number_field.totallyreal_data**
  - Entity: `totallyreal_data`
  - Module: `sage.rings.number_field.totallyreal_data`
  - Type: `module`

- **sage.rings.number_field.totallyreal_phc**
  - Entity: `totallyreal_phc`
  - Module: `sage.rings.number_field.totallyreal_phc`
  - Type: `module`

- **sage.rings.number_field.totallyreal_rel**
  - Entity: `totallyreal_rel`
  - Module: `sage.rings.number_field.totallyreal_rel`
  - Type: `module`

- **sage.rings.number_field.unit_group**
  - Entity: `unit_group`
  - Module: `sage.rings.number_field.unit_group`
  - Type: `module`

- **sage.rings.padics.common_conversion**
  - Entity: `common_conversion`
  - Module: `sage.rings.padics.common_conversion`
  - Type: `module`

- **sage.rings.padics.eisenstein_extension_generic**
  - Entity: `eisenstein_extension_generic`
  - Module: `sage.rings.padics.eisenstein_extension_generic`
  - Type: `module`

- **sage.rings.padics.factory**
  - Entity: `factory`
  - Module: `sage.rings.padics.factory`
  - Type: `module`

- **sage.rings.padics.generic_nodes**
  - Entity: `generic_nodes`
  - Module: `sage.rings.padics.generic_nodes`
  - Type: `module`

- **sage.rings.padics.local_generic**
  - Entity: `local_generic`
  - Module: `sage.rings.padics.local_generic`
  - Type: `module`

- **sage.rings.padics.local_generic_element**
  - Entity: `local_generic_element`
  - Module: `sage.rings.padics.local_generic_element`
  - Type: `module`

- **sage.rings.padics.misc**
  - Entity: `misc`
  - Module: `sage.rings.padics.misc`
  - Type: `module`

- **sage.rings.padics.morphism**
  - Entity: `morphism`
  - Module: `sage.rings.padics.morphism`
  - Type: `module`

- **sage.rings.padics.padic_ZZ_pX_CA_element**
  - Entity: `padic_ZZ_pX_CA_element`
  - Module: `sage.rings.padics.padic_ZZ_pX_CA_element`
  - Type: `module`

- **sage.rings.padics.padic_ZZ_pX_CR_element**
  - Entity: `padic_ZZ_pX_CR_element`
  - Module: `sage.rings.padics.padic_ZZ_pX_CR_element`
  - Type: `module`

- **sage.rings.padics.padic_ZZ_pX_FM_element**
  - Entity: `padic_ZZ_pX_FM_element`
  - Module: `sage.rings.padics.padic_ZZ_pX_FM_element`
  - Type: `module`

- **sage.rings.padics.padic_ZZ_pX_element**
  - Entity: `padic_ZZ_pX_element`
  - Module: `sage.rings.padics.padic_ZZ_pX_element`
  - Type: `module`

- **sage.rings.padics.padic_base_generic**
  - Entity: `padic_base_generic`
  - Module: `sage.rings.padics.padic_base_generic`
  - Type: `module`

- **sage.rings.padics.padic_base_leaves**
  - Entity: `padic_base_leaves`
  - Module: `sage.rings.padics.padic_base_leaves`
  - Type: `module`

- **sage.rings.padics.padic_capped_absolute_element**
  - Entity: `padic_capped_absolute_element`
  - Module: `sage.rings.padics.padic_capped_absolute_element`
  - Type: `module`

- **sage.rings.padics.padic_capped_relative_element**
  - Entity: `padic_capped_relative_element`
  - Module: `sage.rings.padics.padic_capped_relative_element`
  - Type: `module`

- **sage.rings.padics.padic_ext_element**
  - Entity: `padic_ext_element`
  - Module: `sage.rings.padics.padic_ext_element`
  - Type: `module`

- **sage.rings.padics.padic_extension_generic**
  - Entity: `padic_extension_generic`
  - Module: `sage.rings.padics.padic_extension_generic`
  - Type: `module`

- **sage.rings.padics.padic_extension_leaves**
  - Entity: `padic_extension_leaves`
  - Module: `sage.rings.padics.padic_extension_leaves`
  - Type: `module`

- **sage.rings.padics.padic_fixed_mod_element**
  - Entity: `padic_fixed_mod_element`
  - Module: `sage.rings.padics.padic_fixed_mod_element`
  - Type: `module`

- **sage.rings.padics.padic_generic**
  - Entity: `padic_generic`
  - Module: `sage.rings.padics.padic_generic`
  - Type: `module`

- **sage.rings.padics.padic_generic_element**
  - Entity: `padic_generic_element`
  - Module: `sage.rings.padics.padic_generic_element`
  - Type: `module`

- **sage.rings.padics.padic_printing**
  - Entity: `padic_printing`
  - Module: `sage.rings.padics.padic_printing`
  - Type: `module`

- **sage.rings.padics.padic_valuation**
  - Entity: `padic_valuation`
  - Module: `sage.rings.padics.padic_valuation`
  - Type: `module`

- **sage.rings.padics.pow_computer**
  - Entity: `pow_computer`
  - Module: `sage.rings.padics.pow_computer`
  - Type: `module`

- **sage.rings.padics.pow_computer_ext**
  - Entity: `pow_computer_ext`
  - Module: `sage.rings.padics.pow_computer_ext`
  - Type: `module`

- **sage.rings.padics.precision_error**
  - Entity: `precision_error`
  - Module: `sage.rings.padics.precision_error`
  - Type: `module`

- **sage.rings.padics.tutorial**
  - Entity: `tutorial`
  - Module: `sage.rings.padics.tutorial`
  - Type: `module`

- **sage.rings.padics.unramified_extension_generic**
  - Entity: `unramified_extension_generic`
  - Module: `sage.rings.padics.unramified_extension_generic`
  - Type: `module`

- **sage.rings.polynomial.complex_roots**
  - Entity: `complex_roots`
  - Module: `sage.rings.polynomial.complex_roots`
  - Type: `module`

- **sage.rings.polynomial.convolution**
  - Entity: `convolution`
  - Module: `sage.rings.polynomial.convolution`
  - Type: `module`

- **sage.rings.polynomial.cyclotomic**
  - Entity: `cyclotomic`
  - Module: `sage.rings.polynomial.cyclotomic`
  - Type: `module`

- **sage.rings.polynomial.flatten**
  - Entity: `flatten`
  - Module: `sage.rings.polynomial.flatten`
  - Type: `module`

- **sage.rings.polynomial.groebner_fan**
  - Entity: `groebner_fan`
  - Module: `sage.rings.polynomial.groebner_fan`
  - Type: `module`

- **sage.rings.polynomial.hilbert**
  - Entity: `hilbert`
  - Module: `sage.rings.polynomial.hilbert`
  - Type: `module`

- **sage.rings.polynomial.ideal**
  - Entity: `ideal`
  - Module: `sage.rings.polynomial.ideal`
  - Type: `module`

- **sage.rings.polynomial.infinite_polynomial_element**
  - Entity: `infinite_polynomial_element`
  - Module: `sage.rings.polynomial.infinite_polynomial_element`
  - Type: `module`

- **sage.rings.polynomial.infinite_polynomial_ring**
  - Entity: `infinite_polynomial_ring`
  - Module: `sage.rings.polynomial.infinite_polynomial_ring`
  - Type: `module`

- **sage.rings.polynomial.integer_valued_polynomials**
  - Entity: `integer_valued_polynomials`
  - Module: `sage.rings.polynomial.integer_valued_polynomials`
  - Type: `module`

- **sage.rings.polynomial.laurent_polynomial**
  - Entity: `laurent_polynomial`
  - Module: `sage.rings.polynomial.laurent_polynomial`
  - Type: `module`

- **sage.rings.polynomial.laurent_polynomial_ring**
  - Entity: `laurent_polynomial_ring`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring`
  - Type: `module`

- **sage.rings.polynomial.laurent_polynomial_ring_base**
  - Entity: `laurent_polynomial_ring_base`
  - Module: `sage.rings.polynomial.laurent_polynomial_ring_base`
  - Type: `module`

- **sage.rings.polynomial.msolve**
  - Entity: `msolve`
  - Module: `sage.rings.polynomial.msolve`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial**
  - Entity: `multi_polynomial`
  - Module: `sage.rings.polynomial.multi_polynomial`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial_element**
  - Entity: `multi_polynomial_element`
  - Module: `sage.rings.polynomial.multi_polynomial_element`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial_ideal**
  - Entity: `multi_polynomial_ideal`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial_ideal_libsingular**
  - Entity: `multi_polynomial_ideal_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_ideal_libsingular`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial_libsingular**
  - Entity: `multi_polynomial_libsingular`
  - Module: `sage.rings.polynomial.multi_polynomial_libsingular`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial_ring**
  - Entity: `multi_polynomial_ring`
  - Module: `sage.rings.polynomial.multi_polynomial_ring`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial_ring_base**
  - Entity: `multi_polynomial_ring_base`
  - Module: `sage.rings.polynomial.multi_polynomial_ring_base`
  - Type: `module`

- **sage.rings.polynomial.multi_polynomial_sequence**
  - Entity: `multi_polynomial_sequence`
  - Module: `sage.rings.polynomial.multi_polynomial_sequence`
  - Type: `module`

- **sage.rings.polynomial.omega**
  - Entity: `omega`
  - Module: `sage.rings.polynomial.omega`
  - Type: `module`

- **sage.rings.polynomial.ore_function_element**
  - Entity: `ore_function_element`
  - Module: `sage.rings.polynomial.ore_function_element`
  - Type: `module`

- **sage.rings.polynomial.ore_function_field**
  - Entity: `ore_function_field`
  - Module: `sage.rings.polynomial.ore_function_field`
  - Type: `module`

- **sage.rings.polynomial.ore_polynomial_element**
  - Entity: `ore_polynomial_element`
  - Module: `sage.rings.polynomial.ore_polynomial_element`
  - Type: `module`

- **sage.rings.polynomial.ore_polynomial_ring**
  - Entity: `ore_polynomial_ring`
  - Module: `sage.rings.polynomial.ore_polynomial_ring`
  - Type: `module`

- **sage.rings.polynomial.padics.polynomial_padic**
  - Entity: `polynomial_padic`
  - Module: `sage.rings.polynomial.padics.polynomial_padic`
  - Type: `module`

- **sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense**
  - Entity: `polynomial_padic_capped_relative_dense`
  - Module: `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense`
  - Type: `module`

- **sage.rings.polynomial.padics.polynomial_padic_flat**
  - Entity: `polynomial_padic_flat`
  - Module: `sage.rings.polynomial.padics.polynomial_padic_flat`
  - Type: `module`

- **sage.rings.polynomial.pbori.pbori**
  - Entity: `pbori`
  - Module: `sage.rings.polynomial.pbori.pbori`
  - Type: `module`

- **sage.rings.polynomial.plural**
  - Entity: `plural`
  - Module: `sage.rings.polynomial.plural`
  - Type: `module`

- **sage.rings.polynomial.polydict**
  - Entity: `polydict`
  - Module: `sage.rings.polynomial.polydict`
  - Type: `module`

- **sage.rings.polynomial.polynomial_compiled**
  - Entity: `polynomial_compiled`
  - Module: `sage.rings.polynomial.polynomial_compiled`
  - Type: `module`

- **sage.rings.polynomial.polynomial_element**
  - Entity: `polynomial_element`
  - Module: `sage.rings.polynomial.polynomial_element`
  - Type: `module`

- **sage.rings.polynomial.polynomial_element_generic**
  - Entity: `polynomial_element_generic`
  - Module: `sage.rings.polynomial.polynomial_element_generic`
  - Type: `module`

- **sage.rings.polynomial.polynomial_fateman**
  - Entity: `polynomial_fateman`
  - Module: `sage.rings.polynomial.polynomial_fateman`
  - Type: `module`

- **sage.rings.polynomial.polynomial_gf2x**
  - Entity: `polynomial_gf2x`
  - Module: `sage.rings.polynomial.polynomial_gf2x`
  - Type: `module`

- **sage.rings.polynomial.polynomial_integer_dense_flint**
  - Entity: `polynomial_integer_dense_flint`
  - Module: `sage.rings.polynomial.polynomial_integer_dense_flint`
  - Type: `module`

- **sage.rings.polynomial.polynomial_integer_dense_ntl**
  - Entity: `polynomial_integer_dense_ntl`
  - Module: `sage.rings.polynomial.polynomial_integer_dense_ntl`
  - Type: `module`

- **sage.rings.polynomial.polynomial_modn_dense_ntl**
  - Entity: `polynomial_modn_dense_ntl`
  - Module: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - Type: `module`

- **sage.rings.polynomial.polynomial_number_field**
  - Entity: `polynomial_number_field`
  - Module: `sage.rings.polynomial.polynomial_number_field`
  - Type: `module`

- **sage.rings.polynomial.polynomial_quotient_ring**
  - Entity: `polynomial_quotient_ring`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring`
  - Type: `module`

- **sage.rings.polynomial.polynomial_quotient_ring_element**
  - Entity: `polynomial_quotient_ring_element`
  - Module: `sage.rings.polynomial.polynomial_quotient_ring_element`
  - Type: `module`

- **sage.rings.polynomial.polynomial_rational_flint**
  - Entity: `polynomial_rational_flint`
  - Module: `sage.rings.polynomial.polynomial_rational_flint`
  - Type: `module`

- **sage.rings.polynomial.polynomial_real_mpfr_dense**
  - Entity: `polynomial_real_mpfr_dense`
  - Module: `sage.rings.polynomial.polynomial_real_mpfr_dense`
  - Type: `module`

- **sage.rings.polynomial.polynomial_ring**
  - Entity: `polynomial_ring`
  - Module: `sage.rings.polynomial.polynomial_ring`
  - Type: `module`

- **sage.rings.polynomial.polynomial_ring_constructor**
  - Entity: `polynomial_ring_constructor`
  - Module: `sage.rings.polynomial.polynomial_ring_constructor`
  - Type: `module`

- **sage.rings.polynomial.polynomial_ring_homomorphism**
  - Entity: `polynomial_ring_homomorphism`
  - Module: `sage.rings.polynomial.polynomial_ring_homomorphism`
  - Type: `module`

- **sage.rings.polynomial.polynomial_singular_interface**
  - Entity: `polynomial_singular_interface`
  - Module: `sage.rings.polynomial.polynomial_singular_interface`
  - Type: `module`


### Part 13 (443 entries)

#### ATTRIBUTE (9 entries)

- **sage.rings.polynomial.term_order.greater_tuple**
  - Entity: `greater_tuple`
  - Module: `sage.rings.polynomial.term_order`
  - Type: `attribute`

- **sage.rings.polynomial.term_order.sortkey**
  - Entity: `sortkey`
  - Module: `sage.rings.polynomial.term_order`
  - Type: `attribute`

- **sage.rings.real_mpfr.base**
  - Entity: `base`
  - Module: `sage.rings.real_mpfr`
  - Type: `attribute`

- **sage.rings.real_mpfr.literal**
  - Entity: `literal`
  - Module: `sage.rings.real_mpfr`
  - Type: `attribute`

- **sage.schemes.elliptic_curves.period_lattice.is_approximate**
  - Entity: `is_approximate`
  - Module: `sage.schemes.elliptic_curves.period_lattice`
  - Type: `attribute`

- **sage.schemes.elliptic_curves.period_lattice_region.data**
  - Entity: `data`
  - Module: `sage.schemes.elliptic_curves.period_lattice_region`
  - Type: `attribute`

- **sage.schemes.elliptic_curves.period_lattice_region.full**
  - Entity: `full`
  - Module: `sage.schemes.elliptic_curves.period_lattice_region`
  - Type: `attribute`

- **sage.schemes.elliptic_curves.period_lattice_region.w1**
  - Entity: `w1`
  - Module: `sage.schemes.elliptic_curves.period_lattice_region`
  - Type: `attribute`

- **sage.schemes.elliptic_curves.period_lattice_region.w2**
  - Entity: `w2`
  - Module: `sage.schemes.elliptic_curves.period_lattice_region`
  - Type: `attribute`

#### CLASS (123 entries)

- **sage.rings.polynomial.polynomial_singular_interface.Polynomial_singular_repr**
  - Entity: `Polynomial_singular_repr`
  - Module: `sage.rings.polynomial.polynomial_singular_interface`
  - Type: `class`

- **sage.rings.polynomial.polynomial_zmod_flint.Polynomial_template**
  - Entity: `Polynomial_template`
  - Module: `sage.rings.polynomial.polynomial_zmod_flint`
  - Type: `class`

- **sage.rings.polynomial.polynomial_zmod_flint.Polynomial_zmod_flint**
  - Entity: `Polynomial_zmod_flint`
  - Module: `sage.rings.polynomial.polynomial_zmod_flint`
  - Type: `class`

- **sage.rings.polynomial.polynomial_zz_pex.Polynomial_ZZ_pEX**
  - Entity: `Polynomial_ZZ_pEX`
  - Module: `sage.rings.polynomial.polynomial_zz_pex`
  - Type: `class`

- **sage.rings.polynomial.polynomial_zz_pex.Polynomial_template**
  - Entity: `Polynomial_template`
  - Module: `sage.rings.polynomial.polynomial_zz_pex`
  - Type: `class`

- **sage.rings.polynomial.q_integer_valued_polynomials.Bases**
  - Entity: `Bases`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.q_integer_valued_polynomials.Binomial**
  - Entity: `Binomial`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.q_integer_valued_polynomials.Element**
  - Entity: `Element`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.q_integer_valued_polynomials.ElementMethods**
  - Entity: `ElementMethods`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.q_integer_valued_polynomials.ParentMethods**
  - Entity: `ParentMethods`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.q_integer_valued_polynomials.QuantumValuedPolynomialRing**
  - Entity: `QuantumValuedPolynomialRing`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.q_integer_valued_polynomials.Shifted**
  - Entity: `Shifted`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `class`

- **sage.rings.polynomial.real_roots.bernstein_polynomial_factory**
  - Entity: `bernstein_polynomial_factory`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.bernstein_polynomial_factory_ar**
  - Entity: `bernstein_polynomial_factory_ar`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.bernstein_polynomial_factory_intlist**
  - Entity: `bernstein_polynomial_factory_intlist`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.bernstein_polynomial_factory_ratlist**
  - Entity: `bernstein_polynomial_factory_ratlist`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.context**
  - Entity: `context`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.interval_bernstein_polynomial**
  - Entity: `interval_bernstein_polynomial`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.interval_bernstein_polynomial_float**
  - Entity: `interval_bernstein_polynomial_float`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.interval_bernstein_polynomial_integer**
  - Entity: `interval_bernstein_polynomial_integer`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.island**
  - Entity: `island`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.linear_map**
  - Entity: `linear_map`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.ocean**
  - Entity: `ocean`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.rr_gap**
  - Entity: `rr_gap`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.real_roots.warp_map**
  - Entity: `warp_map`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_element.SkewPolynomial_generic_dense**
  - Entity: `SkewPolynomial_generic_dense`
  - Module: `sage.rings.polynomial.skew_polynomial_element`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_finite_field.SkewPolynomial_finite_field_dense**
  - Entity: `SkewPolynomial_finite_field_dense`
  - Module: `sage.rings.polynomial.skew_polynomial_finite_field`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_finite_order.SkewPolynomial_finite_order_dense**
  - Entity: `SkewPolynomial_finite_order_dense`
  - Module: `sage.rings.polynomial.skew_polynomial_finite_order`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_ring.SectionSkewPolynomialCenterInjection**
  - Entity: `SectionSkewPolynomialCenterInjection`
  - Module: `sage.rings.polynomial.skew_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialCenterInjection**
  - Entity: `SkewPolynomialCenterInjection`
  - Module: `sage.rings.polynomial.skew_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing**
  - Entity: `SkewPolynomialRing`
  - Module: `sage.rings.polynomial.skew_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing_finite_field**
  - Entity: `SkewPolynomialRing_finite_field`
  - Module: `sage.rings.polynomial.skew_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing_finite_order**
  - Entity: `SkewPolynomialRing_finite_order`
  - Module: `sage.rings.polynomial.skew_polynomial_ring`
  - Type: `class`

- **sage.rings.polynomial.symmetric_ideal.SymmetricIdeal**
  - Entity: `SymmetricIdeal`
  - Module: `sage.rings.polynomial.symmetric_ideal`
  - Type: `class`

- **sage.rings.polynomial.symmetric_reduction.SymmetricReductionStrategy**
  - Entity: `SymmetricReductionStrategy`
  - Module: `sage.rings.polynomial.symmetric_reduction`
  - Type: `class`

- **sage.rings.polynomial.term_order.TermOrder**
  - Entity: `TermOrder`
  - Module: `sage.rings.polynomial.term_order`
  - Type: `class`

- **sage.rings.power_series_ring.PowerSeriesRing_domain**
  - Entity: `PowerSeriesRing_domain`
  - Module: `sage.rings.power_series_ring`
  - Type: `class`

- **sage.rings.power_series_ring.PowerSeriesRing_generic**
  - Entity: `PowerSeriesRing_generic`
  - Module: `sage.rings.power_series_ring`
  - Type: `class`

- **sage.rings.power_series_ring.PowerSeriesRing_over_field**
  - Entity: `PowerSeriesRing_over_field`
  - Module: `sage.rings.power_series_ring`
  - Type: `class`

- **sage.rings.power_series_ring_element.PowerSeries**
  - Entity: `PowerSeries`
  - Module: `sage.rings.power_series_ring_element`
  - Type: `class`

- **sage.rings.rational.Q_to_Z**
  - Entity: `Q_to_Z`
  - Module: `sage.rings.rational`
  - Type: `class`

- **sage.rings.rational.Rational**
  - Entity: `Rational`
  - Module: `sage.rings.rational`
  - Type: `class`

- **sage.rings.rational.Z_to_Q**
  - Entity: `Z_to_Q`
  - Module: `sage.rings.rational`
  - Type: `class`

- **sage.rings.rational.int_to_Q**
  - Entity: `int_to_Q`
  - Module: `sage.rings.rational`
  - Type: `class`

- **sage.rings.rational_field.RationalField**
  - Entity: `RationalField`
  - Module: `sage.rings.rational_field`
  - Type: `class`

- **sage.rings.real_double.RealDoubleElement**
  - Entity: `RealDoubleElement`
  - Module: `sage.rings.real_double`
  - Type: `class`

- **sage.rings.real_double.RealDoubleField_class**
  - Entity: `RealDoubleField_class`
  - Module: `sage.rings.real_double`
  - Type: `class`

- **sage.rings.real_double.ToRDF**
  - Entity: `ToRDF`
  - Module: `sage.rings.real_double`
  - Type: `class`

- **sage.rings.real_mpfr.QQtoRR**
  - Entity: `QQtoRR`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.rings.real_mpfr.RRtoRR**
  - Entity: `RRtoRR`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.rings.real_mpfr.RealField_class**
  - Entity: `RealField_class`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.rings.real_mpfr.RealLiteral**
  - Entity: `RealLiteral`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.rings.real_mpfr.RealNumber**
  - Entity: `RealNumber`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.rings.real_mpfr.ZZtoRR**
  - Entity: `ZZtoRR`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.rings.real_mpfr.double_toRR**
  - Entity: `double_toRR`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.rings.real_mpfr.int_toRR**
  - Entity: `int_toRR`
  - Module: `sage.rings.real_mpfr`
  - Type: `class`

- **sage.schemes.elliptic_curves.constructor.EllipticCurveFactory**
  - Entity: `EllipticCurveFactory`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `class`

- **sage.schemes.elliptic_curves.ec_database.EllipticCurves**
  - Entity: `EllipticCurves`
  - Module: `sage.schemes.elliptic_curves.ec_database`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.EllipticCurveIsogeny**
  - Entity: `EllipticCurveIsogeny`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_field.EllipticCurve_field**
  - Entity: `EllipticCurve_field`
  - Module: `sage.schemes.elliptic_curves.ell_field`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_finite_field.EllipticCurve_finite_field**
  - Entity: `EllipticCurve_finite_field`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_generic.EllipticCurve_generic**
  - Entity: `EllipticCurve_generic`
  - Module: `sage.schemes.elliptic_curves.ell_generic`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_local_data.EllipticCurveLocalData**
  - Entity: `EllipticCurveLocalData`
  - Module: `sage.schemes.elliptic_curves.ell_local_data`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_modular_symbols.ModularSymbol**
  - Entity: `ModularSymbol`
  - Module: `sage.schemes.elliptic_curves.ell_modular_symbols`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_modular_symbols.ModularSymbolECLIB**
  - Entity: `ModularSymbolECLIB`
  - Module: `sage.schemes.elliptic_curves.ell_modular_symbols`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_modular_symbols.ModularSymbolSage**
  - Entity: `ModularSymbolSage`
  - Module: `sage.schemes.elliptic_curves.ell_modular_symbols`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_number_field.EllipticCurve_number_field**
  - Entity: `EllipticCurve_number_field`
  - Module: `sage.schemes.elliptic_curves.ell_number_field`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_padic_field.EllipticCurve_padic_field**
  - Entity: `EllipticCurve_padic_field`
  - Module: `sage.schemes.elliptic_curves.ell_padic_field`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_point.EllipticCurvePoint**
  - Entity: `EllipticCurvePoint`
  - Module: `sage.schemes.elliptic_curves.ell_point`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_point.EllipticCurvePoint_field**
  - Entity: `EllipticCurvePoint_field`
  - Module: `sage.schemes.elliptic_curves.ell_point`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_point.EllipticCurvePoint_finite_field**
  - Entity: `EllipticCurvePoint_finite_field`
  - Module: `sage.schemes.elliptic_curves.ell_point`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_point.EllipticCurvePoint_number_field**
  - Entity: `EllipticCurvePoint_number_field`
  - Module: `sage.schemes.elliptic_curves.ell_point`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_rational_field.EllipticCurve_rational_field**
  - Entity: `EllipticCurve_rational_field`
  - Module: `sage.schemes.elliptic_curves.ell_rational_field`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_tate_curve.TateCurve**
  - Entity: `TateCurve`
  - Module: `sage.schemes.elliptic_curves.ell_tate_curve`
  - Type: `class`

- **sage.schemes.elliptic_curves.ell_torsion.EllipticCurveTorsionSubgroup**
  - Entity: `EllipticCurveTorsionSubgroup`
  - Module: `sage.schemes.elliptic_curves.ell_torsion`
  - Type: `class`

- **sage.schemes.elliptic_curves.formal_group.EllipticCurveFormalGroup**
  - Entity: `EllipticCurveFormalGroup`
  - Module: `sage.schemes.elliptic_curves.formal_group`
  - Type: `class`

- **sage.schemes.elliptic_curves.gal_reps.GaloisRepresentation**
  - Entity: `GaloisRepresentation`
  - Module: `sage.schemes.elliptic_curves.gal_reps`
  - Type: `class`

- **sage.schemes.elliptic_curves.gal_reps_number_field.GaloisRepresentation**
  - Entity: `GaloisRepresentation`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.GaloisAutomorphism**
  - Entity: `GaloisAutomorphism`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.GaloisAutomorphismComplexConjugation**
  - Entity: `GaloisAutomorphismComplexConjugation`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.GaloisAutomorphismQuadraticForm**
  - Entity: `GaloisAutomorphismQuadraticForm`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.GaloisGroup**
  - Entity: `GaloisGroup`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerPoint**
  - Entity: `HeegnerPoint`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerPointOnEllipticCurve**
  - Entity: `HeegnerPointOnEllipticCurve`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerPointOnX0N**
  - Entity: `HeegnerPointOnX0N`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerPoints**
  - Entity: `HeegnerPoints`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerPoints_level**
  - Entity: `HeegnerPoints_level`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerPoints_level_disc**
  - Entity: `HeegnerPoints_level_disc`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerPoints_level_disc_cond**
  - Entity: `HeegnerPoints_level_disc_cond`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerQuatAlg**
  - Entity: `HeegnerQuatAlg`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.HeegnerQuatAlgEmbedding**
  - Entity: `HeegnerQuatAlgEmbedding`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.KolyvaginCohomologyClass**
  - Entity: `KolyvaginCohomologyClass`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.KolyvaginCohomologyClassEn**
  - Entity: `KolyvaginCohomologyClassEn`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.KolyvaginPoint**
  - Entity: `KolyvaginPoint`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.heegner.RingClassField**
  - Entity: `RingClassField`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `class`

- **sage.schemes.elliptic_curves.height.EllipticCurveCanonicalHeight**
  - Entity: `EllipticCurveCanonicalHeight`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `class`

- **sage.schemes.elliptic_curves.height.UnionOfIntervals**
  - Entity: `UnionOfIntervals`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `class`

- **sage.schemes.elliptic_curves.hom.EllipticCurveHom**
  - Entity: `EllipticCurveHom`
  - Module: `sage.schemes.elliptic_curves.hom`
  - Type: `class`

- **sage.schemes.elliptic_curves.hom_composite.EllipticCurveHom_composite**
  - Entity: `EllipticCurveHom_composite`
  - Module: `sage.schemes.elliptic_curves.hom_composite`
  - Type: `class`

- **sage.schemes.elliptic_curves.hom_frobenius.EllipticCurveHom_frobenius**
  - Entity: `EllipticCurveHom_frobenius`
  - Module: `sage.schemes.elliptic_curves.hom_frobenius`
  - Type: `class`

- **sage.schemes.elliptic_curves.hom_scalar.EllipticCurveHom_scalar**
  - Entity: `EllipticCurveHom_scalar`
  - Module: `sage.schemes.elliptic_curves.hom_scalar`
  - Type: `class`

- **sage.schemes.elliptic_curves.hom_sum.EllipticCurveHom_sum**
  - Entity: `EllipticCurveHom_sum`
  - Module: `sage.schemes.elliptic_curves.hom_sum`
  - Type: `class`

- **sage.schemes.elliptic_curves.hom_velusqrt.EllipticCurveHom_velusqrt**
  - Entity: `EllipticCurveHom_velusqrt`
  - Module: `sage.schemes.elliptic_curves.hom_velusqrt`
  - Type: `class`

- **sage.schemes.elliptic_curves.hom_velusqrt.FastEllipticPolynomial**
  - Entity: `FastEllipticPolynomial`
  - Module: `sage.schemes.elliptic_curves.hom_velusqrt`
  - Type: `class`

- **sage.schemes.elliptic_curves.isogeny_class.IsogenyClass_EC**
  - Entity: `IsogenyClass_EC`
  - Module: `sage.schemes.elliptic_curves.isogeny_class`
  - Type: `class`

- **sage.schemes.elliptic_curves.isogeny_class.IsogenyClass_EC_NumberField**
  - Entity: `IsogenyClass_EC_NumberField`
  - Module: `sage.schemes.elliptic_curves.isogeny_class`
  - Type: `class`

- **sage.schemes.elliptic_curves.isogeny_class.IsogenyClass_EC_Rational**
  - Entity: `IsogenyClass_EC_Rational`
  - Module: `sage.schemes.elliptic_curves.isogeny_class`
  - Type: `class`

- **sage.schemes.elliptic_curves.kodaira_symbol.KodairaSymbol_class**
  - Entity: `KodairaSymbol_class`
  - Module: `sage.schemes.elliptic_curves.kodaira_symbol`
  - Type: `class`

- **sage.schemes.elliptic_curves.lseries_ell.Lseries_ell**
  - Entity: `Lseries_ell`
  - Module: `sage.schemes.elliptic_curves.lseries_ell`
  - Type: `class`

- **sage.schemes.elliptic_curves.mod_sym_num.ModularSymbolNumerical**
  - Entity: `ModularSymbolNumerical`
  - Module: `sage.schemes.elliptic_curves.mod_sym_num`
  - Type: `class`

- **sage.schemes.elliptic_curves.modular_parametrization.ModularParameterization**
  - Entity: `ModularParameterization`
  - Module: `sage.schemes.elliptic_curves.modular_parametrization`
  - Type: `class`

- **sage.schemes.elliptic_curves.padic_lseries.pAdicLseries**
  - Entity: `pAdicLseries`
  - Module: `sage.schemes.elliptic_curves.padic_lseries`
  - Type: `class`

- **sage.schemes.elliptic_curves.padic_lseries.pAdicLseriesOrdinary**
  - Entity: `pAdicLseriesOrdinary`
  - Module: `sage.schemes.elliptic_curves.padic_lseries`
  - Type: `class`

- **sage.schemes.elliptic_curves.padic_lseries.pAdicLseriesSupersingular**
  - Entity: `pAdicLseriesSupersingular`
  - Module: `sage.schemes.elliptic_curves.padic_lseries`
  - Type: `class`

- **sage.schemes.elliptic_curves.period_lattice.PeriodLattice**
  - Entity: `PeriodLattice`
  - Module: `sage.schemes.elliptic_curves.period_lattice`
  - Type: `class`

- **sage.schemes.elliptic_curves.period_lattice.PeriodLattice_ell**
  - Entity: `PeriodLattice_ell`
  - Module: `sage.schemes.elliptic_curves.period_lattice`
  - Type: `class`

- **sage.schemes.elliptic_curves.period_lattice_region.PeriodicRegion**
  - Entity: `PeriodicRegion`
  - Module: `sage.schemes.elliptic_curves.period_lattice_region`
  - Type: `class`

- **sage.schemes.elliptic_curves.saturation.EllipticCurveSaturator**
  - Entity: `EllipticCurveSaturator`
  - Module: `sage.schemes.elliptic_curves.saturation`
  - Type: `class`

- **sage.schemes.elliptic_curves.sha_tate.Sha**
  - Entity: `Sha`
  - Module: `sage.schemes.elliptic_curves.sha_tate`
  - Type: `class`

- **sage.schemes.elliptic_curves.weierstrass_morphism.WeierstrassIsomorphism**
  - Entity: `WeierstrassIsomorphism`
  - Module: `sage.schemes.elliptic_curves.weierstrass_morphism`
  - Type: `class`

- **sage.schemes.elliptic_curves.weierstrass_morphism.baseWI**
  - Entity: `baseWI`
  - Module: `sage.schemes.elliptic_curves.weierstrass_morphism`
  - Type: `class`

- **sage.schemes.elliptic_curves.weierstrass_transform.WeierstrassTransformation**
  - Entity: `WeierstrassTransformation`
  - Module: `sage.schemes.elliptic_curves.weierstrass_transform`
  - Type: `class`

- **sage.schemes.elliptic_curves.weierstrass_transform.WeierstrassTransformationWithInverse_class**
  - Entity: `WeierstrassTransformationWithInverse_class`
  - Module: `sage.schemes.elliptic_curves.weierstrass_transform`
  - Type: `class`

#### FUNCTION (243 entries)

- **sage.rings.polynomial.polynomial_singular_interface.can_convert_to_singular**
  - Entity: `can_convert_to_singular`
  - Module: `sage.rings.polynomial.polynomial_singular_interface`
  - Type: `function`

- **sage.rings.polynomial.polynomial_zmod_flint.make_element**
  - Entity: `make_element`
  - Module: `sage.rings.polynomial.polynomial_zmod_flint`
  - Type: `function`

- **sage.rings.polynomial.polynomial_zz_pex.make_element**
  - Entity: `make_element`
  - Module: `sage.rings.polynomial.polynomial_zz_pex`
  - Type: `function`

- **sage.rings.polynomial.q_integer_valued_polynomials.q_binomial_x**
  - Entity: `q_binomial_x`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `function`

- **sage.rings.polynomial.q_integer_valued_polynomials.q_int_x**
  - Entity: `q_int_x`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `function`

- **sage.rings.polynomial.real_roots.bernstein_down**
  - Entity: `bernstein_down`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.bernstein_expand**
  - Entity: `bernstein_expand`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.bernstein_up**
  - Entity: `bernstein_up`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.bitsize_doctest**
  - Entity: `bitsize_doctest`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.cl_maximum_root**
  - Entity: `cl_maximum_root`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.cl_maximum_root_first_lambda**
  - Entity: `cl_maximum_root_first_lambda`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.cl_maximum_root_local_max**
  - Entity: `cl_maximum_root_local_max`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.de_casteljau_doublevec**
  - Entity: `de_casteljau_doublevec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.de_casteljau_intvec**
  - Entity: `de_casteljau_intvec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.degree_reduction_next_size**
  - Entity: `degree_reduction_next_size`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.dprod_imatrow_vec**
  - Entity: `dprod_imatrow_vec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.get_realfield_rndu**
  - Entity: `get_realfield_rndu`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.intvec_to_doublevec**
  - Entity: `intvec_to_doublevec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.max_abs_doublevec**
  - Entity: `max_abs_doublevec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.max_bitsize_intvec_doctest**
  - Entity: `max_bitsize_intvec_doctest`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.maximum_root_first_lambda**
  - Entity: `maximum_root_first_lambda`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.maximum_root_local_max**
  - Entity: `maximum_root_local_max`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.min_max_delta_intvec**
  - Entity: `min_max_delta_intvec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.min_max_diff_doublevec**
  - Entity: `min_max_diff_doublevec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.min_max_diff_intvec**
  - Entity: `min_max_diff_intvec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.mk_context**
  - Entity: `mk_context`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.mk_ibpf**
  - Entity: `mk_ibpf`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.mk_ibpi**
  - Entity: `mk_ibpi`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.precompute_degree_reduction_cache**
  - Entity: `precompute_degree_reduction_cache`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.pseudoinverse**
  - Entity: `pseudoinverse`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.rational_root_bounds**
  - Entity: `rational_root_bounds`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.real_roots**
  - Entity: `real_roots`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.relative_bounds**
  - Entity: `relative_bounds`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.reverse_intvec**
  - Entity: `reverse_intvec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.root_bounds**
  - Entity: `root_bounds`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.scale_intvec_var**
  - Entity: `scale_intvec_var`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.split_for_targets**
  - Entity: `split_for_targets`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.subsample_vec_doctest**
  - Entity: `subsample_vec_doctest`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.taylor_shift1_intvec**
  - Entity: `taylor_shift1_intvec`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.to_bernstein**
  - Entity: `to_bernstein`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.to_bernstein_warp**
  - Entity: `to_bernstein_warp`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.real_roots.wordsize_rational**
  - Entity: `wordsize_rational`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `function`

- **sage.rings.polynomial.refine_root.refine_root**
  - Entity: `refine_root`
  - Module: `sage.rings.polynomial.refine_root`
  - Type: `function`

- **sage.rings.polynomial.term_order.termorder_from_singular**
  - Entity: `termorder_from_singular`
  - Module: `sage.rings.polynomial.term_order`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.LCM**
  - Entity: `LCM`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.LM**
  - Entity: `LM`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.LT**
  - Entity: `LT`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.buchberger**
  - Entity: `buchberger`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.buchberger_improved**
  - Entity: `buchberger_improved`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.inter_reduction**
  - Entity: `inter_reduction`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.select**
  - Entity: `select`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.spol**
  - Entity: `spol`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_buchberger.update**
  - Entity: `update`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `function`

- **sage.rings.polynomial.toy_d_basis.LC**
  - Entity: `LC`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `function`

- **sage.rings.polynomial.toy_d_basis.LM**
  - Entity: `LM`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `function`

- **sage.rings.polynomial.toy_d_basis.d_basis**
  - Entity: `d_basis`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `function`

- **sage.rings.polynomial.toy_d_basis.gpol**
  - Entity: `gpol`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `function`

- **sage.rings.polynomial.toy_d_basis.select**
  - Entity: `select`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `function`

- **sage.rings.polynomial.toy_d_basis.spol**
  - Entity: `spol`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `function`

- **sage.rings.polynomial.toy_d_basis.update**
  - Entity: `update`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `function`

- **sage.rings.polynomial.toy_variety.coefficient_matrix**
  - Entity: `coefficient_matrix`
  - Module: `sage.rings.polynomial.toy_variety`
  - Type: `function`

- **sage.rings.polynomial.toy_variety.elim_pol**
  - Entity: `elim_pol`
  - Module: `sage.rings.polynomial.toy_variety`
  - Type: `function`

- **sage.rings.polynomial.toy_variety.is_linearly_dependent**
  - Entity: `is_linearly_dependent`
  - Module: `sage.rings.polynomial.toy_variety`
  - Type: `function`

- **sage.rings.polynomial.toy_variety.is_triangular**
  - Entity: `is_triangular`
  - Module: `sage.rings.polynomial.toy_variety`
  - Type: `function`

- **sage.rings.polynomial.toy_variety.linear_representation**
  - Entity: `linear_representation`
  - Module: `sage.rings.polynomial.toy_variety`
  - Type: `function`

- **sage.rings.polynomial.toy_variety.triangular_factorization**
  - Entity: `triangular_factorization`
  - Module: `sage.rings.polynomial.toy_variety`
  - Type: `function`

- **sage.rings.power_series_ring.PowerSeriesRing**
  - Entity: `PowerSeriesRing`
  - Module: `sage.rings.power_series_ring`
  - Type: `function`

- **sage.rings.power_series_ring.is_PowerSeriesRing**
  - Entity: `is_PowerSeriesRing`
  - Module: `sage.rings.power_series_ring`
  - Type: `function`

- **sage.rings.power_series_ring.unpickle_power_series_ring_v0**
  - Entity: `unpickle_power_series_ring_v0`
  - Module: `sage.rings.power_series_ring`
  - Type: `function`

- **sage.rings.power_series_ring_element.is_PowerSeries**
  - Entity: `is_PowerSeries`
  - Module: `sage.rings.power_series_ring_element`
  - Type: `function`

- **sage.rings.power_series_ring_element.make_element_from_parent_v0**
  - Entity: `make_element_from_parent_v0`
  - Module: `sage.rings.power_series_ring_element`
  - Type: `function`

- **sage.rings.power_series_ring_element.make_powerseries_poly_v0**
  - Entity: `make_powerseries_poly_v0`
  - Module: `sage.rings.power_series_ring_element`
  - Type: `function`

- **sage.rings.rational.integer_rational_power**
  - Entity: `integer_rational_power`
  - Module: `sage.rings.rational`
  - Type: `function`

- **sage.rings.rational.is_Rational**
  - Entity: `is_Rational`
  - Module: `sage.rings.rational`
  - Type: `function`

- **sage.rings.rational.make_rational**
  - Entity: `make_rational`
  - Module: `sage.rings.rational`
  - Type: `function`

- **sage.rings.rational.rational_power_parts**
  - Entity: `rational_power_parts`
  - Module: `sage.rings.rational`
  - Type: `function`

- **sage.rings.rational_field.frac**
  - Entity: `frac`
  - Module: `sage.rings.rational_field`
  - Type: `function`

- **sage.rings.rational_field.is_RationalField**
  - Entity: `is_RationalField`
  - Module: `sage.rings.rational_field`
  - Type: `function`

- **sage.rings.real_double.RealDoubleField**
  - Entity: `RealDoubleField`
  - Module: `sage.rings.real_double`
  - Type: `function`

- **sage.rings.real_double.is_RealDoubleElement**
  - Entity: `is_RealDoubleElement`
  - Module: `sage.rings.real_double`
  - Type: `function`

- **sage.rings.real_mpfr.RealField**
  - Entity: `RealField`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.create_RealNumber**
  - Entity: `create_RealNumber`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.is_RealNumber**
  - Entity: `is_RealNumber`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_get_exp_max**
  - Entity: `mpfr_get_exp_max`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_get_exp_max_max**
  - Entity: `mpfr_get_exp_max_max`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_get_exp_min**
  - Entity: `mpfr_get_exp_min`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_get_exp_min_min**
  - Entity: `mpfr_get_exp_min_min`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_prec_max**
  - Entity: `mpfr_prec_max`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_prec_min**
  - Entity: `mpfr_prec_min`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_set_exp_max**
  - Entity: `mpfr_set_exp_max`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.rings.real_mpfr.mpfr_set_exp_min**
  - Entity: `mpfr_set_exp_min`
  - Module: `sage.rings.real_mpfr`
  - Type: `function`

- **sage.schemes.elliptic_curves.Qcurves.Step4Test**
  - Entity: `Step4Test`
  - Module: `sage.schemes.elliptic_curves.Qcurves`
  - Type: `function`

- **sage.schemes.elliptic_curves.Qcurves.conjugacy_test**
  - Entity: `conjugacy_test`
  - Module: `sage.schemes.elliptic_curves.Qcurves`
  - Type: `function`

- **sage.schemes.elliptic_curves.Qcurves.is_Q_curve**
  - Entity: `is_Q_curve`
  - Module: `sage.schemes.elliptic_curves.Qcurves`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.OrderClassNumber**
  - Entity: `OrderClassNumber`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.cm_j_invariants**
  - Entity: `cm_j_invariants`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.cm_j_invariants_and_orders**
  - Entity: `cm_j_invariants_and_orders`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.cm_orders**
  - Entity: `cm_orders`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.discriminants_with_bounded_class_number**
  - Entity: `discriminants_with_bounded_class_number`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.hilbert_class_polynomial**
  - Entity: `hilbert_class_polynomial`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.is_HCP**
  - Entity: `is_HCP`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.is_cm_j_invariant**
  - Entity: `is_cm_j_invariant`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.largest_disc_with_class_number**
  - Entity: `largest_disc_with_class_number`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.cm.largest_fundamental_disc_with_class_number**
  - Entity: `largest_fundamental_disc_with_class_number`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.EllipticCurve_from_Weierstrass_polynomial**
  - Entity: `EllipticCurve_from_Weierstrass_polynomial`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.EllipticCurve_from_c4c6**
  - Entity: `EllipticCurve_from_c4c6`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.EllipticCurve_from_cubic**
  - Entity: `EllipticCurve_from_cubic`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.EllipticCurve_from_j**
  - Entity: `EllipticCurve_from_j`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.EllipticCurves_with_good_reduction_outside_S**
  - Entity: `EllipticCurves_with_good_reduction_outside_S`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.are_projectively_equivalent**
  - Entity: `are_projectively_equivalent`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.chord_and_tangent**
  - Entity: `chord_and_tangent`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.coefficients_from_Weierstrass_polynomial**
  - Entity: `coefficients_from_Weierstrass_polynomial`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.coefficients_from_j**
  - Entity: `coefficients_from_j`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.projective_point**
  - Entity: `projective_point`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.constructor.tangent_at_smooth_point**
  - Entity: `tangent_at_smooth_point`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `function`

- **sage.schemes.elliptic_curves.descent_two_isogeny.test_els**
  - Entity: `test_els`
  - Module: `sage.schemes.elliptic_curves.descent_two_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.descent_two_isogeny.test_padic_square**
  - Entity: `test_padic_square`
  - Module: `sage.schemes.elliptic_curves.descent_two_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.descent_two_isogeny.test_qpls**
  - Entity: `test_qpls`
  - Module: `sage.schemes.elliptic_curves.descent_two_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.descent_two_isogeny.test_valuation**
  - Entity: `test_valuation`
  - Module: `sage.schemes.elliptic_curves.descent_two_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.descent_two_isogeny.two_descent_by_two_isogeny**
  - Entity: `two_descent_by_two_isogeny`
  - Module: `sage.schemes.elliptic_curves.descent_two_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.descent_two_isogeny.two_descent_by_two_isogeny_work**
  - Entity: `two_descent_by_two_isogeny_work`
  - Module: `sage.schemes.elliptic_curves.descent_two_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_codomain_formula**
  - Entity: `compute_codomain_formula`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_codomain_kohel**
  - Entity: `compute_codomain_kohel`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_intermediate_curves**
  - Entity: `compute_intermediate_curves`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_isogeny_bmss**
  - Entity: `compute_isogeny_bmss`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_isogeny_kernel_polynomial**
  - Entity: `compute_isogeny_kernel_polynomial`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_isogeny_stark**
  - Entity: `compute_isogeny_stark`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_sequence_of_maps**
  - Entity: `compute_sequence_of_maps`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_vw_kohel_even_deg1**
  - Entity: `compute_vw_kohel_even_deg1`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_vw_kohel_even_deg3**
  - Entity: `compute_vw_kohel_even_deg3`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.compute_vw_kohel_odd**
  - Entity: `compute_vw_kohel_odd`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.fill_isogeny_matrix**
  - Entity: `fill_isogeny_matrix`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.isogeny_codomain_from_kernel**
  - Entity: `isogeny_codomain_from_kernel`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.two_torsion_part**
  - Entity: `two_torsion_part`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_curve_isogeny.unfill_isogeny_matrix**
  - Entity: `unfill_isogeny_matrix`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_egros.curve_key**
  - Entity: `curve_key`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_egros.egros_from_j**
  - Entity: `egros_from_j`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_egros.egros_from_j_0**
  - Entity: `egros_from_j_0`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_egros.egros_from_j_1728**
  - Entity: `egros_from_j_1728`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_egros.egros_from_jlist**
  - Entity: `egros_from_jlist`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_egros.egros_get_j**
  - Entity: `egros_get_j`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_egros.is_possible_j**
  - Entity: `is_possible_j`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_field.compute_model**
  - Entity: `compute_model`
  - Module: `sage.schemes.elliptic_curves.ell_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_field.point_of_order**
  - Entity: `point_of_order`
  - Module: `sage.schemes.elliptic_curves.ell_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.EllipticCurve_with_order**
  - Entity: `EllipticCurve_with_order`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.EllipticCurve_with_prime_order**
  - Entity: `EllipticCurve_with_prime_order`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.curves_with_j_0**
  - Entity: `curves_with_j_0`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.curves_with_j_0_char2**
  - Entity: `curves_with_j_0_char2`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.curves_with_j_0_char3**
  - Entity: `curves_with_j_0_char3`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.curves_with_j_1728**
  - Entity: `curves_with_j_1728`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.fill_ss_j_dict**
  - Entity: `fill_ss_j_dict`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.is_j_supersingular**
  - Entity: `is_j_supersingular`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.special_supersingular_curve**
  - Entity: `special_supersingular_curve`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_finite_field.supersingular_j_polynomial**
  - Entity: `supersingular_j_polynomial`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_generic.is_EllipticCurve**
  - Entity: `is_EllipticCurve`
  - Module: `sage.schemes.elliptic_curves.ell_generic`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_local_data.check_prime**
  - Entity: `check_prime`
  - Module: `sage.schemes.elliptic_curves.ell_local_data`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_modular_symbols.modular_symbol_space**
  - Entity: `modular_symbol_space`
  - Module: `sage.schemes.elliptic_curves.ell_modular_symbols`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_rational_field.cremona_curves**
  - Entity: `cremona_curves`
  - Module: `sage.schemes.elliptic_curves.ell_rational_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_rational_field.cremona_optimal_curves**
  - Entity: `cremona_optimal_curves`
  - Module: `sage.schemes.elliptic_curves.ell_rational_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_rational_field.elliptic_curve_congruence_graph**
  - Entity: `elliptic_curve_congruence_graph`
  - Module: `sage.schemes.elliptic_curves.ell_rational_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_rational_field.integral_points_with_bounded_mw_coeffs**
  - Entity: `integral_points_with_bounded_mw_coeffs`
  - Module: `sage.schemes.elliptic_curves.ell_rational_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_torsion.torsion_bound**
  - Entity: `torsion_bound`
  - Module: `sage.schemes.elliptic_curves.ell_torsion`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_wp.compute_wp_fast**
  - Entity: `compute_wp_fast`
  - Module: `sage.schemes.elliptic_curves.ell_wp`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_wp.compute_wp_pari**
  - Entity: `compute_wp_pari`
  - Module: `sage.schemes.elliptic_curves.ell_wp`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_wp.compute_wp_quadratic**
  - Entity: `compute_wp_quadratic`
  - Module: `sage.schemes.elliptic_curves.ell_wp`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_wp.solve_linear_differential_system**
  - Entity: `solve_linear_differential_system`
  - Module: `sage.schemes.elliptic_curves.ell_wp`
  - Type: `function`

- **sage.schemes.elliptic_curves.ell_wp.weierstrass_p**
  - Entity: `weierstrass_p`
  - Module: `sage.schemes.elliptic_curves.ell_wp`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.Billerey_B_bound**
  - Entity: `Billerey_B_bound`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.Billerey_B_l**
  - Entity: `Billerey_B_l`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.Billerey_P_l**
  - Entity: `Billerey_P_l`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.Billerey_R_bound**
  - Entity: `Billerey_R_bound`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.Billerey_R_q**
  - Entity: `Billerey_R_q`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.Frobenius_filter**
  - Entity: `Frobenius_filter`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.deg_one_primes_iter**
  - Entity: `deg_one_primes_iter`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.reducible_primes_Billerey**
  - Entity: `reducible_primes_Billerey`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gal_reps_number_field.reducible_primes_naive**
  - Entity: `reducible_primes_naive`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `function`

- **sage.schemes.elliptic_curves.gp_simon.simon_two_descent**
  - Entity: `simon_two_descent`
  - Module: `sage.schemes.elliptic_curves.gp_simon`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.class_number**
  - Entity: `class_number`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.ell_heegner_discriminants**
  - Entity: `ell_heegner_discriminants`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.ell_heegner_discriminants_list**
  - Entity: `ell_heegner_discriminants_list`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.ell_heegner_point**
  - Entity: `ell_heegner_point`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.heegner_index**
  - Entity: `heegner_index`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.heegner_index_bound**
  - Entity: `heegner_index_bound`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.heegner_point**
  - Entity: `heegner_point`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.heegner_point_height**
  - Entity: `heegner_point_height`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.heegner_points**
  - Entity: `heegner_points`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.heegner_sha_an**
  - Entity: `heegner_sha_an`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.is_inert**
  - Entity: `is_inert`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.is_kolyvagin_conductor**
  - Entity: `is_kolyvagin_conductor`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.is_ramified**
  - Entity: `is_ramified`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.is_split**
  - Entity: `is_split`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.kolyvagin_point**
  - Entity: `kolyvagin_point`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.kolyvagin_reduction_data**
  - Entity: `kolyvagin_reduction_data`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.make_monic**
  - Entity: `make_monic`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.nearby_rational_poly**
  - Entity: `nearby_rational_poly`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.quadratic_order**
  - Entity: `quadratic_order`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.satisfies_heegner_hypothesis**
  - Entity: `satisfies_heegner_hypothesis`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.satisfies_weak_heegner_hypothesis**
  - Entity: `satisfies_weak_heegner_hypothesis`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.heegner.simplest_rational_poly**
  - Entity: `simplest_rational_poly`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `function`

- **sage.schemes.elliptic_curves.height.eps**
  - Entity: `eps`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `function`

- **sage.schemes.elliptic_curves.height.inf_max_abs**
  - Entity: `inf_max_abs`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `function`

- **sage.schemes.elliptic_curves.height.min_on_disk**
  - Entity: `min_on_disk`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `function`

- **sage.schemes.elliptic_curves.height.nonneg_region**
  - Entity: `nonneg_region`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `function`

- **sage.schemes.elliptic_curves.height.rat_term_CIF**
  - Entity: `rat_term_CIF`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `function`

- **sage.schemes.elliptic_curves.hom.compare_via_evaluation**
  - Entity: `compare_via_evaluation`
  - Module: `sage.schemes.elliptic_curves.hom`
  - Type: `function`

- **sage.schemes.elliptic_curves.hom.compute_trace_generic**
  - Entity: `compute_trace_generic`
  - Module: `sage.schemes.elliptic_curves.hom`
  - Type: `function`

- **sage.schemes.elliptic_curves.hom.find_post_isomorphism**
  - Entity: `find_post_isomorphism`
  - Module: `sage.schemes.elliptic_curves.hom`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_class.isogeny_degrees_cm**
  - Entity: `isogeny_degrees_cm`
  - Module: `sage.schemes.elliptic_curves.isogeny_class`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_class.possible_isogeny_degrees**
  - Entity: `possible_isogeny_degrees`
  - Module: `sage.schemes.elliptic_curves.isogeny_class`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.Fricke_module**
  - Entity: `Fricke_module`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.Fricke_polynomial**
  - Entity: `Fricke_polynomial`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.Psi**
  - Entity: `Psi`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.Psi2**
  - Entity: `Psi2`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.is_kernel_polynomial**
  - Entity: `is_kernel_polynomial`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_13_0**
  - Entity: `isogenies_13_0`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_13_1728**
  - Entity: `isogenies_13_1728`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_2**
  - Entity: `isogenies_2`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_3**
  - Entity: `isogenies_3`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_5_0**
  - Entity: `isogenies_5_0`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_5_1728**
  - Entity: `isogenies_5_1728`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_7_0**
  - Entity: `isogenies_7_0`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_7_1728**
  - Entity: `isogenies_7_1728`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_prime_degree**
  - Entity: `isogenies_prime_degree`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_prime_degree_general**
  - Entity: `isogenies_prime_degree_general`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_prime_degree_genus_0**
  - Entity: `isogenies_prime_degree_genus_0`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_prime_degree_genus_plus_0**
  - Entity: `isogenies_prime_degree_genus_plus_0`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_prime_degree_genus_plus_0_j0**
  - Entity: `isogenies_prime_degree_genus_plus_0_j0`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_prime_degree_genus_plus_0_j1728**
  - Entity: `isogenies_prime_degree_genus_plus_0_j1728`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.isogeny_small_degree.isogenies_sporadic_Q**
  - Entity: `isogenies_sporadic_Q`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `function`

- **sage.schemes.elliptic_curves.jacobian.Jacobian**
  - Entity: `Jacobian`
  - Module: `sage.schemes.elliptic_curves.jacobian`
  - Type: `function`

- **sage.schemes.elliptic_curves.jacobian.Jacobian_of_curve**
  - Entity: `Jacobian_of_curve`
  - Module: `sage.schemes.elliptic_curves.jacobian`
  - Type: `function`

- **sage.schemes.elliptic_curves.jacobian.Jacobian_of_equation**
  - Entity: `Jacobian_of_equation`
  - Module: `sage.schemes.elliptic_curves.jacobian`
  - Type: `function`

- **sage.schemes.elliptic_curves.kodaira_symbol.KodairaSymbol**
  - Entity: `KodairaSymbol`
  - Module: `sage.schemes.elliptic_curves.kodaira_symbol`
  - Type: `function`

- **sage.schemes.elliptic_curves.mod5family.mod5family**
  - Entity: `mod5family`
  - Module: `sage.schemes.elliptic_curves.mod5family`
  - Type: `function`

- **sage.schemes.elliptic_curves.mod_poly.classical_modular_polynomial**
  - Entity: `classical_modular_polynomial`
  - Module: `sage.schemes.elliptic_curves.mod_poly`
  - Type: `function`

- **sage.schemes.elliptic_curves.period_lattice.extended_agm_iteration**
  - Entity: `extended_agm_iteration`
  - Module: `sage.schemes.elliptic_curves.period_lattice`
  - Type: `function`

- **sage.schemes.elliptic_curves.period_lattice.normalise_periods**
  - Entity: `normalise_periods`
  - Module: `sage.schemes.elliptic_curves.period_lattice`
  - Type: `function`

- **sage.schemes.elliptic_curves.period_lattice.reduce_tau**
  - Entity: `reduce_tau`
  - Module: `sage.schemes.elliptic_curves.period_lattice`
  - Type: `function`

- **sage.schemes.elliptic_curves.saturation.p_projections**
  - Entity: `p_projections`
  - Module: `sage.schemes.elliptic_curves.saturation`
  - Type: `function`

- **sage.schemes.elliptic_curves.saturation.reduce_mod_q**
  - Entity: `reduce_mod_q`
  - Module: `sage.schemes.elliptic_curves.saturation`
  - Type: `function`

- **sage.schemes.elliptic_curves.weierstrass_morphism.identity_morphism**
  - Entity: `identity_morphism`
  - Module: `sage.schemes.elliptic_curves.weierstrass_morphism`
  - Type: `function`

- **sage.schemes.elliptic_curves.weierstrass_morphism.negation_morphism**
  - Entity: `negation_morphism`
  - Module: `sage.schemes.elliptic_curves.weierstrass_morphism`
  - Type: `function`

- **sage.schemes.elliptic_curves.weierstrass_transform.WeierstrassTransformationWithInverse**
  - Entity: `WeierstrassTransformationWithInverse`
  - Module: `sage.schemes.elliptic_curves.weierstrass_transform`
  - Type: `function`

#### MODULE (68 entries)

- **sage.rings.polynomial.polynomial_zmod_flint**
  - Entity: `polynomial_zmod_flint`
  - Module: `sage.rings.polynomial.polynomial_zmod_flint`
  - Type: `module`

- **sage.rings.polynomial.polynomial_zz_pex**
  - Entity: `polynomial_zz_pex`
  - Module: `sage.rings.polynomial.polynomial_zz_pex`
  - Type: `module`

- **sage.rings.polynomial.q_integer_valued_polynomials**
  - Entity: `q_integer_valued_polynomials`
  - Module: `sage.rings.polynomial.q_integer_valued_polynomials`
  - Type: `module`

- **sage.rings.polynomial.real_roots**
  - Entity: `real_roots`
  - Module: `sage.rings.polynomial.real_roots`
  - Type: `module`

- **sage.rings.polynomial.refine_root**
  - Entity: `refine_root`
  - Module: `sage.rings.polynomial.refine_root`
  - Type: `module`

- **sage.rings.polynomial.skew_polynomial_element**
  - Entity: `skew_polynomial_element`
  - Module: `sage.rings.polynomial.skew_polynomial_element`
  - Type: `module`

- **sage.rings.polynomial.skew_polynomial_finite_field**
  - Entity: `skew_polynomial_finite_field`
  - Module: `sage.rings.polynomial.skew_polynomial_finite_field`
  - Type: `module`

- **sage.rings.polynomial.skew_polynomial_finite_order**
  - Entity: `skew_polynomial_finite_order`
  - Module: `sage.rings.polynomial.skew_polynomial_finite_order`
  - Type: `module`

- **sage.rings.polynomial.skew_polynomial_ring**
  - Entity: `skew_polynomial_ring`
  - Module: `sage.rings.polynomial.skew_polynomial_ring`
  - Type: `module`

- **sage.rings.polynomial.symmetric_ideal**
  - Entity: `symmetric_ideal`
  - Module: `sage.rings.polynomial.symmetric_ideal`
  - Type: `module`

- **sage.rings.polynomial.symmetric_reduction**
  - Entity: `symmetric_reduction`
  - Module: `sage.rings.polynomial.symmetric_reduction`
  - Type: `module`

- **sage.rings.polynomial.term_order**
  - Entity: `term_order`
  - Module: `sage.rings.polynomial.term_order`
  - Type: `module`

- **sage.rings.polynomial.toy_buchberger**
  - Entity: `toy_buchberger`
  - Module: `sage.rings.polynomial.toy_buchberger`
  - Type: `module`

- **sage.rings.polynomial.toy_d_basis**
  - Entity: `toy_d_basis`
  - Module: `sage.rings.polynomial.toy_d_basis`
  - Type: `module`

- **sage.rings.polynomial.toy_variety**
  - Entity: `toy_variety`
  - Module: `sage.rings.polynomial.toy_variety`
  - Type: `module`

- **sage.rings.power_series_ring**
  - Entity: `power_series_ring`
  - Module: `sage.rings.power_series_ring`
  - Type: `module`

- **sage.rings.power_series_ring_element**
  - Entity: `power_series_ring_element`
  - Module: `sage.rings.power_series_ring_element`
  - Type: `module`

- **sage.rings.rational**
  - Entity: `rational`
  - Module: `sage.rings.rational`
  - Type: `module`

- **sage.rings.rational_field**
  - Entity: `rational_field`
  - Module: `sage.rings.rational_field`
  - Type: `module`

- **sage.rings.real_double**
  - Entity: `real_double`
  - Module: `sage.rings.real_double`
  - Type: `module`

- **sage.rings.real_mpfr**
  - Entity: `real_mpfr`
  - Module: `sage.rings.real_mpfr`
  - Type: `module`

- **sage.schemes.elliptic_curves.Qcurves**
  - Entity: `Qcurves`
  - Module: `sage.schemes.elliptic_curves.Qcurves`
  - Type: `module`

- **sage.schemes.elliptic_curves.cm**
  - Entity: `cm`
  - Module: `sage.schemes.elliptic_curves.cm`
  - Type: `module`

- **sage.schemes.elliptic_curves.constructor**
  - Entity: `constructor`
  - Module: `sage.schemes.elliptic_curves.constructor`
  - Type: `module`

- **sage.schemes.elliptic_curves.descent_two_isogeny**
  - Entity: `descent_two_isogeny`
  - Module: `sage.schemes.elliptic_curves.descent_two_isogeny`
  - Type: `module`

- **sage.schemes.elliptic_curves.ec_database**
  - Entity: `ec_database`
  - Module: `sage.schemes.elliptic_curves.ec_database`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_curve_isogeny**
  - Entity: `ell_curve_isogeny`
  - Module: `sage.schemes.elliptic_curves.ell_curve_isogeny`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_egros**
  - Entity: `ell_egros`
  - Module: `sage.schemes.elliptic_curves.ell_egros`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_field**
  - Entity: `ell_field`
  - Module: `sage.schemes.elliptic_curves.ell_field`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_finite_field**
  - Entity: `ell_finite_field`
  - Module: `sage.schemes.elliptic_curves.ell_finite_field`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_generic**
  - Entity: `ell_generic`
  - Module: `sage.schemes.elliptic_curves.ell_generic`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_local_data**
  - Entity: `ell_local_data`
  - Module: `sage.schemes.elliptic_curves.ell_local_data`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_modular_symbols**
  - Entity: `ell_modular_symbols`
  - Module: `sage.schemes.elliptic_curves.ell_modular_symbols`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_number_field**
  - Entity: `ell_number_field`
  - Module: `sage.schemes.elliptic_curves.ell_number_field`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_padic_field**
  - Entity: `ell_padic_field`
  - Module: `sage.schemes.elliptic_curves.ell_padic_field`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_point**
  - Entity: `ell_point`
  - Module: `sage.schemes.elliptic_curves.ell_point`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_rational_field**
  - Entity: `ell_rational_field`
  - Module: `sage.schemes.elliptic_curves.ell_rational_field`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_tate_curve**
  - Entity: `ell_tate_curve`
  - Module: `sage.schemes.elliptic_curves.ell_tate_curve`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_torsion**
  - Entity: `ell_torsion`
  - Module: `sage.schemes.elliptic_curves.ell_torsion`
  - Type: `module`

- **sage.schemes.elliptic_curves.ell_wp**
  - Entity: `ell_wp`
  - Module: `sage.schemes.elliptic_curves.ell_wp`
  - Type: `module`

- **sage.schemes.elliptic_curves.formal_group**
  - Entity: `formal_group`
  - Module: `sage.schemes.elliptic_curves.formal_group`
  - Type: `module`

- **sage.schemes.elliptic_curves.gal_reps**
  - Entity: `gal_reps`
  - Module: `sage.schemes.elliptic_curves.gal_reps`
  - Type: `module`

- **sage.schemes.elliptic_curves.gal_reps_number_field**
  - Entity: `gal_reps_number_field`
  - Module: `sage.schemes.elliptic_curves.gal_reps_number_field`
  - Type: `module`

- **sage.schemes.elliptic_curves.gp_simon**
  - Entity: `gp_simon`
  - Module: `sage.schemes.elliptic_curves.gp_simon`
  - Type: `module`

- **sage.schemes.elliptic_curves.heegner**
  - Entity: `heegner`
  - Module: `sage.schemes.elliptic_curves.heegner`
  - Type: `module`

- **sage.schemes.elliptic_curves.height**
  - Entity: `height`
  - Module: `sage.schemes.elliptic_curves.height`
  - Type: `module`

- **sage.schemes.elliptic_curves.hom**
  - Entity: `hom`
  - Module: `sage.schemes.elliptic_curves.hom`
  - Type: `module`

- **sage.schemes.elliptic_curves.hom_composite**
  - Entity: `hom_composite`
  - Module: `sage.schemes.elliptic_curves.hom_composite`
  - Type: `module`

- **sage.schemes.elliptic_curves.hom_frobenius**
  - Entity: `hom_frobenius`
  - Module: `sage.schemes.elliptic_curves.hom_frobenius`
  - Type: `module`

- **sage.schemes.elliptic_curves.hom_scalar**
  - Entity: `hom_scalar`
  - Module: `sage.schemes.elliptic_curves.hom_scalar`
  - Type: `module`

- **sage.schemes.elliptic_curves.hom_sum**
  - Entity: `hom_sum`
  - Module: `sage.schemes.elliptic_curves.hom_sum`
  - Type: `module`

- **sage.schemes.elliptic_curves.hom_velusqrt**
  - Entity: `hom_velusqrt`
  - Module: `sage.schemes.elliptic_curves.hom_velusqrt`
  - Type: `module`

- **sage.schemes.elliptic_curves.isogeny_class**
  - Entity: `isogeny_class`
  - Module: `sage.schemes.elliptic_curves.isogeny_class`
  - Type: `module`

- **sage.schemes.elliptic_curves.isogeny_small_degree**
  - Entity: `isogeny_small_degree`
  - Module: `sage.schemes.elliptic_curves.isogeny_small_degree`
  - Type: `module`

- **sage.schemes.elliptic_curves.jacobian**
  - Entity: `jacobian`
  - Module: `sage.schemes.elliptic_curves.jacobian`
  - Type: `module`

- **sage.schemes.elliptic_curves.kodaira_symbol**
  - Entity: `kodaira_symbol`
  - Module: `sage.schemes.elliptic_curves.kodaira_symbol`
  - Type: `module`

- **sage.schemes.elliptic_curves.lseries_ell**
  - Entity: `lseries_ell`
  - Module: `sage.schemes.elliptic_curves.lseries_ell`
  - Type: `module`

- **sage.schemes.elliptic_curves.mod5family**
  - Entity: `mod5family`
  - Module: `sage.schemes.elliptic_curves.mod5family`
  - Type: `module`

- **sage.schemes.elliptic_curves.mod_poly**
  - Entity: `mod_poly`
  - Module: `sage.schemes.elliptic_curves.mod_poly`
  - Type: `module`

- **sage.schemes.elliptic_curves.mod_sym_num**
  - Entity: `mod_sym_num`
  - Module: `sage.schemes.elliptic_curves.mod_sym_num`
  - Type: `module`

- **sage.schemes.elliptic_curves.modular_parametrization**
  - Entity: `modular_parametrization`
  - Module: `sage.schemes.elliptic_curves.modular_parametrization`
  - Type: `module`

- **sage.schemes.elliptic_curves.padic_lseries**
  - Entity: `padic_lseries`
  - Module: `sage.schemes.elliptic_curves.padic_lseries`
  - Type: `module`

- **sage.schemes.elliptic_curves.period_lattice**
  - Entity: `period_lattice`
  - Module: `sage.schemes.elliptic_curves.period_lattice`
  - Type: `module`

- **sage.schemes.elliptic_curves.period_lattice_region**
  - Entity: `period_lattice_region`
  - Module: `sage.schemes.elliptic_curves.period_lattice_region`
  - Type: `module`

- **sage.schemes.elliptic_curves.saturation**
  - Entity: `saturation`
  - Module: `sage.schemes.elliptic_curves.saturation`
  - Type: `module`

- **sage.schemes.elliptic_curves.sha_tate**
  - Entity: `sha_tate`
  - Module: `sage.schemes.elliptic_curves.sha_tate`
  - Type: `module`

- **sage.schemes.elliptic_curves.weierstrass_morphism**
  - Entity: `weierstrass_morphism`
  - Module: `sage.schemes.elliptic_curves.weierstrass_morphism`
  - Type: `module`

- **sage.schemes.elliptic_curves.weierstrass_transform**
  - Entity: `weierstrass_transform`
  - Module: `sage.schemes.elliptic_curves.weierstrass_transform`
  - Type: `module`


### Part 14 (163 entries)

#### ATTRIBUTE (25 entries)

- **sage.stats.distributions.discrete_gaussian_integer.algorithm**
  - Entity: `algorithm`
  - Module: `sage.stats.distributions.discrete_gaussian_integer`
  - Type: `attribute`

- **sage.stats.distributions.discrete_gaussian_integer.c**
  - Entity: `c`
  - Module: `sage.stats.distributions.discrete_gaussian_integer`
  - Type: `attribute`

- **sage.stats.distributions.discrete_gaussian_integer.sigma**
  - Entity: `sigma`
  - Module: `sage.stats.distributions.discrete_gaussian_integer`
  - Type: `attribute`

- **sage.stats.distributions.discrete_gaussian_integer.table_cutoff**
  - Entity: `table_cutoff`
  - Module: `sage.stats.distributions.discrete_gaussian_integer`
  - Type: `attribute`

- **sage.stats.distributions.discrete_gaussian_integer.tau**
  - Entity: `tau`
  - Module: `sage.stats.distributions.discrete_gaussian_integer`
  - Type: `attribute`

- **sage.symbolic.expression.op**
  - Entity: `op`
  - Module: `sage.symbolic.expression`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.cos**
  - Entity: `cos`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.cosh**
  - Entity: `cosh`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.cot**
  - Entity: `cot`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.coth**
  - Entity: `coth`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.csc**
  - Entity: `csc`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.csch**
  - Entity: `csch`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.e**
  - Entity: `e`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.half**
  - Entity: `half`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.halfx**
  - Entity: `halfx`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.one**
  - Entity: `one`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.sec**
  - Entity: `sec`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.sech**
  - Entity: `sech`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.sin**
  - Entity: `sin`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.sinh**
  - Entity: `sinh`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.tan**
  - Entity: `tan`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.tanh**
  - Entity: `tanh`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.two**
  - Entity: `two`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.expression_conversions.x**
  - Entity: `x`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `attribute`

- **sage.symbolic.ring.symbols**
  - Entity: `symbols`
  - Module: `sage.symbolic.ring`
  - Type: `attribute`

#### CLASS (39 entries)

- **sage.stats.distributions.discrete_gaussian_integer.DiscreteGaussianDistributionIntegerSampler**
  - Entity: `DiscreteGaussianDistributionIntegerSampler`
  - Module: `sage.stats.distributions.discrete_gaussian_integer`
  - Type: `class`

- **sage.stats.distributions.discrete_gaussian_lattice.DiscreteGaussianDistributionLatticeSampler**
  - Entity: `DiscreteGaussianDistributionLatticeSampler`
  - Module: `sage.stats.distributions.discrete_gaussian_lattice`
  - Type: `class`

- **sage.stats.distributions.discrete_gaussian_polynomial.DiscreteGaussianDistributionPolynomialSampler**
  - Entity: `DiscreteGaussianDistributionPolynomialSampler`
  - Module: `sage.stats.distributions.discrete_gaussian_polynomial`
  - Type: `class`

- **sage.stats.hmm.chmm.GaussianHiddenMarkovModel**
  - Entity: `GaussianHiddenMarkovModel`
  - Module: `sage.stats.hmm.chmm`
  - Type: `class`

- **sage.stats.hmm.chmm.GaussianMixtureHiddenMarkovModel**
  - Entity: `GaussianMixtureHiddenMarkovModel`
  - Module: `sage.stats.hmm.chmm`
  - Type: `class`

- **sage.stats.hmm.distributions.DiscreteDistribution**
  - Entity: `DiscreteDistribution`
  - Module: `sage.stats.hmm.distributions`
  - Type: `class`

- **sage.stats.hmm.distributions.Distribution**
  - Entity: `Distribution`
  - Module: `sage.stats.hmm.distributions`
  - Type: `class`

- **sage.stats.hmm.distributions.GaussianDistribution**
  - Entity: `GaussianDistribution`
  - Module: `sage.stats.hmm.distributions`
  - Type: `class`

- **sage.stats.hmm.distributions.GaussianMixtureDistribution**
  - Entity: `GaussianMixtureDistribution`
  - Module: `sage.stats.hmm.distributions`
  - Type: `class`

- **sage.stats.hmm.hmm.DiscreteHiddenMarkovModel**
  - Entity: `DiscreteHiddenMarkovModel`
  - Module: `sage.stats.hmm.hmm`
  - Type: `class`

- **sage.stats.hmm.hmm.HiddenMarkovModel**
  - Entity: `HiddenMarkovModel`
  - Module: `sage.stats.hmm.hmm`
  - Type: `class`

- **sage.stats.hmm.util.HMM_Util**
  - Entity: `HMM_Util`
  - Module: `sage.stats.hmm.util`
  - Type: `class`

- **sage.stats.intlist.IntList**
  - Entity: `IntList`
  - Module: `sage.stats.intlist`
  - Type: `class`

- **sage.symbolic.expression.E**
  - Entity: `E`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression.Expression**
  - Entity: `Expression`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression.ExpressionIterator**
  - Entity: `ExpressionIterator`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression.OperandsWrapper**
  - Entity: `OperandsWrapper`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression.PynacConstant**
  - Entity: `PynacConstant`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression.SubstitutionMap**
  - Entity: `SubstitutionMap`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression.SymbolicSeries**
  - Entity: `SymbolicSeries`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression.hold_class**
  - Entity: `hold_class`
  - Module: `sage.symbolic.expression`
  - Type: `class`

- **sage.symbolic.expression_conversions.Converter**
  - Entity: `Converter`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.DeMoivre**
  - Entity: `DeMoivre`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.Exponentialize**
  - Entity: `Exponentialize`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.ExpressionTreeWalker**
  - Entity: `ExpressionTreeWalker`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.FakeExpression**
  - Entity: `FakeExpression`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.FastCallableConverter**
  - Entity: `FastCallableConverter`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.FriCASConverter**
  - Entity: `FriCASConverter`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.HalfAngle**
  - Entity: `HalfAngle`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.HoldRemover**
  - Entity: `HoldRemover`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.InterfaceInit**
  - Entity: `InterfaceInit`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.LaurentPolynomialConverter**
  - Entity: `LaurentPolynomialConverter`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.PolynomialConverter**
  - Entity: `PolynomialConverter`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.RingConverter**
  - Entity: `RingConverter`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.expression_conversions.SubstituteFunction**
  - Entity: `SubstituteFunction`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `class`

- **sage.symbolic.ring.NumpyToSRMorphism**
  - Entity: `NumpyToSRMorphism`
  - Module: `sage.symbolic.ring`
  - Type: `class`

- **sage.symbolic.ring.SymbolicRing**
  - Entity: `SymbolicRing`
  - Module: `sage.symbolic.ring`
  - Type: `class`

- **sage.symbolic.ring.TemporaryVariables**
  - Entity: `TemporaryVariables`
  - Module: `sage.symbolic.ring`
  - Type: `class`

- **sage.symbolic.ring.UnderscoreSageMorphism**
  - Entity: `UnderscoreSageMorphism`
  - Module: `sage.symbolic.ring`
  - Type: `class`

#### FUNCTION (84 entries)

- **sage.stats.basic_stats.mean**
  - Entity: `mean`
  - Module: `sage.stats.basic_stats`
  - Type: `function`

- **sage.stats.basic_stats.median**
  - Entity: `median`
  - Module: `sage.stats.basic_stats`
  - Type: `function`

- **sage.stats.basic_stats.mode**
  - Entity: `mode`
  - Module: `sage.stats.basic_stats`
  - Type: `function`

- **sage.stats.basic_stats.moving_average**
  - Entity: `moving_average`
  - Module: `sage.stats.basic_stats`
  - Type: `function`

- **sage.stats.basic_stats.std**
  - Entity: `std`
  - Module: `sage.stats.basic_stats`
  - Type: `function`

- **sage.stats.basic_stats.variance**
  - Entity: `variance`
  - Module: `sage.stats.basic_stats`
  - Type: `function`

- **sage.stats.hmm.chmm.unpickle_gaussian_hmm_v0**
  - Entity: `unpickle_gaussian_hmm_v0`
  - Module: `sage.stats.hmm.chmm`
  - Type: `function`

- **sage.stats.hmm.chmm.unpickle_gaussian_hmm_v1**
  - Entity: `unpickle_gaussian_hmm_v1`
  - Module: `sage.stats.hmm.chmm`
  - Type: `function`

- **sage.stats.hmm.chmm.unpickle_gaussian_mixture_hmm_v1**
  - Entity: `unpickle_gaussian_mixture_hmm_v1`
  - Module: `sage.stats.hmm.chmm`
  - Type: `function`

- **sage.stats.hmm.distributions.unpickle_gaussian_mixture_distribution_v1**
  - Entity: `unpickle_gaussian_mixture_distribution_v1`
  - Module: `sage.stats.hmm.distributions`
  - Type: `function`

- **sage.stats.hmm.hmm.unpickle_discrete_hmm_v0**
  - Entity: `unpickle_discrete_hmm_v0`
  - Module: `sage.stats.hmm.hmm`
  - Type: `function`

- **sage.stats.hmm.hmm.unpickle_discrete_hmm_v1**
  - Entity: `unpickle_discrete_hmm_v1`
  - Module: `sage.stats.hmm.hmm`
  - Type: `function`

- **sage.stats.intlist.unpickle_intlist_v1**
  - Entity: `unpickle_intlist_v1`
  - Module: `sage.stats.intlist`
  - Type: `function`

- **sage.stats.r.ttest**
  - Entity: `ttest`
  - Module: `sage.stats.r`
  - Type: `function`

- **sage.symbolic.expression.call_registered_function**
  - Entity: `call_registered_function`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.doublefactorial**
  - Entity: `doublefactorial`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.find_registered_function**
  - Entity: `find_registered_function`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.get_fn_serial**
  - Entity: `get_fn_serial`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.get_ginac_serial**
  - Entity: `get_ginac_serial`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.get_sfunction_from_hash**
  - Entity: `get_sfunction_from_hash`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.get_sfunction_from_serial**
  - Entity: `get_sfunction_from_serial`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.init_function_table**
  - Entity: `init_function_table`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.init_pynac_I**
  - Entity: `init_pynac_I`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.is_SymbolicEquation**
  - Entity: `is_SymbolicEquation`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.make_map**
  - Entity: `make_map`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.math_sorted**
  - Entity: `math_sorted`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.mixed_order**
  - Entity: `mixed_order`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.mixed_sorted**
  - Entity: `mixed_sorted`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.new_Expression**
  - Entity: `new_Expression`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.new_Expression_from_pyobject**
  - Entity: `new_Expression_from_pyobject`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.new_Expression_symbol**
  - Entity: `new_Expression_symbol`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.new_Expression_wild**
  - Entity: `new_Expression_wild`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.normalize_index_for_doctests**
  - Entity: `normalize_index_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.paramset_from_Expression**
  - Entity: `paramset_from_Expression`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.print_order**
  - Entity: `print_order`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.print_sorted**
  - Entity: `print_sorted`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_atan2_for_doctests**
  - Entity: `py_atan2_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_denom_for_doctests**
  - Entity: `py_denom_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_eval_infinity_for_doctests**
  - Entity: `py_eval_infinity_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_eval_neg_infinity_for_doctests**
  - Entity: `py_eval_neg_infinity_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_eval_unsigned_infinity_for_doctests**
  - Entity: `py_eval_unsigned_infinity_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_exp_for_doctests**
  - Entity: `py_exp_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_factorial_py**
  - Entity: `py_factorial_py`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_float_for_doctests**
  - Entity: `py_float_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_imag_for_doctests**
  - Entity: `py_imag_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_is_cinteger_for_doctest**
  - Entity: `py_is_cinteger_for_doctest`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_is_crational_for_doctest**
  - Entity: `py_is_crational_for_doctest`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_is_integer_for_doctests**
  - Entity: `py_is_integer_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_latex_fderivative_for_doctests**
  - Entity: `py_latex_fderivative_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_latex_function_pystring**
  - Entity: `py_latex_function_pystring`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_latex_variable_for_doctests**
  - Entity: `py_latex_variable_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_lgamma_for_doctests**
  - Entity: `py_lgamma_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_li2_for_doctests**
  - Entity: `py_li2_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_li_for_doctests**
  - Entity: `py_li_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_log_for_doctests**
  - Entity: `py_log_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_mod_for_doctests**
  - Entity: `py_mod_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_numer_for_doctests**
  - Entity: `py_numer_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_print_fderivative_for_doctests**
  - Entity: `py_print_fderivative_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_print_function_pystring**
  - Entity: `py_print_function_pystring`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_psi2_for_doctests**
  - Entity: `py_psi2_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_psi_for_doctests**
  - Entity: `py_psi_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_real_for_doctests**
  - Entity: `py_real_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_stieltjes_for_doctests**
  - Entity: `py_stieltjes_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_tgamma_for_doctests**
  - Entity: `py_tgamma_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.py_zeta_for_doctests**
  - Entity: `py_zeta_for_doctests`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.register_or_update_function**
  - Entity: `register_or_update_function`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.restore_op_wrapper**
  - Entity: `restore_op_wrapper`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.solve_diophantine**
  - Entity: `solve_diophantine`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.test_binomial**
  - Entity: `test_binomial`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.tolerant_is_symbol**
  - Entity: `tolerant_is_symbol`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression.unpack_operands**
  - Entity: `unpack_operands`
  - Module: `sage.symbolic.expression`
  - Type: `function`

- **sage.symbolic.expression_conversions.fast_callable**
  - Entity: `fast_callable`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `function`

- **sage.symbolic.expression_conversions.laurent_polynomial**
  - Entity: `laurent_polynomial`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `function`

- **sage.symbolic.expression_conversions.polynomial**
  - Entity: `polynomial`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `function`

- **sage.symbolic.relation.solve**
  - Entity: `solve`
  - Module: `sage.symbolic.relation`
  - Type: `function`

- **sage.symbolic.relation.solve_ineq**
  - Entity: `solve_ineq`
  - Module: `sage.symbolic.relation`
  - Type: `function`

- **sage.symbolic.relation.solve_ineq_fourier**
  - Entity: `solve_ineq_fourier`
  - Module: `sage.symbolic.relation`
  - Type: `function`

- **sage.symbolic.relation.solve_ineq_univar**
  - Entity: `solve_ineq_univar`
  - Module: `sage.symbolic.relation`
  - Type: `function`

- **sage.symbolic.relation.solve_mod**
  - Entity: `solve_mod`
  - Module: `sage.symbolic.relation`
  - Type: `function`

- **sage.symbolic.relation.string_to_list_of_solutions**
  - Entity: `string_to_list_of_solutions`
  - Module: `sage.symbolic.relation`
  - Type: `function`

- **sage.symbolic.relation.test_relation_maxima**
  - Entity: `test_relation_maxima`
  - Module: `sage.symbolic.relation`
  - Type: `function`

- **sage.symbolic.ring.isidentifier**
  - Entity: `isidentifier`
  - Module: `sage.symbolic.ring`
  - Type: `function`

- **sage.symbolic.ring.the_SymbolicRing**
  - Entity: `the_SymbolicRing`
  - Module: `sage.symbolic.ring`
  - Type: `function`

- **sage.symbolic.ring.var**
  - Entity: `var`
  - Module: `sage.symbolic.ring`
  - Type: `function`

#### MODULE (15 entries)

- **sage.stats**
  - Entity: `stats`
  - Module: `sage.stats`
  - Type: `module`

- **sage.stats.basic_stats**
  - Entity: `basic_stats`
  - Module: `sage.stats.basic_stats`
  - Type: `module`

- **sage.stats.distributions.discrete_gaussian_integer**
  - Entity: `discrete_gaussian_integer`
  - Module: `sage.stats.distributions.discrete_gaussian_integer`
  - Type: `module`

- **sage.stats.distributions.discrete_gaussian_lattice**
  - Entity: `discrete_gaussian_lattice`
  - Module: `sage.stats.distributions.discrete_gaussian_lattice`
  - Type: `module`

- **sage.stats.distributions.discrete_gaussian_polynomial**
  - Entity: `discrete_gaussian_polynomial`
  - Module: `sage.stats.distributions.discrete_gaussian_polynomial`
  - Type: `module`

- **sage.stats.hmm.chmm**
  - Entity: `chmm`
  - Module: `sage.stats.hmm.chmm`
  - Type: `module`

- **sage.stats.hmm.distributions**
  - Entity: `distributions`
  - Module: `sage.stats.hmm.distributions`
  - Type: `module`

- **sage.stats.hmm.hmm**
  - Entity: `hmm`
  - Module: `sage.stats.hmm.hmm`
  - Type: `module`

- **sage.stats.hmm.util**
  - Entity: `util`
  - Module: `sage.stats.hmm.util`
  - Type: `module`

- **sage.stats.intlist**
  - Entity: `intlist`
  - Module: `sage.stats.intlist`
  - Type: `module`

- **sage.stats.r**
  - Entity: `r`
  - Module: `sage.stats.r`
  - Type: `module`

- **sage.symbolic.expression**
  - Entity: `expression`
  - Module: `sage.symbolic.expression`
  - Type: `module`

- **sage.symbolic.expression_conversions**
  - Entity: `expression_conversions`
  - Module: `sage.symbolic.expression_conversions`
  - Type: `module`

- **sage.symbolic.relation**
  - Entity: `relation`
  - Module: `sage.symbolic.relation`
  - Type: `module`

- **sage.symbolic.ring**
  - Entity: `ring`
  - Module: `sage.symbolic.ring`
  - Type: `module`


