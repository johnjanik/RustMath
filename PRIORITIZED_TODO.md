# RustMath Development TODO - Prioritized Partial Features

**Total Features:** 970

This list organizes the 970 partial SageMath features by implementation priority.

## Priority: HIGH (141 features)

These are fundamental operations used frequently. Implementing these will have the highest impact.

### sage.arith.functions (2 features)

- [ ] **LCM_list** `(function)`
  - Full name: `sage.arith.functions.LCM_list`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/functions.pyx#L127)

- [ ] **lcm** `(function)`
  - Full name: `sage.arith.functions.lcm`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/functions.pyx#L22)

### sage.arith.misc (20 features)

- [ ] **Euler_Phi** `(class)`
  - Full name: `sage.arith.misc.Euler_Phi`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3073)

- [ ] **GCD** `(function)`
  - Full name: `sage.arith.misc.GCD`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1718)

- [ ] **XGCD** `(function)`
  - Full name: `sage.arith.misc.XGCD`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1939)

- [ ] **binomial** `(function)`
  - Full name: `sage.arith.misc.binomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3724)

- [ ] **binomial_coefficients** `(function)`
  - Full name: `sage.arith.misc.binomial_coefficients`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4063)

- [ ] **carmichael_lambda** `(function)`
  - Full name: `sage.arith.misc.carmichael_lambda`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3206)

- [ ] **divisors** `(function)`
  - Full name: `sage.arith.misc.divisors`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1489)

- [ ] **factor** `(function)`
  - Full name: `sage.arith.misc.factor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2563)

- [ ] **factorial** `(function)`
  - Full name: `sage.arith.misc.factorial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L399)

- [ ] **falling_factorial** `(function)`
  - Full name: `sage.arith.misc.falling_factorial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5123)

- [ ] **gcd** `(function)`
  - Full name: `sage.arith.misc.gcd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1718)

- [ ] **get_gcd** `(function)`
  - Full name: `sage.arith.misc.get_gcd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2231)

- [ ] **number_of_divisors** `(function)`
  - Full name: `sage.arith.misc.number_of_divisors`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4833)

- [ ] **prime_divisors** `(function)`
  - Full name: `sage.arith.misc.prime_divisors`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2802)

- [ ] **prime_factors** `(function)`
  - Full name: `sage.arith.misc.prime_factors`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2802)

- [ ] **rising_factorial** `(function)`
  - Full name: `sage.arith.misc.rising_factorial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5224)

- [ ] **squarefree_divisors** `(function)`
  - Full name: `sage.arith.misc.squarefree_divisors`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6110)

- [ ] **subfactorial** `(function)`
  - Full name: `sage.arith.misc.subfactorial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5858)

- [ ] **xgcd** `(function)`
  - Full name: `sage.arith.misc.xgcd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1939)

- [ ] **xlcm** `(function)`
  - Full name: `sage.arith.misc.xlcm`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1899)

### sage.calculus.calculus (3 features)

- [ ] **limit** `(function)`
  - Full name: `sage.calculus.calculus.limit`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1159)

- [ ] **mma_free_limit** `(function)`
  - Full name: `sage.calculus.calculus.mma_free_limit`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1488)

- [ ] **nintegral** `(function)`
  - Full name: `sage.calculus.calculus.nintegral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L683)

### sage.calculus.functional (4 features)

- [ ] **derivative** `(function)`
  - Full name: `sage.calculus.functional.derivative`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L69)

- [ ] **integral** `(function)`
  - Full name: `sage.calculus.functional.integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L175)

- [ ] **limit** `(function)`
  - Full name: `sage.calculus.functional.limit`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L325)

- [ ] **taylor** `(function)`
  - Full name: `sage.calculus.functional.taylor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L381)

### sage.calculus.integration (2 features)

- [ ] **monte_carlo_integral** `(function)`
  - Full name: `sage.calculus.integration.monte_carlo_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/integration.pyx#L443)

- [ ] **numerical_integral** `(function)`
  - Full name: `sage.calculus.integration.numerical_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/integration.pyx#L74)

### sage.calculus.riemann (1 features)

- [ ] **get_derivatives** `(function)`
  - Full name: `sage.calculus.riemann.get_derivatives`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1092)

### sage.functions.exp_integral (11 features)

- [ ] **Function_cos_integral** `(class)`
  - Full name: `sage.functions.exp_integral.Function_cos_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L882)

- [ ] **Function_cosh_integral** `(class)`
  - Full name: `sage.functions.exp_integral.Function_cosh_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1168)

- [ ] **Function_exp_integral** `(class)`
  - Full name: `sage.functions.exp_integral.Function_exp_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1302)

- [ ] **Function_exp_integral_e** `(class)`
  - Full name: `sage.functions.exp_integral.Function_exp_integral_e`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L72)

- [ ] **Function_exp_integral_e1** `(class)`
  - Full name: `sage.functions.exp_integral.Function_exp_integral_e1`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L263)

- [ ] **Function_log_integral** `(class)`
  - Full name: `sage.functions.exp_integral.Function_log_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L370)

- [ ] **Function_log_integral_offset** `(class)`
  - Full name: `sage.functions.exp_integral.Function_log_integral_offset`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L499)

- [ ] **Function_sin_integral** `(class)`
  - Full name: `sage.functions.exp_integral.Function_sin_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L691)

- [ ] **Function_sinh_integral** `(class)`
  - Full name: `sage.functions.exp_integral.Function_sinh_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1021)

- [ ] **exp_integral** `(module)`
  - Full name: `sage.functions.exp_integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L882)

- [ ] **exponential_integral_1** `(function)`
  - Full name: `sage.functions.exp_integral.exponential_integral_1`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/exp_integral.py#L1412)

### sage.functions.hyperbolic (4 features)

- [ ] **Function_arccosh** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_arccosh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L432)

- [ ] **Function_arcsinh** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_arcsinh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L369)

- [ ] **Function_cosh** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_cosh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L87)

- [ ] **Function_sinh** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_sinh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L49)

### sage.functions.log (10 features)

- [ ] **Function_dilog** `(class)`
  - Full name: `sage.functions.log.Function_dilog`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L449)

- [ ] **Function_exp** `(class)`
  - Full name: `sage.functions.log.Function_exp`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L41)

- [ ] **Function_exp_polar** `(class)`
  - Full name: `sage.functions.log.Function_exp_polar`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L916)

- [ ] **Function_harmonic_number** `(class)`
  - Full name: `sage.functions.log.Function_harmonic_number`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L1306)

- [ ] **Function_harmonic_number_generalized** `(class)`
  - Full name: `sage.functions.log.Function_harmonic_number_generalized`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L1027)

- [ ] **Function_lambert_w** `(class)`
  - Full name: `sage.functions.log.Function_lambert_w`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L564)

- [ ] **Function_log1** `(class)`
  - Full name: `sage.functions.log.Function_log1`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L184)

- [ ] **Function_log2** `(class)`
  - Full name: `sage.functions.log.Function_log2`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L266)

- [ ] **Function_polylog** `(class)`
  - Full name: `sage.functions.log.Function_polylog`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L301)

- [ ] **log** `(module)`
  - Full name: `sage.functions.log`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/log.py#L449)

### sage.functions.trig (4 features)

- [ ] **Function_arccos** `(class)`
  - Full name: `sage.functions.trig.Function_arccos`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L589)

- [ ] **Function_arcsin** `(class)`
  - Full name: `sage.functions.trig.Function_arcsin`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L525)

- [ ] **Function_cos** `(class)`
  - Full name: `sage.functions.trig.Function_cos`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L138)

- [ ] **Function_sin** `(class)`
  - Full name: `sage.functions.trig.Function_sin`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L9)

### sage.rings.polynomial.complex_roots (4 features)

- [ ] **complex_roots** `(module)`
  - Full name: `sage.rings.polynomial.complex_roots`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L159)

- [ ] **complex_roots** `(function)`
  - Full name: `sage.rings.polynomial.complex_roots.complex_roots`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L159)

- [ ] **interval_roots** `(function)`
  - Full name: `sage.rings.polynomial.complex_roots.interval_roots`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L51)

- [ ] **intervals_disjoint** `(function)`
  - Full name: `sage.rings.polynomial.complex_roots.intervals_disjoint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/complex_roots.py#L94)

### sage.rings.polynomial.infinite_polynomial_ring (1 features)

- [ ] **InfinitePolynomialRingFactory** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRingFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L279)

### sage.rings.polynomial.pbori.pbori (4 features)

- [ ] **MonomialFactory** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.MonomialFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7955)

- [ ] **PolynomialFactory** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.PolynomialFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L8028)

- [ ] **VariableFactory** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.VariableFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7907)

- [ ] **easy_linear_factors** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.easy_linear_factors`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7715)

### sage.rings.polynomial.plural (1 features)

- [ ] **G_AlgFactory** `(class)`
  - Full name: `sage.rings.polynomial.plural.G_AlgFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L150)

### sage.rings.polynomial.polynomial_element (1 features)

- [ ] **universal_discriminant** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_element.universal_discriminant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12592)

### sage.rings.polynomial.polynomial_modn_dense_ntl (1 features)

- [ ] **small_roots** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_modn_dense_ntl.small_roots`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L472)

### sage.rings.polynomial.polynomial_quotient_ring (1 features)

- [ ] **PolynomialQuotientRingFactory** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRingFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L65)

### sage.rings.polynomial.real_roots (51 features)

- [ ] **bernstein_down** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.bernstein_down`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2067)

- [ ] **bernstein_expand** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.bernstein_expand`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4463)

- [ ] **bernstein_polynomial_factory** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.bernstein_polynomial_factory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2530)

- [ ] **bernstein_polynomial_factory_ar** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.bernstein_polynomial_factory_ar`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2734)

- [ ] **bernstein_polynomial_factory_intlist** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.bernstein_polynomial_factory_intlist`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2570)

- [ ] **bernstein_polynomial_factory_ratlist** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.bernstein_polynomial_factory_ratlist`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2649)

- [ ] **bernstein_up** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.bernstein_up`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2109)

- [ ] **bitsize_doctest** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.bitsize_doctest`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1968)

- [ ] **cl_maximum_root** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.cl_maximum_root`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2325)

- [ ] **cl_maximum_root_first_lambda** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.cl_maximum_root_first_lambda`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2201)

- [ ] **cl_maximum_root_local_max** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.cl_maximum_root_local_max`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2289)

- [ ] **context** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.context`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4318)

- [ ] **de_casteljau_doublevec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.de_casteljau_doublevec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1712)

- [ ] **de_casteljau_intvec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.de_casteljau_intvec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L995)

- [ ] **degree_reduction_next_size** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.degree_reduction_next_size`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1972)

- [ ] **dprod_imatrow_vec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.dprod_imatrow_vec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4560)

- [ ] **get_realfield_rndu** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.get_realfield_rndu`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4295)

- [ ] **interval_bernstein_polynomial** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.interval_bernstein_polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L168)

- [ ] **interval_bernstein_polynomial_float** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.interval_bernstein_polynomial_float`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1335)

- [ ] **interval_bernstein_polynomial_integer** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.interval_bernstein_polynomial_integer`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L437)

- [ ] **intvec_to_doublevec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.intvec_to_doublevec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1270)

- [ ] **island** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.island`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3260)

- [ ] **linear_map** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.linear_map`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3782)

- [ ] **max_abs_doublevec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.max_abs_doublevec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1797)

- [ ] **max_bitsize_intvec_doctest** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.max_bitsize_intvec_doctest`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4556)

- [ ] **maximum_root_first_lambda** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.maximum_root_first_lambda`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2174)

- [ ] **maximum_root_local_max** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.maximum_root_local_max`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2264)

- [ ] **min_max_delta_intvec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.min_max_delta_intvec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4609)

- [ ] **min_max_diff_doublevec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.min_max_diff_doublevec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4675)

- [ ] **min_max_diff_intvec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.min_max_diff_intvec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4644)

- [ ] **mk_context** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.mk_context`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4379)

- [ ] **mk_ibpf** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.mk_ibpf`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1683)

- [ ] **mk_ibpi** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.mk_ibpi`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L978)

- [ ] **ocean** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.ocean`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2971)

- [ ] **precompute_degree_reduction_cache** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.precompute_degree_reduction_cache`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2010)

- [ ] **pseudoinverse** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.pseudoinverse`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2104)

- [ ] **rational_root_bounds** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.rational_root_bounds`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2418)

- [ ] **real_roots** `(module)`
  - Full name: `sage.rings.polynomial.real_roots`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2067)

- [ ] **real_roots** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.real_roots`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3835)

- [ ] **relative_bounds** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.relative_bounds`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1918)

- [ ] **reverse_intvec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.reverse_intvec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4270)

- [ ] **root_bounds** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.root_bounds`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2347)

- [ ] **rr_gap** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.rr_gap`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3763)

- [ ] **scale_intvec_var** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.scale_intvec_var`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4211)

- [ ] **split_for_targets** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.split_for_targets`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2846)

- [ ] **subsample_vec_doctest** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.subsample_vec_doctest`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L2170)

- [ ] **taylor_shift1_intvec** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.taylor_shift1_intvec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4239)

- [ ] **to_bernstein** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.to_bernstein`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4395)

- [ ] **to_bernstein_warp** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.to_bernstein_warp`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L4445)

- [ ] **warp_map** `(class)`
  - Full name: `sage.rings.polynomial.real_roots.warp_map`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L3806)

- [ ] **wordsize_rational** `(function)`
  - Full name: `sage.rings.polynomial.real_roots.wordsize_rational`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/real_roots.pyx#L1821)

### sage.rings.polynomial.toy_variety (1 features)

- [ ] **triangular_factorization** `(function)`
  - Full name: `sage.rings.polynomial.toy_variety.triangular_factorization`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L230)

### sage.symbolic.callable (1 features)

- [ ] **CallableSymbolicExpressionRingFactory** `(class)`
  - Full name: `sage.symbolic.callable.CallableSymbolicExpressionRingFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L426)

### sage.symbolic.expression (3 features)

- [ ] **doublefactorial** `(function)`
  - Full name: `sage.symbolic.expression.doublefactorial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1517)

- [ ] **py_factorial_py** `(function)`
  - Full name: `sage.symbolic.expression.py_factorial_py`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1495)

- [ ] **solve_diophantine** `(function)`
  - Full name: `sage.symbolic.expression.solve_diophantine`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13522)

### sage.symbolic.function_factory (4 features)

- [ ] **function** `(function)`
  - Full name: `sage.symbolic.function_factory.function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L152)

- [ ] **function_factory** `(module)`
  - Full name: `sage.symbolic.function_factory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L152)

- [ ] **function_factory** `(function)`
  - Full name: `sage.symbolic.function_factory.function_factory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L18)

- [ ] **unpickle_function** `(function)`
  - Full name: `sage.symbolic.function_factory.unpickle_function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function_factory.py#L113)

### sage.symbolic.integration.integral (1 features)

- [ ] **integrate** `(function)`
  - Full name: `sage.symbolic.integration.integral.integrate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L446)

### sage.symbolic.relation (5 features)

- [ ] **solve** `(function)`
  - Full name: `sage.symbolic.relation.solve`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L584)

- [ ] **solve_ineq** `(function)`
  - Full name: `sage.symbolic.relation.solve_ineq`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1829)

- [ ] **solve_ineq_fourier** `(function)`
  - Full name: `sage.symbolic.relation.solve_ineq_fourier`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1755)

- [ ] **solve_ineq_univar** `(function)`
  - Full name: `sage.symbolic.relation.solve_ineq_univar`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1708)

- [ ] **solve_mod** `(function)`
  - Full name: `sage.symbolic.relation.solve_mod`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L1473)

### sage.symbolic.subring (1 features)

- [ ] **SymbolicSubringFactory** `(class)`
  - Full name: `sage.symbolic.subring.SymbolicSubringFactory`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L106)


---

## Priority: MEDIUM (810 features)

Useful features that enhance functionality but are not critical.

### sage.arith (1 features)

- [ ] **arith** `(module)`
  - Full name: `sage.arith`

### sage.arith.functions (1 features)

- [ ] **functions** `(module)`
  - Full name: `sage.arith.functions`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/functions.pyx#L127)

### sage.arith.misc (67 features)

- [ ] **CRT** `(function)`
  - Full name: `sage.arith.misc.CRT`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3354)

- [ ] **CRT_basis** `(function)`
  - Full name: `sage.arith.misc.CRT_basis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3641)

- [ ] **CRT_list** `(function)`
  - Full name: `sage.arith.misc.CRT_list`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3500)

- [ ] **CRT_vectors** `(function)`
  - Full name: `sage.arith.misc.CRT_vectors`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3688)

- [ ] **Moebius** `(class)`
  - Full name: `sage.arith.misc.Moebius`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4566)

- [ ] **Sigma** `(class)`
  - Full name: `sage.arith.misc.Sigma`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1580)

- [ ] **algdep** `(function)`
  - Full name: `sage.arith.misc.algdep`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L43)

- [ ] **algebraic_dependency** `(function)`
  - Full name: `sage.arith.misc.algebraic_dependency`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L43)

- [ ] **bernoulli** `(function)`
  - Full name: `sage.arith.misc.bernoulli`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L280)

- [ ] **continuant** `(function)`
  - Full name: `sage.arith.misc.continuant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4749)

- [ ] **coprime_part** `(function)`
  - Full name: `sage.arith.misc.coprime_part`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6470)

- [ ] **crt** `(function)`
  - Full name: `sage.arith.misc.crt`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3354)

- [ ] **dedekind_psi** `(function)`
  - Full name: `sage.arith.misc.dedekind_psi`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6393)

- [ ] **dedekind_sum** `(function)`
  - Full name: `sage.arith.misc.dedekind_sum`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6167)

- [ ] **differences** `(function)`
  - Full name: `sage.arith.misc.differences`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5933)

- [ ] **eratosthenes** `(function)`
  - Full name: `sage.arith.misc.eratosthenes`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L963)

- [ ] **four_squares** `(function)`
  - Full name: `sage.arith.misc.four_squares`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5677)

- [ ] **fundamental_discriminant** `(function)`
  - Full name: `sage.arith.misc.fundamental_discriminant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6072)

- [ ] **gauss_sum** `(function)`
  - Full name: `sage.arith.misc.gauss_sum`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6279)

- [ ] **get_inverse_mod** `(function)`
  - Full name: `sage.arith.misc.get_inverse_mod`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2253)

- [ ] **hilbert_conductor** `(function)`
  - Full name: `sage.arith.misc.hilbert_conductor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4981)

- [ ] **hilbert_conductor_inverse** `(function)`
  - Full name: `sage.arith.misc.hilbert_conductor_inverse`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5025)

- [ ] **hilbert_symbol** `(function)`
  - Full name: `sage.arith.misc.hilbert_symbol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4866)

- [ ] **integer_ceil** `(function)`
  - Full name: `sage.arith.misc.integer_ceil`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5316)

- [ ] **integer_floor** `(function)`
  - Full name: `sage.arith.misc.integer_floor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5348)

- [ ] **integer_trunc** `(function)`
  - Full name: `sage.arith.misc.integer_trunc`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5394)

- [ ] **inverse_mod** `(function)`
  - Full name: `sage.arith.misc.inverse_mod`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2195)

- [ ] **is_power_of_two** `(function)`
  - Full name: `sage.arith.misc.is_power_of_two`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5894)

- [ ] **is_prime** `(function)`
  - Full name: `sage.arith.misc.is_prime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L479)

- [ ] **is_prime_power** `(function)`
  - Full name: `sage.arith.misc.is_prime_power`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L610)

- [ ] **is_pseudoprime** `(function)`
  - Full name: `sage.arith.misc.is_pseudoprime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L575)

- [ ] **is_pseudoprime_power** `(function)`
  - Full name: `sage.arith.misc.is_pseudoprime_power`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L680)

- [ ] **is_square** `(function)`
  - Full name: `sage.arith.misc.is_square`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2928)

- [ ] **is_squarefree** `(function)`
  - Full name: `sage.arith.misc.is_squarefree`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3007)

- [ ] **jacobi_symbol** `(function)`
  - Full name: `sage.arith.misc.jacobi_symbol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4318)

- [ ] **kronecker** `(function)`
  - Full name: `sage.arith.misc.kronecker`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4218)

- [ ] **kronecker_symbol** `(function)`
  - Full name: `sage.arith.misc.kronecker_symbol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4218)

- [ ] **legendre_symbol** `(function)`
  - Full name: `sage.arith.misc.legendre_symbol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4266)

- [ ] **misc** `(module)`
  - Full name: `sage.arith.misc`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3354)

- [ ] **mqrr_rational_reconstruction** `(function)`
  - Full name: `sage.arith.misc.mqrr_rational_reconstruction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2458)

- [ ] **multinomial** `(function)`
  - Full name: `sage.arith.misc.multinomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L3995)

- [ ] **multinomial_coefficients** `(function)`
  - Full name: `sage.arith.misc.multinomial_coefficients`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4108)

- [ ] **next_prime** `(function)`
  - Full name: `sage.arith.misc.next_prime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1200)

- [ ] **next_prime_power** `(function)`
  - Full name: `sage.arith.misc.next_prime_power`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1108)

- [ ] **next_probable_prime** `(function)`
  - Full name: `sage.arith.misc.next_probable_prime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1168)

- [ ] **nth_prime** `(function)`
  - Full name: `sage.arith.misc.nth_prime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4490)

- [ ] **odd_part** `(function)`
  - Full name: `sage.arith.misc.odd_part`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2858)

- [ ] **power_mod** `(function)`
  - Full name: `sage.arith.misc.power_mod`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2285)

- [ ] **previous_prime** `(function)`
  - Full name: `sage.arith.misc.previous_prime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1249)

- [ ] **previous_prime_power** `(function)`
  - Full name: `sage.arith.misc.previous_prime_power`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1303)

- [ ] **prime_powers** `(function)`
  - Full name: `sage.arith.misc.prime_powers`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L809)

- [ ] **prime_to_m_part** `(function)`
  - Full name: `sage.arith.misc.prime_to_m_part`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2886)

- [ ] **primes** `(function)`
  - Full name: `sage.arith.misc.primes`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1020)

- [ ] **primes_first_n** `(function)`
  - Full name: `sage.arith.misc.primes_first_n`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L931)

- [ ] **primitive_root** `(function)`
  - Full name: `sage.arith.misc.primitive_root`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4369)

- [ ] **quadratic_residues** `(function)`
  - Full name: `sage.arith.misc.quadratic_residues`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L4533)

- [ ] **radical** `(function)`
  - Full name: `sage.arith.misc.radical`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2762)

- [ ] **random_prime** `(function)`
  - Full name: `sage.arith.misc.random_prime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L1376)

- [ ] **rational_reconstruction** `(function)`
  - Full name: `sage.arith.misc.rational_reconstruction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2360)

- [ ] **smooth_part** `(function)`
  - Full name: `sage.arith.misc.smooth_part`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6421)

- [ ] **sort_complex_numbers_for_display** `(function)`
  - Full name: `sage.arith.misc.sort_complex_numbers_for_display`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L6011)

- [ ] **sum_of_k_squares** `(function)`
  - Full name: `sage.arith.misc.sum_of_k_squares`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5752)

- [ ] **three_squares** `(function)`
  - Full name: `sage.arith.misc.three_squares`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5531)

- [ ] **trial_division** `(function)`
  - Full name: `sage.arith.misc.trial_division`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2518)

- [ ] **two_squares** `(function)`
  - Full name: `sage.arith.misc.two_squares`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L5412)

- [ ] **valuation** `(function)`
  - Full name: `sage.arith.misc.valuation`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L738)

- [ ] **xkcd** `(function)`
  - Full name: `sage.arith.misc.xkcd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/misc.py#L2136)

### sage.arith.multi_modular (4 features)

- [ ] **MultiModularBasis** `(class)`
  - Full name: `sage.arith.multi_modular.MultiModularBasis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L851)

- [ ] **MultiModularBasis_base** `(class)`
  - Full name: `sage.arith.multi_modular.MultiModularBasis_base`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L37)

- [ ] **MutableMultiModularBasis** `(class)`
  - Full name: `sage.arith.multi_modular.MutableMultiModularBasis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L917)

- [ ] **multi_modular** `(module)`
  - Full name: `sage.arith.multi_modular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/multi_modular.pyx#L851)

### sage.arith.power (2 features)

- [ ] **generic_power** `(function)`
  - Full name: `sage.arith.power.generic_power`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/power.pyx#L24)

- [ ] **power** `(module)`
  - Full name: `sage.arith.power`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/power.pyx#L24)

### sage.arith.srange (5 features)

- [ ] **ellipsis_iter** `(function)`
  - Full name: `sage.arith.srange.ellipsis_iter`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L321)

- [ ] **ellipsis_range** `(function)`
  - Full name: `sage.arith.srange.ellipsis_range`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L449)

- [ ] **srange** `(module)`
  - Full name: `sage.arith.srange`

- [ ] **srange** `(function)`
  - Full name: `sage.arith.srange.srange`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L175)

- [ ] **xsrange** `(function)`
  - Full name: `sage.arith.srange.xsrange`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/arith/srange.pyx#L30)

### sage.calculus (1 features)

- [ ] **calculus** `(module)`
  - Full name: `sage.calculus`

### sage.calculus.calculus (18 features)

- [ ] **at** `(function)`
  - Full name: `sage.calculus.calculus.at`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1964)

- [ ] **calculus** `(module)`
  - Full name: `sage.calculus.calculus`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1964)

- [ ] **dummy_diff** `(function)`
  - Full name: `sage.calculus.calculus.dummy_diff`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2042)

- [ ] **dummy_integrate** `(function)`
  - Full name: `sage.calculus.calculus.dummy_integrate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2067)

- [ ] **dummy_inverse_laplace** `(function)`
  - Full name: `sage.calculus.calculus.dummy_inverse_laplace`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2104)

- [ ] **dummy_laplace** `(function)`
  - Full name: `sage.calculus.calculus.dummy_laplace`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2088)

- [ ] **dummy_pochhammer** `(function)`
  - Full name: `sage.calculus.calculus.dummy_pochhammer`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2120)

- [ ] **inverse_laplace** `(function)`
  - Full name: `sage.calculus.calculus.inverse_laplace`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1779)

- [ ] **laplace** `(function)`
  - Full name: `sage.calculus.calculus.laplace`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1537)

- [ ] **lim** `(function)`
  - Full name: `sage.calculus.calculus.lim`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L1159)

- [ ] **mapped_opts** `(function)`
  - Full name: `sage.calculus.calculus.mapped_opts`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2427)

- [ ] **maxima_options** `(function)`
  - Full name: `sage.calculus.calculus.maxima_options`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2455)

- [ ] **minpoly** `(function)`
  - Full name: `sage.calculus.calculus.minpoly`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L936)

- [ ] **nintegrate** `(function)`
  - Full name: `sage.calculus.calculus.nintegrate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L683)

- [ ] **symbolic_expression_from_maxima_string** `(function)`
  - Full name: `sage.calculus.calculus.symbolic_expression_from_maxima_string`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2236)

- [ ] **symbolic_expression_from_string** `(function)`
  - Full name: `sage.calculus.calculus.symbolic_expression_from_string`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L2564)

- [ ] **symbolic_product** `(function)`
  - Full name: `sage.calculus.calculus.symbolic_product`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L828)

- [ ] **symbolic_sum** `(function)`
  - Full name: `sage.calculus.calculus.symbolic_sum`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/calculus.py#L445)

### sage.calculus.desolvers (15 features)

- [ ] **desolve** `(function)`
  - Full name: `sage.calculus.desolvers.desolve`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L179)

- [ ] **desolve_laplace** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_laplace`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L661)

- [ ] **desolve_mintides** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_mintides`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1684)

- [ ] **desolve_odeint** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_odeint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1486)

- [ ] **desolve_rk4** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_rk4`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1228)

- [ ] **desolve_rk4_determine_bounds** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_rk4_determine_bounds`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1184)

- [ ] **desolve_system** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_system`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L789)

- [ ] **desolve_system_rk4** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_system_rk4`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1382)

- [ ] **desolve_tides_mpfr** `(function)`
  - Full name: `sage.calculus.desolvers.desolve_tides_mpfr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1771)

- [ ] **desolvers** `(module)`
  - Full name: `sage.calculus.desolvers`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L179)

- [ ] **eulers_method** `(function)`
  - Full name: `sage.calculus.desolvers.eulers_method`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L950)

- [ ] **eulers_method_2x2** `(function)`
  - Full name: `sage.calculus.desolvers.eulers_method_2x2`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1038)

- [ ] **eulers_method_2x2_plot** `(function)`
  - Full name: `sage.calculus.desolvers.eulers_method_2x2_plot`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L1141)

- [ ] **fricas_desolve** `(function)`
  - Full name: `sage.calculus.desolvers.fricas_desolve`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L88)

- [ ] **fricas_desolve_system** `(function)`
  - Full name: `sage.calculus.desolvers.fricas_desolve_system`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/desolvers.py#L125)

### sage.calculus.expr (2 features)

- [ ] **expr** `(module)`
  - Full name: `sage.calculus.expr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/expr.py#L14)

- [ ] **symbolic_expression** `(function)`
  - Full name: `sage.calculus.expr.symbolic_expression`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/expr.py#L14)

### sage.calculus.functional (6 features)

- [ ] **diff** `(function)`
  - Full name: `sage.calculus.functional.diff`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L69)

- [ ] **expand** `(function)`
  - Full name: `sage.calculus.functional.expand`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L425)

- [ ] **functional** `(module)`
  - Full name: `sage.calculus.functional`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L69)

- [ ] **integrate** `(function)`
  - Full name: `sage.calculus.functional.integrate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L175)

- [ ] **lim** `(function)`
  - Full name: `sage.calculus.functional.lim`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L325)

- [ ] **simplify** `(function)`
  - Full name: `sage.calculus.functional.simplify`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functional.py#L32)

### sage.calculus.functions (3 features)

- [ ] **functions** `(module)`
  - Full name: `sage.calculus.functions`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functions.py#L109)

- [ ] **jacobian** `(function)`
  - Full name: `sage.calculus.functions.jacobian`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functions.py#L109)

- [ ] **wronskian** `(function)`
  - Full name: `sage.calculus.functions.wronskian`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/functions.py#L13)

### sage.calculus.integration (3 features)

- [ ] **PyFunctionWrapper** `(class)`
  - Full name: `sage.calculus.integration.PyFunctionWrapper`

- [ ] **compiled_integrand** `(class)`
  - Full name: `sage.calculus.integration.compiled_integrand`

- [ ] **integration** `(module)`
  - Full name: `sage.calculus.integration`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/integration.pyx#L443)

### sage.calculus.interpolation (3 features)

- [ ] **Spline** `(class)`
  - Full name: `sage.calculus.interpolation.Spline`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolation.pyx#L9)

- [ ] **interpolation** `(module)`
  - Full name: `sage.calculus.interpolation`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolation.pyx#L9)

- [ ] **spline** `(attribute)`
  - Full name: `sage.calculus.interpolation.spline`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolation.pyx#L9)

### sage.calculus.interpolators (5 features)

- [ ] **CCSpline** `(class)`
  - Full name: `sage.calculus.interpolators.CCSpline`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L207)

- [ ] **PSpline** `(class)`
  - Full name: `sage.calculus.interpolators.PSpline`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L71)

- [ ] **complex_cubic_spline** `(function)`
  - Full name: `sage.calculus.interpolators.complex_cubic_spline`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L172)

- [ ] **interpolators** `(module)`
  - Full name: `sage.calculus.interpolators`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L207)

- [ ] **polygon_spline** `(function)`
  - Full name: `sage.calculus.interpolators.polygon_spline`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/interpolators.pyx#L37)

### sage.calculus.ode (4 features)

- [ ] **PyFunctionWrapper** `(class)`
  - Full name: `sage.calculus.ode.PyFunctionWrapper`

- [ ] **ode** `(module)`
  - Full name: `sage.calculus.ode`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/ode.pyx#L106)

- [ ] **ode_solver** `(class)`
  - Full name: `sage.calculus.ode.ode_solver`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/ode.pyx#L106)

- [ ] **ode_system** `(class)`
  - Full name: `sage.calculus.ode.ode_system`

### sage.calculus.riemann (7 features)

- [ ] **Riemann_Map** `(class)`
  - Full name: `sage.calculus.riemann.Riemann_Map`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L70)

- [ ] **analytic_boundary** `(function)`
  - Full name: `sage.calculus.riemann.analytic_boundary`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1371)

- [ ] **analytic_interior** `(function)`
  - Full name: `sage.calculus.riemann.analytic_interior`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1467)

- [ ] **cauchy_kernel** `(function)`
  - Full name: `sage.calculus.riemann.cauchy_kernel`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1421)

- [ ] **complex_to_rgb** `(function)`
  - Full name: `sage.calculus.riemann.complex_to_rgb`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1266)

- [ ] **complex_to_spiderweb** `(function)`
  - Full name: `sage.calculus.riemann.complex_to_spiderweb`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L1147)

- [ ] **riemann** `(module)`
  - Full name: `sage.calculus.riemann`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/riemann.pyx#L70)

### sage.calculus.test_sympy (1 features)

- [ ] **test_sympy** `(module)`
  - Full name: `sage.calculus.test_sympy`

### sage.calculus.tests (1 features)

- [ ] **tests** `(module)`
  - Full name: `sage.calculus.tests`

### sage.calculus.transforms.dft (2 features)

- [ ] **IndexedSequence** `(class)`
  - Full name: `sage.calculus.transforms.dft.IndexedSequence`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dft.py#L90)

- [ ] **dft** `(module)`
  - Full name: `sage.calculus.transforms.dft`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dft.py#L90)

### sage.calculus.transforms.dwt (5 features)

- [ ] **DWT** `(function)`
  - Full name: `sage.calculus.transforms.dwt.DWT`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L24)

- [ ] **DiscreteWaveletTransform** `(class)`
  - Full name: `sage.calculus.transforms.dwt.DiscreteWaveletTransform`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L102)

- [ ] **WaveletTransform** `(function)`
  - Full name: `sage.calculus.transforms.dwt.WaveletTransform`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L24)

- [ ] **dwt** `(module)`
  - Full name: `sage.calculus.transforms.dwt`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L24)

- [ ] **is2pow** `(function)`
  - Full name: `sage.calculus.transforms.dwt.is2pow`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/dwt.pyx#L156)

### sage.calculus.transforms.fft (7 features)

- [ ] **FFT** `(function)`
  - Full name: `sage.calculus.transforms.fft.FFT`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L29)

- [ ] **FastFourierTransform** `(function)`
  - Full name: `sage.calculus.transforms.fft.FastFourierTransform`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L29)

- [ ] **FastFourierTransform_base** `(class)`
  - Full name: `sage.calculus.transforms.fft.FastFourierTransform_base`

- [ ] **FastFourierTransform_complex** `(class)`
  - Full name: `sage.calculus.transforms.fft.FastFourierTransform_complex`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L85)

- [ ] **FourierTransform_complex** `(class)`
  - Full name: `sage.calculus.transforms.fft.FourierTransform_complex`

- [ ] **FourierTransform_real** `(class)`
  - Full name: `sage.calculus.transforms.fft.FourierTransform_real`

- [ ] **fft** `(module)`
  - Full name: `sage.calculus.transforms.fft`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/transforms/fft.pyx#L29)

### sage.calculus.var (4 features)

- [ ] **clear_vars** `(function)`
  - Full name: `sage.calculus.var.clear_vars`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L362)

- [ ] **function** `(function)`
  - Full name: `sage.calculus.var.function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L133)

- [ ] **var** `(module)`
  - Full name: `sage.calculus.var`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L362)

- [ ] **var** `(function)`
  - Full name: `sage.calculus.var.var`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/calculus/var.pyx#L10)

### sage.calculus.wester (1 features)

- [ ] **wester** `(module)`
  - Full name: `sage.calculus.wester`

### sage.categories.fields (4 features)

- [ ] **ElementMethods** `(class)`
  - Full name: `sage.categories.fields.ElementMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L652)

- [ ] **Fields** `(class)`
  - Full name: `sage.categories.fields.Fields`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L27)

- [ ] **ParentMethods** `(class)`
  - Full name: `sage.categories.fields.ParentMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L194)

- [ ] **fields** `(module)`
  - Full name: `sage.categories.fields`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/fields.py#L27)

### sage.categories.groups (7 features)

- [ ] **CartesianProducts** `(class)`
  - Full name: `sage.categories.groups.CartesianProducts`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L548)

- [ ] **Commutative** `(class)`
  - Full name: `sage.categories.groups.Commutative`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L496)

- [ ] **ElementMethods** `(class)`
  - Full name: `sage.categories.groups.ElementMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L454)

- [ ] **Groups** `(class)`
  - Full name: `sage.categories.groups.Groups`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L24)

- [ ] **ParentMethods** `(class)`
  - Full name: `sage.categories.groups.ParentMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L573)

- [ ] **Topological** `(class)`
  - Full name: `sage.categories.groups.Topological`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L655)

- [ ] **groups** `(module)`
  - Full name: `sage.categories.groups`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/groups.py#L24)

### sage.categories.modules (11 features)

- [ ] **CartesianProducts** `(class)`
  - Full name: `sage.categories.modules.CartesianProducts`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L835)

- [ ] **ElementMethods** `(class)`
  - Full name: `sage.categories.modules.ElementMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L918)

- [ ] **Endset** `(class)`
  - Full name: `sage.categories.modules.Endset`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L813)

- [ ] **FiniteDimensional** `(class)`
  - Full name: `sage.categories.modules.FiniteDimensional`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L515)

- [ ] **FinitelyPresented** `(class)`
  - Full name: `sage.categories.modules.FinitelyPresented`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L564)

- [ ] **Homsets** `(class)`
  - Full name: `sage.categories.modules.Homsets`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L724)

- [ ] **Modules** `(class)`
  - Full name: `sage.categories.modules.Modules`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L34)

- [ ] **ParentMethods** `(class)`
  - Full name: `sage.categories.modules.ParentMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L860)

- [ ] **SubcategoryMethods** `(class)`
  - Full name: `sage.categories.modules.SubcategoryMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L199)

- [ ] **TensorProducts** `(class)`
  - Full name: `sage.categories.modules.TensorProducts`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L547)

- [ ] **modules** `(module)`
  - Full name: `sage.categories.modules`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules.py#L34)

### sage.categories.modules_with_basis (9 features)

- [ ] **CartesianProducts** `(class)`
  - Full name: `sage.categories.modules_with_basis.CartesianProducts`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2547)

- [ ] **DualObjects** `(class)`
  - Full name: `sage.categories.modules_with_basis.DualObjects`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2753)

- [ ] **ElementMethods** `(class)`
  - Full name: `sage.categories.modules_with_basis.ElementMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L1441)

- [ ] **Homsets** `(class)`
  - Full name: `sage.categories.modules_with_basis.Homsets`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2429)

- [ ] **ModulesWithBasis** `(class)`
  - Full name: `sage.categories.modules_with_basis.ModulesWithBasis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L45)

- [ ] **MorphismMethods** `(class)`
  - Full name: `sage.categories.modules_with_basis.MorphismMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2500)

- [ ] **ParentMethods** `(class)`
  - Full name: `sage.categories.modules_with_basis.ParentMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2566)

- [ ] **TensorProducts** `(class)`
  - Full name: `sage.categories.modules_with_basis.TensorProducts`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L2592)

- [ ] **modules_with_basis** `(module)`
  - Full name: `sage.categories.modules_with_basis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/modules_with_basis.py#L45)

### sage.categories.rings (6 features)

- [ ] **ElementMethods** `(class)`
  - Full name: `sage.categories.rings.ElementMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L1598)

- [ ] **MorphismMethods** `(class)`
  - Full name: `sage.categories.rings.MorphismMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L66)

- [ ] **ParentMethods** `(class)`
  - Full name: `sage.categories.rings.ParentMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L317)

- [ ] **Rings** `(class)`
  - Full name: `sage.categories.rings.Rings`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L27)

- [ ] **SubcategoryMethods** `(class)`
  - Full name: `sage.categories.rings.SubcategoryMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L266)

- [ ] **rings** `(module)`
  - Full name: `sage.categories.rings`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/categories/rings.py#L27)

### sage.functions.hyperbolic (9 features)

- [ ] **Function_arccoth** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_arccoth`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L584)

- [ ] **Function_arccsch** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_arccsch`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L694)

- [ ] **Function_arcsech** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_arcsech`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L646)

- [ ] **Function_arctanh** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_arctanh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L523)

- [ ] **Function_coth** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_coth`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L193)

- [ ] **Function_csch** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_csch`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L309)

- [ ] **Function_sech** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_sech`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L252)

- [ ] **Function_tanh** `(class)`
  - Full name: `sage.functions.hyperbolic.Function_tanh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L125)

- [ ] **hyperbolic** `(module)`
  - Full name: `sage.functions.hyperbolic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/hyperbolic.py#L432)

### sage.functions.other (20 features)

- [ ] **Function_Order** `(class)`
  - Full name: `sage.functions.other.Function_Order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L641)

- [ ] **Function_abs** `(class)`
  - Full name: `sage.functions.other.Function_abs`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L42)

- [ ] **Function_arg** `(class)`
  - Full name: `sage.functions.other.Function_arg`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L964)

- [ ] **Function_binomial** `(class)`
  - Full name: `sage.functions.other.Function_binomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1567)

- [ ] **Function_cases** `(class)`
  - Full name: `sage.functions.other.Function_cases`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L2006)

- [ ] **Function_ceil** `(class)`
  - Full name: `sage.functions.other.Function_ceil`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L300)

- [ ] **Function_conjugate** `(class)`
  - Full name: `sage.functions.other.Function_conjugate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1283)

- [ ] **Function_crootof** `(class)`
  - Full name: `sage.functions.other.Function_crootof`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L2126)

- [ ] **Function_elementof** `(class)`
  - Full name: `sage.functions.other.Function_elementof`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L2219)

- [ ] **Function_factorial** `(class)`
  - Full name: `sage.functions.other.Function_factorial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1372)

- [ ] **Function_floor** `(class)`
  - Full name: `sage.functions.other.Function_floor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L473)

- [ ] **Function_frac** `(class)`
  - Full name: `sage.functions.other.Function_frac`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L706)

- [ ] **Function_imag_part** `(class)`
  - Full name: `sage.functions.other.Function_imag_part`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1216)

- [ ] **Function_limit** `(class)`
  - Full name: `sage.functions.other.Function_limit`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1900)

- [ ] **Function_prod** `(class)`
  - Full name: `sage.functions.other.Function_prod`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1835)

- [ ] **Function_real_nth_root** `(class)`
  - Full name: `sage.functions.other.Function_real_nth_root`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L789)

- [ ] **Function_real_part** `(class)`
  - Full name: `sage.functions.other.Function_real_part`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1119)

- [ ] **Function_sqrt** `(class)`
  - Full name: `sage.functions.other.Function_sqrt`

- [ ] **Function_sum** `(class)`
  - Full name: `sage.functions.other.Function_sum`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L1778)

- [ ] **other** `(module)`
  - Full name: `sage.functions.other`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/other.py#L641)

### sage.functions.trig (10 features)

- [ ] **Function_arccot** `(class)`
  - Full name: `sage.functions.trig.Function_arccot`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L732)

- [ ] **Function_arccsc** `(class)`
  - Full name: `sage.functions.trig.Function_arccsc`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L797)

- [ ] **Function_arcsec** `(class)`
  - Full name: `sage.functions.trig.Function_arcsec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L858)

- [ ] **Function_arctan** `(class)`
  - Full name: `sage.functions.trig.Function_arctan`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L658)

- [ ] **Function_arctan2** `(class)`
  - Full name: `sage.functions.trig.Function_arctan2`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L921)

- [ ] **Function_cot** `(class)`
  - Full name: `sage.functions.trig.Function_cot`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L272)

- [ ] **Function_csc** `(class)`
  - Full name: `sage.functions.trig.Function_csc`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L447)

- [ ] **Function_sec** `(class)`
  - Full name: `sage.functions.trig.Function_sec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L373)

- [ ] **Function_tan** `(class)`
  - Full name: `sage.functions.trig.Function_tan`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L206)

- [ ] **trig** `(module)`
  - Full name: `sage.functions.trig`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/functions/trig.py#L589)

### sage.libs.flint.arith_sage (1 features)

- [ ] **arith_sage** `(module)`
  - Full name: `sage.libs.flint.arith_sage`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/libs/flint/arith_sage.pyx#L28)

### sage.rings.polynomial.convolution (2 features)

- [ ] **convolution** `(module)`
  - Full name: `sage.rings.polynomial.convolution`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/convolution.py#L57)

- [ ] **convolution** `(function)`
  - Full name: `sage.rings.polynomial.convolution.convolution`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/convolution.py#L57)

### sage.rings.polynomial.cyclotomic (4 features)

- [ ] **bateman_bound** `(function)`
  - Full name: `sage.rings.polynomial.cyclotomic.bateman_bound`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L387)

- [ ] **cyclotomic** `(module)`
  - Full name: `sage.rings.polynomial.cyclotomic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L387)

- [ ] **cyclotomic_coeffs** `(function)`
  - Full name: `sage.rings.polynomial.cyclotomic.cyclotomic_coeffs`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L44)

- [ ] **cyclotomic_value** `(function)`
  - Full name: `sage.rings.polynomial.cyclotomic.cyclotomic_value`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/cyclotomic.pyx#L201)

### sage.rings.polynomial.flatten (5 features)

- [ ] **FlatteningMorphism** `(class)`
  - Full name: `sage.rings.polynomial.flatten.FlatteningMorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L48)

- [ ] **FractionSpecializationMorphism** `(class)`
  - Full name: `sage.rings.polynomial.flatten.FractionSpecializationMorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L663)

- [ ] **SpecializationMorphism** `(class)`
  - Full name: `sage.rings.polynomial.flatten.SpecializationMorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L417)

- [ ] **UnflatteningMorphism** `(class)`
  - Full name: `sage.rings.polynomial.flatten.UnflatteningMorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L286)

- [ ] **flatten** `(module)`
  - Full name: `sage.rings.polynomial.flatten`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/flatten.py#L48)

### sage.rings.polynomial.groebner_fan (12 features)

- [ ] **GroebnerFan** `(class)`
  - Full name: `sage.rings.polynomial.groebner_fan.GroebnerFan`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L772)

- [ ] **InitialForm** `(class)`
  - Full name: `sage.rings.polynomial.groebner_fan.InitialForm`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L534)

- [ ] **PolyhedralCone** `(class)`
  - Full name: `sage.rings.polynomial.groebner_fan.PolyhedralCone`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L157)

- [ ] **PolyhedralFan** `(class)`
  - Full name: `sage.rings.polynomial.groebner_fan.PolyhedralFan`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L284)

- [ ] **ReducedGroebnerBasis** `(class)`
  - Full name: `sage.rings.polynomial.groebner_fan.ReducedGroebnerBasis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L1762)

- [ ] **TropicalPrevariety** `(class)`
  - Full name: `sage.rings.polynomial.groebner_fan.TropicalPrevariety`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L645)

- [ ] **groebner_fan** `(module)`
  - Full name: `sage.rings.polynomial.groebner_fan`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L772)

- [ ] **ideal_to_gfan_format** `(function)`
  - Full name: `sage.rings.polynomial.groebner_fan.ideal_to_gfan_format`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L744)

- [ ] **max_degree** `(function)`
  - Full name: `sage.rings.polynomial.groebner_fan.max_degree`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L104)

- [ ] **prefix_check** `(function)`
  - Full name: `sage.rings.polynomial.groebner_fan.prefix_check`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L83)

- [ ] **ring_to_gfan_format** `(function)`
  - Full name: `sage.rings.polynomial.groebner_fan.ring_to_gfan_format`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L721)

- [ ] **verts_for_normal** `(function)`
  - Full name: `sage.rings.polynomial.groebner_fan.verts_for_normal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/groebner_fan.py#L623)

### sage.rings.polynomial.hilbert (4 features)

- [ ] **Node** `(class)`
  - Full name: `sage.rings.polynomial.hilbert.Node`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L36)

- [ ] **first_hilbert_series** `(function)`
  - Full name: `sage.rings.polynomial.hilbert.first_hilbert_series`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L425)

- [ ] **hilbert** `(module)`
  - Full name: `sage.rings.polynomial.hilbert`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L36)

- [ ] **hilbert_poincare_series** `(function)`
  - Full name: `sage.rings.polynomial.hilbert.hilbert_poincare_series`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/hilbert.pyx#L556)

### sage.rings.polynomial.ideal (2 features)

- [ ] **Ideal_1poly_field** `(class)`
  - Full name: `sage.rings.polynomial.ideal.Ideal_1poly_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ideal.py#L23)

- [ ] **ideal** `(module)`
  - Full name: `sage.rings.polynomial.ideal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ideal.py#L23)

### sage.rings.polynomial.infinite_polynomial_element (4 features)

- [ ] **InfinitePolynomial** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L114)

- [ ] **InfinitePolynomial_dense** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L1582)

- [ ] **InfinitePolynomial_sparse** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_element.InfinitePolynomial_sparse`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L1286)

- [ ] **infinite_polynomial_element** `(module)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_element.py#L114)

### sage.rings.polynomial.infinite_polynomial_ring (6 features)

- [ ] **GenDictWithBasering** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_ring.GenDictWithBasering`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L497)

- [ ] **InfiniteGenDict** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_ring.InfiniteGenDict`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L390)

- [ ] **InfinitePolynomialGen** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialGen`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L1376)

- [ ] **InfinitePolynomialRing_dense** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRing_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L1537)

- [ ] **InfinitePolynomialRing_sparse** `(class)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_ring.InfinitePolynomialRing_sparse`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L621)

- [ ] **infinite_polynomial_ring** `(module)`
  - Full name: `sage.rings.polynomial.infinite_polynomial_ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/infinite_polynomial_ring.py#L497)

### sage.rings.polynomial.integer_valued_polynomials (8 features)

- [ ] **Bases** `(class)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials.Bases`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L119)

- [ ] **Binomial** `(class)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials.Binomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L898)

- [ ] **Element** `(class)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials.Element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L1188)

- [ ] **ElementMethods** `(class)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials.ElementMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L282)

- [ ] **IntegerValuedPolynomialRing** `(class)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials.IntegerValuedPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L34)

- [ ] **ParentMethods** `(class)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials.ParentMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L144)

- [ ] **Shifted** `(class)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials.Shifted`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L393)

- [ ] **integer_valued_polynomials** `(module)`
  - Full name: `sage.rings.polynomial.integer_valued_polynomials`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/integer_valued_polynomials.py#L34)

### sage.rings.polynomial.laurent_polynomial (3 features)

- [ ] **LaurentPolynomial** `(class)`
  - Full name: `sage.rings.polynomial.laurent_polynomial.LaurentPolynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial.pyx#L21)

- [ ] **LaurentPolynomial_univariate** `(class)`
  - Full name: `sage.rings.polynomial.laurent_polynomial.LaurentPolynomial_univariate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial.pyx#L303)

- [ ] **laurent_polynomial** `(module)`
  - Full name: `sage.rings.polynomial.laurent_polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial.pyx#L21)

### sage.rings.polynomial.laurent_polynomial_ring (6 features)

- [ ] **LaurentPolynomialRing** `(function)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L82)

- [ ] **LaurentPolynomialRing_mpair** `(class)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing_mpair`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L590)

- [ ] **LaurentPolynomialRing_univariate** `(class)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring.LaurentPolynomialRing_univariate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L432)

- [ ] **from_fraction_field** `(function)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring.from_fraction_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L399)

- [ ] **is_LaurentPolynomialRing** `(function)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring.is_LaurentPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L52)

- [ ] **laurent_polynomial_ring** `(module)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring.py#L82)

### sage.rings.polynomial.laurent_polynomial_ring_base (2 features)

- [ ] **LaurentPolynomialRing_generic** `(class)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring_base.LaurentPolynomialRing_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring_base.py#L34)

- [ ] **laurent_polynomial_ring_base** `(module)`
  - Full name: `sage.rings.polynomial.laurent_polynomial_ring_base`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/laurent_polynomial_ring_base.py#L34)

### sage.rings.polynomial.msolve (3 features)

- [ ] **groebner_basis_degrevlex** `(function)`
  - Full name: `sage.rings.polynomial.msolve.groebner_basis_degrevlex`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/msolve.py#L68)

- [ ] **msolve** `(module)`
  - Full name: `sage.rings.polynomial.msolve`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/msolve.py#L68)

- [ ] **variety** `(function)`
  - Full name: `sage.rings.polynomial.msolve.variety`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/msolve.py#L117)

### sage.rings.polynomial.multi_polynomial (4 features)

- [ ] **MPolynomial** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial.MPolynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L369)

- [ ] **MPolynomial_libsingular** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial.MPolynomial_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L2993)

- [ ] **is_MPolynomial** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial.is_MPolynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L24)

- [ ] **multi_polynomial** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial.pyx#L369)

### sage.rings.polynomial.multi_polynomial_element (4 features)

- [ ] **MPolynomial_element** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_element.MPolynomial_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L77)

- [ ] **MPolynomial_polydict** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_element.MPolynomial_polydict`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L419)

- [ ] **degree_lowest_rational_function** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_element.degree_lowest_rational_function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L2517)

- [ ] **multi_polynomial_element** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_element.py#L77)

### sage.rings.polynomial.multi_polynomial_ideal (12 features)

- [ ] **MPolynomialIdeal** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3876)

- [ ] **MPolynomialIdeal_macaulay2_repr** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_macaulay2_repr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3382)

- [ ] **MPolynomialIdeal_magma_repr** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_magma_repr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L349)

- [ ] **MPolynomialIdeal_quotient** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_quotient`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L5646)

- [ ] **MPolynomialIdeal_singular_base_repr** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_singular_base_repr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L454)

- [ ] **MPolynomialIdeal_singular_repr** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal_singular_repr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L617)

- [ ] **NCPolynomialIdeal** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.NCPolynomialIdeal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3495)

- [ ] **RequireField** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.RequireField`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L273)

- [ ] **basis** `(attribute)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.basis`

- [ ] **is_MPolynomialIdeal** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.is_MPolynomialIdeal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L310)

- [ ] **multi_polynomial_ideal** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L3876)

- [ ] **require_field** `(attribute)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal.require_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal.py#L273)

### sage.rings.polynomial.multi_polynomial_ideal_libsingular (5 features)

- [ ] **interred_libsingular** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal_libsingular.interred_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L278)

- [ ] **kbase_libsingular** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal_libsingular.kbase_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L150)

- [ ] **multi_polynomial_ideal_libsingular** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L278)

- [ ] **slimgb_libsingular** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal_libsingular.slimgb_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L238)

- [ ] **std_libsingular** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ideal_libsingular.std_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ideal_libsingular.pyx#L207)

### sage.rings.polynomial.multi_polynomial_libsingular (5 features)

- [ ] **MPolynomialRing_libsingular** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_libsingular.MPolynomialRing_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L264)

- [ ] **MPolynomial_libsingular** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L1896)

- [ ] **multi_polynomial_libsingular** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L264)

- [ ] **unpickle_MPolynomialRing_libsingular** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_libsingular.unpickle_MPolynomialRing_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L1880)

- [ ] **unpickle_MPolynomial_libsingular** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_libsingular.unpickle_MPolynomial_libsingular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx#L6179)

### sage.rings.polynomial.multi_polynomial_ring (4 features)

- [ ] **MPolynomialRing_macaulay2_repr** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_macaulay2_repr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L79)

- [ ] **MPolynomialRing_polydict** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_polydict`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L97)

- [ ] **MPolynomialRing_polydict_domain** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring.MPolynomialRing_polydict_domain`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L937)

- [ ] **multi_polynomial_ring** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring.py#L79)

### sage.rings.polynomial.multi_polynomial_ring_base (6 features)

- [ ] **BooleanPolynomialRing_base** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring_base.BooleanPolynomialRing_base`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1805)

- [ ] **MPolynomialRing_base** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring_base.MPolynomialRing_base`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L40)

- [ ] **is_MPolynomialRing** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring_base.is_MPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L32)

- [ ] **multi_polynomial_ring_base** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring_base`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1805)

- [ ] **unpickle_MPolynomialRing_generic** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring_base.unpickle_MPolynomialRing_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1835)

- [ ] **unpickle_MPolynomialRing_generic_v1** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_ring_base.unpickle_MPolynomialRing_generic_v1`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_ring_base.pyx#L1830)

### sage.rings.polynomial.multi_polynomial_sequence (6 features)

- [ ] **PolynomialSequence** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L212)

- [ ] **PolynomialSequence_generic** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L397)

- [ ] **PolynomialSequence_gf2** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_gf2`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L1311)

- [ ] **PolynomialSequence_gf2e** `(class)`
  - Full name: `sage.rings.polynomial.multi_polynomial_sequence.PolynomialSequence_gf2e`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L1826)

- [ ] **is_PolynomialSequence** `(function)`
  - Full name: `sage.rings.polynomial.multi_polynomial_sequence.is_PolynomialSequence`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L186)

- [ ] **multi_polynomial_sequence** `(module)`
  - Full name: `sage.rings.polynomial.multi_polynomial_sequence`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_sequence.py#L212)

### sage.rings.polynomial.omega (4 features)

- [ ] **MacMahonOmega** `(function)`
  - Full name: `sage.rings.polynomial.omega.MacMahonOmega`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L56)

- [ ] **Omega_ge** `(function)`
  - Full name: `sage.rings.polynomial.omega.Omega_ge`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L470)

- [ ] **homogeneous_symmetric_function** `(function)`
  - Full name: `sage.rings.polynomial.omega.homogeneous_symmetric_function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L949)

- [ ] **omega** `(module)`
  - Full name: `sage.rings.polynomial.omega`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/omega.py#L56)

### sage.rings.polynomial.ore_function_element (5 features)

- [ ] **ConstantOreFunctionSection** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_element.ConstantOreFunctionSection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L668)

- [ ] **OreFunction** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_element.OreFunction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L28)

- [ ] **OreFunctionBaseringInjection** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_element.OreFunctionBaseringInjection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L718)

- [ ] **OreFunction_with_large_center** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_element.OreFunction_with_large_center`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L815)

- [ ] **ore_function_element** `(module)`
  - Full name: `sage.rings.polynomial.ore_function_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_element.py#L668)

### sage.rings.polynomial.ore_function_field (5 features)

- [ ] **OreFunctionCenterInjection** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_field.OreFunctionCenterInjection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L773)

- [ ] **OreFunctionField** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_field.OreFunctionField`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L186)

- [ ] **OreFunctionField_with_large_center** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_field.OreFunctionField_with_large_center`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L876)

- [ ] **SectionOreFunctionCenterInjection** `(class)`
  - Full name: `sage.rings.polynomial.ore_function_field.SectionOreFunctionCenterInjection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L694)

- [ ] **ore_function_field** `(module)`
  - Full name: `sage.rings.polynomial.ore_function_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_function_field.py#L773)

### sage.rings.polynomial.ore_polynomial_element (5 features)

- [ ] **ConstantOrePolynomialSection** `(class)`
  - Full name: `sage.rings.polynomial.ore_polynomial_element.ConstantOrePolynomialSection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L2968)

- [ ] **OrePolynomial** `(class)`
  - Full name: `sage.rings.polynomial.ore_polynomial_element.OrePolynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L50)

- [ ] **OrePolynomialBaseringInjection** `(class)`
  - Full name: `sage.rings.polynomial.ore_polynomial_element.OrePolynomialBaseringInjection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L3021)

- [ ] **OrePolynomial_generic_dense** `(class)`
  - Full name: `sage.rings.polynomial.ore_polynomial_element.OrePolynomial_generic_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L2212)

- [ ] **ore_polynomial_element** `(module)`
  - Full name: `sage.rings.polynomial.ore_polynomial_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_element.pyx#L2968)

### sage.rings.polynomial.ore_polynomial_ring (2 features)

- [ ] **OrePolynomialRing** `(class)`
  - Full name: `sage.rings.polynomial.ore_polynomial_ring.OrePolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_ring.py#L73)

- [ ] **ore_polynomial_ring** `(module)`
  - Full name: `sage.rings.polynomial.ore_polynomial_ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/ore_polynomial_ring.py#L73)

### sage.rings.polynomial.padics.polynomial_padic (2 features)

- [ ] **Polynomial_padic** `(class)`
  - Full name: `sage.rings.polynomial.padics.polynomial_padic.Polynomial_padic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic.py#L30)

- [ ] **polynomial_padic** `(module)`
  - Full name: `sage.rings.polynomial.padics.polynomial_padic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic.py#L30)

### sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense (3 features)

- [ ] **Polynomial_padic_capped_relative_dense** `(class)`
  - Full name: `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense.Polynomial_padic_capped_relative_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_capped_relative_dense.py#L39)

- [ ] **make_padic_poly** `(function)`
  - Full name: `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense.make_padic_poly`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_capped_relative_dense.py#L1319)

- [ ] **polynomial_padic_capped_relative_dense** `(module)`
  - Full name: `sage.rings.polynomial.padics.polynomial_padic_capped_relative_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_capped_relative_dense.py#L39)

### sage.rings.polynomial.padics.polynomial_padic_flat (2 features)

- [ ] **Polynomial_padic_flat** `(class)`
  - Full name: `sage.rings.polynomial.padics.polynomial_padic_flat.Polynomial_padic_flat`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_flat.py#L18)

- [ ] **polynomial_padic_flat** `(module)`
  - Full name: `sage.rings.polynomial.padics.polynomial_padic_flat`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/padics/polynomial_padic_flat.py#L18)

### sage.rings.polynomial.pbori.pbori (53 features)

- [ ] **BooleConstant** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleConstant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7766)

- [ ] **BooleSet** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleSet`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5237)

- [ ] **BooleSetIterator** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleSetIterator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5863)

- [ ] **BooleanMonomial** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanMonomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L2211)

- [ ] **BooleanMonomialIterator** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanMonomialIterator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L2874)

- [ ] **BooleanMonomialMonoid** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanMonomialMonoid`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L1848)

- [ ] **BooleanMonomialVariableIterator** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanMonomialVariableIterator`

- [ ] **BooleanMulAction** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanMulAction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L78)

- [ ] **BooleanPolynomial** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanPolynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L2922)

- [ ] **BooleanPolynomialEntry** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanPolynomialEntry`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6425)

- [ ] **BooleanPolynomialIdeal** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanPolynomialIdeal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4801)

- [ ] **BooleanPolynomialIterator** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanPolynomialIterator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4750)

- [ ] **BooleanPolynomialRing** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L254)

- [ ] **BooleanPolynomialVector** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5960)

- [ ] **BooleanPolynomialVectorIterator** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.BooleanPolynomialVectorIterator`

- [ ] **CCuddNavigator** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.CCuddNavigator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L5928)

- [ ] **FGLMStrategy** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.FGLMStrategy`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6430)

- [ ] **GroebnerStrategy** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.GroebnerStrategy`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6502)

- [ ] **MonomialConstruct** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.MonomialConstruct`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4687)

- [ ] **PolynomialConstruct** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.PolynomialConstruct`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4636)

- [ ] **ReductionStrategy** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.ReductionStrategy`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L6153)

- [ ] **TermOrder_from_pb_order** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.TermOrder_from_pb_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7891)

- [ ] **VariableBlock** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.VariableBlock`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7014)

- [ ] **VariableConstruct** `(class)`
  - Full name: `sage.rings.polynomial.pbori.pbori.VariableConstruct`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L4725)

- [ ] **add_up_polynomials** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.add_up_polynomials`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7030)

- [ ] **contained_vars** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.contained_vars`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7277)

- [ ] **gauss_on_polys** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.gauss_on_polys`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7603)

- [ ] **get_var_mapping** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.get_var_mapping`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L1794)

- [ ] **if_then_else** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.if_then_else`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7412)

- [ ] **interpolate** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.interpolate`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7144)

- [ ] **interpolate_smallest_lex** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.interpolate_smallest_lex`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7199)

- [ ] **ll_red_nf_noredsb** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.ll_red_nf_noredsb`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7340)

- [ ] **ll_red_nf_noredsb_single_recursive_call** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.ll_red_nf_noredsb_single_recursive_call`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7367)

- [ ] **ll_red_nf_redsb** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.ll_red_nf_redsb`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7297)

- [ ] **map_every_x_to_x_plus_one** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.map_every_x_to_x_plus_one`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7080)

- [ ] **mod_mon_set** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.mod_mon_set`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7400)

- [ ] **mod_var_set** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.mod_var_set`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7282)

- [ ] **mult_fact_sim_C** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.mult_fact_sim_C`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7286)

- [ ] **nf3** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.nf3`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7051)

- [ ] **p** `(attribute)`
  - Full name: `sage.rings.polynomial.pbori.pbori.p`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L69)

- [ ] **parallel_reduce** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.parallel_reduce`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7406)

- [ ] **pbori** `(module)`
  - Full name: `sage.rings.polynomial.pbori.pbori`

- [ ] **random_set** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.random_set`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7695)

- [ ] **recursively_insert** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.recursively_insert`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7290)

- [ ] **red_tail** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.red_tail`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7056)

- [ ] **reduction_strategy** `(attribute)`
  - Full name: `sage.rings.polynomial.pbori.pbori.reduction_strategy`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L84)

- [ ] **set_random_seed** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.set_random_seed`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7670)

- [ ] **substitute_variables** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.substitute_variables`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7631)

- [ ] **top_index** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.top_index`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7495)

- [ ] **unpickle_BooleanPolynomial** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7720)

- [ ] **unpickle_BooleanPolynomial0** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomial0`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7735)

- [ ] **unpickle_BooleanPolynomialRing** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.unpickle_BooleanPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7751)

- [ ] **zeros** `(function)`
  - Full name: `sage.rings.polynomial.pbori.pbori.zeros`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/pbori/pbori.pyx#L7100)

### sage.rings.polynomial.plural (10 features)

- [ ] **ExteriorAlgebra** `(function)`
  - Full name: `sage.rings.polynomial.plural.ExteriorAlgebra`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3115)

- [ ] **ExteriorAlgebra_plural** `(class)`
  - Full name: `sage.rings.polynomial.plural.ExteriorAlgebra_plural`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L246)

- [ ] **NCPolynomialRing_plural** `(class)`
  - Full name: `sage.rings.polynomial.plural.NCPolynomialRing_plural`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L246)

- [ ] **NCPolynomial_plural** `(class)`
  - Full name: `sage.rings.polynomial.plural.NCPolynomial_plural`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L1422)

- [ ] **SCA** `(function)`
  - Full name: `sage.rings.polynomial.plural.SCA`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3061)

- [ ] **new_CRing** `(function)`
  - Full name: `sage.rings.polynomial.plural.new_CRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L2865)

- [ ] **new_NRing** `(function)`
  - Full name: `sage.rings.polynomial.plural.new_NRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L2936)

- [ ] **new_Ring** `(function)`
  - Full name: `sage.rings.polynomial.plural.new_Ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3005)

- [ ] **plural** `(module)`
  - Full name: `sage.rings.polynomial.plural`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L3115)

- [ ] **unpickle_NCPolynomial_plural** `(function)`
  - Full name: `sage.rings.polynomial.plural.unpickle_NCPolynomial_plural`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/plural.pyx#L1383)

### sage.rings.polynomial.polydict (7 features)

- [ ] **ETuple** `(class)`
  - Full name: `sage.rings.polynomial.polydict.ETuple`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L1425)

- [ ] **PolyDict** `(class)`
  - Full name: `sage.rings.polynomial.polydict.PolyDict`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L96)

- [ ] **gen_index** `(function)`
  - Full name: `sage.rings.polynomial.polydict.gen_index`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L47)

- [ ] **make_ETuple** `(function)`
  - Full name: `sage.rings.polynomial.polydict.make_ETuple`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L2696)

- [ ] **make_PolyDict** `(function)`
  - Full name: `sage.rings.polynomial.polydict.make_PolyDict`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L2689)

- [ ] **monomial_exponent** `(function)`
  - Full name: `sage.rings.polynomial.polydict.monomial_exponent`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L72)

- [ ] **polydict** `(module)`
  - Full name: `sage.rings.polynomial.polydict`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polydict.pyx#L1425)

### sage.rings.polynomial.polynomial_compiled (14 features)

- [ ] **CompiledPolynomialFunction** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.CompiledPolynomialFunction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L21)

- [ ] **abc_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.abc_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L495)

- [ ] **add_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.add_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L455)

- [ ] **binary_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.binary_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L455)

- [ ] **coeff_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.coeff_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L404)

- [ ] **dummy_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.dummy_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L371)

- [ ] **generic_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.generic_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L354)

- [ ] **mul_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.mul_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L455)

- [ ] **polynomial_compiled** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_compiled`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L21)

- [ ] **pow_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.pow_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L441)

- [ ] **sqr_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.sqr_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L417)

- [ ] **unary_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.unary_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L417)

- [ ] **univar_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.univar_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L394)

- [ ] **var_pd** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_compiled.var_pd`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_compiled.pyx#L383)

### sage.rings.polynomial.polynomial_element (10 features)

- [ ] **ConstantPolynomialSection** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element.ConstantPolynomialSection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12874)

- [ ] **Polynomial** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element.Polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L187)

- [ ] **PolynomialBaseringInjection** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element.PolynomialBaseringInjection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12928)

- [ ] **Polynomial_generic_dense** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element.Polynomial_generic_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L11952)

- [ ] **Polynomial_generic_dense_inexact** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element.Polynomial_generic_dense_inexact`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12736)

- [ ] **generic_power_trunc** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_element.generic_power_trunc`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12632)

- [ ] **is_Polynomial** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_element.is_Polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L135)

- [ ] **make_generic_polynomial** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_element.make_generic_polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12588)

- [ ] **polynomial_element** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L12874)

- [ ] **polynomial_is_variable** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_element.polynomial_is_variable`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element.pyx#L13124)

### sage.rings.polynomial.polynomial_element_generic (15 features)

- [ ] **Polynomial_generic_cdv** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdv`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1155)

- [ ] **Polynomial_generic_cdvf** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdvf`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1610)

- [ ] **Polynomial_generic_cdvr** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_cdvr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1598)

- [ ] **Polynomial_generic_dense_cdv** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdv`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1590)

- [ ] **Polynomial_generic_dense_cdvf** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdvf`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1614)

- [ ] **Polynomial_generic_dense_cdvr** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_cdvr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1602)

- [ ] **Polynomial_generic_dense_field** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_dense_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1146)

- [ ] **Polynomial_generic_domain** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_domain`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1053)

- [ ] **Polynomial_generic_field** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1087)

- [ ] **Polynomial_generic_sparse** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L52)

- [ ] **Polynomial_generic_sparse_cdv** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdv`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1594)

- [ ] **Polynomial_generic_sparse_cdvf** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdvf`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1618)

- [ ] **Polynomial_generic_sparse_cdvr** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_cdvr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1606)

- [ ] **Polynomial_generic_sparse_field** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic.Polynomial_generic_sparse_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1131)

- [ ] **polynomial_element_generic** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_element_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_element_generic.py#L1155)

### sage.rings.polynomial.polynomial_fateman (1 features)

- [ ] **polynomial_fateman** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_fateman`

### sage.rings.polynomial.polynomial_gf2x (7 features)

- [ ] **GF2X_BuildIrred_list** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_gf2x.GF2X_BuildIrred_list`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L295)

- [ ] **GF2X_BuildRandomIrred_list** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_gf2x.GF2X_BuildRandomIrred_list`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L340)

- [ ] **GF2X_BuildSparseIrred_list** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_gf2x.GF2X_BuildSparseIrred_list`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L319)

- [ ] **Polynomial_GF2X** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_gf2x.Polynomial_GF2X`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L36)

- [ ] **Polynomial_template** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_gf2x.Polynomial_template`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L60)

- [ ] **make_element** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_gf2x.make_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L28)

- [ ] **polynomial_gf2x** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_gf2x`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_gf2x.pyx#L295)

### sage.rings.polynomial.polynomial_integer_dense_flint (2 features)

- [ ] **Polynomial_integer_dense_flint** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_integer_dense_flint.Polynomial_integer_dense_flint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_flint.pyx#L81)

- [ ] **polynomial_integer_dense_flint** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_integer_dense_flint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_flint.pyx#L81)

### sage.rings.polynomial.polynomial_integer_dense_ntl (2 features)

- [ ] **Polynomial_integer_dense_ntl** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_integer_dense_ntl.Polynomial_integer_dense_ntl`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_ntl.pyx#L74)

- [ ] **polynomial_integer_dense_ntl** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_integer_dense_ntl`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_integer_dense_ntl.pyx#L74)

### sage.rings.polynomial.polynomial_modn_dense_ntl (6 features)

- [ ] **Polynomial_dense_mod_n** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_mod_n`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L66)

- [ ] **Polynomial_dense_mod_p** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_mod_p`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L1868)

- [ ] **Polynomial_dense_modn_ntl_ZZ** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_modn_ntl_ZZ`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L1296)

- [ ] **Polynomial_dense_modn_ntl_zz** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_modn_dense_ntl.Polynomial_dense_modn_ntl_zz`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L671)

- [ ] **make_element** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_modn_dense_ntl.make_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L60)

- [ ] **polynomial_modn_dense_ntl** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_modn_dense_ntl`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_modn_dense_ntl.pyx#L66)

### sage.rings.polynomial.polynomial_number_field (3 features)

- [ ] **Polynomial_absolute_number_field_dense** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_number_field.Polynomial_absolute_number_field_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_number_field.pyx#L81)

- [ ] **Polynomial_relative_number_field_dense** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_number_field.Polynomial_relative_number_field_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_number_field.pyx#L215)

- [ ] **polynomial_number_field** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_number_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_number_field.pyx#L81)

### sage.rings.polynomial.polynomial_quotient_ring (6 features)

- [ ] **PolynomialQuotientRing_coercion** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_coercion`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L2170)

- [ ] **PolynomialQuotientRing_domain** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_domain`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L2280)

- [ ] **PolynomialQuotientRing_field** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L2417)

- [ ] **PolynomialQuotientRing_generic** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring.PolynomialQuotientRing_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L274)

- [ ] **is_PolynomialQuotientRing** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring.is_PolynomialQuotientRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L266)

- [ ] **polynomial_quotient_ring** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring.py#L65)

### sage.rings.polynomial.polynomial_quotient_ring_element (2 features)

- [ ] **PolynomialQuotientRingElement** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring_element.PolynomialQuotientRingElement`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring_element.py#L93)

- [ ] **polynomial_quotient_ring_element** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_quotient_ring_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_quotient_ring_element.py#L93)

### sage.rings.polynomial.polynomial_rational_flint (2 features)

- [ ] **Polynomial_rational_flint** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_rational_flint.Polynomial_rational_flint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_rational_flint.pyx#L83)

- [ ] **polynomial_rational_flint** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_rational_flint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_rational_flint.pyx#L83)

### sage.rings.polynomial.polynomial_real_mpfr_dense (3 features)

- [ ] **PolynomialRealDense** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_real_mpfr_dense.PolynomialRealDense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_real_mpfr_dense.pyx#L49)

- [ ] **make_PolynomialRealDense** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_real_mpfr_dense.make_PolynomialRealDense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_real_mpfr_dense.pyx#L779)

- [ ] **polynomial_real_mpfr_dense** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_real_mpfr_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_real_mpfr_dense.pyx#L49)

### sage.rings.polynomial.polynomial_ring (19 features)

- [ ] **PolynomialRing_cdvf** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_cdvf`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3056)

- [ ] **PolynomialRing_cdvr** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_cdvr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3024)

- [ ] **PolynomialRing_commutative** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_commutative`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L1800)

- [ ] **PolynomialRing_dense_finite_field** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_finite_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L2601)

- [ ] **PolynomialRing_dense_mod_n** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_mod_n`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3228)

- [ ] **PolynomialRing_dense_mod_p** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_mod_p`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3404)

- [ ] **PolynomialRing_dense_padic_field_capped_relative** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_field_capped_relative`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3207)

- [ ] **PolynomialRing_dense_padic_field_generic** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_field_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3116)

- [ ] **PolynomialRing_dense_padic_ring_capped_absolute** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_capped_absolute`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3166)

- [ ] **PolynomialRing_dense_padic_ring_capped_relative** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_capped_relative`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3145)

- [ ] **PolynomialRing_dense_padic_ring_fixed_mod** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_fixed_mod`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3186)

- [ ] **PolynomialRing_dense_padic_ring_generic** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_dense_padic_ring_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3087)

- [ ] **PolynomialRing_field** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L2159)

- [ ] **PolynomialRing_generic** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_generic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L240)

- [ ] **PolynomialRing_integral_domain** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring.PolynomialRing_integral_domain`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L1929)

- [ ] **is_PolynomialRing** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_ring.is_PolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L184)

- [ ] **polygen** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_ring.polygen`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3700)

- [ ] **polygens** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_ring.polygens`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3742)

- [ ] **polynomial_ring** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring.py#L3056)

### sage.rings.polynomial.polynomial_ring_constructor (5 features)

- [ ] **BooleanPolynomialRing_constructor** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_ring_constructor.BooleanPolynomialRing_constructor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L973)

- [ ] **PolynomialRing** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_ring_constructor.PolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L60)

- [ ] **polynomial_default_category** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_ring_constructor.polynomial_default_category`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L909)

- [ ] **polynomial_ring_constructor** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_ring_constructor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L973)

- [ ] **unpickle_PolynomialRing** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_ring_constructor.unpickle_PolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_constructor.py#L731)

### sage.rings.polynomial.polynomial_ring_homomorphism (2 features)

- [ ] **PolynomialRingHomomorphism_from_base** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_ring_homomorphism.PolynomialRingHomomorphism_from_base`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_homomorphism.pyx#L21)

- [ ] **polynomial_ring_homomorphism** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_ring_homomorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_ring_homomorphism.pyx#L21)

### sage.rings.polynomial.polynomial_singular_interface (4 features)

- [ ] **PolynomialRing_singular_repr** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_singular_interface.PolynomialRing_singular_repr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L172)

- [ ] **Polynomial_singular_repr** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_singular_interface.Polynomial_singular_repr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L451)

- [ ] **can_convert_to_singular** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_singular_interface.can_convert_to_singular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L381)

- [ ] **polynomial_singular_interface** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_singular_interface`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_singular_interface.py#L172)

### sage.rings.polynomial.polynomial_zmod_flint (4 features)

- [ ] **Polynomial_template** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_zmod_flint.Polynomial_template`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L60)

- [ ] **Polynomial_zmod_flint** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_zmod_flint.Polynomial_zmod_flint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L71)

- [ ] **make_element** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_zmod_flint.make_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L28)

- [ ] **polynomial_zmod_flint** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_zmod_flint`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zmod_flint.pyx#L60)

### sage.rings.polynomial.polynomial_zz_pex (4 features)

- [ ] **Polynomial_ZZ_pEX** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_zz_pex.Polynomial_ZZ_pEX`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L68)

- [ ] **Polynomial_template** `(class)`
  - Full name: `sage.rings.polynomial.polynomial_zz_pex.Polynomial_template`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L60)

- [ ] **make_element** `(function)`
  - Full name: `sage.rings.polynomial.polynomial_zz_pex.make_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L28)

- [ ] **polynomial_zz_pex** `(module)`
  - Full name: `sage.rings.polynomial.polynomial_zz_pex`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/polynomial_zz_pex.pyx#L68)

### sage.rings.polynomial.q_integer_valued_polynomials (10 features)

- [ ] **Bases** `(class)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.Bases`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L265)

- [ ] **Binomial** `(class)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.Binomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L986)

- [ ] **Element** `(class)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.Element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L1221)

- [ ] **ElementMethods** `(class)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.ElementMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L447)

- [ ] **ParentMethods** `(class)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.ParentMethods`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L293)

- [ ] **QuantumValuedPolynomialRing** `(class)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.QuantumValuedPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L117)

- [ ] **Shifted** `(class)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.Shifted`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L533)

- [ ] **q_binomial_x** `(function)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.q_binomial_x`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L73)

- [ ] **q_int_x** `(function)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials.q_int_x`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L42)

- [ ] **q_integer_valued_polynomials** `(module)`
  - Full name: `sage.rings.polynomial.q_integer_valued_polynomials`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/q_integer_valued_polynomials.py#L117)

### sage.rings.polynomial.refine_root (2 features)

- [ ] **refine_root** `(module)`
  - Full name: `sage.rings.polynomial.refine_root`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/refine_root.pyx#L28)

- [ ] **refine_root** `(function)`
  - Full name: `sage.rings.polynomial.refine_root.refine_root`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/refine_root.pyx#L28)

### sage.rings.polynomial.skew_polynomial_element (2 features)

- [ ] **SkewPolynomial_generic_dense** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_element.SkewPolynomial_generic_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_element.pyx#L62)

- [ ] **skew_polynomial_element** `(module)`
  - Full name: `sage.rings.polynomial.skew_polynomial_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_element.pyx#L62)

### sage.rings.polynomial.skew_polynomial_finite_field (2 features)

- [ ] **SkewPolynomial_finite_field_dense** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_finite_field.SkewPolynomial_finite_field_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_field.pyx#L28)

- [ ] **skew_polynomial_finite_field** `(module)`
  - Full name: `sage.rings.polynomial.skew_polynomial_finite_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_field.pyx#L28)

### sage.rings.polynomial.skew_polynomial_finite_order (2 features)

- [ ] **SkewPolynomial_finite_order_dense** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_finite_order.SkewPolynomial_finite_order_dense`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_order.pyx#L28)

- [ ] **skew_polynomial_finite_order** `(module)`
  - Full name: `sage.rings.polynomial.skew_polynomial_finite_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_finite_order.pyx#L28)

### sage.rings.polynomial.skew_polynomial_ring (6 features)

- [ ] **SectionSkewPolynomialCenterInjection** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_ring.SectionSkewPolynomialCenterInjection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L369)

- [ ] **SkewPolynomialCenterInjection** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialCenterInjection`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L446)

- [ ] **SkewPolynomialRing** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L218)

- [ ] **SkewPolynomialRing_finite_field** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing_finite_field`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L764)

- [ ] **SkewPolynomialRing_finite_order** `(class)`
  - Full name: `sage.rings.polynomial.skew_polynomial_ring.SkewPolynomialRing_finite_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L561)

- [ ] **skew_polynomial_ring** `(module)`
  - Full name: `sage.rings.polynomial.skew_polynomial_ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/skew_polynomial_ring.py#L369)

### sage.rings.polynomial.symmetric_ideal (2 features)

- [ ] **SymmetricIdeal** `(class)`
  - Full name: `sage.rings.polynomial.symmetric_ideal.SymmetricIdeal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_ideal.py#L68)

- [ ] **symmetric_ideal** `(module)`
  - Full name: `sage.rings.polynomial.symmetric_ideal`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_ideal.py#L68)

### sage.rings.polynomial.symmetric_reduction (2 features)

- [ ] **SymmetricReductionStrategy** `(class)`
  - Full name: `sage.rings.polynomial.symmetric_reduction.SymmetricReductionStrategy`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_reduction.pyx#L126)

- [ ] **symmetric_reduction** `(module)`
  - Full name: `sage.rings.polynomial.symmetric_reduction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/symmetric_reduction.pyx#L126)

### sage.rings.polynomial.term_order (5 features)

- [ ] **TermOrder** `(class)`
  - Full name: `sage.rings.polynomial.term_order.TermOrder`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/term_order.py#L543)

- [ ] **greater_tuple** `(attribute)`
  - Full name: `sage.rings.polynomial.term_order.greater_tuple`

- [ ] **sortkey** `(attribute)`
  - Full name: `sage.rings.polynomial.term_order.sortkey`

- [ ] **term_order** `(module)`
  - Full name: `sage.rings.polynomial.term_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/term_order.py#L543)

- [ ] **termorder_from_singular** `(function)`
  - Full name: `sage.rings.polynomial.term_order.termorder_from_singular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/term_order.py#L2158)

### sage.rings.polynomial.toy_buchberger (10 features)

- [ ] **LCM** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.LCM`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L149)

- [ ] **LM** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.LM`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L150)

- [ ] **LT** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.LT`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L151)

- [ ] **buchberger** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.buchberger`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L175)

- [ ] **buchberger_improved** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.buchberger_improved`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L231)

- [ ] **inter_reduction** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.inter_reduction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L398)

- [ ] **select** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.select`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L375)

- [ ] **spol** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.spol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L154)

- [ ] **toy_buchberger** `(module)`
  - Full name: `sage.rings.polynomial.toy_buchberger`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L149)

- [ ] **update** `(function)`
  - Full name: `sage.rings.polynomial.toy_buchberger.update`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_buchberger.py#L292)

### sage.rings.polynomial.toy_d_basis (8 features)

- [ ] **LC** `(function)`
  - Full name: `sage.rings.polynomial.toy_d_basis.LC`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L196)

- [ ] **LM** `(function)`
  - Full name: `sage.rings.polynomial.toy_d_basis.LM`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L192)

- [ ] **d_basis** `(function)`
  - Full name: `sage.rings.polynomial.toy_d_basis.d_basis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L200)

- [ ] **gpol** `(function)`
  - Full name: `sage.rings.polynomial.toy_d_basis.gpol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L160)

- [ ] **select** `(function)`
  - Full name: `sage.rings.polynomial.toy_d_basis.select`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L273)

- [ ] **spol** `(function)`
  - Full name: `sage.rings.polynomial.toy_d_basis.spol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L127)

- [ ] **toy_d_basis** `(module)`
  - Full name: `sage.rings.polynomial.toy_d_basis`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L196)

- [ ] **update** `(function)`
  - Full name: `sage.rings.polynomial.toy_d_basis.update`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_d_basis.py#L305)

### sage.rings.polynomial.toy_variety (6 features)

- [ ] **coefficient_matrix** `(function)`
  - Full name: `sage.rings.polynomial.toy_variety.coefficient_matrix`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L82)

- [ ] **elim_pol** `(function)`
  - Full name: `sage.rings.polynomial.toy_variety.elim_pol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L300)

- [ ] **is_linearly_dependent** `(function)`
  - Full name: `sage.rings.polynomial.toy_variety.is_linearly_dependent`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L128)

- [ ] **is_triangular** `(function)`
  - Full name: `sage.rings.polynomial.toy_variety.is_triangular`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L33)

- [ ] **linear_representation** `(function)`
  - Full name: `sage.rings.polynomial.toy_variety.linear_representation`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L182)

- [ ] **toy_variety** `(module)`
  - Full name: `sage.rings.polynomial.toy_variety`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/toy_variety.py#L82)

### sage.rings.real_lazy (15 features)

- [ ] **ComplexLazyField** `(function)`
  - Full name: `sage.rings.real_lazy.ComplexLazyField`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L513)

- [ ] **ComplexLazyField_class** `(class)`
  - Full name: `sage.rings.real_lazy.ComplexLazyField_class`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L371)

- [ ] **LazyAlgebraic** `(class)`
  - Full name: `sage.rings.real_lazy.LazyAlgebraic`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1588)

- [ ] **LazyBinop** `(class)`
  - Full name: `sage.rings.real_lazy.LazyBinop`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1064)

- [ ] **LazyConstant** `(class)`
  - Full name: `sage.rings.real_lazy.LazyConstant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1445)

- [ ] **LazyField** `(class)`
  - Full name: `sage.rings.real_lazy.LazyField`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L82)

- [ ] **LazyFieldElement** `(class)`
  - Full name: `sage.rings.real_lazy.LazyFieldElement`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L369)

- [ ] **LazyNamedUnop** `(class)`
  - Full name: `sage.rings.real_lazy.LazyNamedUnop`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1305)

- [ ] **LazyUnop** `(class)`
  - Full name: `sage.rings.real_lazy.LazyUnop`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1201)

- [ ] **LazyWrapper** `(class)`
  - Full name: `sage.rings.real_lazy.LazyWrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L920)

- [ ] **LazyWrapperMorphism** `(class)`
  - Full name: `sage.rings.real_lazy.LazyWrapperMorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L1713)

- [ ] **RealLazyField** `(function)`
  - Full name: `sage.rings.real_lazy.RealLazyField`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L357)

- [ ] **RealLazyField_class** `(class)`
  - Full name: `sage.rings.real_lazy.RealLazyField_class`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L220)

- [ ] **make_element** `(function)`
  - Full name: `sage.rings.real_lazy.make_element`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L907)

- [ ] **real_lazy** `(module)`
  - Full name: `sage.rings.real_lazy`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/rings/real_lazy.pyx#L513)

### sage.symbolic (1 features)

- [ ] **symbolic** `(module)`
  - Full name: `sage.symbolic`

### sage.symbolic.benchmark (1 features)

- [ ] **benchmark** `(module)`
  - Full name: `sage.symbolic.benchmark`

### sage.symbolic.callable (3 features)

- [ ] **CallableSymbolicExpressionFunctor** `(class)`
  - Full name: `sage.symbolic.callable.CallableSymbolicExpressionFunctor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L73)

- [ ] **CallableSymbolicExpressionRing_class** `(class)`
  - Full name: `sage.symbolic.callable.CallableSymbolicExpressionRing_class`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L227)

- [ ] **callable** `(module)`
  - Full name: `sage.symbolic.callable`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/callable.py#L73)

### sage.symbolic.complexity_measures (2 features)

- [ ] **complexity_measures** `(module)`
  - Full name: `sage.symbolic.complexity_measures`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/complexity_measures.py#L10)

- [ ] **string_length** `(function)`
  - Full name: `sage.symbolic.complexity_measures.string_length`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/complexity_measures.py#L10)

### sage.symbolic.constants (14 features)

- [ ] **Catalan** `(class)`
  - Full name: `sage.symbolic.constants.Catalan`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1014)

- [ ] **Constant** `(class)`
  - Full name: `sage.symbolic.constants.Constant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L273)

- [ ] **EulerGamma** `(class)`
  - Full name: `sage.symbolic.constants.EulerGamma`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L935)

- [ ] **Glaisher** `(class)`
  - Full name: `sage.symbolic.constants.Glaisher`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1233)

- [ ] **GoldenRatio** `(class)`
  - Full name: `sage.symbolic.constants.GoldenRatio`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L754)

- [ ] **Khinchin** `(class)`
  - Full name: `sage.symbolic.constants.Khinchin`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1084)

- [ ] **Log2** `(class)`
  - Full name: `sage.symbolic.constants.Log2`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L858)

- [ ] **Mertens** `(class)`
  - Full name: `sage.symbolic.constants.Mertens`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1185)

- [ ] **NotANumber** `(class)`
  - Full name: `sage.symbolic.constants.NotANumber`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L692)

- [ ] **Pi** `(class)`
  - Full name: `sage.symbolic.constants.Pi`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L548)

- [ ] **TwinPrime** `(class)`
  - Full name: `sage.symbolic.constants.TwinPrime`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1137)

- [ ] **constants** `(module)`
  - Full name: `sage.symbolic.constants`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L1014)

- [ ] **pi** `(attribute)`
  - Full name: `sage.symbolic.constants.pi`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L697)

- [ ] **unpickle_Constant** `(function)`
  - Full name: `sage.symbolic.constants.unpickle_Constant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/constants.py#L244)

### sage.symbolic.expression (64 features)

- [ ] **E** `(class)`
  - Full name: `sage.symbolic.expression.E`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L33)

- [ ] **Expression** `(class)`
  - Full name: `sage.symbolic.expression.Expression`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L697)

- [ ] **ExpressionIterator** `(class)`
  - Full name: `sage.symbolic.expression.ExpressionIterator`

- [ ] **OperandsWrapper** `(class)`
  - Full name: `sage.symbolic.expression.OperandsWrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L65)

- [ ] **PynacConstant** `(class)`
  - Full name: `sage.symbolic.expression.PynacConstant`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L112)

- [ ] **SubstitutionMap** `(class)`
  - Full name: `sage.symbolic.expression.SubstitutionMap`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L36)

- [ ] **SymbolicSeries** `(class)`
  - Full name: `sage.symbolic.expression.SymbolicSeries`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L129)

- [ ] **call_registered_function** `(function)`
  - Full name: `sage.symbolic.expression.call_registered_function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1)

- [ ] **expression** `(module)`
  - Full name: `sage.symbolic.expression`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L33)

- [ ] **find_registered_function** `(function)`
  - Full name: `sage.symbolic.expression.find_registered_function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L67)

- [ ] **get_fn_serial** `(function)`
  - Full name: `sage.symbolic.expression.get_fn_serial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L270)

- [ ] **get_ginac_serial** `(function)`
  - Full name: `sage.symbolic.expression.get_ginac_serial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L251)

- [ ] **get_sfunction_from_hash** `(function)`
  - Full name: `sage.symbolic.expression.get_sfunction_from_hash`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L190)

- [ ] **get_sfunction_from_serial** `(function)`
  - Full name: `sage.symbolic.expression.get_sfunction_from_serial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L174)

- [ ] **hold_class** `(class)`
  - Full name: `sage.symbolic.expression.hold_class`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L14082)

- [ ] **init_function_table** `(function)`
  - Full name: `sage.symbolic.expression.init_function_table`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2477)

- [ ] **init_pynac_I** `(function)`
  - Full name: `sage.symbolic.expression.init_pynac_I`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2380)

- [ ] **is_SymbolicEquation** `(function)`
  - Full name: `sage.symbolic.expression.is_SymbolicEquation`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L412)

- [ ] **make_map** `(function)`
  - Full name: `sage.symbolic.expression.make_map`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L76)

- [ ] **math_sorted** `(function)`
  - Full name: `sage.symbolic.expression.math_sorted`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L216)

- [ ] **mixed_order** `(function)`
  - Full name: `sage.symbolic.expression.mixed_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L240)

- [ ] **mixed_sorted** `(function)`
  - Full name: `sage.symbolic.expression.mixed_sorted`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L403)

- [ ] **new_Expression** `(function)`
  - Full name: `sage.symbolic.expression.new_Expression`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13696)

- [ ] **new_Expression_from_pyobject** `(function)`
  - Full name: `sage.symbolic.expression.new_Expression_from_pyobject`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13786)

- [ ] **new_Expression_symbol** `(function)`
  - Full name: `sage.symbolic.expression.new_Expression_symbol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13885)

- [ ] **new_Expression_wild** `(function)`
  - Full name: `sage.symbolic.expression.new_Expression_wild`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L13857)

- [ ] **normalize_index_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.normalize_index_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L53)

- [ ] **op** `(attribute)`
  - Full name: `sage.symbolic.expression.op`

- [ ] **paramset_from_Expression** `(function)`
  - Full name: `sage.symbolic.expression.paramset_from_Expression`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L213)

- [ ] **print_order** `(function)`
  - Full name: `sage.symbolic.expression.print_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L48)

- [ ] **print_sorted** `(function)`
  - Full name: `sage.symbolic.expression.print_sorted`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L136)

- [ ] **py_atan2_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_atan2_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1900)

- [ ] **py_denom_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_denom_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1318)

- [ ] **py_eval_infinity_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_eval_infinity_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2301)

- [ ] **py_eval_neg_infinity_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_eval_neg_infinity_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2321)

- [ ] **py_eval_unsigned_infinity_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_eval_unsigned_infinity_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2281)

- [ ] **py_exp_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_exp_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1715)

- [ ] **py_float_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_float_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1394)

- [ ] **py_imag_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_imag_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1082)

- [ ] **py_is_cinteger_for_doctest** `(function)`
  - Full name: `sage.symbolic.expression.py_is_cinteger_for_doctest`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1335)

- [ ] **py_is_crational_for_doctest** `(function)`
  - Full name: `sage.symbolic.expression.py_is_crational_for_doctest`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1188)

- [ ] **py_is_integer_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_is_integer_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1155)

- [ ] **py_latex_fderivative_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_latex_fderivative_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L726)

- [ ] **py_latex_function_pystring** `(function)`
  - Full name: `sage.symbolic.expression.py_latex_function_pystring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L511)

- [ ] **py_latex_variable_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_latex_variable_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L426)

- [ ] **py_lgamma_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_lgamma_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2003)

- [ ] **py_li2_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_li2_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2236)

- [ ] **py_li_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_li_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2142)

- [ ] **py_log_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_log_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1785)

- [ ] **py_mod_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_mod_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2066)

- [ ] **py_numer_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_numer_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1282)

- [ ] **py_print_fderivative_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_print_fderivative_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L650)

- [ ] **py_print_function_pystring** `(function)`
  - Full name: `sage.symbolic.expression.py_print_function_pystring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L442)

- [ ] **py_psi2_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_psi2_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2207)

- [ ] **py_psi_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_psi_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L2178)

- [ ] **py_real_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_real_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1025)

- [ ] **py_stieltjes_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_stieltjes_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1638)

- [ ] **py_tgamma_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_tgamma_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1455)

- [ ] **py_zeta_for_doctests** `(function)`
  - Full name: `sage.symbolic.expression.py_zeta_for_doctests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L1673)

- [ ] **register_or_update_function** `(function)`
  - Full name: `sage.symbolic.expression.register_or_update_function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L91)

- [ ] **restore_op_wrapper** `(function)`
  - Full name: `sage.symbolic.expression.restore_op_wrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L192)

- [ ] **test_binomial** `(function)`
  - Full name: `sage.symbolic.expression.test_binomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L915)

- [ ] **tolerant_is_symbol** `(function)`
  - Full name: `sage.symbolic.expression.tolerant_is_symbol`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L601)

- [ ] **unpack_operands** `(function)`
  - Full name: `sage.symbolic.expression.unpack_operands`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression.pyx#L115)

### sage.symbolic.expression_conversions (36 features)

- [ ] **Converter** `(class)`
  - Full name: `sage.symbolic.expression_conversions.Converter`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L131)

- [ ] **DeMoivre** `(class)`
  - Full name: `sage.symbolic.expression_conversions.DeMoivre`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1712)

- [ ] **Exponentialize** `(class)`
  - Full name: `sage.symbolic.expression_conversions.Exponentialize`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1646)

- [ ] **ExpressionTreeWalker** `(class)`
  - Full name: `sage.symbolic.expression_conversions.ExpressionTreeWalker`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1448)

- [ ] **FakeExpression** `(class)`
  - Full name: `sage.symbolic.expression_conversions.FakeExpression`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L33)

- [ ] **FastCallableConverter** `(class)`
  - Full name: `sage.symbolic.expression_conversions.FastCallableConverter`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1153)

- [ ] **FriCASConverter** `(class)`
  - Full name: `sage.symbolic.expression_conversions.FriCASConverter`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L660)

- [ ] **HalfAngle** `(class)`
  - Full name: `sage.symbolic.expression_conversions.HalfAngle`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1774)

- [ ] **HoldRemover** `(class)`
  - Full name: `sage.symbolic.expression_conversions.HoldRemover`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1841)

- [ ] **InterfaceInit** `(class)`
  - Full name: `sage.symbolic.expression_conversions.InterfaceInit`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L387)

- [ ] **LaurentPolynomialConverter** `(class)`
  - Full name: `sage.symbolic.expression_conversions.LaurentPolynomialConverter`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1083)

- [ ] **PolynomialConverter** `(class)`
  - Full name: `sage.symbolic.expression_conversions.PolynomialConverter`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L844)

- [ ] **RingConverter** `(class)`
  - Full name: `sage.symbolic.expression_conversions.RingConverter`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1333)

- [ ] **SubstituteFunction** `(class)`
  - Full name: `sage.symbolic.expression_conversions.SubstituteFunction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1562)

- [ ] **cos** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.cos`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L138)

- [ ] **cosh** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.cosh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L87)

- [ ] **cot** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.cot`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L272)

- [ ] **coth** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.coth`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L193)

- [ ] **csc** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.csc`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L447)

- [ ] **csch** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.csch`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L309)

- [ ] **e** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.e`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L33)

- [ ] **expression_conversions** `(module)`
  - Full name: `sage.symbolic.expression_conversions`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L131)

- [ ] **fast_callable** `(function)`
  - Full name: `sage.symbolic.expression_conversions.fast_callable`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1312)

- [ ] **half** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.half`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L414)

- [ ] **halfx** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.halfx`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L697)

- [ ] **laurent_polynomial** `(function)`
  - Full name: `sage.symbolic.expression_conversions.laurent_polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1108)

- [ ] **one** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.one`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L404)

- [ ] **polynomial** `(function)`
  - Full name: `sage.symbolic.expression_conversions.polynomial`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L1021)

- [ ] **sec** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.sec`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L373)

- [ ] **sech** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.sech`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L252)

- [ ] **sin** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.sin`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L9)

- [ ] **sinh** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.sinh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L49)

- [ ] **tan** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.tan`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L206)

- [ ] **tanh** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.tanh`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L125)

- [ ] **two** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.two`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L404)

- [ ] **x** `(attribute)`
  - Full name: `sage.symbolic.expression_conversions.x`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/expression_conversions.py#L697)

### sage.symbolic.function (7 features)

- [ ] **BuiltinFunction** `(class)`
  - Full name: `sage.symbolic.function.BuiltinFunction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L867)

- [ ] **Function** `(class)`
  - Full name: `sage.symbolic.function.Function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L166)

- [ ] **GinacFunction** `(class)`
  - Full name: `sage.symbolic.function.GinacFunction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L818)

- [ ] **SymbolicFunction** `(class)`
  - Full name: `sage.symbolic.function.SymbolicFunction`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L1174)

- [ ] **function** `(module)`
  - Full name: `sage.symbolic.function`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L867)

- [ ] **pickle_wrapper** `(function)`
  - Full name: `sage.symbolic.function.pickle_wrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L1407)

- [ ] **unpickle_wrapper** `(function)`
  - Full name: `sage.symbolic.function.unpickle_wrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/function.pyx#L1429)

### sage.symbolic.integration.external (6 features)

- [ ] **external** `(module)`
  - Full name: `sage.symbolic.integration.external`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L131)

- [ ] **fricas_integrator** `(function)`
  - Full name: `sage.symbolic.integration.external.fricas_integrator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L131)

- [ ] **libgiac_integrator** `(function)`
  - Full name: `sage.symbolic.integration.external.libgiac_integrator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L217)

- [ ] **maxima_integrator** `(function)`
  - Full name: `sage.symbolic.integration.external.maxima_integrator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L13)

- [ ] **mma_free_integrator** `(function)`
  - Full name: `sage.symbolic.integration.external.mma_free_integrator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L72)

- [ ] **sympy_integrator** `(function)`
  - Full name: `sage.symbolic.integration.external.sympy_integrator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/external.py#L50)

### sage.symbolic.integration.integral (4 features)

- [ ] **DefiniteIntegral** `(class)`
  - Full name: `sage.symbolic.integration.integral.DefiniteIntegral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L196)

- [ ] **IndefiniteIntegral** `(class)`
  - Full name: `sage.symbolic.integration.integral.IndefiniteIntegral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L41)

- [ ] **integral** `(module)`
  - Full name: `sage.symbolic.integration.integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L196)

- [ ] **integral** `(function)`
  - Full name: `sage.symbolic.integration.integral.integral`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/integration/integral.py#L446)

### sage.symbolic.maxima_wrapper (3 features)

- [ ] **MaximaFunctionElementWrapper** `(class)`
  - Full name: `sage.symbolic.maxima_wrapper.MaximaFunctionElementWrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/maxima_wrapper.py#L16)

- [ ] **MaximaWrapper** `(class)`
  - Full name: `sage.symbolic.maxima_wrapper.MaximaWrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/maxima_wrapper.py#L34)

- [ ] **maxima_wrapper** `(module)`
  - Full name: `sage.symbolic.maxima_wrapper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/maxima_wrapper.py#L16)

### sage.symbolic.operators (6 features)

- [ ] **DerivativeOperator** `(class)`
  - Full name: `sage.symbolic.operators.DerivativeOperator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L204)

- [ ] **DerivativeOperatorWithParameters** `(class)`
  - Full name: `sage.symbolic.operators.DerivativeOperatorWithParameters`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L229)

- [ ] **FDerivativeOperator** `(class)`
  - Full name: `sage.symbolic.operators.FDerivativeOperator`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L75)

- [ ] **add_vararg** `(function)`
  - Full name: `sage.symbolic.operators.add_vararg`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L8)

- [ ] **mul_vararg** `(function)`
  - Full name: `sage.symbolic.operators.mul_vararg`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L33)

- [ ] **operators** `(module)`
  - Full name: `sage.symbolic.operators`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/operators.py#L204)

### sage.symbolic.random_tests (8 features)

- [ ] **assert_strict_weak_order** `(function)`
  - Full name: `sage.symbolic.random_tests.assert_strict_weak_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L329)

- [ ] **choose_from_prob_list** `(function)`
  - Full name: `sage.symbolic.random_tests.choose_from_prob_list`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L130)

- [ ] **normalize_prob_list** `(function)`
  - Full name: `sage.symbolic.random_tests.normalize_prob_list`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L79)

- [ ] **random_expr** `(function)`
  - Full name: `sage.symbolic.random_tests.random_expr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L270)

- [ ] **random_expr_helper** `(function)`
  - Full name: `sage.symbolic.random_tests.random_expr_helper`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L224)

- [ ] **random_integer_vector** `(function)`
  - Full name: `sage.symbolic.random_tests.random_integer_vector`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L169)

- [ ] **random_tests** `(module)`
  - Full name: `sage.symbolic.random_tests`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L329)

- [ ] **test_symbolic_expression_order** `(function)`
  - Full name: `sage.symbolic.random_tests.test_symbolic_expression_order`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/random_tests.py#L421)

### sage.symbolic.relation (3 features)

- [ ] **relation** `(module)`
  - Full name: `sage.symbolic.relation`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L584)

- [ ] **string_to_list_of_solutions** `(function)`
  - Full name: `sage.symbolic.relation.string_to_list_of_solutions`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L550)

- [ ] **test_relation_maxima** `(function)`
  - Full name: `sage.symbolic.relation.test_relation_maxima`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/relation.py#L363)

### sage.symbolic.ring (9 features)

- [ ] **NumpyToSRMorphism** `(class)`
  - Full name: `sage.symbolic.ring.NumpyToSRMorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1148)

- [ ] **SymbolicRing** `(class)`
  - Full name: `sage.symbolic.ring.SymbolicRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L66)

- [ ] **TemporaryVariables** `(class)`
  - Full name: `sage.symbolic.ring.TemporaryVariables`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1374)

- [ ] **UnderscoreSageMorphism** `(class)`
  - Full name: `sage.symbolic.ring.UnderscoreSageMorphism`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1243)

- [ ] **isidentifier** `(function)`
  - Full name: `sage.symbolic.ring.isidentifier`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1331)

- [ ] **ring** `(module)`
  - Full name: `sage.symbolic.ring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1148)

- [ ] **symbols** `(attribute)`
  - Full name: `sage.symbolic.ring.symbols`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L4)

- [ ] **the_SymbolicRing** `(function)`
  - Full name: `sage.symbolic.ring.the_SymbolicRing`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1282)

- [ ] **var** `(function)`
  - Full name: `sage.symbolic.ring.var`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/ring.pyx#L1300)

### sage.symbolic.subring (10 features)

- [ ] **GenericSymbolicSubring** `(class)`
  - Full name: `sage.symbolic.subring.GenericSymbolicSubring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L249)

- [ ] **GenericSymbolicSubringFunctor** `(class)`
  - Full name: `sage.symbolic.subring.GenericSymbolicSubringFunctor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L476)

- [ ] **SymbolicConstantsSubring** `(class)`
  - Full name: `sage.symbolic.subring.SymbolicConstantsSubring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L968)

- [ ] **SymbolicSubringAcceptingVars** `(class)`
  - Full name: `sage.symbolic.subring.SymbolicSubringAcceptingVars`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L613)

- [ ] **SymbolicSubringAcceptingVarsFunctor** `(class)`
  - Full name: `sage.symbolic.subring.SymbolicSubringAcceptingVarsFunctor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L714)

- [ ] **SymbolicSubringRejectingVars** `(class)`
  - Full name: `sage.symbolic.subring.SymbolicSubringRejectingVars`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L779)

- [ ] **SymbolicSubringRejectingVarsFunctor** `(class)`
  - Full name: `sage.symbolic.subring.SymbolicSubringRejectingVarsFunctor`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L903)

- [ ] **coercion_reversed** `(attribute)`
  - Full name: `sage.symbolic.subring.coercion_reversed`

- [ ] **rank** `(attribute)`
  - Full name: `sage.symbolic.subring.rank`

- [ ] **subring** `(module)`
  - Full name: `sage.symbolic.subring`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/subring.py#L249)


---

## Priority: LOW (19 features)

Specialized or advanced features for specific use cases.

### sage.symbolic.assumptions (7 features)

- [ ] **GenericDeclaration** `(class)`
  - Full name: `sage.symbolic.assumptions.GenericDeclaration`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L90)

- [ ] **assume** `(function)`
  - Full name: `sage.symbolic.assumptions.assume`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L436)

- [ ] **assuming** `(class)`
  - Full name: `sage.symbolic.assumptions.assuming`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L834)

- [ ] **assumptions** `(module)`
  - Full name: `sage.symbolic.assumptions`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L90)

- [ ] **assumptions** `(function)`
  - Full name: `sage.symbolic.assumptions.assumptions`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L726)

- [ ] **forget** `(function)`
  - Full name: `sage.symbolic.assumptions.forget`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L675)

- [ ] **preprocess_assumptions** `(function)`
  - Full name: `sage.symbolic.assumptions.preprocess_assumptions`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/assumptions.py#L405)

### sage.symbolic.units (12 features)

- [ ] **UnitExpression** `(class)`
  - Full name: `sage.symbolic.units.UnitExpression`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L989)

- [ ] **Units** `(class)`
  - Full name: `sage.symbolic.units.Units`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1042)

- [ ] **base_units** `(function)`
  - Full name: `sage.symbolic.units.base_units`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1355)

- [ ] **convert** `(function)`
  - Full name: `sage.symbolic.units.convert`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1263)

- [ ] **convert_temperature** `(function)`
  - Full name: `sage.symbolic.units.convert_temperature`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1407)

- [ ] **evalunitdict** `(function)`
  - Full name: `sage.symbolic.units.evalunitdict`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L493)

- [ ] **is_unit** `(function)`
  - Full name: `sage.symbolic.units.is_unit`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1232)

- [ ] **str_to_unit** `(function)`
  - Full name: `sage.symbolic.units.str_to_unit`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1020)

- [ ] **unit_derivations_expr** `(function)`
  - Full name: `sage.symbolic.units.unit_derivations_expr`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L952)

- [ ] **unitdocs** `(function)`
  - Full name: `sage.symbolic.units.unitdocs`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L1203)

- [ ] **units** `(module)`
  - Full name: `sage.symbolic.units`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L989)

- [ ] **vars_in_str** `(function)`
  - Full name: `sage.symbolic.units.vars_in_str`
  - [Source](https://github.com/sagemath/sage/blob/develop/src/sage/symbolic/units.py#L933)


---

