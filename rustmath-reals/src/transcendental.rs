//! Transcendental functions for real numbers

use crate::Real;

impl Real {
    /// Compute sine
    pub fn sin(&self) -> Self {
        Real {
            value: self.value.sin(),
        }
    }

    /// Compute cosine
    pub fn cos(&self) -> Self {
        Real {
            value: self.value.cos(),
        }
    }

    /// Compute tangent
    pub fn tan(&self) -> Self {
        Real {
            value: self.value.tan(),
        }
    }

    /// Compute arcsine
    pub fn asin(&self) -> Self {
        Real {
            value: self.value.asin(),
        }
    }

    /// Compute arccosine
    pub fn acos(&self) -> Self {
        Real {
            value: self.value.acos(),
        }
    }

    /// Compute arctangent
    pub fn atan(&self) -> Self {
        Real {
            value: self.value.atan(),
        }
    }

    /// Compute arctangent of y/x
    pub fn atan2(&self, x: &Self) -> Self {
        Real {
            value: self.value.atan2(x.value),
        }
    }

    /// Compute hyperbolic sine
    pub fn sinh(&self) -> Self {
        Real {
            value: self.value.sinh(),
        }
    }

    /// Compute hyperbolic cosine
    pub fn cosh(&self) -> Self {
        Real {
            value: self.value.cosh(),
        }
    }

    /// Compute hyperbolic tangent
    pub fn tanh(&self) -> Self {
        Real {
            value: self.value.tanh(),
        }
    }

    /// Compute inverse hyperbolic sine
    pub fn asinh(&self) -> Self {
        Real {
            value: self.value.asinh(),
        }
    }

    /// Compute inverse hyperbolic cosine
    pub fn acosh(&self) -> Self {
        Real {
            value: self.value.acosh(),
        }
    }

    /// Compute inverse hyperbolic tangent
    pub fn atanh(&self) -> Self {
        Real {
            value: self.value.atanh(),
        }
    }

    /// Compute exponential (e^x)
    pub fn exp(&self) -> Self {
        Real {
            value: self.value.exp(),
        }
    }

    /// Compute natural logarithm
    pub fn ln(&self) -> Self {
        Real {
            value: self.value.ln(),
        }
    }

    /// Compute logarithm base 2
    pub fn log2(&self) -> Self {
        Real {
            value: self.value.log2(),
        }
    }

    /// Compute logarithm base 10
    pub fn log10(&self) -> Self {
        Real {
            value: self.value.log10(),
        }
    }

    /// Compute logarithm with arbitrary base
    pub fn log(&self, base: &Self) -> Self {
        Real {
            value: self.value.log(base.value),
        }
    }

    /// Compute square root
    pub fn sqrt(&self) -> Self {
        Real {
            value: self.value.sqrt(),
        }
    }

    /// Compute cube root
    pub fn cbrt(&self) -> Self {
        Real {
            value: self.value.cbrt(),
        }
    }

    /// Compute e^x - 1 (more accurate for small x)
    pub fn exp_m1(&self) -> Self {
        Real {
            value: self.value.exp_m1(),
        }
    }

    /// Compute ln(1 + x) (more accurate for small x)
    pub fn ln_1p(&self) -> Self {
        Real {
            value: self.value.ln_1p(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trig_functions() {
        let x = Real::from(std::f64::consts::PI / 4.0);

        let sin_x = x.sin();
        let cos_x = x.cos();

        // sin(π/4) ≈ √2/2 ≈ 0.707...
        assert!((sin_x.to_f64() - 0.7071067811865476).abs() < 1e-10);

        // cos(π/4) ≈ √2/2 ≈ 0.707...
        assert!((cos_x.to_f64() - 0.7071067811865476).abs() < 1e-10);

        // sin²(x) + cos²(x) = 1
        let sin_sq = sin_x.clone() * sin_x;
        let cos_sq = cos_x.clone() * cos_x;
        let sum = sin_sq + cos_sq;
        assert!((sum.to_f64() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_log() {
        let x = Real::from(2.0);

        let exp_x = x.exp();
        let log_exp_x = exp_x.ln();

        assert!((log_exp_x.to_f64() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let x = Real::from(16.0);
        let sqrt_x = x.sqrt();

        assert!((sqrt_x.to_f64() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic() {
        let x = Real::from(1.0);

        let sinh_x = x.sinh();
        let cosh_x = x.cosh();

        // cosh²(x) - sinh²(x) = 1
        let cosh_sq = cosh_x.clone() * cosh_x;
        let sinh_sq = sinh_x.clone() * sinh_x;
        let diff = cosh_sq - sinh_sq;

        assert!((diff.to_f64() - 1.0).abs() < 1e-10);
    }
}
