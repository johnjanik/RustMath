//! Numerical integration

/// Integrate function f from a to b using Simpson's rule
pub fn integrate<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    simpson(f, a, b, n)
}

/// Simpson's rule for numerical integration
pub fn simpson<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (b - a) / n as f64;

    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;
        if i % 2 == 0 {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }

    sum * h / 3.0
}

/// Trapezoidal rule for numerical integration
pub fn trapezoid<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(x);
    }

    sum * h
}

/// Romberg integration
pub fn romberg<F>(f: F, a: f64, b: f64, max_steps: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut r = vec![vec![0.0; max_steps]; max_steps];

    // First estimate using trapezoidal rule
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));

    for i in 1..max_steps {
        let h = (b - a) / 2.0_f64.powi(i as i32);
        let mut sum = 0.0;

        for k in 0..2_usize.pow(i as u32 - 1) {
            sum += f(a + (2 * k + 1) as f64 * h);
        }

        r[i][0] = 0.5 * r[i - 1][0] + h * sum;

        for j in 1..=i {
            let factor = 4.0_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }
    }

    r[max_steps - 1][max_steps - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpson() {
        // Integrate x^2 from 0 to 1, exact answer is 1/3
        let f = |x: f64| x * x;
        let result = simpson(f, 0.0, 1.0, 100);

        assert!((result - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_trapezoid() {
        let f = |x: f64| x * x;
        let result = trapezoid(f, 0.0, 1.0, 1000);

        assert!((result - 1.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_romberg() {
        let f = |x: f64| x * x;
        let result = romberg(f, 0.0, 1.0, 10);

        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }
}
