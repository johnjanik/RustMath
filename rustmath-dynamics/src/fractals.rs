//! Fractal generation
//!
//! Mandelbrot set, Julia sets, and other fractals

use rustmath_complex::Complex;

/// Result of Mandelbrot set iteration
#[derive(Clone, Debug, PartialEq)]
pub struct MandelbrotResult {
    pub c: Complex,
    pub iterations: usize,
    pub escaped: bool,
}

/// Result of Julia set iteration
#[derive(Clone, Debug, PartialEq)]
pub struct JuliaResult {
    pub z0: Complex,
    pub iterations: usize,
    pub escaped: bool,
}

/// Check if a complex number is in the Mandelbrot set
/// Returns the number of iterations before escape (or max_iter if no escape)
pub fn mandelbrot(c: Complex, max_iter: usize, escape_radius: f64) -> MandelbrotResult {
    let mut z = Complex::new(0.0, 0.0);
    let radius_sq = escape_radius * escape_radius;

    for i in 0..max_iter {
        if z.abs_sq() > radius_sq {
            return MandelbrotResult {
                c: c.clone(),
                iterations: i,
                escaped: true,
            };
        }
        z = z.clone() * z.clone() + c.clone();
    }

    MandelbrotResult {
        c,
        iterations: max_iter,
        escaped: false,
    }
}

/// Check if a point is in the Julia set for parameter c
/// Iterates z_{n+1} = z_n^2 + c starting from z0
pub fn julia_set(z0: Complex, c: Complex, max_iter: usize, escape_radius: f64) -> JuliaResult {
    let mut z = z0.clone();
    let radius_sq = escape_radius * escape_radius;

    for i in 0..max_iter {
        if z.abs_sq() > radius_sq {
            return JuliaResult {
                z0: z0.clone(),
                iterations: i,
                escaped: true,
            };
        }
        z = z.clone() * z.clone() + c.clone();
    }

    JuliaResult {
        z0,
        iterations: max_iter,
        escaped: false,
    }
}

/// Generate a grid of Mandelbrot set values
pub fn mandelbrot_grid(
    x_range: (f64, f64),
    y_range: (f64, f64),
    width: usize,
    height: usize,
    max_iter: usize,
    escape_radius: f64,
) -> Vec<Vec<MandelbrotResult>> {
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    let mut grid = Vec::with_capacity(height);

    for row in 0..height {
        let mut row_data = Vec::with_capacity(width);
        let y = y_max - (row as f64 / height as f64) * (y_max - y_min);

        for col in 0..width {
            let x = x_min + (col as f64 / width as f64) * (x_max - x_min);
            let c = Complex::new(x, y);
            row_data.push(mandelbrot(c, max_iter, escape_radius));
        }

        grid.push(row_data);
    }

    grid
}

/// Generate a grid of Julia set values
pub fn julia_grid(
    c: Complex,
    x_range: (f64, f64),
    y_range: (f64, f64),
    width: usize,
    height: usize,
    max_iter: usize,
    escape_radius: f64,
) -> Vec<Vec<JuliaResult>> {
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;

    let mut grid = Vec::with_capacity(height);

    for row in 0..height {
        let mut row_data = Vec::with_capacity(width);
        let y = y_max - (row as f64 / height as f64) * (y_max - y_min);

        for col in 0..width {
            let x = x_min + (col as f64 / width as f64) * (x_max - x_min);
            let z0 = Complex::new(x, y);
            row_data.push(julia_set(z0, c.clone(), max_iter, escape_radius));
        }

        grid.push(row_data);
    }

    grid
}

/// Burning Ship fractal: z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)^2 + c
pub fn burning_ship(c: Complex, max_iter: usize, escape_radius: f64) -> MandelbrotResult {
    let mut z = Complex::new(0.0, 0.0);
    let radius_sq = escape_radius * escape_radius;

    for i in 0..max_iter {
        if z.abs_sq() > radius_sq {
            return MandelbrotResult {
                c: c.clone(),
                iterations: i,
                escaped: true,
            };
        }
        // Take absolute values of real and imaginary parts
        let z_abs = Complex::new(z.real().abs(), z.imag().abs());
        z = z_abs.clone() * z_abs + c.clone();
    }

    MandelbrotResult {
        c,
        iterations: max_iter,
        escaped: false,
    }
}

/// Multibrot set: z_{n+1} = z_n^d + c (generalization of Mandelbrot)
pub fn multibrot(c: Complex, d: i32, max_iter: usize, escape_radius: f64) -> MandelbrotResult {
    let mut z = Complex::new(0.0, 0.0);
    let radius_sq = escape_radius * escape_radius;

    for i in 0..max_iter {
        if z.abs_sq() > radius_sq {
            return MandelbrotResult {
                c: c.clone(),
                iterations: i,
                escaped: true,
            };
        }
        z = z.pow(&Complex::new(d as f64, 0.0)) + c.clone();
    }

    MandelbrotResult {
        c,
        iterations: max_iter,
        escaped: false,
    }
}

/// Newton fractal for finding roots of z^3 - 1
pub fn newton_fractal(z0: Complex, max_iter: usize, tolerance: f64) -> NewtonResult {
    let mut z = z0.clone();

    for i in 0..max_iter {
        // f(z) = z^3 - 1
        // f'(z) = 3z^2
        let f = z.pow(&Complex::new(3.0, 0.0)) - Complex::new(1.0, 0.0);

        if f.abs() < tolerance {
            return NewtonResult {
                z0: z0.clone(),
                iterations: i,
                converged: true,
                root: z,
            };
        }

        let fprime = z.pow(&Complex::new(2.0, 0.0)) * Complex::new(3.0, 0.0);

        if fprime.abs() < 1e-10 {
            break;
        }

        z = z - f / fprime;
    }

    NewtonResult {
        z0,
        iterations: max_iter,
        converged: false,
        root: z,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NewtonResult {
    pub z0: Complex,
    pub iterations: usize,
    pub converged: bool,
    pub root: Complex,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mandelbrot_inside() {
        // Point (0, 0) is in the Mandelbrot set
        let c = Complex::new(0.0, 0.0);
        let result = mandelbrot(c, 100, 2.0);
        assert!(!result.escaped);
        assert_eq!(result.iterations, 100);
    }

    #[test]
    fn test_mandelbrot_outside() {
        // Point (2, 0) is outside the Mandelbrot set
        let c = Complex::new(2.0, 0.0);
        let result = mandelbrot(c, 100, 2.0);
        assert!(result.escaped);
        assert!(result.iterations < 100);
    }

    #[test]
    fn test_mandelbrot_boundary() {
        // Point on the boundary should take many iterations
        let c = Complex::new(-0.5, 0.0);
        let result = mandelbrot(c, 100, 2.0);
        // This point is actually inside
        assert!(!result.escaped);
    }

    #[test]
    fn test_julia_set() {
        // Julia set for c = -0.4 + 0.6i
        let c = Complex::new(-0.4, 0.6);
        let z0 = Complex::new(0.0, 0.0);
        let result = julia_set(z0, c, 100, 2.0);

        // Origin behavior depends on c
        assert!(result.iterations <= 100);
    }

    #[test]
    fn test_julia_set_escape() {
        // Far point should escape quickly
        let c = Complex::new(0.0, 0.0);
        let z0 = Complex::new(10.0, 10.0);
        let result = julia_set(z0, c, 100, 2.0);

        assert!(result.escaped);
        assert!(result.iterations < 10);
    }

    #[test]
    fn test_mandelbrot_grid() {
        let grid = mandelbrot_grid((-2.0, 1.0), (-1.0, 1.0), 10, 10, 50, 2.0);

        assert_eq!(grid.len(), 10);
        assert_eq!(grid[0].len(), 10);
    }

    #[test]
    fn test_julia_grid() {
        let c = Complex::new(-0.8, 0.156);
        let grid = julia_grid(c, (-2.0, 2.0), (-2.0, 2.0), 5, 5, 50, 2.0);

        assert_eq!(grid.len(), 5);
        assert_eq!(grid[0].len(), 5);
    }

    #[test]
    fn test_burning_ship() {
        let c = Complex::new(0.0, 0.0);
        let result = burning_ship(c, 100, 2.0);
        assert!(!result.escaped);
    }

    #[test]
    fn test_multibrot_d2() {
        // d=2 should be same as standard Mandelbrot
        let c = Complex::new(0.0, 0.0);
        let result = multibrot(c, 2, 100, 2.0);
        assert!(!result.escaped);
    }

    #[test]
    fn test_multibrot_d3() {
        // d=3 (cubic Mandelbrot)
        let c = Complex::new(0.0, 0.0);
        let result = multibrot(c, 3, 100, 2.0);
        assert!(!result.escaped);
    }

    #[test]
    fn test_newton_fractal() {
        // Start near root z=1
        let z0 = Complex::new(0.9, 0.1);
        let result = newton_fractal(z0, 100, 1e-6);

        assert!(result.converged);
        // Should converge to one of the roots: 1, e^(2πi/3), e^(4πi/3)
        let root_norm = result.root.abs();
        assert!((root_norm - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_newton_fractal_root() {
        // Start exactly at root z=1
        let z0 = Complex::new(1.0, 0.0);
        let result = newton_fractal(z0, 100, 1e-6);

        assert!(result.converged);
        assert!(result.iterations < 5);
    }
}
