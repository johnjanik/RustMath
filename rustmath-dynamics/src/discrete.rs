//! Discrete dynamical systems
//!
//! Systems defined by iteration of a function: x_{n+1} = f(x_n)

/// A point in an orbit
#[derive(Clone, Debug, PartialEq)]
pub struct OrbitPoint {
    pub iteration: usize,
    pub value: f64,
}

/// A discrete dynamical system
pub struct DiscreteSystem<F>
where
    F: Fn(f64) -> f64,
{
    pub map: F,
}

impl<F> DiscreteSystem<F>
where
    F: Fn(f64) -> f64,
{
    pub fn new(map: F) -> Self {
        DiscreteSystem { map }
    }

    /// Iterate the map n times starting from x0
    pub fn iterate(&self, x0: f64, n: usize) -> Vec<OrbitPoint> {
        let mut orbit = Vec::with_capacity(n + 1);
        let mut x = x0;

        orbit.push(OrbitPoint {
            iteration: 0,
            value: x,
        });

        for i in 1..=n {
            x = (self.map)(x);
            orbit.push(OrbitPoint {
                iteration: i,
                value: x,
            });
        }

        orbit
    }

    /// Find fixed points using numerical search
    /// Returns points where |f(x) - x| < tolerance
    pub fn find_fixed_points(&self, search_range: (f64, f64), num_samples: usize, tolerance: f64) -> Vec<f64> {
        let (min, max) = search_range;
        let step = (max - min) / num_samples as f64;
        let mut fixed_points = Vec::new();

        for i in 0..num_samples {
            let x = min + i as f64 * step;
            let fx = (self.map)(x);

            if (fx - x).abs() < tolerance {
                // Check if this is a new fixed point
                if !fixed_points.iter().any(|fp| (fp - x).abs() < tolerance) {
                    fixed_points.push(x);
                }
            }
        }

        fixed_points
    }

    /// Check if a point is periodic with period n
    pub fn is_periodic(&self, x0: f64, period: usize, tolerance: f64) -> bool {
        let mut x = x0;
        for _ in 0..period {
            x = (self.map)(x);
        }
        (x - x0).abs() < tolerance
    }

    /// Find the eventual behavior (limit, cycle, or chaotic)
    pub fn analyze_orbit(&self, x0: f64, transient: usize, check_length: usize, tolerance: f64) -> OrbitBehavior {
        // Skip transient
        let mut x = x0;
        for _ in 0..transient {
            x = (self.map)(x);
        }

        // Collect orbit after transient
        let mut orbit = Vec::new();
        for _ in 0..check_length {
            x = (self.map)(x);
            orbit.push(x);
        }

        // Check for fixed point
        if orbit.iter().all(|&val| (val - orbit[0]).abs() < tolerance) {
            return OrbitBehavior::FixedPoint(orbit[0]);
        }

        // Check for periodic orbit (periods 2-10)
        for period in 2..=10.min(check_length / 2) {
            let mut is_periodic = true;
            for i in 0..(check_length - period) {
                if (orbit[i] - orbit[i + period]).abs() > tolerance {
                    is_periodic = false;
                    break;
                }
            }
            if is_periodic {
                return OrbitBehavior::Periodic {
                    period,
                    cycle: orbit[0..period].to_vec(),
                };
            }
        }

        OrbitBehavior::Chaotic
    }
}

/// The behavior of an orbit
#[derive(Clone, Debug, PartialEq)]
pub enum OrbitBehavior {
    FixedPoint(f64),
    Periodic { period: usize, cycle: Vec<f64> },
    Chaotic,
}

/// Iterate a function n times from initial value
pub fn iterate<F>(f: F, x0: f64, n: usize) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let mut result = Vec::with_capacity(n + 1);
    let mut x = x0;
    result.push(x);

    for _ in 0..n {
        x = f(x);
        result.push(x);
    }

    result
}

/// Find fixed points of a function
pub fn fixed_points<F>(f: F, range: (f64, f64), samples: usize, tol: f64) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let system = DiscreteSystem::new(f);
    system.find_fixed_points(range, samples, tol)
}

/// Logistic map: f(x) = r * x * (1 - x)
pub fn logistic_map(r: f64) -> impl Fn(f64) -> f64 {
    move |x| r * x * (1.0 - x)
}

/// Tent map: f(x) = r * min(x, 1-x)
pub fn tent_map(r: f64) -> impl Fn(f64) -> f64 {
    move |x| r * x.min(1.0 - x)
}

/// Sine map: f(x) = r * sin(π * x)
pub fn sine_map(r: f64) -> impl Fn(f64) -> f64 {
    use std::f64::consts::PI;
    move |x| r * (PI * x).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iterate_simple() {
        let f = |x: f64| x * 2.0;
        let orbit = iterate(f, 1.0, 5);

        assert_eq!(orbit.len(), 6);
        assert_eq!(orbit[0], 1.0);
        assert_eq!(orbit[1], 2.0);
        assert_eq!(orbit[2], 4.0);
        assert_eq!(orbit[5], 32.0);
    }

    #[test]
    fn test_discrete_system_iterate() {
        let system = DiscreteSystem::new(|x| x * 0.5);
        let orbit = system.iterate(8.0, 3);

        assert_eq!(orbit.len(), 4);
        assert_eq!(orbit[0].value, 8.0);
        assert_eq!(orbit[1].value, 4.0);
        assert_eq!(orbit[2].value, 2.0);
        assert_eq!(orbit[3].value, 1.0);
    }

    #[test]
    fn test_fixed_point() {
        // f(x) = x/2 + 1 has fixed point at x = 2
        let f = |x: f64| x / 2.0 + 1.0;
        let fps = fixed_points(f, (0.0, 10.0), 1000, 0.01);

        assert!(!fps.is_empty());
        assert!(fps.iter().any(|&x| (x - 2.0).abs() < 0.1));
    }

    #[test]
    fn test_logistic_map_fixed_point() {
        // For r=2, logistic map has fixed point at x=0.5
        let system = DiscreteSystem::new(logistic_map(2.0));
        let orbit = system.iterate(0.3, 100);

        // Should converge to 0.5
        assert!((orbit.last().unwrap().value - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_is_periodic() {
        // f(x) = -x has period 2 for any x != 0
        let system = DiscreteSystem::new(|x| -x);
        assert!(system.is_periodic(1.0, 2, 0.0001));
        assert!(!system.is_periodic(1.0, 3, 0.0001));
    }

    #[test]
    fn test_analyze_orbit_fixed_point() {
        let system = DiscreteSystem::new(|x| x * 0.5);
        let behavior = system.analyze_orbit(10.0, 50, 50, 0.01);

        match behavior {
            OrbitBehavior::FixedPoint(x) => {
                assert!(x.abs() < 0.1); // Should converge to 0
            }
            _ => panic!("Expected fixed point"),
        }
    }

    #[test]
    fn test_analyze_orbit_periodic() {
        // f(x) = -x has period 2
        let system = DiscreteSystem::new(|x| -x);
        let behavior = system.analyze_orbit(1.0, 0, 20, 0.0001);

        match behavior {
            OrbitBehavior::Periodic { period, .. } => {
                assert_eq!(period, 2);
            }
            _ => panic!("Expected periodic orbit"),
        }
    }

    #[test]
    fn test_logistic_map_creation() {
        let f = logistic_map(3.5);
        let x = 0.5;
        let fx = f(x);
        assert_eq!(fx, 3.5 * 0.5 * 0.5);
    }

    #[test]
    fn test_tent_map() {
        let f = tent_map(2.0);
        assert!((f(0.3) - 0.6).abs() < 1e-10);
        assert!((f(0.7) - 0.6).abs() < 1e-10);
    }
}
