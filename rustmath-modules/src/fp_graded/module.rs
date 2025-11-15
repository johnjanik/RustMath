//! Finitely presented graded modules

/// A finitely presented graded module
#[derive(Clone, Debug)]
pub struct FPGradedModule {
    /// Number of generators in each degree
    generators_by_degree: Vec<usize>,
    /// Relations
    relations: Vec<(usize, Vec<i32>)>,
}

impl FPGradedModule {
    pub fn new(generators_by_degree: Vec<usize>) -> Self {
        Self {
            generators_by_degree,
            relations: Vec::new(),
        }
    }

    pub fn add_relation(&mut self, degree: usize, coeffs: Vec<i32>) {
        self.relations.push((degree, coeffs));
    }

    pub fn num_generators_in_degree(&self, degree: usize) -> usize {
        if degree < self.generators_by_degree.len() {
            self.generators_by_degree[degree]
        } else {
            0
        }
    }

    pub fn hilbert_series(&self, max_degree: usize) -> Vec<i32> {
        let mut series = vec![0; max_degree + 1];
        for (deg, count) in self.generators_by_degree.iter().enumerate() {
            if deg <= max_degree {
                series[deg] = *count as i32;
            }
        }
        series
    }
}
