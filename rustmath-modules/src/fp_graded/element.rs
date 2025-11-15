//! Elements of graded modules

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FPGradedElement {
    degree: usize,
    coefficients: Vec<i32>,
}

impl FPGradedElement {
    pub fn new(degree: usize, coefficients: Vec<i32>) -> Self {
        Self { degree, coefficients }
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn coefficients(&self) -> &[i32] {
        &self.coefficients
    }

    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|&x| x == 0)
    }
}
