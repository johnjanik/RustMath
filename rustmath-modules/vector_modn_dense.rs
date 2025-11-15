//! modn_dense vector implementation

#[derive(Clone, Debug, PartialEq)]
pub struct VectorModn_dense {
    dimension: usize,
    data: Vec<f64>, // Placeholder - would use appropriate type
}

impl VectorModn_dense {
    pub fn new(dimension: usize) -> Self {
        Self { dimension, data: vec![0.0; dimension] }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn get(&self, index: usize) -> f64 {
        self.data.get(index).copied().unwrap_or(0.0)
    }

    pub fn set(&mut self, index: usize, value: f64) {
        if index < self.dimension {
            self.data[index] = value;
        }
    }
}
