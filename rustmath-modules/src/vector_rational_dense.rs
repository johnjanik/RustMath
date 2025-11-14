//! vector_rational_dense implementation

// Vector implementation for sage.modules.vector_rational_dense

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_rational_dense<T> {
    data: Vec<T>,
}
