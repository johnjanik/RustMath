//! vector_double_dense implementation

// Vector implementation for sage.modules.vector_double_dense

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_double_dense<T> {
    data: Vec<T>,
}
