//! vector_real_double_dense implementation

// Vector implementation for sage.modules.vector_real_double_dense

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_real_double_dense<T> {
    data: Vec<T>,
}
