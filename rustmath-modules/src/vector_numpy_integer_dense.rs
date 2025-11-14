//! vector_numpy_integer_dense implementation

// Vector implementation for sage.modules.vector_numpy_integer_dense

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_numpy_integer_dense<T> {
    data: Vec<T>,
}
