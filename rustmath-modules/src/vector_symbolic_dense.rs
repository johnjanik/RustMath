//! vector_symbolic_dense implementation

// Vector implementation for sage.modules.vector_symbolic_dense

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_symbolic_dense<T> {
    data: Vec<T>,
}
