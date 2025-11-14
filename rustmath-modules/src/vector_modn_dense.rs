//! vector_modn_dense implementation

// Vector implementation for sage.modules.vector_modn_dense

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_modn_dense<T> {
    data: Vec<T>,
}
