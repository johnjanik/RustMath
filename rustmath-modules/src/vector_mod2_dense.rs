//! vector_mod2_dense implementation

// Vector implementation for sage.modules.vector_mod2_dense

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_mod2_dense<T> {
    data: Vec<T>,
}
