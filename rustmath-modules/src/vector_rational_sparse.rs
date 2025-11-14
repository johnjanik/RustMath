//! vector_rational_sparse implementation

// Vector implementation for sage.modules.vector_rational_sparse

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_rational_sparse<T> {
    data: Vec<T>,
}
