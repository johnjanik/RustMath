//! vector_integer_sparse implementation

// Vector implementation for sage.modules.vector_integer_sparse

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_integer_sparse<T> {
    data: Vec<T>,
}
