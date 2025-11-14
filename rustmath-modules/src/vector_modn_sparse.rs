//! vector_modn_sparse implementation

// Vector implementation for sage.modules.vector_modn_sparse

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_modn_sparse<T> {
    data: Vec<T>,
}
