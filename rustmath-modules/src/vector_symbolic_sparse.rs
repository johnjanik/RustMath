//! vector_symbolic_sparse implementation

// Vector implementation for sage.modules.vector_symbolic_sparse

#[derive(Clone, Debug, PartialEq)]
pub struct Vector_symbolic_sparse<T> {
    data: Vec<T>,
}
