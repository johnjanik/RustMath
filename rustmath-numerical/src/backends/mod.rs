//! Backends for numerical optimization and linear programming

pub mod generic_backend;
pub mod generic_sdp_backend;
pub mod cvxopt_backend;
pub mod cvxopt_sdp_backend;
pub mod glpk_backend;

pub use generic_backend::GenericBackend;
pub use generic_sdp_backend::GenericSDPBackend;
pub use cvxopt_backend::CVXOPTBackend;
pub use cvxopt_sdp_backend::CVXOPTSDPBackend;
pub use glpk_backend::GLPKBackend;
