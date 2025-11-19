//! Conditional compilation utilities
//!
//! This module provides macros and utilities for conditional compilation
//! based on feature flags.

/// Macro to define a feature-gated module
///
/// # Example
///
/// ```ignore
/// use rustmath_features::feature_module;
///
/// feature_module! {
///     #[cfg(feature = "gmp")]
///     pub mod gmp_impl {
///         pub fn factorial(n: u64) -> u64 {
///             // GMP implementation
///             n
///         }
///     }
/// }
/// ```
#[macro_export]
macro_rules! feature_module {
    (
        #[cfg(feature = $feature:literal)]
        $vis:vis mod $name:ident $body:tt
    ) => {
        #[cfg(feature = $feature)]
        $vis mod $name $body
    };
}

/// Macro to conditionally compile a function based on features
///
/// # Example
///
/// ```ignore
/// use rustmath_features::feature_fn;
///
/// feature_fn! {
///     #[cfg(feature = "parallel")]
///     pub fn parallel_sum(data: &[i32]) -> i32 {
///         use rayon::prelude::*;
///         data.par_iter().sum()
///     }
/// }
/// ```
#[macro_export]
macro_rules! feature_fn {
    (
        #[cfg(feature = $feature:literal)]
        $vis:vis fn $name:ident($($args:tt)*) -> $ret:ty $body:block
    ) => {
        #[cfg(feature = $feature)]
        $vis fn $name($($args)*) -> $ret $body
    };
}

/// Macro to provide multiple implementations of the same function for different features
///
/// # Example
///
/// ```ignore
/// use rustmath_features::multi_impl;
///
/// multi_impl! {
///     pub fn factorial(n: u64) -> u64;
///
///     #[cfg(feature = "gmp")]
///     impl {
///         // GMP implementation
///         n
///     }
///
///     #[cfg(not(feature = "gmp"))]
///     impl {
///         // Pure Rust implementation
///         (1..=n).product()
///     }
/// }
/// ```
#[macro_export]
macro_rules! multi_impl {
    (
        $vis:vis fn $name:ident($($args:tt)*) -> $ret:ty;

        $(
            #[cfg($($cfg:tt)*)]
            impl $body:block
        )+
    ) => {
        $(
            #[cfg($($cfg)*)]
            $vis fn $name($($args)*) -> $ret $body
        )+
    };
}

/// Macro to import feature-specific dependencies
///
/// # Example
///
/// ```ignore
/// use rustmath_features::feature_use;
///
/// feature_use! {
///     #[cfg(feature = "gmp")]
///     use gmp::mpz::Mpz;
///
///     #[cfg(not(feature = "gmp"))]
///     use num_bigint::BigInt as Mpz;
/// }
/// ```
#[macro_export]
macro_rules! feature_use {
    (
        $(
            #[cfg($($cfg:tt)*)]
            use $($path:tt)*;
        )+
    ) => {
        $(
            #[cfg($($cfg)*)]
            use $($path)*;
        )+
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_conditional_macros() {
        // Test that macros compile
        feature_module! {
            #[cfg(feature = "std")]
            pub mod test_mod {
                pub fn test_fn() -> i32 { 42 }
            }
        }

        #[cfg(feature = "std")]
        {
            assert_eq!(test_mod::test_fn(), 42);
        }
    }
}
