//! Fallback implementation framework
//!
//! This module provides traits and utilities for implementing feature-dependent
//! operations with automatic fallbacks.

use crate::{Feature, has_feature};

/// Strategy for selecting implementations based on available features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Use the first available implementation from a priority list
    FirstAvailable,
    /// Use the fastest available implementation
    Fastest,
    /// Use the most accurate available implementation
    MostAccurate,
    /// Always use the fallback (for testing)
    AlwaysFallback,
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        Self::FirstAvailable
    }
}

/// Context for feature-dependent execution
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Strategy for selecting implementations
    pub strategy: SelectionStrategy,
    /// Preferred features in priority order
    pub preferred_features: Vec<Feature>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            strategy: SelectionStrategy::default(),
            preferred_features: vec![
                Feature::Gmp,
                Feature::Mpfr,
                Feature::Flint,
                Feature::Parallel,
                Feature::Simd,
            ],
        }
    }
}

impl ExecutionContext {
    /// Create a new execution context with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the selection strategy
    pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set preferred features
    pub fn with_features(mut self, features: Vec<Feature>) -> Self {
        self.preferred_features = features;
        self
    }

    /// Select the best available feature based on the strategy
    pub fn select_feature(&self) -> Option<Feature> {
        match self.strategy {
            SelectionStrategy::AlwaysFallback => None,
            SelectionStrategy::FirstAvailable => {
                self.preferred_features
                    .iter()
                    .copied()
                    .find(|&f| has_feature(f))
            }
            SelectionStrategy::Fastest => {
                // Priority: Native > SIMD > Parallel > External libs
                let priority = vec![
                    Feature::Native,
                    Feature::Simd,
                    Feature::Parallel,
                    Feature::Gmp,
                    Feature::Flint,
                ];

                priority.iter().copied().find(|&f| has_feature(f))
            }
            SelectionStrategy::MostAccurate => {
                // Priority: External libs (arbitrary precision) > Native
                let priority = vec![
                    Feature::Mpfr,
                    Feature::Gmp,
                    Feature::Flint,
                    Feature::Pari,
                ];

                priority.iter().copied().find(|&f| has_feature(f))
            }
        }
    }
}

/// Trait for operations with feature-dependent implementations and fallbacks
///
/// This trait enables implementing algorithms with multiple backends:
/// - Feature-specific optimized implementations
/// - Pure Rust fallback
/// - Automatic selection based on availability
pub trait FallbackOperation {
    /// Input type for the operation
    type Input;
    /// Output type for the operation
    type Output;
    /// Error type
    type Error;

    /// Execute with a specific feature
    fn execute_with_feature(
        &self,
        input: &Self::Input,
        feature: Feature,
    ) -> Result<Self::Output, Self::Error>;

    /// Fallback implementation (pure Rust)
    fn fallback(&self, input: &Self::Input) -> Result<Self::Output, Self::Error>;

    /// Execute with automatic feature selection
    fn execute(&self, input: &Self::Input, ctx: &ExecutionContext) -> Result<Self::Output, Self::Error> {
        // Try to use a feature-specific implementation
        if let Some(feature) = ctx.select_feature() {
            if let Ok(result) = self.execute_with_feature(input, feature) {
                return Ok(result);
            }
        }

        // Fall back to pure Rust implementation
        self.fallback(input)
    }
}

/// Builder pattern for configuring feature-dependent operations
pub struct OperationBuilder<T> {
    operation: T,
    context: ExecutionContext,
}

impl<T> OperationBuilder<T> {
    /// Create a new operation builder
    pub fn new(operation: T) -> Self {
        Self {
            operation,
            context: ExecutionContext::default(),
        }
    }

    /// Set the selection strategy
    pub fn strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.context.strategy = strategy;
        self
    }

    /// Set preferred features
    pub fn prefer(mut self, features: Vec<Feature>) -> Self {
        self.context.preferred_features = features;
        self
    }

    /// Build and return the operation with context
    pub fn build(self) -> (T, ExecutionContext) {
        (self.operation, self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_context() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.strategy, SelectionStrategy::FirstAvailable);
        assert!(!ctx.preferred_features.is_empty());
    }

    #[test]
    fn test_selection_strategy() {
        let ctx = ExecutionContext::new()
            .with_strategy(SelectionStrategy::AlwaysFallback);
        assert_eq!(ctx.strategy, SelectionStrategy::AlwaysFallback);
        assert_eq!(ctx.select_feature(), None);
    }

    #[test]
    fn test_operation_builder() {
        let op = 42i32;
        let builder = OperationBuilder::new(op)
            .strategy(SelectionStrategy::Fastest);

        let (operation, context) = builder.build();
        assert_eq!(operation, 42);
        assert_eq!(context.strategy, SelectionStrategy::Fastest);
    }
}
