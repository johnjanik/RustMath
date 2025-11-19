//! Integration tests for rustmath-features

use rustmath_features::{Feature, has_feature, enabled_features};
use rustmath_features::fallback::{
    FallbackOperation, ExecutionContext, SelectionStrategy
};

#[test]
fn test_feature_detection() {
    // std should be enabled by default
    #[cfg(feature = "std")]
    assert!(has_feature(Feature::Std));

    // All features should report consistently
    for &feature in Feature::all() {
        let runtime_enabled = has_feature(feature);
        let compile_time_enabled = feature.is_enabled();
        assert_eq!(
            runtime_enabled, compile_time_enabled,
            "Runtime and compile-time detection mismatch for {:?}",
            feature
        );
    }
}

#[test]
fn test_enabled_features_list() {
    let features = enabled_features();

    // Should not exceed total number of features
    assert!(features.len() <= Feature::all().len());

    // All returned features should actually be enabled
    for feature in features {
        assert!(has_feature(feature));
    }
}

#[test]
#[cfg(feature = "std")]
fn test_feature_summary() {
    let summary = rustmath_features::feature_summary();
    assert!(!summary.is_empty());

    // Should contain "feature" somewhere
    assert!(summary.to_lowercase().contains("feature"));
}

#[test]
fn test_capabilities_detection() {
    let caps = rustmath_features::capabilities();

    // Should have at least 1 core
    assert!(caps.num_cores >= 1);

    // SIMD detection should be consistent
    let has_any_simd = caps.has_avx || caps.has_avx2 || caps.has_avx512 || caps.has_neon;
    assert_eq!(caps.has_simd(), has_any_simd);
}

// Test operation for fallback tests
struct TestOperation;

impl FallbackOperation for TestOperation {
    type Input = i32;
    type Output = i32;
    type Error = ();

    fn execute_with_feature(
        &self,
        input: &Self::Input,
        _feature: Feature,
    ) -> Result<Self::Output, Self::Error> {
        Ok(input * 2)
    }

    fn fallback(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        Ok(input * 3)
    }
}

#[test]
fn test_fallback_operation() {
    let op = TestOperation;
    let input = 10;

    // Test with fallback strategy
    let ctx = ExecutionContext::new()
        .with_strategy(SelectionStrategy::AlwaysFallback);
    let result = op.execute(&input, &ctx).unwrap();
    assert_eq!(result, 30); // Should use fallback (input * 3)

    // Test with default strategy
    let ctx = ExecutionContext::default();
    let result = op.execute(&input, &ctx).unwrap();
    // Result depends on available features, but should be either 20 or 30
    assert!(result == 20 || result == 30);
}

#[test]
fn test_selection_strategies() {
    let ctx_first = ExecutionContext::new()
        .with_strategy(SelectionStrategy::FirstAvailable);
    assert_eq!(ctx_first.strategy, SelectionStrategy::FirstAvailable);

    let ctx_fastest = ExecutionContext::new()
        .with_strategy(SelectionStrategy::Fastest);
    assert_eq!(ctx_fastest.strategy, SelectionStrategy::Fastest);

    let ctx_accurate = ExecutionContext::new()
        .with_strategy(SelectionStrategy::MostAccurate);
    assert_eq!(ctx_accurate.strategy, SelectionStrategy::MostAccurate);

    let ctx_fallback = ExecutionContext::new()
        .with_strategy(SelectionStrategy::AlwaysFallback);
    assert_eq!(ctx_fallback.select_feature(), None);
}

#[test]
fn test_custom_feature_preferences() {
    let preferred = vec![Feature::Gmp, Feature::Parallel];
    let ctx = ExecutionContext::new()
        .with_features(preferred.clone());

    assert_eq!(ctx.preferred_features, preferred);
}

#[test]
fn test_feature_metadata() {
    for &feature in Feature::all() {
        // Name should not be empty
        assert!(!feature.name().is_empty());

        // Description should not be empty
        assert!(!feature.description().is_empty());

        // Name should be lowercase
        assert_eq!(feature.name(), feature.name().to_lowercase());
    }
}

#[test]
fn test_with_fallback_macro() {
    use rustmath_features::with_fallback;

    fn optimized() -> i32 {
        100
    }

    fn fallback_impl() -> i32 {
        200
    }

    // This should compile and execute
    let result = with_fallback!(
        Feature::Gmp => optimized(),
        fallback_impl()
    );

    // Result depends on whether GMP is enabled
    assert!(result == 100 || result == 200);
}

#[test]
fn test_operation_builder() {
    use rustmath_features::fallback::OperationBuilder;

    let op = TestOperation;
    let (operation, context) = OperationBuilder::new(op)
        .strategy(SelectionStrategy::Fastest)
        .prefer(vec![Feature::Simd, Feature::Parallel])
        .build();

    // Context should have the specified settings
    assert_eq!(context.strategy, SelectionStrategy::Fastest);
    assert_eq!(context.preferred_features, vec![Feature::Simd, Feature::Parallel]);

    // Operation should execute correctly
    let result = operation.execute(&5, &context).unwrap();
    assert!(result == 10 || result == 15);
}

#[test]
fn test_feature_all_unique() {
    use std::collections::HashSet;

    let features: HashSet<_> = Feature::all().iter().collect();
    assert_eq!(features.len(), Feature::all().len(), "Features should be unique");
}

#[test]
fn test_feature_equality() {
    assert_eq!(Feature::Gmp, Feature::Gmp);
    assert_ne!(Feature::Gmp, Feature::Mpfr);
}
