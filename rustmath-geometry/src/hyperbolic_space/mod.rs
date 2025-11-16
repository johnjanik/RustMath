//! Hyperbolic space and geometry
//!
//! This module provides comprehensive support for hyperbolic geometry through
//! multiple coordinate models:
//! - **UHP** (Upper Half Plane): Points z = x + iy with y > 0
//! - **PD** (Poincaré Disk): Points inside the unit disk
//! - **KM** (Klein Model): Projective disk model
//! - **HM** (Hyperboloid Model): Points on a hyperboloid in Minkowski space
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperbolic_space::hyperbolic_point::HyperbolicPointUHP;
//! use rustmath_geometry::hyperbolic_space::hyperbolic_interface::{HyperbolicSpace, ModelType};
//!
//! // Create a point in the upper half plane
//! let point = HyperbolicPointUHP::new(1.0, 2.0);
//!
//! // Create hyperbolic space
//! let space = HyperbolicSpace::new(ModelType::UHP);
//! assert_eq!(space.dimension(), 2);
//! ```

pub mod hyperbolic_point;
pub mod hyperbolic_model;
pub mod hyperbolic_geodesic;
pub mod hyperbolic_isometry;
pub mod hyperbolic_interface;

pub use hyperbolic_point::{HyperbolicPoint, HyperbolicPointUHP, HyperbolicPointPD};
pub use hyperbolic_model::{
    HyperbolicModel, HyperbolicModelUHP, HyperbolicModelPD,
    HyperbolicModelKM, HyperbolicModelHM,
};
pub use hyperbolic_geodesic::{
    HyperbolicGeodesic, HyperbolicGeodesicUHP, HyperbolicGeodesicPD,
    HyperbolicGeodesicKM, HyperbolicGeodesicHM,
};
pub use hyperbolic_isometry::{
    HyperbolicIsometry, HyperbolicIsometryUHP, HyperbolicIsometryPD,
    HyperbolicIsometryKM, moebius_transform,
};
pub use hyperbolic_interface::{
    HyperbolicSpace, HyperbolicPlane, HyperbolicModels,
    ParentMethods, ModelType,
};
