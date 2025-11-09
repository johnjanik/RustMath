//! Cremona elliptic curve database interface
//!
//! Provides access to John Cremona's database of elliptic curves over Q.
//! The database contains all elliptic curves of bounded conductor with
//! their Weierstrass equations, ranks, torsion, and generators.
//!
//! # Example
//!
//! ```
//! use rustmath_databases::cremona::{CremonaDatabase, CurveLabel};
//!
//! let db = CremonaDatabase::new();
//!
//! // Look up curve 11a1 (first curve of conductor 11)
//! if let Some(curve) = db.lookup_curve("11a1") {
//!     println!("Curve: {}", curve.label);
//!     println!("Equation: y^2 + y = x^3 - x^2 - 10x - 20");
//!     println!("Rank: {}", curve.rank);
//!     println!("Torsion: {}", curve.torsion_order);
//! }
//! ```

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;

/// Error type for Cremona database operations
#[derive(Debug)]
pub enum CremonaError {
    /// File I/O error
    IoError(String),
    /// Parse error
    ParseError(String),
    /// Curve not found
    NotFound(String),
    /// Invalid curve label
    InvalidLabel(String),
}

impl fmt::Display for CremonaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CremonaError::IoError(msg) => write!(f, "I/O error: {}", msg),
            CremonaError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            CremonaError::NotFound(msg) => write!(f, "Not found: {}", msg),
            CremonaError::InvalidLabel(msg) => write!(f, "Invalid label: {}", msg),
        }
    }
}

impl Error for CremonaError {}

/// Result type for Cremona operations
pub type Result<T> = std::result::Result<T, CremonaError>;

/// Represents a curve label in Cremona notation
///
/// Format: <conductor><isogeny_class><curve_number>
/// Examples: "11a1", "37a1", "389a1", "5077a1"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CurveLabel {
    /// Conductor
    pub conductor: u64,
    /// Isogeny class (e.g., 'a', 'b', 'c')
    pub isogeny_class: String,
    /// Curve number within the isogeny class
    pub curve_number: u32,
}

impl CurveLabel {
    /// Parse a curve label from string
    ///
    /// # Arguments
    ///
    /// * `label` - Curve label string (e.g., "11a1", "389b2")
    ///
    /// # Returns
    ///
    /// Parsed CurveLabel or error
    pub fn parse(label: &str) -> Result<Self> {
        // Find where the letters start
        let mut letter_start = 0;
        for (i, c) in label.chars().enumerate() {
            if c.is_alphabetic() {
                letter_start = i;
                break;
            }
        }

        if letter_start == 0 {
            return Err(CremonaError::InvalidLabel(
                format!("No conductor found in label: {}", label)
            ));
        }

        // Extract conductor (numeric part)
        let conductor_str = &label[..letter_start];
        let conductor = conductor_str.parse::<u64>()
            .map_err(|_| CremonaError::InvalidLabel(
                format!("Invalid conductor: {}", conductor_str)
            ))?;

        // Find where the final number starts
        let chars: Vec<char> = label.chars().collect();
        let mut number_start = chars.len();
        for i in (0..chars.len()).rev() {
            if !chars[i].is_numeric() {
                number_start = i + 1;
                break;
            }
        }

        // Extract isogeny class (letter part)
        let isogeny_class = label[letter_start..number_start].to_string();
        if isogeny_class.is_empty() {
            return Err(CremonaError::InvalidLabel(
                format!("No isogeny class in label: {}", label)
            ));
        }

        // Extract curve number
        let number_str = &label[number_start..];
        let curve_number = if number_str.is_empty() {
            1  // Default to 1 if no number specified
        } else {
            number_str.parse::<u32>()
                .map_err(|_| CremonaError::InvalidLabel(
                    format!("Invalid curve number: {}", number_str)
                ))?
        };

        Ok(CurveLabel {
            conductor,
            isogeny_class,
            curve_number,
        })
    }

    /// Get the full label as a string
    pub fn to_string(&self) -> String {
        format!("{}{}{}", self.conductor, self.isogeny_class, self.curve_number)
    }
}

impl fmt::Display for CurveLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}", self.conductor, self.isogeny_class, self.curve_number)
    }
}

/// Weierstrass equation coefficients
///
/// Represents the equation: y^2 + a1*xy + a3*y = x^3 + a2*x^2 + a4*x + a6
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeierstrassEquation {
    pub a1: i64,
    pub a2: i64,
    pub a3: i64,
    pub a4: i64,
    pub a6: i64,
}

impl WeierstrassEquation {
    /// Create a new Weierstrass equation
    pub fn new(a1: i64, a2: i64, a3: i64, a4: i64, a6: i64) -> Self {
        WeierstrassEquation { a1, a2, a3, a4, a6 }
    }

    /// Parse from coefficient array [a1, a2, a3, a4, a6]
    pub fn from_array(coeffs: &[i64; 5]) -> Self {
        WeierstrassEquation {
            a1: coeffs[0],
            a2: coeffs[1],
            a3: coeffs[2],
            a4: coeffs[3],
            a6: coeffs[4],
        }
    }

    /// Get coefficients as array
    pub fn to_array(&self) -> [i64; 5] {
        [self.a1, self.a2, self.a3, self.a4, self.a6]
    }
}

impl fmt::Display for WeierstrassEquation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format: y^2 + a1*xy + a3*y = x^3 + a2*x^2 + a4*x + a6
        write!(f, "y^2")?;

        if self.a1 != 0 {
            if self.a1 == 1 {
                write!(f, " + xy")?;
            } else if self.a1 == -1 {
                write!(f, " - xy")?;
            } else if self.a1 > 0 {
                write!(f, " + {}xy", self.a1)?;
            } else {
                write!(f, " - {}xy", -self.a1)?;
            }
        }

        if self.a3 != 0 {
            if self.a3 == 1 {
                write!(f, " + y")?;
            } else if self.a3 == -1 {
                write!(f, " - y")?;
            } else if self.a3 > 0 {
                write!(f, " + {}y", self.a3)?;
            } else {
                write!(f, " - {}y", -self.a3)?;
            }
        }

        write!(f, " = x^3")?;

        if self.a2 != 0 {
            if self.a2 == 1 {
                write!(f, " + x^2")?;
            } else if self.a2 == -1 {
                write!(f, " - x^2")?;
            } else if self.a2 > 0 {
                write!(f, " + {}x^2", self.a2)?;
            } else {
                write!(f, " - {}x^2", -self.a2)?;
            }
        }

        if self.a4 != 0 {
            if self.a4 == 1 {
                write!(f, " + x")?;
            } else if self.a4 == -1 {
                write!(f, " - x")?;
            } else if self.a4 > 0 {
                write!(f, " + {}x", self.a4)?;
            } else {
                write!(f, " - {}x", -self.a4)?;
            }
        }

        if self.a6 != 0 {
            if self.a6 > 0 {
                write!(f, " + {}", self.a6)?;
            } else {
                write!(f, " - {}", -self.a6)?;
            }
        }

        Ok(())
    }
}

/// Represents a point on an elliptic curve
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    /// x-coordinate (numerator/denominator)
    pub x: (i64, i64),
    /// y-coordinate (numerator/denominator)
    pub y: (i64, i64),
}

impl Point {
    /// Create a new point
    pub fn new(x_num: i64, x_den: i64, y_num: i64, y_den: i64) -> Self {
        Point {
            x: (x_num, x_den),
            y: (y_num, y_den),
        }
    }

    /// Create a point from integers (denominator = 1)
    pub fn from_integers(x: i64, y: i64) -> Self {
        Point {
            x: (x, 1),
            y: (y, 1),
        }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x_str = if self.x.1 == 1 {
            format!("{}", self.x.0)
        } else {
            format!("{}/{}", self.x.0, self.x.1)
        };

        let y_str = if self.y.1 == 1 {
            format!("{}", self.y.0)
        } else {
            format!("{}/{}", self.y.0, self.y.1)
        };

        write!(f, "({}, {})", x_str, y_str)
    }
}

/// Represents an elliptic curve from the Cremona database
#[derive(Debug, Clone)]
pub struct EllipticCurve {
    /// Curve label (e.g., "11a1")
    pub label: CurveLabel,

    /// Conductor
    pub conductor: u64,

    /// Weierstrass equation coefficients
    pub equation: WeierstrassEquation,

    /// Rank (dimension of the Mordell-Weil group modulo torsion)
    pub rank: u32,

    /// Torsion order
    pub torsion_order: u32,

    /// Generators of the Mordell-Weil group (if rank > 0)
    pub generators: Vec<Point>,

    /// Modular degree (if available)
    pub modular_degree: Option<u64>,

    /// Real period (if available)
    pub real_period: Option<f64>,

    /// Regulator (if available)
    pub regulator: Option<f64>,

    /// Order of Sha (Shafarevich-Tate group, if known)
    pub sha_order: Option<u64>,
}

impl EllipticCurve {
    /// Create a minimal curve entry with just label and equation
    pub fn new(label: CurveLabel, equation: WeierstrassEquation) -> Self {
        let conductor = label.conductor;
        EllipticCurve {
            label,
            conductor,
            equation,
            rank: 0,
            torsion_order: 1,
            generators: Vec::new(),
            modular_degree: None,
            real_period: None,
            regulator: None,
            sha_order: None,
        }
    }

    /// Get the j-invariant (placeholder - would need actual computation)
    pub fn j_invariant(&self) -> Option<String> {
        // Actual computation requires more complex arithmetic
        None
    }

    /// Check if the curve is in the database
    pub fn is_known(&self) -> bool {
        true  // If we have it, it's known
    }
}

impl fmt::Display for EllipticCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Elliptic Curve {}", self.label)?;
        writeln!(f, "Conductor: {}", self.conductor)?;
        writeln!(f, "Equation: {}", self.equation)?;
        writeln!(f, "Rank: {}", self.rank)?;
        writeln!(f, "Torsion: {}", self.torsion_order)?;
        if !self.generators.is_empty() {
            writeln!(f, "Generators:")?;
            for gen in &self.generators {
                writeln!(f, "  {}", gen)?;
            }
        }
        Ok(())
    }
}

/// Client for accessing the Cremona elliptic curve database
pub struct CremonaDatabase {
    /// In-memory cache of curves
    curves: HashMap<String, EllipticCurve>,
    /// Path to data directory (if using local files)
    data_path: Option<String>,
}

impl CremonaDatabase {
    /// Create a new Cremona database client with built-in curves
    pub fn new() -> Self {
        let mut db = CremonaDatabase {
            curves: HashMap::new(),
            data_path: None,
        };

        // Load some well-known curves
        db.load_builtin_curves();
        db
    }

    /// Create a client that loads data from a directory
    pub fn from_directory<P: AsRef<Path>>(data_path: P) -> Result<Self> {
        let mut db = CremonaDatabase {
            curves: HashMap::new(),
            data_path: Some(data_path.as_ref().to_string_lossy().to_string()),
        };

        db.load_builtin_curves();
        // Would load from files here
        Ok(db)
    }

    /// Load built-in well-known curves
    fn load_builtin_curves(&mut self) {
        // Curve 11a1: y^2 + y = x^3 - x^2 - 10x - 20
        // Rank 0, torsion order 5
        self.add_builtin("11a1", [0, -1, 1, -10, -20], 0, 5, vec![]);

        // Curve 11a2: y^2 + y = x^3 - x^2 - 7820x - 263580
        self.add_builtin("11a2", [0, -1, 1, -7820, -263580], 0, 5, vec![]);

        // Curve 11a3: y^2 + y = x^3 - x^2
        self.add_builtin("11a3", [0, -1, 1, 0, 0], 0, 5, vec![]);

        // Curve 37a1: y^2 + y = x^3 - x
        // Rank 1, torsion order 1
        self.add_builtin("37a1", [0, 0, 1, -1, 0], 1, 1,
                        vec![Point::from_integers(0, 0)]);

        // Curve 37b1: y^2 + y = x^3 + x^2 - 23x - 50
        self.add_builtin("37b1", [0, 1, 1, -23, -50], 0, 3, vec![]);

        // Curve 389a1: y^2 + y = x^3 + x^2 - 2x
        // Rank 2, torsion order 1
        self.add_builtin("389a1", [0, 1, 1, -2, 0], 2, 1,
                        vec![Point::from_integers(-1, 1),
                             Point::from_integers(0, 0)]);

        // Curve 5077a1: famous rank 3 curve
        // y^2 + y = x^3 - 7x + 6
        self.add_builtin("5077a1", [0, 0, 1, -7, 6], 3, 1, vec![]);

        // Some curves with torsion
        // Curve 14a1: Z/6Z torsion
        self.add_builtin("14a1", [0, 1, 1, 4, -6], 0, 6, vec![]);

        // Curve 15a1: Z/8Z torsion
        self.add_builtin("15a1", [0, 1, 1, -10, -10], 0, 8, vec![]);

        // Curve 17a1: y^2 + xy + y = x^3 - x^2 - x - 14
        self.add_builtin("17a1", [1, -1, 1, -1, -14], 0, 4, vec![]);

        // Curve 19a1: y^2 + y = x^3 + x^2 - 9x - 15
        self.add_builtin("19a1", [0, 1, 1, -9, -15], 0, 3, vec![]);

        // Curve 20a1: y^2 = x^3 + x^2 - x
        self.add_builtin("20a1", [0, 1, 0, -1, 0], 0, 2, vec![]);
    }

    /// Add a built-in curve
    fn add_builtin(&mut self, label: &str, coeffs: [i64; 5], rank: u32,
                   torsion: u32, generators: Vec<Point>) {
        let curve_label = CurveLabel::parse(label).unwrap();
        let equation = WeierstrassEquation::from_array(&coeffs);

        let mut curve = EllipticCurve::new(curve_label, equation);
        curve.rank = rank;
        curve.torsion_order = torsion;
        curve.generators = generators;

        self.curves.insert(label.to_string(), curve);
    }

    /// Look up a curve by its label
    ///
    /// # Arguments
    ///
    /// * `label` - Curve label string (e.g., "11a1")
    ///
    /// # Returns
    ///
    /// The elliptic curve if found in the database
    pub fn lookup_curve(&self, label: &str) -> Option<&EllipticCurve> {
        self.curves.get(label)
    }

    /// Get all curves of a given conductor
    ///
    /// # Arguments
    ///
    /// * `conductor` - The conductor value
    ///
    /// # Returns
    ///
    /// Vector of all curves with this conductor
    pub fn curves_of_conductor(&self, conductor: u64) -> Vec<&EllipticCurve> {
        self.curves
            .values()
            .filter(|c| c.conductor == conductor)
            .collect()
    }

    /// Get all curves in a given isogeny class
    ///
    /// # Arguments
    ///
    /// * `conductor` - The conductor value
    /// * `isogeny_class` - The isogeny class (e.g., "a", "b")
    ///
    /// # Returns
    ///
    /// Vector of all curves in this isogeny class
    pub fn curves_in_isogeny_class(&self, conductor: u64, isogeny_class: &str)
        -> Vec<&EllipticCurve> {
        self.curves
            .values()
            .filter(|c| c.conductor == conductor &&
                       c.label.isogeny_class == isogeny_class)
            .collect()
    }

    /// Find curves with a given rank
    ///
    /// # Arguments
    ///
    /// * `rank` - The desired rank
    ///
    /// # Returns
    ///
    /// Vector of curves with the specified rank
    pub fn curves_of_rank(&self, rank: u32) -> Vec<&EllipticCurve> {
        self.curves
            .values()
            .filter(|c| c.rank == rank)
            .collect()
    }

    /// Find curves with a given torsion order
    pub fn curves_with_torsion(&self, torsion: u32) -> Vec<&EllipticCurve> {
        self.curves
            .values()
            .filter(|c| c.torsion_order == torsion)
            .collect()
    }

    /// Add a curve to the database
    pub fn add_curve(&mut self, curve: EllipticCurve) {
        let label = curve.label.to_string();
        self.curves.insert(label, curve);
    }

    /// Get the total number of curves in the database
    pub fn curve_count(&self) -> usize {
        self.curves.len()
    }

    /// Get a sorted list of all conductors in the database
    pub fn all_conductors(&self) -> Vec<u64> {
        let mut conductors: Vec<u64> = self.curves
            .values()
            .map(|c| c.conductor)
            .collect();
        conductors.sort_unstable();
        conductors.dedup();
        conductors
    }
}

impl Default for CremonaDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_curve_label() {
        let label = CurveLabel::parse("11a1").unwrap();
        assert_eq!(label.conductor, 11);
        assert_eq!(label.isogeny_class, "a");
        assert_eq!(label.curve_number, 1);

        let label2 = CurveLabel::parse("389b2").unwrap();
        assert_eq!(label2.conductor, 389);
        assert_eq!(label2.isogeny_class, "b");
        assert_eq!(label2.curve_number, 2);

        let label3 = CurveLabel::parse("5077a1").unwrap();
        assert_eq!(label3.conductor, 5077);
        assert_eq!(label3.isogeny_class, "a");
        assert_eq!(label3.curve_number, 1);
    }

    #[test]
    fn test_curve_label_display() {
        let label = CurveLabel::parse("11a1").unwrap();
        assert_eq!(format!("{}", label), "11a1");
    }

    #[test]
    fn test_weierstrass_equation() {
        // y^2 + y = x^3 - x^2 - 10x - 20
        let eq = WeierstrassEquation::new(0, -1, 1, -10, -20);
        let display = format!("{}", eq);
        assert!(display.contains("y^2"));
        assert!(display.contains("x^3"));
    }

    #[test]
    fn test_builtin_curves() {
        let db = CremonaDatabase::new();

        // Test curve 11a1
        let curve = db.lookup_curve("11a1").unwrap();
        assert_eq!(curve.conductor, 11);
        assert_eq!(curve.rank, 0);
        assert_eq!(curve.torsion_order, 5);

        // Test curve 37a1 (rank 1)
        let curve37 = db.lookup_curve("37a1").unwrap();
        assert_eq!(curve37.rank, 1);
        assert_eq!(curve37.generators.len(), 1);

        // Test curve 389a1 (rank 2)
        let curve389 = db.lookup_curve("389a1").unwrap();
        assert_eq!(curve389.rank, 2);
        assert_eq!(curve389.generators.len(), 2);
    }

    #[test]
    fn test_curves_of_conductor() {
        let db = CremonaDatabase::new();
        let curves = db.curves_of_conductor(11);
        assert_eq!(curves.len(), 3);  // 11a1, 11a2, 11a3
    }

    #[test]
    fn test_curves_in_isogeny_class() {
        let db = CremonaDatabase::new();
        let curves = db.curves_in_isogeny_class(11, "a");
        assert_eq!(curves.len(), 3);

        for curve in curves {
            assert_eq!(curve.label.isogeny_class, "a");
        }
    }

    #[test]
    fn test_curves_of_rank() {
        let db = CremonaDatabase::new();

        let rank0 = db.curves_of_rank(0);
        assert!(!rank0.is_empty());

        let rank1 = db.curves_of_rank(1);
        assert!(!rank1.is_empty());

        let rank2 = db.curves_of_rank(2);
        assert!(!rank2.is_empty());

        let rank3 = db.curves_of_rank(3);
        assert!(!rank3.is_empty());
    }

    #[test]
    fn test_curves_with_torsion() {
        let db = CremonaDatabase::new();

        let torsion1 = db.curves_with_torsion(1);
        assert!(!torsion1.is_empty());

        let torsion5 = db.curves_with_torsion(5);
        assert!(!torsion5.is_empty());
    }

    #[test]
    fn test_all_conductors() {
        let db = CremonaDatabase::new();
        let conductors = db.all_conductors();
        assert!(conductors.contains(&11));
        assert!(conductors.contains(&37));
        assert!(conductors.contains(&389));
        assert!(conductors.contains(&5077));

        // Should be sorted
        for i in 1..conductors.len() {
            assert!(conductors[i] > conductors[i - 1]);
        }
    }

    #[test]
    fn test_point_display() {
        let p = Point::from_integers(0, 0);
        assert_eq!(format!("{}", p), "(0, 0)");

        let p2 = Point::new(1, 2, 3, 4);
        assert_eq!(format!("{}", p2), "(1/2, 3/4)");
    }

    #[test]
    fn test_curve_display() {
        let db = CremonaDatabase::new();
        let curve = db.lookup_curve("11a1").unwrap();
        let display = format!("{}", curve);
        assert!(display.contains("11a1"));
        assert!(display.contains("Conductor: 11"));
        assert!(display.contains("Rank: 0"));
    }
}
