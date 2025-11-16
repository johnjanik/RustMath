//! Voronoi diagram computation using brute-force half-plane intersections
//!
//! This module implements Voronoi diagrams using a brute-force approach:
//! 1. For each site, compute perpendicular bisectors to all other sites
//! 2. Each bisector defines a half-plane (points closer to this site)
//! 3. The Voronoi cell is the intersection of all such half-planes
//! 4. Uses Sutherland-Hodgman algorithm for polygon clipping
//!
//! Time complexity: O(n³) where n is the number of sites
//! This is suitable for small numbers of sites (n < 100)

use crate::point::Point2D;

/// A Voronoi cell represented as a convex polygon
#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCell {
    /// The site (generator point) for this cell
    pub site: Point2D,
    /// The vertices of the cell polygon in counter-clockwise order
    pub vertices: Vec<Point2D>,
}

impl VoronoiCell {
    /// Create a new Voronoi cell
    pub fn new(site: Point2D, vertices: Vec<Point2D>) -> Self {
        VoronoiCell { site, vertices }
    }

    /// Calculate the area of the cell using the shoelace formula
    pub fn area(&self) -> f64 {
        if self.vertices.len() < 3 {
            return 0.0;
        }

        let n = self.vertices.len();
        let mut area = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }

        (area / 2.0).abs()
    }

    /// Calculate the perimeter of the cell
    pub fn perimeter(&self) -> f64 {
        if self.vertices.len() < 2 {
            return 0.0;
        }

        let n = self.vertices.len();
        let mut perim = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            perim += self.vertices[i].distance(&self.vertices[j]);
        }

        perim
    }

    /// Check if a point is inside this cell
    pub fn contains(&self, p: &Point2D) -> bool {
        if self.vertices.len() < 3 {
            return false;
        }

        let n = self.vertices.len();
        let mut inside = false;
        let mut j = n - 1;

        for i in 0..n {
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];

            if ((vi.y > p.y) != (vj.y > p.y)) &&
               (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x) {
                inside = !inside;
            }
            j = i;
        }

        inside
    }
}

/// A complete Voronoi diagram
#[derive(Debug, Clone)]
pub struct VoronoiDiagram {
    /// The Voronoi cells, one for each site
    pub cells: Vec<VoronoiCell>,
    /// The original input points
    points: Vec<Point2D>,
    /// Ambient dimension (always 2 for this implementation)
    ambient_dim: usize,
}

impl VoronoiDiagram {
    /// Create a new Voronoi diagram
    ///
    /// This is typically called by `voronoi_brute_force` rather than directly.
    pub fn new(cells: Vec<VoronoiCell>, points: Vec<Point2D>) -> Self {
        Self {
            cells,
            ambient_dim: 2,
            points,
        }
    }

    /// Get the number of cells in the diagram
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Get the input points
    ///
    /// Returns the original generator points used to create this diagram.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::{Point2D, voronoi::voronoi_brute_force};
    ///
    /// let sites = vec![
    ///     Point2D::new(0.0, 0.0),
    ///     Point2D::new(1.0, 0.0),
    /// ];
    ///
    /// let diagram = voronoi_brute_force(&sites, None);
    /// let points = diagram.points();
    /// assert_eq!(points.len(), 2);
    /// ```
    pub fn points(&self) -> &[Point2D] {
        &self.points
    }

    /// Get the ambient dimension
    ///
    /// Returns the dimension of the space in which the diagram lives (always 2).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::{Point2D, voronoi::voronoi_brute_force};
    ///
    /// let sites = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0)];
    /// let diagram = voronoi_brute_force(&sites, None);
    /// assert_eq!(diagram.ambient_dim(), 2);
    /// ```
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Get the Voronoi regions
    ///
    /// Returns a vector of cells, where each cell represents a Voronoi region.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::{Point2D, voronoi::voronoi_brute_force};
    ///
    /// let sites = vec![
    ///     Point2D::new(0.0, 0.0),
    ///     Point2D::new(1.0, 0.0),
    /// ];
    ///
    /// let diagram = voronoi_brute_force(&sites, None);
    /// let regions = diagram.regions();
    /// assert_eq!(regions.len(), 2);
    /// ```
    pub fn regions(&self) -> &[VoronoiCell] {
        &self.cells
    }

    /// Get a string representing the base ring
    ///
    /// In SageMath, this would return QQ, RDF, or AA.
    /// For this implementation, we use f64, so this returns "RDF" (Real Double Field).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::{Point2D, voronoi::voronoi_brute_force};
    ///
    /// let sites = vec![Point2D::new(0.0, 0.0)];
    /// let diagram = voronoi_brute_force(&sites, None);
    /// assert_eq!(diagram.base_ring(), "RDF");
    /// ```
    pub fn base_ring(&self) -> &str {
        "RDF" // Real Double Field (f64)
    }

    /// Find the cell containing a given point
    /// Returns the index of the cell, or None if the point is outside all cells
    pub fn find_cell(&self, p: &Point2D) -> Option<usize> {
        for (i, cell) in self.cells.iter().enumerate() {
            if cell.contains(p) {
                return Some(i);
            }
        }
        None
    }

    /// Get the nearest site to a given point
    pub fn nearest_site(&self, p: &Point2D) -> Option<&Point2D> {
        if self.cells.is_empty() {
            return None;
        }

        let mut min_dist_sq = f64::INFINITY;
        let mut nearest = None;

        for cell in &self.cells {
            let dist_sq = cell.site.distance_squared(p);
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                nearest = Some(&cell.site);
            }
        }

        nearest
    }

    /// Validate that each point lies within its corresponding region
    ///
    /// This is a consistency check that verifies the diagram was computed correctly.
    ///
    /// # Returns
    ///
    /// `true` if each site is contained in its own Voronoi cell, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::{Point2D, voronoi::voronoi_brute_force};
    ///
    /// let sites = vec![
    ///     Point2D::new(0.0, 0.0),
    ///     Point2D::new(2.0, 0.0),
    /// ];
    ///
    /// let diagram = voronoi_brute_force(&sites, None);
    /// assert!(diagram.are_points_in_regions());
    /// ```
    pub fn are_points_in_regions(&self) -> bool {
        for (i, cell) in self.cells.iter().enumerate() {
            if i < self.points.len() {
                if !cell.contains(&self.points[i]) {
                    return false;
                }
            }
        }
        true
    }

    /// Generate a string representation of the diagram
    ///
    /// Returns a human-readable description like:
    /// "The Voronoi diagram of 3 points of dimension 2 in the Real Double Field"
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::{Point2D, voronoi::voronoi_brute_force};
    ///
    /// let sites = vec![
    ///     Point2D::new(0.0, 0.0),
    ///     Point2D::new(1.0, 0.0),
    ///     Point2D::new(0.5, 1.0),
    /// ];
    ///
    /// let diagram = voronoi_brute_force(&sites, None);
    /// let desc = diagram.description();
    /// assert!(desc.contains("3 points"));
    /// assert!(desc.contains("dimension 2"));
    /// ```
    pub fn description(&self) -> String {
        format!(
            "The Voronoi diagram of {} points of dimension {} in the {}",
            self.points.len(),
            self.ambient_dim,
            self.base_ring()
        )
    }
}

/// Clip a polygon by a half-plane using the Sutherland-Hodgman algorithm
///
/// The half-plane is defined by the line A*x + B*y + C = 0,
/// where the "inside" region is A*x + B*y + C < 0
///
/// # Arguments
/// * `polygon` - The polygon to clip (vertices in counter-clockwise order)
/// * `a` - Coefficient A of the half-plane equation
/// * `b` - Coefficient B of the half-plane equation
/// * `c` - Coefficient C of the half-plane equation
///
/// # Returns
/// The clipped polygon vertices
fn clip_polygon_by_half_plane(polygon: &[Point2D], a: f64, b: f64, c: f64) -> Vec<Point2D> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::new();
    let n = polygon.len();

    for i in 0..n {
        let v0 = polygon[i];
        let v1 = polygon[(i + 1) % n];

        let d0 = a * v0.x + b * v0.y + c;
        let d1 = a * v1.x + b * v1.y + c;

        let inside_v0 = d0 < 0.0;
        let inside_v1 = d1 < 0.0;

        if inside_v0 && inside_v1 {
            // Both inside: add v1
            output.push(v1);
        } else if inside_v0 && !inside_v1 {
            // From inside to outside: add intersection
            let denom = a * (v1.x - v0.x) + b * (v1.y - v0.y);
            if denom.abs() > 1e-10 {
                let t = -d0 / denom;
                let intersection = Point2D::new(
                    v0.x + t * (v1.x - v0.x),
                    v0.y + t * (v1.y - v0.y),
                );
                output.push(intersection);
            }
        } else if !inside_v0 && inside_v1 {
            // From outside to inside: add intersection and v1
            let denom = a * (v1.x - v0.x) + b * (v1.y - v0.y);
            if denom.abs() > 1e-10 {
                let t = -d0 / denom;
                let intersection = Point2D::new(
                    v0.x + t * (v1.x - v0.x),
                    v0.y + t * (v1.y - v0.y),
                );
                output.push(intersection);
            }
            output.push(v1);
        }
        // Both outside: add nothing
    }

    output
}

/// Compute the Voronoi diagram using brute-force half-plane intersections
///
/// This algorithm:
/// 1. For each site, starts with a bounding box
/// 2. Clips the box against half-planes defined by perpendicular bisectors to other sites
/// 3. The result is a convex polygon representing the Voronoi cell
///
/// # Arguments
/// * `sites` - The generator points for the Voronoi diagram
/// * `bounds` - Optional bounding box (min, max). If None, computed from sites with margin
///
/// # Returns
/// A VoronoiDiagram containing all cells
///
/// # Example
/// ```
/// use rustmath_geometry::{Point2D, voronoi::voronoi_brute_force};
///
/// let sites = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.5, 1.0),
/// ];
///
/// let diagram = voronoi_brute_force(&sites, None);
/// assert_eq!(diagram.num_cells(), 3);
/// ```
pub fn voronoi_brute_force(sites: &[Point2D], bounds: Option<(Point2D, Point2D)>) -> VoronoiDiagram {
    if sites.is_empty() {
        return VoronoiDiagram::new(Vec::new(), Vec::new());
    }

    // Compute bounding box
    let (min_bound, max_bound) = if let Some((min, max)) = bounds {
        (min, max)
    } else {
        // Compute from sites with a margin
        let mut min_x = sites[0].x;
        let mut max_x = sites[0].x;
        let mut min_y = sites[0].y;
        let mut max_y = sites[0].y;

        for site in sites {
            min_x = min_x.min(site.x);
            max_x = max_x.max(site.x);
            min_y = min_y.min(site.y);
            max_y = max_y.max(site.y);
        }

        let width = max_x - min_x;
        let height = max_y - min_y;
        let margin = (width.max(height) * 0.5).max(10.0);

        (
            Point2D::new(min_x - margin, min_y - margin),
            Point2D::new(max_x + margin, max_y + margin),
        )
    };

    // Initial bounding box polygon (counter-clockwise)
    let bounding_poly = vec![
        Point2D::new(min_bound.x, min_bound.y),
        Point2D::new(max_bound.x, min_bound.y),
        Point2D::new(max_bound.x, max_bound.y),
        Point2D::new(min_bound.x, max_bound.y),
    ];

    let mut cells = Vec::new();

    // For each site, compute its Voronoi cell
    for (i, &site) in sites.iter().enumerate() {
        let mut cell_polygon = bounding_poly.clone();

        // Clip against half-planes from all other sites
        for (j, &other) in sites.iter().enumerate() {
            if i == j {
                continue;
            }

            // Compute the perpendicular bisector between site and other
            // The bisector is the set of points equidistant from site and other
            // The half-plane we want contains points closer to site than to other
            //
            // Distance condition: (x - site.x)² + (y - site.y)² < (x - other.x)² + (y - other.y)²
            // Expanding and simplifying:
            // 2(other.x - site.x)*x + 2(other.y - site.y)*y + (site.x² + site.y² - other.x² - other.y²) < 0
            //
            // So the half-plane is: A*x + B*y + C < 0
            let a = 2.0 * (other.x - site.x);
            let b = 2.0 * (other.y - site.y);
            let c = site.x * site.x + site.y * site.y - other.x * other.x - other.y * other.y;

            // Clip the cell polygon by this half-plane
            cell_polygon = clip_polygon_by_half_plane(&cell_polygon, a, b, c);

            // Early exit if polygon becomes empty
            if cell_polygon.is_empty() {
                break;
            }
        }

        cells.push(VoronoiCell::new(site, cell_polygon));
    }

    VoronoiDiagram::new(cells, sites.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_voronoi_two_sites() {
        // Two sites should create two half-planes
        let sites = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
        ];

        let diagram = voronoi_brute_force(&sites, Some((
            Point2D::new(-1.0, -1.0),
            Point2D::new(3.0, 1.0),
        )));

        assert_eq!(diagram.num_cells(), 2);

        // The bisector should be at x = 1.0
        // First cell should be on the left (x < 1)
        // Second cell should be on the right (x > 1)

        // Check that each site is in its own cell
        assert!(diagram.cells[0].contains(&sites[0]));
        assert!(diagram.cells[1].contains(&sites[1]));

        // Check that cells don't contain the other site
        assert!(!diagram.cells[0].contains(&sites[1]));
        assert!(!diagram.cells[1].contains(&sites[0]));
    }

    #[test]
    fn test_voronoi_three_sites_triangle() {
        // Three sites forming a triangle
        let sites = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 0.866025403784439), // sqrt(3)/2
        ];

        let diagram = voronoi_brute_force(&sites, None);

        assert_eq!(diagram.num_cells(), 3);

        // Each site should be in its own cell
        for (i, site) in sites.iter().enumerate() {
            assert!(diagram.cells[i].contains(site));
        }

        // All cells should have positive area
        for cell in &diagram.cells {
            assert!(cell.area() > 0.0);
        }

        // Verify that each cell's site is the nearest site to any point in that cell
        // Test the centroid of each cell
        for (i, cell) in diagram.cells.iter().enumerate() {
            if cell.vertices.len() < 3 {
                continue;
            }

            // Compute centroid
            let sum_x: f64 = cell.vertices.iter().map(|p| p.x).sum();
            let sum_y: f64 = cell.vertices.iter().map(|p| p.y).sum();
            let centroid = Point2D::new(
                sum_x / cell.vertices.len() as f64,
                sum_y / cell.vertices.len() as f64,
            );

            // The nearest site to the centroid should be this cell's site
            let nearest = diagram.nearest_site(&centroid);
            assert_eq!(nearest, Some(&sites[i]));
        }
    }

    #[test]
    fn test_voronoi_four_sites_square() {
        // Four sites at corners of a unit square
        let sites = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];

        let diagram = voronoi_brute_force(&sites, Some((
            Point2D::new(-0.5, -0.5),
            Point2D::new(1.5, 1.5),
        )));

        assert_eq!(diagram.num_cells(), 4);

        // The center point (0.5, 0.5) should be equidistant from all sites
        let center = Point2D::new(0.5, 0.5);
        for site in &sites {
            let dist = site.distance(&center);
            assert!(approx_eq(dist, 0.7071067811865476)); // sqrt(2)/2
        }

        // Each cell should contain its corresponding site
        for (i, site) in sites.iter().enumerate() {
            assert!(diagram.cells[i].contains(site));
        }
    }

    #[test]
    fn test_voronoi_collinear_sites() {
        // Three collinear sites
        let sites = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
        ];

        let diagram = voronoi_brute_force(&sites, Some((
            Point2D::new(-1.0, -1.0),
            Point2D::new(3.0, 1.0),
        )));

        assert_eq!(diagram.num_cells(), 3);

        // Middle cell should be a vertical strip
        let middle_cell = &diagram.cells[1];

        // Points at x=1 should be in the middle cell
        assert!(middle_cell.contains(&Point2D::new(1.0, 0.5)));
        assert!(middle_cell.contains(&Point2D::new(1.0, -0.5)));
    }

    #[test]
    fn test_clip_polygon_by_half_plane() {
        // Unit square
        let square = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];

        // Clip by vertical line x = 0.5, keeping left side (x < 0.5)
        // Half-plane: x - 0.5 < 0, so A=1, B=0, C=-0.5
        let clipped = clip_polygon_by_half_plane(&square, 1.0, 0.0, -0.5);

        // Should get a rectangle from (0,0) to (0.5, 1)
        assert_eq!(clipped.len(), 4);

        // Check area
        let mut area = 0.0;
        for i in 0..clipped.len() {
            let j = (i + 1) % clipped.len();
            area += clipped[i].x * clipped[j].y;
            area -= clipped[j].x * clipped[i].y;
        }
        area = (area / 2.0).abs();
        assert!(approx_eq(area, 0.5)); // Half of the original square
    }

    #[test]
    fn test_voronoi_cell_area() {
        let cell = VoronoiCell::new(
            Point2D::new(0.5, 0.5),
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(1.0, 0.0),
                Point2D::new(1.0, 1.0),
                Point2D::new(0.0, 1.0),
            ],
        );

        assert!(approx_eq(cell.area(), 1.0));
    }

    #[test]
    fn test_voronoi_cell_perimeter() {
        let cell = VoronoiCell::new(
            Point2D::new(0.5, 0.5),
            vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(1.0, 0.0),
                Point2D::new(1.0, 1.0),
                Point2D::new(0.0, 1.0),
            ],
        );

        assert!(approx_eq(cell.perimeter(), 4.0));
    }

    #[test]
    fn test_nearest_site() {
        let sites = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(1.0, 2.0),
        ];

        let diagram = voronoi_brute_force(&sites, None);

        // Point near first site
        let nearest = diagram.nearest_site(&Point2D::new(0.1, 0.1));
        assert_eq!(nearest, Some(&sites[0]));

        // Point near second site
        let nearest = diagram.nearest_site(&Point2D::new(1.9, 0.1));
        assert_eq!(nearest, Some(&sites[1]));

        // Point near third site
        let nearest = diagram.nearest_site(&Point2D::new(1.0, 1.9));
        assert_eq!(nearest, Some(&sites[2]));
    }

    #[test]
    fn test_single_site() {
        let sites = vec![Point2D::new(0.0, 0.0)];

        let diagram = voronoi_brute_force(&sites, Some((
            Point2D::new(-10.0, -10.0),
            Point2D::new(10.0, 10.0),
        )));

        assert_eq!(diagram.num_cells(), 1);

        // The cell should be the entire bounding box
        let cell = &diagram.cells[0];
        assert!(approx_eq(cell.area(), 400.0)); // 20x20 square
    }

    #[test]
    fn test_empty_sites() {
        let sites: Vec<Point2D> = vec![];
        let diagram = voronoi_brute_force(&sites, None);
        assert_eq!(diagram.num_cells(), 0);
    }
}
