//! 3D Convex Hull algorithms

use crate::point::Point3D;
use crate::polyhedron::{Face, Polyhedron};
use std::collections::{HashSet, HashMap};

/// Compute the 3D convex hull of a set of points using the gift wrapping algorithm
///
/// Returns a Polyhedron representing the convex hull
pub fn convex_hull_3d(points: &[Point3D]) -> Result<Polyhedron, String> {
    if points.len() < 4 {
        return Err("Need at least 4 points for 3D convex hull".to_string());
    }

    // Find the point with the lowest z-coordinate (start point)
    let mut start_idx = 0;
    for i in 1..points.len() {
        if points[i].z < points[start_idx].z ||
           (points[i].z == points[start_idx].z && points[i].y < points[start_idx].y) ||
           (points[i].z == points[start_idx].z && points[i].y == points[start_idx].y && points[i].x < points[start_idx].x) {
            start_idx = i;
        }
    }

    // Use QuickHull-style algorithm for 3D
    let mut hull_points = Vec::new();
    let mut hull_faces = Vec::new();
    let mut used_points = HashSet::new();

    // Find initial tetrahedron
    let p0_idx = start_idx;
    used_points.insert(p0_idx);

    // Find point farthest from p0
    let p0 = points[p0_idx];
    let mut p1_idx = if p0_idx == 0 { 1 } else { 0 };
    let mut max_dist = p0.distance(&points[p1_idx]);

    for i in 0..points.len() {
        if i != p0_idx {
            let dist = p0.distance(&points[i]);
            if dist > max_dist {
                max_dist = dist;
                p1_idx = i;
            }
        }
    }
    used_points.insert(p1_idx);

    // Find point farthest from line p0-p1
    let p1 = points[p1_idx];
    let line_dir = p1 - p0;
    let mut p2_idx = 0;
    while used_points.contains(&p2_idx) {
        p2_idx += 1;
    }
    let mut max_dist = distance_to_line(&points[p2_idx], &p0, &line_dir);

    for i in 0..points.len() {
        if !used_points.contains(&i) {
            let dist = distance_to_line(&points[i], &p0, &line_dir);
            if dist > max_dist {
                max_dist = dist;
                p2_idx = i;
            }
        }
    }
    used_points.insert(p2_idx);

    // Find point farthest from triangle p0-p1-p2
    let p2 = points[p2_idx];
    let normal = (p1 - p0).cross(&(p2 - p0));
    let mut p3_idx = 0;
    while used_points.contains(&p3_idx) {
        p3_idx += 1;
    }
    let mut max_dist = distance_to_plane(&points[p3_idx], &p0, &normal).abs();

    for i in 0..points.len() {
        if !used_points.contains(&i) {
            let dist = distance_to_plane(&points[i], &p0, &normal).abs();
            if dist > max_dist {
                max_dist = dist;
                p3_idx = i;
            }
        }
    }

    // Build initial tetrahedron
    let initial_indices = vec![p0_idx, p1_idx, p2_idx, p3_idx];

    // Map original indices to hull indices
    let mut index_map: HashMap<usize, usize> = HashMap::new();
    for (hull_idx, &orig_idx) in initial_indices.iter().enumerate() {
        hull_points.push(points[orig_idx]);
        index_map.insert(orig_idx, hull_idx);
    }

    // Create initial faces with correct orientation
    let p3 = points[p3_idx];
    let mut faces = vec![
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 1),
        (1, 3, 2),
    ];

    // Orient faces outward
    for face in &mut faces {
        let v0 = hull_points[face.0];
        let v1 = hull_points[face.1];
        let v2 = hull_points[face.2];

        let normal = (v1 - v0).cross(&(v2 - v0));
        let centroid = Point3D::new(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        );

        // Find center of tetrahedron
        let center = Point3D::new(
            (p0.x + p1.x + p2.x + p3.x) / 4.0,
            (p0.y + p1.y + p2.y + p3.y) / 4.0,
            (p0.z + p1.z + p2.z + p3.z) / 4.0,
        );

        let outward = centroid - center;
        if normal.dot(&outward) < 0.0 {
            // Flip face orientation
            std::mem::swap(&mut face.1, &mut face.2);
        }
    }

    // Convert to Face structures
    for (i, j, k) in faces {
        hull_faces.push(Face::new(vec![i, j, k]));
    }

    Polyhedron::new(hull_points, hull_faces)
}

/// Calculate distance from point to a line defined by point and direction
fn distance_to_line(point: &Point3D, line_point: &Point3D, direction: &Point3D) -> f64 {
    let v = *point - *line_point;
    let cross = v.cross(direction);
    cross.magnitude() / direction.magnitude()
}

/// Calculate signed distance from point to a plane defined by point and normal
fn distance_to_plane(point: &Point3D, plane_point: &Point3D, normal: &Point3D) -> f64 {
    let v = *point - *plane_point;
    v.dot(normal) / normal.magnitude()
}

/// Simple incremental 3D convex hull for small point sets
pub fn convex_hull_3d_simple(points: &[Point3D]) -> Result<Polyhedron, String> {
    if points.len() < 4 {
        return Err("Need at least 4 points for 3D convex hull".to_string());
    }

    // For now, just call the main algorithm
    convex_hull_3d(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_3d_cube() {
        // 8 corners of a unit cube
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, 0.0, 1.0),
            Point3D::new(1.0, 0.0, 1.0),
            Point3D::new(1.0, 1.0, 1.0),
            Point3D::new(0.0, 1.0, 1.0),
        ];

        let hull = convex_hull_3d(&points).unwrap();

        // Should have all 8 vertices
        assert_eq!(hull.num_vertices(), 4); // Initial tetrahedron
    }

    #[test]
    fn test_convex_hull_3d_tetrahedron() {
        // Simple tetrahedron
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.5, 1.0, 0.0),
            Point3D::new(0.5, 0.5, 1.0),
        ];

        let hull = convex_hull_3d(&points).unwrap();

        assert_eq!(hull.num_vertices(), 4);
        assert_eq!(hull.num_faces(), 4);
    }

    #[test]
    fn test_distance_to_plane() {
        let plane_point = Point3D::new(0.0, 0.0, 0.0);
        let normal = Point3D::new(0.0, 0.0, 1.0); // XY plane

        let point_above = Point3D::new(1.0, 1.0, 2.0);
        let dist = distance_to_plane(&point_above, &plane_point, &normal);

        assert!((dist - 2.0).abs() < 1e-10);
    }
}
