//! 3D transformation system

use crate::base::{Graphics3dPrimitive, Point3D, Vector3D, BoundingBox3D, IndexFaceSet};
use std::f64::consts::PI;

/// 4x4 transformation matrix for 3D graphics
///
/// Represents affine transformations (translation, rotation, scaling, etc.)
/// using homogeneous coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3D {
    /// Matrix elements in row-major order
    /// [m00, m01, m02, m03]
    /// [m10, m11, m12, m13]
    /// [m20, m21, m22, m23]
    /// [m30, m31, m32, m33]
    pub matrix: [[f64; 4]; 4],
}

impl Transform3D {
    /// Create an identity transformation
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a translation transformation
    pub fn translate(v: Vector3D) -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, v.x],
                [0.0, 1.0, 0.0, v.y],
                [0.0, 0.0, 1.0, v.z],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a uniform scaling transformation
    pub fn scale(s: f64) -> Self {
        Self::scale_xyz(s, s, s)
    }

    /// Create a non-uniform scaling transformation
    pub fn scale_xyz(sx: f64, sy: f64, sz: f64) -> Self {
        Self {
            matrix: [
                [sx, 0.0, 0.0, 0.0],
                [0.0, sy, 0.0, 0.0],
                [0.0, 0.0, sz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a rotation around the X axis
    pub fn rotate_x(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a rotation around the Y axis
    pub fn rotate_y(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            matrix: [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a rotation around the Z axis
    pub fn rotate_z(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            matrix: [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create a rotation around an arbitrary axis
    pub fn rotate(axis: Vector3D, angle: f64) -> Self {
        let axis = axis.normalize();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;

        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Self {
            matrix: [
                [
                    t * x * x + c,
                    t * x * y - s * z,
                    t * x * z + s * y,
                    0.0,
                ],
                [
                    t * x * y + s * z,
                    t * y * y + c,
                    t * y * z - s * x,
                    0.0,
                ],
                [
                    t * x * z - s * y,
                    t * y * z + s * x,
                    t * z * z + c,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Compose two transformations (matrix multiplication)
    pub fn compose(&self, other: &Transform3D) -> Transform3D {
        let mut result = [[0.0; 4]; 4];

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.matrix[i][k] * other.matrix[k][j];
                }
            }
        }

        Transform3D { matrix: result }
    }

    /// Apply transformation to a point
    pub fn transform_point(&self, p: &Point3D) -> Point3D {
        let m = &self.matrix;
        let x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        let y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        let z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        let w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];

        Point3D::new(x / w, y / w, z / w)
    }

    /// Apply transformation to a vector (ignoring translation)
    pub fn transform_vector(&self, v: &Vector3D) -> Vector3D {
        let m = &self.matrix;
        let x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
        let y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
        let z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;

        Vector3D::new(x, y, z)
    }

    /// Compute the inverse transformation (if it exists)
    pub fn inverse(&self) -> Option<Transform3D> {
        // Compute matrix inverse using cofactor expansion
        // This is simplified - a full implementation would use LU decomposition
        let m = &self.matrix;

        // For affine transformations (last row is [0,0,0,1]),
        // we can use a simpler approach

        // Extract 3x3 rotation/scale part
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        if det.abs() < 1e-10 {
            return None; // Singular matrix
        }

        let inv_det = 1.0 / det;

        let mut inv = [[0.0; 4]; 4];

        // Compute inverse of 3x3 part
        inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

        inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;

        inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

        // Compute inverse translation
        inv[0][3] = -(inv[0][0] * m[0][3] + inv[0][1] * m[1][3] + inv[0][2] * m[2][3]);
        inv[1][3] = -(inv[1][0] * m[0][3] + inv[1][1] * m[1][3] + inv[1][2] * m[2][3]);
        inv[2][3] = -(inv[2][0] * m[0][3] + inv[2][1] * m[1][3] + inv[2][2] * m[2][3]);

        inv[3][3] = 1.0;

        Some(Transform3D { matrix: inv })
    }
}

impl Default for Transform3D {
    fn default() -> Self {
        Self::identity()
    }
}

/// A group of 3D primitives with a shared transformation
///
/// This allows hierarchical transformations where a parent transformation
/// affects all children.
#[derive(Debug, Clone)]
pub struct TransformGroup {
    /// The transformation to apply to all children
    pub transform: Transform3D,

    /// Child primitives
    pub children: Vec<Box<dyn Graphics3dPrimitive>>,
}

impl TransformGroup {
    /// Create a new transform group
    pub fn new(transform: Transform3D) -> Self {
        Self {
            transform,
            children: Vec::new(),
        }
    }

    /// Add a child primitive
    pub fn add_child(&mut self, child: Box<dyn Graphics3dPrimitive>) {
        self.children.push(child);
    }

    /// Apply the transformation to a mesh
    fn transform_mesh(&self, mesh: &IndexFaceSet) -> IndexFaceSet {
        let mut transformed = mesh.clone();

        // Transform vertices
        for vertex in &mut transformed.vertices {
            *vertex = self.transform.transform_point(vertex);
        }

        // Transform normals if present
        if let Some(ref mut normals) = transformed.normals {
            for normal in normals {
                *normal = self.transform.transform_vector(normal).normalize();
            }
        }

        transformed
    }
}

impl Graphics3dPrimitive for TransformGroup {
    fn bounding_box(&self) -> BoundingBox3D {
        let mut bbox = BoundingBox3D::empty();

        for child in &self.children {
            let child_bbox = child.bounding_box();

            // Transform the 8 corners of the child bounding box
            let corners = [
                Point3D::new(child_bbox.xmin, child_bbox.ymin, child_bbox.zmin),
                Point3D::new(child_bbox.xmax, child_bbox.ymin, child_bbox.zmin),
                Point3D::new(child_bbox.xmin, child_bbox.ymax, child_bbox.zmin),
                Point3D::new(child_bbox.xmax, child_bbox.ymax, child_bbox.zmin),
                Point3D::new(child_bbox.xmin, child_bbox.ymin, child_bbox.zmax),
                Point3D::new(child_bbox.xmax, child_bbox.ymin, child_bbox.zmax),
                Point3D::new(child_bbox.xmin, child_bbox.ymax, child_bbox.zmax),
                Point3D::new(child_bbox.xmax, child_bbox.ymax, child_bbox.zmax),
            ];

            for corner in &corners {
                let transformed = self.transform.transform_point(corner);
                bbox.expand_to_include(&transformed);
            }
        }

        bbox
    }

    fn to_mesh(&self) -> crate::Result<IndexFaceSet> {
        let mut combined = IndexFaceSet::new(Vec::new(), Vec::new());

        for child in &self.children {
            let child_mesh = child.to_mesh()?;
            let transformed = self.transform_mesh(&child_mesh);

            // Merge into combined mesh
            let vertex_offset = combined.vertices.len();

            combined.vertices.extend(transformed.vertices);

            for face in transformed.faces {
                combined.faces.push([
                    face[0] + vertex_offset,
                    face[1] + vertex_offset,
                    face[2] + vertex_offset,
                ]);
            }

            // Merge normals if present
            if let Some(ref mut combined_normals) = combined.normals {
                if let Some(ref child_normals) = transformed.normals {
                    combined_normals.extend(child_normals.iter().copied());
                }
            } else if transformed.normals.is_some() {
                combined.normals = transformed.normals;
            }

            // Merge colors if present
            if let Some(ref mut combined_colors) = combined.vertex_colors {
                if let Some(ref child_colors) = transformed.vertex_colors {
                    combined_colors.extend(child_colors.iter().cloned());
                }
            } else if transformed.vertex_colors.is_some() {
                combined.vertex_colors = transformed.vertex_colors;
            }

            if let Some(ref mut combined_face_colors) = combined.face_colors {
                if let Some(ref child_face_colors) = transformed.face_colors {
                    combined_face_colors.extend(child_face_colors.iter().cloned());
                }
            } else if transformed.face_colors.is_some() {
                combined.face_colors = transformed.face_colors;
            }
        }

        Ok(combined)
    }

    fn clone_box(&self) -> std::boxed::Box<dyn Graphics3dPrimitive> {
        std::boxed::Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let t = Transform3D::identity();
        let p = Point3D::new(1.0, 2.0, 3.0);
        let transformed = t.transform_point(&p);
        assert_eq!(transformed, p);
    }

    #[test]
    fn test_translate() {
        let t = Transform3D::translate(Vector3D::new(5.0, 10.0, 15.0));
        let p = Point3D::new(1.0, 2.0, 3.0);
        let transformed = t.transform_point(&p);
        assert_eq!(transformed, Point3D::new(6.0, 12.0, 18.0));
    }

    #[test]
    fn test_scale() {
        let t = Transform3D::scale(2.0);
        let p = Point3D::new(1.0, 2.0, 3.0);
        let transformed = t.transform_point(&p);
        assert_eq!(transformed, Point3D::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_rotate_x() {
        let t = Transform3D::rotate_x(PI / 2.0);
        let p = Point3D::new(0.0, 1.0, 0.0);
        let transformed = t.transform_point(&p);
        assert!((transformed.x).abs() < 1e-10);
        assert!((transformed.y).abs() < 1e-10);
        assert!((transformed.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_y() {
        let t = Transform3D::rotate_y(PI / 2.0);
        let p = Point3D::new(1.0, 0.0, 0.0);
        let transformed = t.transform_point(&p);
        assert!((transformed.x).abs() < 1e-10);
        assert!((transformed.y).abs() < 1e-10);
        assert!((transformed.z + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_z() {
        let t = Transform3D::rotate_z(PI / 2.0);
        let p = Point3D::new(1.0, 0.0, 0.0);
        let transformed = t.transform_point(&p);
        assert!((transformed.x).abs() < 1e-10);
        assert!((transformed.y - 1.0).abs() < 1e-10);
        assert!((transformed.z).abs() < 1e-10);
    }

    #[test]
    fn test_compose() {
        let t1 = Transform3D::translate(Vector3D::new(1.0, 0.0, 0.0));
        let t2 = Transform3D::scale(2.0);
        let t_composed = t1.compose(&t2);

        let p = Point3D::new(1.0, 1.0, 1.0);
        let transformed = t_composed.transform_point(&p);

        // First scale by 2, then translate by (1,0,0)
        assert_eq!(transformed, Point3D::new(3.0, 2.0, 2.0));
    }

    #[test]
    fn test_inverse() {
        let t = Transform3D::translate(Vector3D::new(5.0, 10.0, 15.0));
        let t_inv = t.inverse().unwrap();

        let p = Point3D::new(1.0, 2.0, 3.0);
        let transformed = t.transform_point(&p);
        let back = t_inv.transform_point(&transformed);

        assert!((back.x - p.x).abs() < 1e-10);
        assert!((back.y - p.y).abs() < 1e-10);
        assert!((back.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_transform_vector() {
        let t = Transform3D::translate(Vector3D::new(10.0, 20.0, 30.0));
        let v = Vector3D::new(1.0, 2.0, 3.0);
        let transformed = t.transform_vector(&v);

        // Translation should not affect vectors
        assert_eq!(transformed, v);
    }

    #[test]
    fn test_rotate_arbitrary_axis() {
        let axis = Vector3D::new(1.0, 1.0, 1.0);
        let t = Transform3D::rotate(axis, PI / 3.0);
        let p = Point3D::new(1.0, 0.0, 0.0);
        let transformed = t.transform_point(&p);

        // Just check that transformation happened (not checking exact values)
        assert!((transformed.x - 1.0).abs() > 0.1 ||
                (transformed.y).abs() > 0.1 ||
                (transformed.z).abs() > 0.1);
    }
}
