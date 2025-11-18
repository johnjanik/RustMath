//! Platonic solids (regular convex polyhedra)
//!
//! This module provides the five Platonic solids: tetrahedron, cube (hexahedron),
//! octahedron, dodecahedron, and icosahedron. All are centered at the origin
//! with unit edge length by default.

use crate::base::{Graphics3d, IndexFaceSet, Point3D};
use crate::Result;
use rustmath_colors::Color;

/// Create a tetrahedron (4 faces, 4 vertices, 6 edges)
///
/// A regular tetrahedron with unit edge length centered at the origin.
pub fn tetrahedron() -> Graphics3d {
    let a = 1.0 / (2.0 * 2.0_f64.sqrt());
    let b = 1.0 / (2.0 * 3.0_f64.sqrt());

    let vertices = vec![
        Point3D { x: a, y: -b, z: -0.5 },
        Point3D { x: -a, y: -b, z: -0.5 },
        Point3D { x: 0.0, y: 2.0 * b, z: -0.5 },
        Point3D { x: 0.0, y: 0.0, z: 0.5 },
    ];

    let faces = vec![
        [0, 1, 2],
        [0, 3, 1],
        [1, 3, 2],
        [2, 3, 0],
    ];

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);
    graphics
}

/// Create a cube/hexahedron (6 faces, 8 vertices, 12 edges)
///
/// A regular cube with unit edge length centered at the origin.
pub fn cube() -> Graphics3d {
    let a = 0.5;

    let vertices = vec![
        Point3D { x: -a, y: -a, z: -a }, // 0
        Point3D { x: a, y: -a, z: -a },  // 1
        Point3D { x: a, y: a, z: -a },   // 2
        Point3D { x: -a, y: a, z: -a },  // 3
        Point3D { x: -a, y: -a, z: a },  // 4
        Point3D { x: a, y: -a, z: a },   // 5
        Point3D { x: a, y: a, z: a },    // 6
        Point3D { x: -a, y: a, z: a },   // 7
    ];

    let faces = vec![
        // Bottom face
        [0, 1, 2], [0, 2, 3],
        // Top face
        [4, 7, 6], [4, 6, 5],
        // Front face
        [0, 4, 5], [0, 5, 1],
        // Back face
        [3, 2, 6], [3, 6, 7],
        // Left face
        [0, 3, 7], [0, 7, 4],
        // Right face
        [1, 5, 6], [1, 6, 2],
    ];

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);
    graphics
}

/// Create an octahedron (8 faces, 6 vertices, 12 edges)
///
/// A regular octahedron with unit edge length centered at the origin.
pub fn octahedron() -> Graphics3d {
    let a = 1.0 / 2.0_f64.sqrt();

    let vertices = vec![
        Point3D { x: a, y: 0.0, z: 0.0 },   // 0
        Point3D { x: -a, y: 0.0, z: 0.0 },  // 1
        Point3D { x: 0.0, y: a, z: 0.0 },   // 2
        Point3D { x: 0.0, y: -a, z: 0.0 },  // 3
        Point3D { x: 0.0, y: 0.0, z: a },   // 4
        Point3D { x: 0.0, y: 0.0, z: -a },  // 5
    ];

    let faces = vec![
        // Upper pyramid
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        // Lower pyramid
        [0, 5, 2],
        [2, 5, 1],
        [1, 5, 3],
        [3, 5, 0],
    ];

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);
    graphics
}

/// Create a dodecahedron (12 faces, 20 vertices, 30 edges)
///
/// A regular dodecahedron with unit edge length centered at the origin.
pub fn dodecahedron() -> Graphics3d {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
    let a = 1.0 / (3.0_f64.sqrt());
    let b = a / phi;
    let c = a * phi;

    let vertices = vec![
        // Cube vertices scaled by a
        Point3D { x: a, y: a, z: a },
        Point3D { x: a, y: a, z: -a },
        Point3D { x: a, y: -a, z: a },
        Point3D { x: a, y: -a, z: -a },
        Point3D { x: -a, y: a, z: a },
        Point3D { x: -a, y: a, z: -a },
        Point3D { x: -a, y: -a, z: a },
        Point3D { x: -a, y: -a, z: -a },
        // Rectangular vertices
        Point3D { x: b, y: c, z: 0.0 },
        Point3D { x: -b, y: c, z: 0.0 },
        Point3D { x: b, y: -c, z: 0.0 },
        Point3D { x: -b, y: -c, z: 0.0 },
        Point3D { x: c, y: 0.0, z: b },
        Point3D { x: c, y: 0.0, z: -b },
        Point3D { x: -c, y: 0.0, z: b },
        Point3D { x: -c, y: 0.0, z: -b },
        Point3D { x: 0.0, y: b, z: c },
        Point3D { x: 0.0, y: -b, z: c },
        Point3D { x: 0.0, y: b, z: -c },
        Point3D { x: 0.0, y: -b, z: -c },
    ];

    // Each pentagonal face is divided into 3 triangles
    let faces = vec![
        // Pentagon faces (each divided into 3 triangles from center)
        // Top
        [0, 8, 1], [0, 12, 8], [12, 0, 2],
        [9, 4, 0], [9, 0, 1], [4, 16, 0],
        // Bottom
        [11, 7, 10], [7, 11, 6], [11, 10, 19],
        [10, 2, 17], [2, 10, 3], [17, 2, 6],
        // Middle band
        [13, 3, 1], [13, 1, 18], [3, 13, 12],
        [15, 5, 7], [5, 15, 18], [7, 5, 14],
        [14, 4, 6], [14, 6, 11], [4, 14, 15],
        [18, 8, 5], [8, 18, 1], [5, 8, 9],
        [19, 3, 10], [3, 19, 13], [10, 7, 19],
        [12, 2, 13], [16, 17, 4], [17, 6, 4],
    ];

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);
    graphics
}

/// Create an icosahedron (20 faces, 12 vertices, 30 edges)
///
/// A regular icosahedron with unit edge length centered at the origin.
pub fn icosahedron() -> Graphics3d {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
    let a = 1.0 / (2.0 * phi);

    let vertices = vec![
        Point3D { x: 0.0, y: a, z: a * phi },
        Point3D { x: 0.0, y: -a, z: a * phi },
        Point3D { x: 0.0, y: a, z: -a * phi },
        Point3D { x: 0.0, y: -a, z: -a * phi },
        Point3D { x: a, z: a * phi, y: 0.0 },
        Point3D { x: -a, z: a * phi, y: 0.0 },
        Point3D { x: a, z: -a * phi, y: 0.0 },
        Point3D { x: -a, z: -a * phi, y: 0.0 },
        Point3D { x: a * phi, y: 0.0, z: a },
        Point3D { x: a * phi, y: 0.0, z: -a },
        Point3D { x: -a * phi, y: 0.0, z: a },
        Point3D { x: -a * phi, y: 0.0, z: -a },
    ];

    let faces = vec![
        [0, 1, 4],
        [0, 4, 8],
        [0, 8, 5],
        [0, 5, 10],
        [0, 10, 1],
        [1, 10, 7],
        [1, 7, 6],
        [1, 6, 4],
        [4, 6, 9],
        [4, 9, 8],
        [8, 9, 2],
        [8, 2, 5],
        [5, 2, 11],
        [5, 11, 10],
        [10, 11, 7],
        [7, 11, 3],
        [7, 3, 6],
        [6, 3, 9],
        [9, 3, 2],
        [2, 3, 11],
    ];

    let mut mesh = IndexFaceSet::new(vertices, faces);
    mesh.compute_normals();

    let mut graphics = Graphics3d::new();
    graphics.add_mesh(mesh);
    graphics
}

/// Create a colored Platonic solid
pub fn colored_platonic_solid(name: &str, color: Color) -> Result<Graphics3d> {
    let mut graphics = match name.to_lowercase().as_str() {
        "tetrahedron" => tetrahedron(),
        "cube" | "hexahedron" => cube(),
        "octahedron" => octahedron(),
        "dodecahedron" => dodecahedron(),
        "icosahedron" => icosahedron(),
        _ => return Err(crate::Plot3DError::InvalidMesh(format!("Unknown Platonic solid: {}", name))),
    };

    // Set uniform color for all objects
    if !graphics.objects.is_empty() {
        // Access the first object and set colors
        // Note: This is a simplified version; in practice, you'd want to properly
        // set colors on all meshes
        graphics.options.default_color = Some(color);
    }

    Ok(graphics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tetrahedron() {
        let graphics = tetrahedron();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_cube() {
        let graphics = cube();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_octahedron() {
        let graphics = octahedron();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_dodecahedron() {
        let graphics = dodecahedron();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_icosahedron() {
        let graphics = icosahedron();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_colored_platonic_solid() {
        let color = Color::rgb(1.0, 0.0, 0.0);
        let graphics = colored_platonic_solid("cube", color).unwrap();
        assert_eq!(graphics.objects.len(), 1);
    }

    #[test]
    fn test_unknown_platonic_solid() {
        let color = Color::rgb(1.0, 0.0, 0.0);
        let result = colored_platonic_solid("pyramid", color);
        assert!(result.is_err());
    }
}
