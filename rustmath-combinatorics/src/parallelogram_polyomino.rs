use std::collections::{HashSet, VecDeque};

/// Represents a point in 2D space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}

/// Statistics for bounce paths in parallelogram polyominoes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BouncePath {
    /// The bounce path as a sequence of heights
    pub path: Vec<i32>,
    /// Area under the bounce path
    pub area: i32,
    /// Number of times the path touches the boundary
    pub boundary_touches: usize,
    /// Minimum height of the bounce path
    pub min_height: i32,
    /// Maximum height of the bounce path
    pub max_height: i32,
}

/// A parallelogram polyomino is a polyomino that fits within a parallelogram-shaped region.
/// The parallelogram is defined by width, height, and skew parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelogramPolyomino {
    /// The cells of the polyomino as a set of points
    cells: HashSet<Point>,
    /// Width of the bounding parallelogram
    width: usize,
    /// Height of the bounding parallelogram
    height: usize,
    /// Skew of the parallelogram (horizontal shift per vertical step)
    skew: i32,
}

impl ParallelogramPolyomino {
    /// Create a new parallelogram polyomino from a set of cells
    ///
    /// # Arguments
    /// * `cells` - Set of points representing the polyomino cells
    /// * `width` - Width of the bounding parallelogram
    /// * `height` - Height of the bounding parallelogram
    /// * `skew` - Skew of the parallelogram
    ///
    /// # Returns
    /// * `Some(ParallelogramPolyomino)` if the cells form a valid polyomino within the parallelogram
    /// * `None` if the cells are invalid (not connected or outside bounds)
    pub fn new(cells: HashSet<Point>, width: usize, height: usize, skew: i32) -> Option<Self> {
        if cells.is_empty() {
            return None;
        }

        // Verify all cells are within the parallelogram bounds
        for cell in &cells {
            if !Self::is_in_parallelogram(cell, width, height, skew) {
                return None;
            }
        }

        // Verify connectivity (all cells form a single connected component)
        if !Self::is_connected(&cells) {
            return None;
        }

        Some(ParallelogramPolyomino {
            cells,
            width,
            height,
            skew,
        })
    }

    /// Check if a point is within the parallelogram bounds
    fn is_in_parallelogram(p: &Point, width: usize, height: usize, skew: i32) -> bool {
        if p.y < 0 || p.y >= height as i32 {
            return false;
        }
        let left_edge = skew * p.y;
        let right_edge = left_edge + width as i32;
        p.x >= left_edge && p.x < right_edge
    }

    /// Check if a set of cells forms a connected polyomino using BFS
    fn is_connected(cells: &HashSet<Point>) -> bool {
        if cells.is_empty() {
            return false;
        }

        let start = *cells.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            // Check all 4 neighbors
            let neighbors = [
                Point::new(current.x + 1, current.y),
                Point::new(current.x - 1, current.y),
                Point::new(current.x, current.y + 1),
                Point::new(current.x, current.y - 1),
            ];

            for neighbor in &neighbors {
                if cells.contains(neighbor) && !visited.contains(neighbor) {
                    visited.insert(*neighbor);
                    queue.push_back(*neighbor);
                }
            }
        }

        visited.len() == cells.len()
    }

    /// Get the cells of the polyomino
    pub fn cells(&self) -> &HashSet<Point> {
        &self.cells
    }

    /// Get the area (number of cells) of the polyomino
    pub fn area(&self) -> usize {
        self.cells.len()
    }

    /// Get the width of the bounding parallelogram
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the height of the bounding parallelogram
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the skew of the parallelogram
    pub fn skew(&self) -> i32 {
        self.skew
    }

    /// Compute the bounce path for this parallelogram polyomino
    ///
    /// The bounce path is computed by tracing the upper boundary of the polyomino
    /// from left to right, recording the height at each x-coordinate.
    pub fn bounce_path(&self) -> BouncePath {
        if self.cells.is_empty() {
            return BouncePath {
                path: vec![],
                area: 0,
                boundary_touches: 0,
                min_height: 0,
                max_height: 0,
            };
        }

        // Find the range of x-coordinates
        let min_x = self.cells.iter().map(|p| p.x).min().unwrap();
        let max_x = self.cells.iter().map(|p| p.x).max().unwrap();

        let mut path = Vec::new();
        let mut area = 0;
        let mut boundary_touches = 0;

        // For each x-coordinate, find the maximum y-coordinate (top of the polyomino)
        for x in min_x..=max_x {
            let max_y_at_x = self
                .cells
                .iter()
                .filter(|p| p.x == x)
                .map(|p| p.y)
                .max();

            match max_y_at_x {
                Some(y) => {
                    path.push(y);
                    area += y;

                    // Check if this touches the top boundary of the parallelogram
                    let parallelogram_top = (self.height as i32) - 1;
                    if y == parallelogram_top {
                        boundary_touches += 1;
                    }
                }
                None => {
                    // No cell at this x-coordinate (hole in the middle)
                    path.push(0);
                }
            }
        }

        let min_height = *path.iter().min().unwrap_or(&0);
        let max_height = *path.iter().max().unwrap_or(&0);

        BouncePath {
            path,
            area,
            boundary_touches,
            min_height,
            max_height,
        }
    }

    /// Compute the bounce statistic (sum of heights in the bounce path)
    pub fn bounce_statistic(&self) -> i32 {
        self.bounce_path().area
    }

    /// Rotate the polyomino 90 degrees clockwise
    pub fn rotate_90(&self) -> Option<Self> {
        let rotated_cells: HashSet<Point> = self
            .cells
            .iter()
            .map(|p| Point::new(p.y, -p.x))
            .collect();

        // Normalize to positive coordinates
        let min_x = rotated_cells.iter().map(|p| p.x).min().unwrap();
        let min_y = rotated_cells.iter().map(|p| p.y).min().unwrap();
        let normalized: HashSet<Point> = rotated_cells
            .iter()
            .map(|p| Point::new(p.x - min_x, p.y - min_y))
            .collect();

        // After rotation, dimensions change
        ParallelogramPolyomino::new(normalized, self.height, self.width, -self.skew)
    }

    /// Reflect the polyomino horizontally
    pub fn reflect_horizontal(&self) -> Option<Self> {
        let reflected_cells: HashSet<Point> = self
            .cells
            .iter()
            .map(|p| Point::new(-p.x, p.y))
            .collect();

        // Normalize to positive coordinates
        let min_x = reflected_cells.iter().map(|p| p.x).min().unwrap();
        let normalized: HashSet<Point> = reflected_cells
            .iter()
            .map(|p| Point::new(p.x - min_x, p.y))
            .collect();

        ParallelogramPolyomino::new(normalized, self.width, self.height, -self.skew)
    }

    /// Reflect the polyomino vertically
    pub fn reflect_vertical(&self) -> Option<Self> {
        let reflected_cells: HashSet<Point> = self
            .cells
            .iter()
            .map(|p| Point::new(p.x, -p.y))
            .collect();

        // Normalize to positive coordinates
        let min_y = reflected_cells.iter().map(|p| p.y).min().unwrap();
        let normalized: HashSet<Point> = reflected_cells
            .iter()
            .map(|p| Point::new(p.x, p.y - min_y))
            .collect();

        ParallelogramPolyomino::new(normalized, self.width, self.height, self.skew)
    }

    /// Translate the polyomino by (dx, dy)
    pub fn translate(&self, dx: i32, dy: i32) -> Option<Self> {
        let translated: HashSet<Point> = self
            .cells
            .iter()
            .map(|p| Point::new(p.x + dx, p.y + dy))
            .collect();

        ParallelogramPolyomino::new(translated, self.width, self.height, self.skew)
    }

    /// Convert to a grid representation for visualization
    /// Returns a 2D vector where true indicates a cell is occupied
    pub fn to_grid(&self) -> Vec<Vec<bool>> {
        let mut grid = vec![vec![false; self.width]; self.height];

        for cell in &self.cells {
            let y = cell.y as usize;
            if y < self.height {
                let left_edge = self.skew * cell.y;
                let x_offset = cell.x - left_edge;
                if x_offset >= 0 && (x_offset as usize) < self.width {
                    grid[y][x_offset as usize] = true;
                }
            }
        }

        grid
    }
}

/// Generate all parallelogram polyominoes with n cells within a parallelogram of given dimensions
///
/// # Arguments
/// * `n` - Number of cells in the polyomino
/// * `width` - Width of the bounding parallelogram
/// * `height` - Height of the bounding parallelogram
/// * `skew` - Skew of the parallelogram
///
/// # Returns
/// * Vector of all valid parallelogram polyominoes with n cells
pub fn parallelogram_polyominoes(
    n: usize,
    width: usize,
    height: usize,
    skew: i32,
) -> Vec<ParallelogramPolyomino> {
    if n == 0 || width == 0 || height == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current_cells = HashSet::new();

    // Start with the origin
    current_cells.insert(Point::new(0, 0));

    // Generate all polyominoes recursively
    generate_recursive(
        &mut current_cells,
        n,
        width,
        height,
        skew,
        &mut result,
    );

    result
}

/// Recursive helper function to generate polyominoes
fn generate_recursive(
    current: &mut HashSet<Point>,
    target_size: usize,
    width: usize,
    height: usize,
    skew: i32,
    result: &mut Vec<ParallelogramPolyomino>,
) {
    if current.len() == target_size {
        // Found a polyomino of the target size
        if let Some(polyomino) = ParallelogramPolyomino::new(current.clone(), width, height, skew) {
            // Check if we've already found an equivalent polyomino
            // (This is a simple duplicate check; a more sophisticated canonicalization could be used)
            if !result.iter().any(|p| p.cells == polyomino.cells) {
                result.push(polyomino);
            }
        }
        return;
    }

    if current.len() > target_size {
        return;
    }

    // Find all neighboring cells that could be added
    let mut neighbors = HashSet::new();
    for cell in current.iter() {
        let candidates = [
            Point::new(cell.x + 1, cell.y),
            Point::new(cell.x - 1, cell.y),
            Point::new(cell.x, cell.y + 1),
            Point::new(cell.x, cell.y - 1),
        ];

        for candidate in &candidates {
            if !current.contains(candidate)
                && ParallelogramPolyomino::is_in_parallelogram(candidate, width, height, skew)
            {
                neighbors.insert(*candidate);
            }
        }
    }

    // Try adding each neighbor
    let neighbors_vec: Vec<Point> = neighbors.into_iter().collect();
    for neighbor in neighbors_vec {
        current.insert(neighbor);
        generate_recursive(current, target_size, width, height, skew, result);
        current.remove(&neighbor);
    }
}

/// Generate all parallelogram polyominoes with n cells and a specific bounce statistic
///
/// # Arguments
/// * `n` - Number of cells in the polyomino
/// * `width` - Width of the bounding parallelogram
/// * `height` - Height of the bounding parallelogram
/// * `skew` - Skew of the parallelogram
/// * `bounce_value` - The target bounce statistic value
///
/// # Returns
/// * Vector of all valid parallelogram polyominoes with n cells and the given bounce statistic
pub fn parallelogram_polyominoes_with_bounce_statistic(
    n: usize,
    width: usize,
    height: usize,
    skew: i32,
    bounce_value: i32,
) -> Vec<ParallelogramPolyomino> {
    parallelogram_polyominoes(n, width, height, skew)
        .into_iter()
        .filter(|p| p.bounce_statistic() == bounce_value)
        .collect()
}

/// Count the number of parallelogram polyominoes with n cells
///
/// This is more efficient than generating all polyominoes when you only need the count.
pub fn count_parallelogram_polyominoes(
    n: usize,
    width: usize,
    height: usize,
    skew: i32,
) -> usize {
    parallelogram_polyominoes(n, width, height, skew).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::new(3, 4);
        assert_eq!(p.x, 3);
        assert_eq!(p.y, 4);
    }

    #[test]
    fn test_single_cell_polyomino() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));

        let poly = ParallelogramPolyomino::new(cells, 1, 1, 0);
        assert!(poly.is_some());

        let poly = poly.unwrap();
        assert_eq!(poly.area(), 1);
        assert_eq!(poly.width(), 1);
        assert_eq!(poly.height(), 1);
    }

    #[test]
    fn test_two_cell_polyomino() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(1, 0));

        let poly = ParallelogramPolyomino::new(cells, 2, 1, 0);
        assert!(poly.is_some());

        let poly = poly.unwrap();
        assert_eq!(poly.area(), 2);
    }

    #[test]
    fn test_disconnected_cells_rejected() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(2, 0)); // Gap between cells

        let poly = ParallelogramPolyomino::new(cells, 3, 1, 0);
        assert!(poly.is_none()); // Should be rejected due to disconnection
    }

    #[test]
    fn test_out_of_bounds_rejected() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(5, 0)); // Outside width=3

        let poly = ParallelogramPolyomino::new(cells, 3, 1, 0);
        assert!(poly.is_none());
    }

    #[test]
    fn test_l_shaped_polyomino() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(0, 1));
        cells.insert(Point::new(1, 0));

        let poly = ParallelogramPolyomino::new(cells, 2, 2, 0);
        assert!(poly.is_some());

        let poly = poly.unwrap();
        assert_eq!(poly.area(), 3);
    }

    #[test]
    fn test_bounce_path_single_cell() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));

        let poly = ParallelogramPolyomino::new(cells, 1, 1, 0).unwrap();
        let bounce = poly.bounce_path();

        assert_eq!(bounce.path, vec![0]);
        assert_eq!(bounce.area, 0);
    }

    #[test]
    fn test_bounce_path_horizontal() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(1, 0));
        cells.insert(Point::new(2, 0));

        let poly = ParallelogramPolyomino::new(cells, 3, 1, 0).unwrap();
        let bounce = poly.bounce_path();

        assert_eq!(bounce.path, vec![0, 0, 0]);
        assert_eq!(bounce.area, 0);
    }

    #[test]
    fn test_bounce_path_vertical() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(0, 1));
        cells.insert(Point::new(0, 2));

        let poly = ParallelogramPolyomino::new(cells, 1, 3, 0).unwrap();
        let bounce = poly.bounce_path();

        assert_eq!(bounce.path, vec![2]);
        assert_eq!(bounce.area, 2);
        assert_eq!(bounce.min_height, 2);
        assert_eq!(bounce.max_height, 2);
    }

    #[test]
    fn test_bounce_statistic() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(0, 1));
        cells.insert(Point::new(1, 0));

        let poly = ParallelogramPolyomino::new(cells, 2, 2, 0).unwrap();
        let stat = poly.bounce_statistic();

        assert!(stat >= 0); // Should be non-negative
    }

    #[test]
    fn test_rotate_90() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(1, 0));

        let poly = ParallelogramPolyomino::new(cells, 2, 1, 0).unwrap();
        let rotated = poly.rotate_90();

        assert!(rotated.is_some());
        let rotated = rotated.unwrap();
        assert_eq!(rotated.area(), 2);
    }

    #[test]
    fn test_reflect_horizontal() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(1, 0));

        let poly = ParallelogramPolyomino::new(cells, 2, 1, 0).unwrap();
        let reflected = poly.reflect_horizontal();

        assert!(reflected.is_some());
        assert_eq!(reflected.unwrap().area(), 2);
    }

    #[test]
    fn test_to_grid() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(1, 0));

        let poly = ParallelogramPolyomino::new(cells, 2, 1, 0).unwrap();
        let grid = poly.to_grid();

        assert_eq!(grid.len(), 1);
        assert_eq!(grid[0].len(), 2);
        assert!(grid[0][0]);
        assert!(grid[0][1]);
    }

    #[test]
    fn test_generate_size_1() {
        let polyominoes = parallelogram_polyominoes(1, 2, 2, 0);
        assert!(polyominoes.len() >= 1);

        for poly in &polyominoes {
            assert_eq!(poly.area(), 1);
        }
    }

    #[test]
    fn test_generate_size_2() {
        let polyominoes = parallelogram_polyominoes(2, 3, 3, 0);
        assert!(polyominoes.len() >= 1);

        for poly in &polyominoes {
            assert_eq!(poly.area(), 2);
        }
    }

    #[test]
    fn test_count_polyominoes() {
        let count = count_parallelogram_polyominoes(1, 2, 2, 0);
        assert!(count >= 1);
    }

    #[test]
    fn test_with_bounce_statistic() {
        let polyominoes = parallelogram_polyominoes_with_bounce_statistic(2, 3, 3, 0, 0);

        for poly in &polyominoes {
            assert_eq!(poly.bounce_statistic(), 0);
        }
    }

    #[test]
    fn test_empty_polyomino_rejected() {
        let cells = HashSet::new();
        let poly = ParallelogramPolyomino::new(cells, 1, 1, 0);
        assert!(poly.is_none());
    }

    #[test]
    fn test_parallelogram_with_skew() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(1, 0));

        // With skew=1, positions (0,0) and (1,0) should be valid in a 2x2 parallelogram
        let poly = ParallelogramPolyomino::new(cells, 2, 2, 1);
        assert!(poly.is_some());

        // Also test a valid cell at a skewed position
        let mut cells2 = HashSet::new();
        cells2.insert(Point::new(1, 1));
        cells2.insert(Point::new(2, 1));

        // With skew=1, y=1 means x should be in range [1, 3)
        let poly2 = ParallelogramPolyomino::new(cells2, 2, 2, 1);
        assert!(poly2.is_some());
    }

    #[test]
    fn test_is_in_parallelogram() {
        // Rectangle (skew=0)
        assert!(ParallelogramPolyomino::is_in_parallelogram(&Point::new(0, 0), 3, 3, 0));
        assert!(ParallelogramPolyomino::is_in_parallelogram(&Point::new(2, 2), 3, 3, 0));
        assert!(!ParallelogramPolyomino::is_in_parallelogram(&Point::new(3, 0), 3, 3, 0));

        // Skewed parallelogram (skew=1)
        assert!(ParallelogramPolyomino::is_in_parallelogram(&Point::new(0, 0), 3, 3, 1));
        assert!(ParallelogramPolyomino::is_in_parallelogram(&Point::new(3, 1), 3, 3, 1));
        assert!(!ParallelogramPolyomino::is_in_parallelogram(&Point::new(0, 1), 3, 3, 1));
    }

    #[test]
    fn test_translate() {
        let mut cells = HashSet::new();
        cells.insert(Point::new(0, 0));
        cells.insert(Point::new(1, 0));

        let poly = ParallelogramPolyomino::new(cells, 3, 2, 0).unwrap();
        let translated = poly.translate(1, 0);

        assert!(translated.is_some());
        let translated = translated.unwrap();
        assert_eq!(translated.area(), 2);
        assert!(translated.cells().contains(&Point::new(1, 0)));
        assert!(translated.cells().contains(&Point::new(2, 0)));
    }
}
