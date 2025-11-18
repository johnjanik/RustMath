//! Chessboard graph generators
//!
//! This module provides functions to generate graphs based on chessboard piece movements.

use crate::graph::Graph;

/// Generate a Bishop graph
///
/// The bishop graph represents possible moves of a bishop on an m×n chessboard.
/// Two squares are adjacent if a bishop can move between them in one move.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::chessboard::bishop_graph;
///
/// let g = bishop_graph(4, 4);
/// assert_eq!(g.num_vertices(), 16);
/// ```
pub fn bishop_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m * n);

    for i1 in 0..m {
        for j1 in 0..n {
            let v1 = i1 * n + j1;

            for i2 in (i1 + 1)..m {
                for j2 in 0..n {
                    let v2 = i2 * n + j2;

                    // Check if on same diagonal
                    let di = (i2 as i32 - i1 as i32).abs();
                    let dj = (j2 as i32 - j1 as i32).abs();

                    if di == dj {
                        g.add_edge(v1, v2).unwrap();
                    }
                }
            }
        }
    }

    g
}

/// Generate a King graph
///
/// The king graph represents possible moves of a king on an m×n chessboard.
/// Two squares are adjacent if a king can move between them in one move
/// (horizontally, vertically, or diagonally).
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::chessboard::king_graph;
///
/// let g = king_graph(4, 4);
/// assert_eq!(g.num_vertices(), 16);
/// ```
pub fn king_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m * n);

    for i in 0..m {
        for j in 0..n {
            let v = i * n + j;

            // Eight possible king moves
            let moves = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ];

            for (di, dj) in moves.iter() {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;

                if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                    let u = (ni as usize) * n + (nj as usize);
                    if v < u {
                        g.add_edge(v, u).unwrap();
                    }
                }
            }
        }
    }

    g
}

/// Generate a Knight graph
///
/// The knight graph represents possible moves of a knight on an m×n chessboard.
/// Two squares are adjacent if a knight can move between them in one move.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::chessboard::knight_graph;
///
/// let g = knight_graph(5, 5);
/// assert_eq!(g.num_vertices(), 25);
/// ```
pub fn knight_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m * n);

    for i in 0..m {
        for j in 0..n {
            let v = i * n + j;

            // Eight possible knight moves
            let moves = [
                (-2, -1), (-2, 1),
                (-1, -2), (-1, 2),
                (1, -2),  (1, 2),
                (2, -1),  (2, 1),
            ];

            for (di, dj) in moves.iter() {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;

                if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                    let u = (ni as usize) * n + (nj as usize);
                    if v < u {
                        g.add_edge(v, u).unwrap();
                    }
                }
            }
        }
    }

    g
}

/// Generate a Queen graph
///
/// The queen graph represents possible moves of a queen on an m×n chessboard.
/// Two squares are adjacent if a queen can move between them in one move
/// (horizontally, vertically, or diagonally).
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::chessboard::queen_graph;
///
/// let g = queen_graph(4, 4);
/// assert_eq!(g.num_vertices(), 16);
/// ```
pub fn queen_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m * n);

    for i1 in 0..m {
        for j1 in 0..n {
            let v1 = i1 * n + j1;

            for i2 in 0..m {
                for j2 in 0..n {
                    if i1 == i2 && j1 == j2 {
                        continue;
                    }

                    let v2 = i2 * n + j2;

                    if v1 < v2 {
                        // Check if on same row, column, or diagonal
                        let same_row = i1 == i2;
                        let same_col = j1 == j2;
                        let same_diag = (i2 as i32 - i1 as i32).abs() == (j2 as i32 - j1 as i32).abs();

                        if same_row || same_col || same_diag {
                            g.add_edge(v1, v2).unwrap();
                        }
                    }
                }
            }
        }
    }

    g
}

/// Generate a Rook graph
///
/// The rook graph represents possible moves of a rook on an m×n chessboard.
/// Two squares are adjacent if a rook can move between them in one move
/// (horizontally or vertically).
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::chessboard::rook_graph;
///
/// let g = rook_graph(4, 4);
/// assert_eq!(g.num_vertices(), 16);
/// ```
pub fn rook_graph(m: usize, n: usize) -> Graph {
    let mut g = Graph::new(m * n);

    // Connect all squares in the same row
    for i in 0..m {
        for j1 in 0..n {
            for j2 in (j1 + 1)..n {
                let v1 = i * n + j1;
                let v2 = i * n + j2;
                g.add_edge(v1, v2).unwrap();
            }
        }
    }

    // Connect all squares in the same column
    for j in 0..n {
        for i1 in 0..m {
            for i2 in (i1 + 1)..m {
                let v1 = i1 * n + j;
                let v2 = i2 * n + j;
                g.add_edge(v1, v2).unwrap();
            }
        }
    }

    g
}

/// Generate a chessboard graph using a custom movement generator
///
/// This is a general function that creates a graph based on arbitrary piece movements.
///
/// # Arguments
///
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `moves` - Function that returns valid moves from a position
///
/// # Examples
///
/// ```
/// use rustmath_graphs::generators::chessboard::chessboard_graph_generator;
///
/// // Generate a knight graph using the general generator
/// let knight_moves = |_i: usize, _j: usize| vec![
///     (-2, -1), (-2, 1), (-1, -2), (-1, 2),
///     (1, -2), (1, 2), (2, -1), (2, 1)
/// ];
/// let g = chessboard_graph_generator(5, 5, knight_moves);
/// assert_eq!(g.num_vertices(), 25);
/// ```
pub fn chessboard_graph_generator<F>(m: usize, n: usize, moves: F) -> Graph
where
    F: Fn(usize, usize) -> Vec<(i32, i32)>,
{
    let mut g = Graph::new(m * n);

    for i in 0..m {
        for j in 0..n {
            let v = i * n + j;
            let move_list = moves(i, j);

            for (di, dj) in move_list {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;

                if ni >= 0 && ni < m as i32 && nj >= 0 && nj < n as i32 {
                    let u = (ni as usize) * n + (nj as usize);
                    if v < u {
                        g.add_edge(v, u).unwrap();
                    }
                }
            }
        }
    }

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rook_graph() {
        let g = rook_graph(3, 3);
        assert_eq!(g.num_vertices(), 9);
        // Each square can reach 4 others in same row/column
        // Corner squares have degree 4, edge squares 4, center 4
        // Actually in a 3x3: each square connects to 4 others (2 in row, 2 in column)
        for i in 0..9 {
            assert_eq!(g.degree(i), Some(4));
        }
    }

    #[test]
    fn test_bishop_graph() {
        let g = bishop_graph(4, 4);
        assert_eq!(g.num_vertices(), 16);
    }

    #[test]
    fn test_knight_graph() {
        let g = knight_graph(5, 5);
        assert_eq!(g.num_vertices(), 25);
    }

    #[test]
    fn test_queen_graph() {
        let g = queen_graph(3, 3);
        assert_eq!(g.num_vertices(), 9);
        // Center square connects to all 8 others
        assert_eq!(g.degree(4), Some(8));
    }

    #[test]
    fn test_king_graph() {
        let g = king_graph(3, 3);
        assert_eq!(g.num_vertices(), 9);
        // Center square connects to all 8 neighbors
        assert_eq!(g.degree(4), Some(8));
        // Corner squares connect to 3 neighbors
        assert_eq!(g.degree(0), Some(3));
    }

    #[test]
    fn test_chessboard_generator() {
        // Test with knight moves
        let knight_moves = |_i: usize, _j: usize| vec![
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ];
        let g = chessboard_graph_generator(5, 5, knight_moves);
        assert_eq!(g.num_vertices(), 25);
    }
}
