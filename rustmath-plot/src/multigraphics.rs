//! Multi-graphics for arranging multiple plots in a grid
//!
//! Based on SageMath's sage.plot.multigraphics module

use crate::Graphics;
use rustmath_plot_core::{PlotError, RenderFormat, Result};
use std::path::Path;

/// Layout configuration for arranging multiple graphics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridLayout {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

impl GridLayout {
    /// Create a new grid layout
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Panics
    /// Panics if rows or cols is 0
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "rows must be > 0");
        assert!(cols > 0, "cols must be > 0");
        Self { rows, cols }
    }

    /// Get the total number of cells in the grid
    pub fn total_cells(&self) -> usize {
        self.rows * self.cols
    }

    /// Get the (row, col) position for a given index
    ///
    /// # Arguments
    /// * `index` - The linear index (0-based)
    ///
    /// # Returns
    /// `Some((row, col))` if the index is valid, `None` otherwise
    pub fn position(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.total_cells() {
            return None;
        }
        Some((index / self.cols, index % self.cols))
    }

    /// Get the index for a given (row, col) position
    ///
    /// # Returns
    /// `Some(index)` if the position is valid, `None` otherwise
    pub fn index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        Some(row * self.cols + col)
    }
}

/// Alignment for graphics within grid cells
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// Align to top-left
    TopLeft,
    /// Align to top-center
    TopCenter,
    /// Align to top-right
    TopRight,
    /// Align to middle-left
    MiddleLeft,
    /// Align to center
    Center,
    /// Align to middle-right
    MiddleRight,
    /// Align to bottom-left
    BottomLeft,
    /// Align to bottom-center
    BottomCenter,
    /// Align to bottom-right
    BottomRight,
}

/// Options for multi-graphics layout
#[derive(Debug, Clone)]
pub struct MultiGraphicsOptions {
    /// Spacing between graphics (in pixels)
    pub spacing: (usize, usize),

    /// Padding around the entire grid (in pixels)
    pub padding: (usize, usize),

    /// Alignment of graphics within cells
    pub alignment: Alignment,

    /// Whether to use uniform sizing for all graphics
    pub uniform_size: bool,
}

impl MultiGraphicsOptions {
    /// Create default multi-graphics options
    pub fn new() -> Self {
        Self {
            spacing: (10, 10),
            padding: (20, 20),
            alignment: Alignment::Center,
            uniform_size: true,
        }
    }

    /// Set the spacing between graphics
    pub fn with_spacing(mut self, horizontal: usize, vertical: usize) -> Self {
        self.spacing = (horizontal, vertical);
        self
    }

    /// Set the padding around the grid
    pub fn with_padding(mut self, horizontal: usize, vertical: usize) -> Self {
        self.padding = (horizontal, vertical);
        self
    }

    /// Set the alignment
    pub fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set whether to use uniform sizing
    pub fn with_uniform_size(mut self, uniform: bool) -> Self {
        self.uniform_size = uniform;
        self
    }
}

impl Default for MultiGraphicsOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// A container for multiple Graphics arranged in a grid
///
/// This allows you to create subplot-style layouts with multiple
/// plots arranged in rows and columns.
///
/// Based on SageMath's GraphicsArray and MultiGraphics classes.
pub struct MultiGraphics {
    /// The graphics to display in the grid
    graphics: Vec<Option<Graphics>>,

    /// The grid layout
    layout: GridLayout,

    /// Layout options
    options: MultiGraphicsOptions,
}

impl MultiGraphics {
    /// Create a new MultiGraphics with the specified grid layout
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Examples
    /// ```
    /// use rustmath_plot::MultiGraphics;
    ///
    /// let mg = MultiGraphics::new(2, 3); // 2 rows, 3 columns
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        let layout = GridLayout::new(rows, cols);
        let total = layout.total_cells();
        Self {
            graphics: (0..total).map(|_| None).collect(),
            layout,
            options: MultiGraphicsOptions::default(),
        }
    }

    /// Create a new MultiGraphics with custom options
    pub fn with_options(rows: usize, cols: usize, options: MultiGraphicsOptions) -> Self {
        let layout = GridLayout::new(rows, cols);
        let total = layout.total_cells();
        Self {
            graphics: (0..total).map(|_| None).collect(),
            layout,
            options,
        }
    }

    /// Get the grid layout
    pub fn layout(&self) -> &GridLayout {
        &self.layout
    }

    /// Get the options
    pub fn options(&self) -> &MultiGraphicsOptions {
        &self.options
    }

    /// Get a mutable reference to the options
    pub fn options_mut(&mut self) -> &mut MultiGraphicsOptions {
        &mut self.options
    }

    /// Set a graphic at the specified position
    ///
    /// # Arguments
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    /// * `graphic` - The Graphics to place at this position
    ///
    /// # Returns
    /// `Ok(())` if successful, `Err` if the position is invalid
    ///
    /// # Examples
    /// ```ignore
    /// let mut mg = MultiGraphics::new(2, 2);
    /// mg.set(0, 0, g1)?;
    /// mg.set(0, 1, g2)?;
    /// ```
    pub fn set(&mut self, row: usize, col: usize, graphic: Graphics) -> Result<()> {
        let index = self
            .layout
            .index(row, col)
            .ok_or_else(|| PlotError::InvalidOption(format!("Invalid position ({}, {})", row, col)))?;
        self.graphics[index] = Some(graphic);
        Ok(())
    }

    /// Set a graphic at the specified linear index
    ///
    /// # Arguments
    /// * `index` - Linear index (0-based, row-major order)
    /// * `graphic` - The Graphics to place at this position
    ///
    /// # Examples
    /// ```ignore
    /// let mut mg = MultiGraphics::new(2, 2);
    /// mg.set_index(0, g1)?;
    /// mg.set_index(1, g2)?;
    /// ```
    pub fn set_index(&mut self, index: usize, graphic: Graphics) -> Result<()> {
        if index >= self.graphics.len() {
            return Err(PlotError::InvalidOption(format!(
                "Invalid index {} (max {})",
                index,
                self.graphics.len() - 1
            )));
        }
        self.graphics[index] = Some(graphic);
        Ok(())
    }

    /// Get a reference to the graphic at the specified position
    ///
    /// # Returns
    /// `Some(&Graphics)` if a graphic exists at this position, `None` otherwise
    pub fn get(&self, row: usize, col: usize) -> Option<&Graphics> {
        let index = self.layout.index(row, col)?;
        self.graphics[index].as_ref()
    }

    /// Get a mutable reference to the graphic at the specified position
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Graphics> {
        let index = self.layout.index(row, col)?;
        self.graphics[index].as_mut()
    }

    /// Get a reference to the graphic at the specified linear index
    pub fn get_index(&self, index: usize) -> Option<&Graphics> {
        self.graphics.get(index)?.as_ref()
    }

    /// Get a mutable reference to the graphic at the specified linear index
    pub fn get_index_mut(&mut self, index: usize) -> Option<&mut Graphics> {
        self.graphics.get_mut(index)?.as_mut()
    }

    /// Get an iterator over all graphics (including empty cells as None)
    pub fn iter(&self) -> impl Iterator<Item = &Option<Graphics>> {
        self.graphics.iter()
    }

    /// Get a mutable iterator over all graphics
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Option<Graphics>> {
        self.graphics.iter_mut()
    }

    /// Get an iterator over only the non-empty graphics with their positions
    pub fn iter_graphics(&self) -> impl Iterator<Item = (usize, usize, &Graphics)> {
        self.graphics
            .iter()
            .enumerate()
            .filter_map(|(idx, g)| {
                g.as_ref()
                    .and_then(|graphic| self.layout.position(idx).map(|(r, c)| (r, c, graphic)))
            })
    }

    /// Count the number of non-empty cells
    pub fn count_graphics(&self) -> usize {
        self.graphics.iter().filter(|g| g.is_some()).count()
    }

    /// Check if all cells are empty
    pub fn is_empty(&self) -> bool {
        self.graphics.iter().all(|g| g.is_none())
    }

    /// Render this MultiGraphics using the provided backend
    ///
    /// This is a placeholder - actual implementation would require
    /// calculating cell positions and rendering each graphic in its cell.
    pub fn render(&self, _backend: &mut dyn rustmath_plot_core::RenderBackend) -> Result<()> {
        // TODO: Implement proper multi-graphics rendering
        // This would involve:
        // 1. Calculate cell dimensions based on layout and options
        // 2. For each non-empty cell:
        //    a. Set up a transform to position the graphic in its cell
        //    b. Render the graphic with the transform
        // 3. Optionally add borders between cells
        Err(PlotError::RenderError(
            "MultiGraphics rendering not yet implemented".to_string(),
        ))
    }

    /// Save this MultiGraphics to a file
    ///
    /// # Arguments
    /// * `path` - The file path to save to
    /// * `format` - The output format
    pub fn save(&self, _path: impl AsRef<Path>, _format: RenderFormat) -> Result<()> {
        // Placeholder - would need to create appropriate backend
        Err(PlotError::RenderError(
            "MultiGraphics save() not yet implemented".to_string(),
        ))
    }

    /// Display this MultiGraphics
    pub fn show(&self) -> Result<()> {
        // Placeholder - would need display backend
        Err(PlotError::RenderError(
            "MultiGraphics show() not yet implemented".to_string(),
        ))
    }
}

/// Type alias for GraphicsArray (SageMath compatibility)
pub type GraphicsArray = MultiGraphics;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_layout() {
        let layout = GridLayout::new(2, 3);
        assert_eq!(layout.total_cells(), 6);
        assert_eq!(layout.position(0), Some((0, 0)));
        assert_eq!(layout.position(2), Some((0, 2)));
        assert_eq!(layout.position(3), Some((1, 0)));
        assert_eq!(layout.position(5), Some((1, 2)));
        assert_eq!(layout.position(6), None);

        assert_eq!(layout.index(0, 0), Some(0));
        assert_eq!(layout.index(0, 2), Some(2));
        assert_eq!(layout.index(1, 0), Some(3));
        assert_eq!(layout.index(1, 2), Some(5));
        assert_eq!(layout.index(2, 0), None);
    }

    #[test]
    #[should_panic(expected = "rows must be > 0")]
    fn test_grid_layout_zero_rows() {
        GridLayout::new(0, 3);
    }

    #[test]
    #[should_panic(expected = "cols must be > 0")]
    fn test_grid_layout_zero_cols() {
        GridLayout::new(2, 0);
    }

    #[test]
    fn test_multigraphics_new() {
        let mg = MultiGraphics::new(2, 3);
        assert_eq!(mg.layout().rows, 2);
        assert_eq!(mg.layout().cols, 3);
        assert!(mg.is_empty());
        assert_eq!(mg.count_graphics(), 0);
    }

    #[test]
    fn test_multigraphics_set_get() {
        let mut mg = MultiGraphics::new(2, 2);
        let g = Graphics::new();

        mg.set(0, 0, g).unwrap();
        assert_eq!(mg.count_graphics(), 1);
        assert!(!mg.is_empty());
        assert!(mg.get(0, 0).is_some());
        assert!(mg.get(0, 1).is_none());
    }

    #[test]
    fn test_multigraphics_set_index() {
        let mut mg = MultiGraphics::new(2, 2);
        let g = Graphics::new();

        mg.set_index(0, g).unwrap();
        assert!(mg.get_index(0).is_some());
        assert!(mg.get_index(1).is_none());
    }

    #[test]
    fn test_multigraphics_invalid_position() {
        let mut mg = MultiGraphics::new(2, 2);
        let g = Graphics::new();

        assert!(mg.set(2, 0, g).is_err());
    }

    #[test]
    fn test_multigraphics_options() {
        let options = MultiGraphicsOptions::new()
            .with_spacing(20, 20)
            .with_padding(10, 10)
            .with_alignment(Alignment::TopLeft)
            .with_uniform_size(false);

        assert_eq!(options.spacing, (20, 20));
        assert_eq!(options.padding, (10, 10));
        assert_eq!(options.alignment, Alignment::TopLeft);
        assert!(!options.uniform_size);
    }

    #[test]
    fn test_multigraphics_iter() {
        let mut mg = MultiGraphics::new(2, 2);
        mg.set(0, 0, Graphics::new()).unwrap();
        mg.set(1, 1, Graphics::new()).unwrap();

        let count = mg.iter().filter(|g| g.is_some()).count();
        assert_eq!(count, 2);

        let graphics_count = mg.iter_graphics().count();
        assert_eq!(graphics_count, 2);
    }
}
