//! Plotting options and styles
//!
//! Based on SageMath's plotting option system from sage.plot.graphics

use rustmath_colors::Color;

/// Line style for drawing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot pattern
    DashDot,
    /// Dash-dot-dot pattern
    DashDotDot,
    /// No line (invisible)
    None,
}

impl LineStyle {
    /// Get the matplotlib-style line style string
    pub fn to_matplotlib_string(&self) -> &'static str {
        match self {
            LineStyle::Solid => "-",
            LineStyle::Dashed => "--",
            LineStyle::Dotted => ":",
            LineStyle::DashDot => "-.",
            LineStyle::DashDotDot => "-..",
            LineStyle::None => "",
        }
    }
}

/// Marker style for points
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerStyle {
    /// No marker
    None,
    /// Circle
    Circle,
    /// Square
    Square,
    /// Triangle (pointing up)
    Triangle,
    /// Triangle (pointing down)
    TriangleDown,
    /// Diamond
    Diamond,
    /// Plus sign
    Plus,
    /// Cross (x)
    Cross,
    /// Star
    Star,
    /// Pentagon
    Pentagon,
    /// Hexagon
    Hexagon,
}

impl MarkerStyle {
    /// Get the matplotlib-style marker string
    pub fn to_matplotlib_string(&self) -> &'static str {
        match self {
            MarkerStyle::None => "",
            MarkerStyle::Circle => "o",
            MarkerStyle::Square => "s",
            MarkerStyle::Triangle => "^",
            MarkerStyle::TriangleDown => "v",
            MarkerStyle::Diamond => "D",
            MarkerStyle::Plus => "+",
            MarkerStyle::Cross => "x",
            MarkerStyle::Star => "*",
            MarkerStyle::Pentagon => "p",
            MarkerStyle::Hexagon => "h",
        }
    }
}

/// Options for rendering graphics primitives
///
/// Based on SageMath's GraphicPrimitive options
#[derive(Debug, Clone)]
pub struct PlotOptions {
    /// Color of the graphic
    pub color: Color,

    /// Line thickness (width)
    pub thickness: f64,

    /// Alpha transparency (0.0 = fully transparent, 1.0 = fully opaque)
    pub alpha: f64,

    /// Line style
    pub linestyle: LineStyle,

    /// Marker style for points
    pub marker: MarkerStyle,

    /// Marker size
    pub marker_size: f64,

    /// Fill color (for filled shapes)
    pub fill_color: Option<Color>,

    /// Whether to fill the shape
    pub fill: bool,

    /// Z-order for layering (higher values are drawn on top)
    pub zorder: i32,

    /// Label for legend
    pub label: Option<String>,
}

impl PlotOptions {
    /// Create default plot options
    pub fn new() -> Self {
        Self {
            color: Color::blue_color(),
            thickness: 1.0,
            alpha: 1.0,
            linestyle: LineStyle::Solid,
            marker: MarkerStyle::None,
            marker_size: 5.0,
            fill_color: None,
            fill: false,
            zorder: 0,
            label: None,
        }
    }

    /// Set the color
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the thickness
    pub fn with_thickness(mut self, thickness: f64) -> Self {
        self.thickness = thickness;
        self
    }

    /// Set the alpha transparency
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set the line style
    pub fn with_linestyle(mut self, linestyle: LineStyle) -> Self {
        self.linestyle = linestyle;
        self
    }

    /// Set the marker style
    pub fn with_marker(mut self, marker: MarkerStyle) -> Self {
        self.marker = marker;
        self
    }

    /// Set the marker size
    pub fn with_marker_size(mut self, size: f64) -> Self {
        self.marker_size = size;
        self
    }

    /// Set the fill color and enable fill
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill_color = Some(color);
        self.fill = true;
        self
    }

    /// Set the z-order
    pub fn with_zorder(mut self, zorder: i32) -> Self {
        self.zorder = zorder;
        self
    }

    /// Set the label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Text rendering options
#[derive(Debug, Clone)]
pub struct TextOptions {
    /// Font family
    pub font_family: String,

    /// Font size in points
    pub font_size: f64,

    /// Font color
    pub color: Color,

    /// Horizontal alignment
    pub horizontal_align: HorizontalAlign,

    /// Vertical alignment
    pub vertical_align: VerticalAlign,

    /// Rotation angle in degrees
    pub rotation: f64,

    /// Background color (if any)
    pub background_color: Option<Color>,
}

impl TextOptions {
    /// Create default text options
    pub fn new() -> Self {
        Self {
            font_family: "sans-serif".to_string(),
            font_size: 12.0,
            color: Color::black(),
            horizontal_align: HorizontalAlign::Center,
            vertical_align: VerticalAlign::Middle,
            rotation: 0.0,
            background_color: None,
        }
    }
}

impl Default for TextOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Horizontal text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizontalAlign {
    Left,
    Center,
    Right,
}

/// Vertical text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalAlign {
    Top,
    Middle,
    Bottom,
}

/// Options for axes
#[derive(Debug, Clone)]
pub struct AxesOptions {
    /// Whether to show axes
    pub show_axes: bool,

    /// Whether to show a frame around the plot
    pub frame: bool,

    /// Whether to show grid lines
    pub gridlines: bool,

    /// Grid line color
    pub gridline_color: Color,

    /// Grid line style
    pub gridline_style: LineStyle,

    /// X-axis label
    pub xlabel: Option<String>,

    /// Y-axis label
    pub ylabel: Option<String>,

    /// Whether to use equal aspect ratio
    pub aspect_ratio: Option<f64>,
}

impl AxesOptions {
    /// Create default axes options
    pub fn new() -> Self {
        Self {
            show_axes: true,
            frame: false,
            gridlines: false,
            gridline_color: Color::gray_color().with_alpha(0.3),
            gridline_style: LineStyle::Dotted,
            xlabel: None,
            ylabel: None,
            aspect_ratio: None,
        }
    }
}

impl Default for AxesOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for overall graphics
#[derive(Debug, Clone)]
pub struct GraphicsOptions {
    /// Plot title
    pub title: Option<String>,

    /// Axes options
    pub axes: AxesOptions,

    /// Figure size in pixels
    pub figsize: (usize, usize),

    /// DPI (dots per inch) for raster output
    pub dpi: usize,

    /// Background color
    pub background_color: Color,

    /// Whether to show a legend
    pub show_legend: bool,

    /// Legend position
    pub legend_position: LegendPosition,
}

impl GraphicsOptions {
    /// Create default graphics options
    pub fn new() -> Self {
        Self {
            title: None,
            axes: AxesOptions::new(),
            figsize: (800, 600),
            dpi: 96,
            background_color: Color::white(),
            show_legend: false,
            legend_position: LegendPosition::BestFit,
        }
    }
}

impl Default for GraphicsOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Legend position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendPosition {
    BestFit,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_options() {
        let options = PlotOptions::new()
            .with_color(Color::red_color())
            .with_thickness(2.0)
            .with_alpha(0.5);

        assert_eq!(options.color, Color::red_color());
        assert_eq!(options.thickness, 2.0);
        assert_eq!(options.alpha, 0.5);
    }

    #[test]
    fn test_linestyle() {
        assert_eq!(LineStyle::Solid.to_matplotlib_string(), "-");
        assert_eq!(LineStyle::Dashed.to_matplotlib_string(), "--");
        assert_eq!(LineStyle::Dotted.to_matplotlib_string(), ":");
    }

    #[test]
    fn test_marker_style() {
        assert_eq!(MarkerStyle::Circle.to_matplotlib_string(), "o");
        assert_eq!(MarkerStyle::Square.to_matplotlib_string(), "s");
    }

    #[test]
    fn test_text_options() {
        let text_opts = TextOptions::new();
        assert_eq!(text_opts.font_size, 12.0);
        assert_eq!(text_opts.horizontal_align, HorizontalAlign::Center);
    }

    #[test]
    fn test_axes_options() {
        let axes_opts = AxesOptions::new();
        assert!(axes_opts.show_axes);
        assert!(!axes_opts.frame);
        assert!(!axes_opts.gridlines);
    }

    #[test]
    fn test_graphics_options() {
        let gfx_opts = GraphicsOptions::new();
        assert_eq!(gfx_opts.figsize, (800, 600));
        assert_eq!(gfx_opts.dpi, 96);
    }
}
