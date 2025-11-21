//! SVG rendering backend
//!
//! This module provides a backend for rendering graphics to SVG format.

use rustmath_colors::Color;
use rustmath_plot_core::{
    LineStyle, MarkerStyle, PlotError, PlotOptions, Point2D, RenderBackend, Result, TextOptions,
    Transform2D,
};
use svg::node::element::{
    path::Data, Circle as SvgCircle, Ellipse as SvgEllipse, Line as SvgLine, Path, Polygon,
    Rectangle as SvgRect, Text as SvgText,
};
use svg::node::element::{Definitions, Marker};
use svg::Document;

/// SVG rendering backend
///
/// This backend renders graphics primitives to SVG format.
pub struct SvgBackend {
    /// The SVG document being constructed
    document: Document,

    /// Width of the viewport in pixels
    #[allow(dead_code)]
    width: f64,

    /// Height of the viewport in pixels
    #[allow(dead_code)]
    height: f64,

    /// Current transformation matrix
    transform: Transform2D,

    /// Current path data (for path operations)
    current_path: Option<Data>,

    /// Current path options (for stroke/fill)
    current_path_options: Option<PlotOptions>,

    /// Marker counter for unique IDs
    marker_counter: usize,
}

impl SvgBackend {
    /// Create a new SVG backend
    ///
    /// # Arguments
    /// * `width` - Width of the SVG viewport in pixels
    /// * `height` - Height of the SVG viewport in pixels
    pub fn new(width: usize, height: usize) -> Self {
        let document = Document::new()
            .set("width", width)
            .set("height", height)
            .set("viewBox", (0, 0, width, height));

        Self {
            document,
            width: width as f64,
            height: height as f64,
            transform: Transform2D::identity(),
            current_path: None,
            current_path_options: None,
            marker_counter: 0,
        }
    }

    /// Convert a Color to an SVG color string
    fn color_to_svg(color: &Color) -> String {
        let rgb = color.rgb_tuple();
        format!("rgb({},{},{})",
            (rgb.0 * 255.0) as u8,
            (rgb.1 * 255.0) as u8,
            (rgb.2 * 255.0) as u8
        )
    }

    /// Convert a line style to an SVG stroke-dasharray string
    fn line_style_to_dasharray(style: &LineStyle) -> Option<String> {
        match style {
            LineStyle::Solid => None,
            LineStyle::Dashed => Some("5,5".to_string()),
            LineStyle::Dotted => Some("2,3".to_string()),
            LineStyle::DashDot => Some("5,3,2,3".to_string()),
            LineStyle::DashDotDot => Some("5,3,2,3,2,3".to_string()),
            LineStyle::None => None,
        }
    }


    /// Get a unique marker ID
    fn get_marker_id(&mut self) -> String {
        let id = format!("marker{}", self.marker_counter);
        self.marker_counter += 1;
        id
    }

    /// Create an arrowhead marker
    fn create_arrowhead(&mut self, options: &PlotOptions) -> String {
        let marker_id = self.get_marker_id();

        let marker = Marker::new()
            .set("id", marker_id.clone())
            .set("markerWidth", 10)
            .set("markerHeight", 10)
            .set("refX", 9)
            .set("refY", 3)
            .set("orient", "auto")
            .add(
                Path::new()
                    .set("d", "M0,0 L0,6 L9,3 z")
                    .set("fill", Self::color_to_svg(&options.color))
            );

        let defs = Definitions::new().add(marker);
        self.document = self.document.clone().add(defs);

        marker_id
    }
}

impl RenderBackend for SvgBackend {
    fn draw_line(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let mut data = Data::new().move_to((points[0].x, points[0].y));

        for point in &points[1..] {
            data = data.line_to((point.x, point.y));
        }

        let mut path = Path::new()
            .set("d", data)
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("stroke-opacity", options.alpha)
            .set("fill", "none");

        if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
            path = path.set("stroke-dasharray", dasharray);
        }

        self.document = self.document.clone().add(path);
        Ok(())
    }

    fn draw_polygon(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let points_str: Vec<String> = points
            .iter()
            .map(|p| format!("{},{}", p.x, p.y))
            .collect();
        let points_attr = points_str.join(" ");

        let mut polygon = Polygon::new()
            .set("points", points_attr.as_str())
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("stroke-opacity", options.alpha);

        if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
            polygon = polygon.set("stroke-dasharray", dasharray);
        }

        if options.fill {
            let fill_col = options.fill_color.as_ref().unwrap_or(&options.color);
            polygon = polygon
                .set("fill", Self::color_to_svg(fill_col))
                .set("fill-opacity", fill_col.alpha());
        } else {
            polygon = polygon.set("fill", "none");
        }

        self.document = self.document.clone().add(polygon);
        Ok(())
    }

    fn draw_circle(&mut self, center: Point2D, radius: f64, options: &PlotOptions) -> Result<()> {
        let mut circle = SvgCircle::new()
            .set("cx", center.x)
            .set("cy", center.y)
            .set("r", radius)
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("stroke-opacity", options.alpha);

        if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
            circle = circle.set("stroke-dasharray", dasharray);
        }

        if options.fill {
            let fill_col = options.fill_color.as_ref().unwrap_or(&options.color);
            circle = circle
                .set("fill", Self::color_to_svg(fill_col))
                .set("fill-opacity", fill_col.alpha());
        } else {
            circle = circle.set("fill", "none");
        }

        self.document = self.document.clone().add(circle);
        Ok(())
    }

    fn draw_ellipse(
        &mut self,
        center: Point2D,
        rx: f64,
        ry: f64,
        rotation: f64,
        options: &PlotOptions,
    ) -> Result<()> {
        let mut ellipse = SvgEllipse::new()
            .set("cx", center.x)
            .set("cy", center.y)
            .set("rx", rx)
            .set("ry", ry)
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("stroke-opacity", options.alpha);

        if rotation != 0.0 {
            ellipse = ellipse.set(
                "transform",
                format!("rotate({} {} {})", rotation, center.x, center.y),
            );
        }

        if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
            ellipse = ellipse.set("stroke-dasharray", dasharray);
        }

        if options.fill {
            let fill_col = options.fill_color.as_ref().unwrap_or(&options.color);
            ellipse = ellipse
                .set("fill", Self::color_to_svg(fill_col))
                .set("fill-opacity", fill_col.alpha());
        } else {
            ellipse = ellipse.set("fill", "none");
        }

        self.document = self.document.clone().add(ellipse);
        Ok(())
    }

    fn draw_rectangle(&mut self, min: Point2D, max: Point2D, options: &PlotOptions) -> Result<()> {
        let width = max.x - min.x;
        let height = max.y - min.y;

        let mut rect = SvgRect::new()
            .set("x", min.x)
            .set("y", min.y)
            .set("width", width)
            .set("height", height)
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("stroke-opacity", options.alpha);

        if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
            rect = rect.set("stroke-dasharray", dasharray);
        }

        if options.fill {
            let fill_col = options.fill_color.as_ref().unwrap_or(&options.color);
            rect = rect
                .set("fill", Self::color_to_svg(fill_col))
                .set("fill-opacity", fill_col.alpha());
        } else {
            rect = rect.set("fill", "none");
        }

        self.document = self.document.clone().add(rect);
        Ok(())
    }

    fn draw_arc(
        &mut self,
        center: Point2D,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        options: &PlotOptions,
    ) -> Result<()> {
        // Convert angles from degrees to radians
        let start_rad = start_angle.to_radians();
        let end_rad = end_angle.to_radians();

        // Calculate start and end points
        let start_x = center.x + radius * start_rad.cos();
        let start_y = center.y + radius * start_rad.sin();
        let end_x = center.x + radius * end_rad.cos();
        let end_y = center.y + radius * end_rad.sin();

        // Determine if we need the large arc flag
        let large_arc = if (end_angle - start_angle).abs() > 180.0 {
            1
        } else {
            0
        };

        // Create the arc path
        let data = Data::new()
            .move_to((start_x, start_y))
            .elliptical_arc_to((radius, radius, 0, large_arc, 1, end_x, end_y));

        let mut path = Path::new()
            .set("d", data)
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("stroke-opacity", options.alpha)
            .set("fill", "none");

        if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
            path = path.set("stroke-dasharray", dasharray);
        }

        self.document = self.document.clone().add(path);
        Ok(())
    }

    fn draw_bezier(
        &mut self,
        p0: Point2D,
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        options: &PlotOptions,
    ) -> Result<()> {
        let data = Data::new()
            .move_to((p0.x, p0.y))
            .cubic_curve_to((p1.x, p1.y, p2.x, p2.y, p3.x, p3.y));

        let mut path = Path::new()
            .set("d", data)
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("stroke-opacity", options.alpha)
            .set("fill", "none");

        if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
            path = path.set("stroke-dasharray", dasharray);
        }

        self.document = self.document.clone().add(path);
        Ok(())
    }

    fn draw_text(&mut self, position: Point2D, text: &str, options: &TextOptions) -> Result<()> {
        let svg_text = SvgText::new(text)
            .set("x", position.x)
            .set("y", position.y)
            .set("font-size", options.font_size)
            .set("fill", Self::color_to_svg(&options.color))
            .set("text-anchor", "middle");

        self.document = self.document.clone().add(svg_text);
        Ok(())
    }

    fn draw_points(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()> {
        // Draw points as small circles
        let marker_size = options.marker_size;

        for point in points {
            match options.marker {
                MarkerStyle::Circle => {
                    let circle = SvgCircle::new()
                        .set("cx", point.x)
                        .set("cy", point.y)
                        .set("r", marker_size / 2.0)
                        .set("fill", Self::color_to_svg(&options.color))
                        .set("fill-opacity", options.alpha);

                    self.document = self.document.clone().add(circle);
                }
                MarkerStyle::Square => {
                    let rect = SvgRect::new()
                        .set("x", point.x - marker_size / 2.0)
                        .set("y", point.y - marker_size / 2.0)
                        .set("width", marker_size)
                        .set("height", marker_size)
                        .set("fill", Self::color_to_svg(&options.color))
                        .set("fill-opacity", options.alpha);

                    self.document = self.document.clone().add(rect);
                }
                MarkerStyle::Plus => {
                    // Draw a plus sign
                    let line1 = SvgLine::new()
                        .set("x1", point.x - marker_size / 2.0)
                        .set("y1", point.y)
                        .set("x2", point.x + marker_size / 2.0)
                        .set("y2", point.y)
                        .set("stroke", Self::color_to_svg(&options.color))
                        .set("stroke-width", 1);

                    let line2 = SvgLine::new()
                        .set("x1", point.x)
                        .set("y1", point.y - marker_size / 2.0)
                        .set("x2", point.x)
                        .set("y2", point.y + marker_size / 2.0)
                        .set("stroke", Self::color_to_svg(&options.color))
                        .set("stroke-width", 1);

                    self.document = self.document.clone().add(line1).add(line2);
                }
                MarkerStyle::Cross => {
                    // Draw an X
                    let line1 = SvgLine::new()
                        .set("x1", point.x - marker_size / 2.0)
                        .set("y1", point.y - marker_size / 2.0)
                        .set("x2", point.x + marker_size / 2.0)
                        .set("y2", point.y + marker_size / 2.0)
                        .set("stroke", Self::color_to_svg(&options.color))
                        .set("stroke-width", 1);

                    let line2 = SvgLine::new()
                        .set("x1", point.x - marker_size / 2.0)
                        .set("y1", point.y + marker_size / 2.0)
                        .set("x2", point.x + marker_size / 2.0)
                        .set("y2", point.y - marker_size / 2.0)
                        .set("stroke", Self::color_to_svg(&options.color))
                        .set("stroke-width", 1);

                    self.document = self.document.clone().add(line1).add(line2);
                }
                MarkerStyle::None => {}
                _ => {
                    // Other marker styles not yet implemented for SVG backend
                    // Fall back to simple circle
                    let circle = SvgCircle::new()
                        .set("cx", point.x)
                        .set("cy", point.y)
                        .set("r", marker_size / 2.0)
                        .set("fill", Self::color_to_svg(&options.color))
                        .set("fill-opacity", options.alpha);

                    self.document = self.document.clone().add(circle);
                }
            }
        }

        Ok(())
    }

    fn draw_arrow(&mut self, start: Point2D, end: Point2D, options: &PlotOptions) -> Result<()> {
        let marker_id = self.create_arrowhead(options);

        let line = SvgLine::new()
            .set("x1", start.x)
            .set("y1", start.y)
            .set("x2", end.x)
            .set("y2", end.y)
            .set("stroke", Self::color_to_svg(&options.color))
            .set("stroke-width", options.thickness)
            .set("marker-end", format!("url(#{})", marker_id));

        self.document = self.document.clone().add(line);
        Ok(())
    }

    fn begin_path(&mut self) -> Result<()> {
        self.current_path = Some(Data::new());
        self.current_path_options = None;
        Ok(())
    }

    fn move_to(&mut self, point: Point2D) -> Result<()> {
        if let Some(data) = &self.current_path {
            self.current_path = Some(data.clone().move_to((point.x, point.y)));
        } else {
            return Err(PlotError::RenderError(
                "begin_path() must be called before move_to()".to_string(),
            ));
        }
        Ok(())
    }

    fn line_to(&mut self, point: Point2D) -> Result<()> {
        if let Some(data) = &self.current_path {
            self.current_path = Some(data.clone().line_to((point.x, point.y)));
        } else {
            return Err(PlotError::RenderError(
                "begin_path() must be called before line_to()".to_string(),
            ));
        }
        Ok(())
    }

    fn close_path(&mut self) -> Result<()> {
        if let Some(data) = &self.current_path {
            self.current_path = Some(data.clone().close());
        } else {
            return Err(PlotError::RenderError(
                "begin_path() must be called before close_path()".to_string(),
            ));
        }
        Ok(())
    }

    fn stroke(&mut self, options: &PlotOptions) -> Result<()> {
        if let Some(data) = self.current_path.take() {
            let mut path = Path::new()
                .set("d", data)
                .set("stroke", Self::color_to_svg(&options.color))
                .set("stroke-width", options.thickness)
                .set("stroke-opacity", options.alpha)
                .set("fill", "none");

            if let Some(dasharray) = Self::line_style_to_dasharray(&options.linestyle) {
                path = path.set("stroke-dasharray", dasharray);
            }

            self.document = self.document.clone().add(path);
        } else {
            return Err(PlotError::RenderError(
                "No path to stroke".to_string(),
            ));
        }
        Ok(())
    }

    fn fill(&mut self, options: &PlotOptions) -> Result<()> {
        if let Some(data) = self.current_path.take() {
            let fill_col = options.fill_color.as_ref().unwrap_or(&options.color);
            let path = Path::new()
                .set("d", data)
                .set("fill", Self::color_to_svg(fill_col))
                .set("fill-opacity", fill_col.alpha());

            self.document = self.document.clone().add(path);
        } else {
            return Err(PlotError::RenderError(
                "No path to fill".to_string(),
            ));
        }
        Ok(())
    }

    fn get_transform(&self) -> Transform2D {
        self.transform
    }

    fn set_transform(&mut self, transform: Transform2D) {
        self.transform = transform;
    }

    fn finalize(&mut self) -> Result<Vec<u8>> {
        let svg_string = self.document.to_string();
        Ok(svg_string.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_plot_core::PlotOptions;

    #[test]
    fn test_svg_backend_creation() {
        let backend = SvgBackend::new(800, 600);
        assert_eq!(backend.width, 800.0);
        assert_eq!(backend.height, 600.0);
    }

    #[test]
    fn test_svg_draw_line() {
        let mut backend = SvgBackend::new(800, 600);
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(100.0, 100.0)];
        let result = backend.draw_line(&points, &PlotOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_svg_draw_circle() {
        let mut backend = SvgBackend::new(800, 600);
        let result = backend.draw_circle(Point2D::new(50.0, 50.0), 25.0, &PlotOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_svg_finalize() {
        let mut backend = SvgBackend::new(800, 600);
        let result = backend.finalize();
        assert!(result.is_ok());
        let svg_data = result.unwrap();
        assert!(!svg_data.is_empty());

        // Check that it's valid SVG
        let svg_string = String::from_utf8(svg_data).unwrap();
        assert!(svg_string.contains("<svg"));
    }

    #[test]
    fn test_color_to_svg() {
        let color = Color::rgb(1.0, 0.0, 0.0);
        let svg_color = SvgBackend::color_to_svg(&color);
        assert_eq!(svg_color, "rgb(255,0,0)");
    }
}
