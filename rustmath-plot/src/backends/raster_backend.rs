//! Raster rendering backend for PNG/JPEG output
//!
//! This module provides a backend for rendering graphics to raster formats
//! like PNG and JPEG using the tiny-skia library for software rendering.

use rustmath_colors::Color;
use rustmath_plot_core::{
    LineStyle, MarkerStyle, PlotError, PlotOptions, Point2D, RenderBackend, Result, TextOptions,
    Transform2D,
};
use tiny_skia::*;

/// Raster rendering backend
///
/// This backend renders graphics primitives to raster formats (PNG, JPEG)
/// using tiny-skia for software rendering.
pub struct RasterBackend {
    /// The pixmap (image buffer)
    pixmap: Pixmap,

    /// Width of the image in pixels
    width: u32,

    /// Height of the image in pixels
    height: u32,

    /// Current transformation matrix
    transform: Transform2D,

    /// Current path builder
    path_builder: Option<PathBuilder>,
}

impl RasterBackend {
    /// Create a new raster backend
    ///
    /// # Arguments
    /// * `width` - Width of the image in pixels
    /// * `height` - Height of the image in pixels
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let pixmap = Pixmap::new(width, height)
            .ok_or_else(|| PlotError::RenderError("Failed to create pixmap".to_string()))?;

        Ok(Self {
            pixmap,
            width,
            height,
            transform: Transform2D::identity(),
            path_builder: None,
        })
    }

    /// Set the background color
    pub fn set_background(&mut self, color: Color) {
        let rgb = color.rgb_tuple();
        let skia_color = tiny_skia::Color::from_rgba8(
            (rgb.0 * 255.0) as u8,
            (rgb.1 * 255.0) as u8,
            (rgb.2 * 255.0) as u8,
            255,
        );
        self.pixmap.fill(skia_color);
    }

    /// Convert a Color to tiny-skia Color
    fn color_to_skia(color: &Color, alpha: f64) -> tiny_skia::Color {
        let rgb = color.rgb_tuple();
        tiny_skia::Color::from_rgba8(
            (rgb.0 * 255.0) as u8,
            (rgb.1 * 255.0) as u8,
            (rgb.2 * 255.0) as u8,
            (alpha * 255.0) as u8,
        )
    }

    /// Create a paint from plot options for stroking
    fn create_stroke_paint(options: &PlotOptions) -> Paint<'static> {
        let mut paint = Paint::default();
        paint.set_color(Self::color_to_skia(&options.color, options.alpha));
        paint.anti_alias = true;
        paint
    }

    /// Create a paint from plot options for filling
    fn create_fill_paint(options: &PlotOptions) -> Paint<'static> {
        let mut paint = Paint::default();
        let fill_col = options.fill_color.as_ref().unwrap_or(&options.color);
        paint.set_color(Self::color_to_skia(fill_col, fill_col.alpha()));
        paint.anti_alias = true;
        paint
    }

    /// Create a stroke from plot options
    fn create_stroke(options: &PlotOptions) -> Stroke {
        let mut stroke = Stroke::default();
        stroke.width = options.thickness as f32;

        // Handle line styles
        match options.linestyle {
            LineStyle::Solid => {}
            LineStyle::Dashed => {
                stroke.dash = StrokeDash::new(vec![5.0, 5.0], 0.0);
            }
            LineStyle::Dotted => {
                stroke.dash = StrokeDash::new(vec![2.0, 3.0], 0.0);
            }
            LineStyle::DashDot => {
                stroke.dash = StrokeDash::new(vec![5.0, 3.0, 2.0, 3.0], 0.0);
            }
            LineStyle::DashDotDot => {
                stroke.dash = StrokeDash::new(vec![5.0, 3.0, 2.0, 3.0, 2.0, 3.0], 0.0);
            }
            LineStyle::None => {}
        }

        stroke
    }

    /// Convert Transform2D to tiny_skia::Transform
    fn to_skia_transform(t: &Transform2D) -> tiny_skia::Transform {
        tiny_skia::Transform::from_row(
            t.a as f32,
            t.b as f32,
            t.c as f32,
            t.d as f32,
            t.e as f32,
            t.f as f32,
        )
    }

    /// Get the current transform as a tiny-skia transform
    fn get_skia_transform(&self) -> tiny_skia::Transform {
        Self::to_skia_transform(&self.transform)
    }

    /// Export to PNG format
    pub fn to_png(&self) -> Result<Vec<u8>> {
        self.pixmap
            .encode_png()
            .map_err(|e| PlotError::RenderError(format!("Failed to encode PNG: {}", e)))
    }

    /// Export to JPEG format (note: requires conversion to RGB)
    pub fn to_jpeg(&self, quality: u8) -> Result<Vec<u8>> {
        use image::{ImageBuffer, Rgb};

        // Convert pixmap to RGB image
        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.pixmap.pixel(x, y).unwrap();
                img.put_pixel(
                    x,
                    y,
                    Rgb([pixel.red(), pixel.green(), pixel.blue()]),
                );
            }
        }

        // Encode as JPEG
        let mut buffer = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
        img.write_with_encoder(encoder)
            .map_err(|e| PlotError::RenderError(format!("Failed to encode JPEG: {}", e)))?;

        Ok(buffer)
    }
}

impl RenderBackend for RasterBackend {
    fn draw_line(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()> {
        if points.len() < 2 {
            return Ok(());
        }

        let mut path_builder = PathBuilder::new();
        path_builder.move_to(points[0].x as f32, points[0].y as f32);

        for point in &points[1..] {
            path_builder.line_to(point.x as f32, point.y as f32);
        }

        let path = path_builder
            .finish()
            .ok_or_else(|| PlotError::RenderError("Failed to build path".to_string()))?;

        let paint = Self::create_stroke_paint(options);
        let stroke = Self::create_stroke(options);
        let transform = self.get_skia_transform();

        self.pixmap.stroke_path(&path, &paint, &stroke, transform, None);

        Ok(())
    }

    fn draw_polygon(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let mut path_builder = PathBuilder::new();
        path_builder.move_to(points[0].x as f32, points[0].y as f32);

        for point in &points[1..] {
            path_builder.line_to(point.x as f32, point.y as f32);
        }

        path_builder.close();

        let path = path_builder
            .finish()
            .ok_or_else(|| PlotError::RenderError("Failed to build path".to_string()))?;

        let transform = self.get_skia_transform();

        // Fill if requested
        if options.fill {
            let paint = Self::create_fill_paint(options);
            self.pixmap.fill_path(&path, &paint, FillRule::Winding, transform, None);
        }

        // Stroke
        let paint = Self::create_stroke_paint(options);
        let stroke = Self::create_stroke(options);
        self.pixmap.stroke_path(&path, &paint, &stroke, transform, None);

        Ok(())
    }

    fn draw_circle(&mut self, center: Point2D, radius: f64, options: &PlotOptions) -> Result<()> {
        let mut path_builder = PathBuilder::new();
        path_builder.push_circle(center.x as f32, center.y as f32, radius as f32);

        let path = path_builder
            .finish()
            .ok_or_else(|| PlotError::RenderError("Failed to build circle path".to_string()))?;

        let transform = self.get_skia_transform();

        // Fill if requested
        if options.fill {
            let paint = Self::create_fill_paint(options);
            self.pixmap.fill_path(&path, &paint, FillRule::Winding, transform, None);
        }

        // Stroke
        let paint = Self::create_stroke_paint(options);
        let stroke = Self::create_stroke(options);
        self.pixmap.stroke_path(&path, &paint, &stroke, transform, None);

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
        // Create an ellipse by transforming a circle
        let mut path_builder = PathBuilder::new();
        path_builder.push_circle(0.0, 0.0, 1.0);

        let path = path_builder
            .finish()
            .ok_or_else(|| PlotError::RenderError("Failed to build ellipse path".to_string()))?;

        // Create transformation: translate to center, rotate, and scale
        let mut transform = tiny_skia::Transform::from_translate(center.x as f32, center.y as f32);

        if rotation != 0.0 {
            transform = transform.post_rotate(rotation as f32);
        }

        transform = transform.post_scale(rx as f32, ry as f32);
        transform = transform.post_concat(self.get_skia_transform());

        // Fill if requested
        if options.fill {
            let paint = Self::create_fill_paint(options);
            self.pixmap.fill_path(&path, &paint, FillRule::Winding, transform, None);
        }

        // Stroke
        let paint = Self::create_stroke_paint(options);
        let stroke = Self::create_stroke(options);
        self.pixmap.stroke_path(&path, &paint, &stroke, transform, None);

        Ok(())
    }

    fn draw_rectangle(&mut self, min: Point2D, max: Point2D, options: &PlotOptions) -> Result<()> {
        let rect = tiny_skia::Rect::from_ltrb(
            min.x as f32,
            min.y as f32,
            max.x as f32,
            max.y as f32,
        )
        .ok_or_else(|| PlotError::RenderError("Invalid rectangle bounds".to_string()))?;

        let path = PathBuilder::from_rect(rect);
        let transform = self.get_skia_transform();

        // Fill if requested
        if options.fill {
            let paint = Self::create_fill_paint(options);
            self.pixmap.fill_path(&path, &paint, FillRule::Winding, transform, None);
        }

        // Stroke
        let paint = Self::create_stroke_paint(options);
        let stroke = Self::create_stroke(options);
        self.pixmap.stroke_path(&path, &paint, &stroke, transform, None);

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
        let start_rad = start_angle.to_radians() as f32;
        let end_rad = end_angle.to_radians() as f32;

        // Calculate start and end points
        let start_x = center.x as f32 + radius as f32 * start_rad.cos();
        let start_y = center.y as f32 + radius as f32 * start_rad.sin();

        let mut path_builder = PathBuilder::new();
        path_builder.move_to(start_x, start_y);

        // Approximate arc with line segments
        let segments = 32;
        let angle_step = (end_rad - start_rad) / segments as f32;

        for i in 1..=segments {
            let angle = start_rad + angle_step * i as f32;
            let x = center.x as f32 + radius as f32 * angle.cos();
            let y = center.y as f32 + radius as f32 * angle.sin();
            path_builder.line_to(x, y);
        }

        let path = path_builder
            .finish()
            .ok_or_else(|| PlotError::RenderError("Failed to build arc path".to_string()))?;

        let paint = Self::create_stroke_paint(options);
        let stroke = Self::create_stroke(options);
        let transform = self.get_skia_transform();

        self.pixmap.stroke_path(&path, &paint, &stroke, transform, None);

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
        let mut path_builder = PathBuilder::new();
        path_builder.move_to(p0.x as f32, p0.y as f32);
        path_builder.cubic_to(
            p1.x as f32,
            p1.y as f32,
            p2.x as f32,
            p2.y as f32,
            p3.x as f32,
            p3.y as f32,
        );

        let path = path_builder
            .finish()
            .ok_or_else(|| PlotError::RenderError("Failed to build bezier path".to_string()))?;

        let paint = Self::create_stroke_paint(options);
        let stroke = Self::create_stroke(options);
        let transform = self.get_skia_transform();

        self.pixmap.stroke_path(&path, &paint, &stroke, transform, None);

        Ok(())
    }

    fn draw_text(&mut self, _position: Point2D, _text: &str, _options: &TextOptions) -> Result<()> {
        // Text rendering is not supported in tiny-skia
        // Would need to integrate a text rendering library like fontdue or rusttype
        // For now, we'll just skip text rendering
        Ok(())
    }

    fn draw_points(&mut self, points: &[Point2D], options: &PlotOptions) -> Result<()> {
        let marker_size = options.marker_size as f32;
        let paint = Self::create_fill_paint(options);

        for point in points {
            match options.marker {
                MarkerStyle::Circle => {
                    let mut path_builder = PathBuilder::new();
                    path_builder.push_circle(
                        point.x as f32,
                        point.y as f32,
                        marker_size / 2.0,
                    );

                    if let Some(path) = path_builder.finish() {
                        self.pixmap.fill_path(
                            &path,
                            &paint,
                            FillRule::Winding,
                            self.get_skia_transform(),
                            None,
                        );
                    }
                }
                MarkerStyle::Square => {
                    let half_size = marker_size / 2.0;
                    if let Some(rect) = tiny_skia::Rect::from_ltrb(
                        point.x as f32 - half_size,
                        point.y as f32 - half_size,
                        point.x as f32 + half_size,
                        point.y as f32 + half_size,
                    ) {
                        let path = PathBuilder::from_rect(rect);
                        self.pixmap.fill_path(
                            &path,
                            &paint,
                            FillRule::Winding,
                            self.get_skia_transform(),
                            None,
                        );
                    }
                }
                MarkerStyle::Plus | MarkerStyle::Cross => {
                    // Draw as small lines
                    let half_size = marker_size / 2.0;
                    let stroke = Stroke {
                        width: 1.0,
                        ..Default::default()
                    };

                    if options.marker == MarkerStyle::Plus {
                        // Horizontal line
                        let mut pb = PathBuilder::new();
                        pb.move_to(point.x as f32 - half_size, point.y as f32);
                        pb.line_to(point.x as f32 + half_size, point.y as f32);
                        if let Some(path) = pb.finish() {
                            self.pixmap.stroke_path(
                                &path,
                                &paint,
                                &stroke,
                                self.get_skia_transform(),
                                None,
                            );
                        }

                        // Vertical line
                        let mut pb = PathBuilder::new();
                        pb.move_to(point.x as f32, point.y as f32 - half_size);
                        pb.line_to(point.x as f32, point.y as f32 + half_size);
                        if let Some(path) = pb.finish() {
                            self.pixmap.stroke_path(
                                &path,
                                &paint,
                                &stroke,
                                self.get_skia_transform(),
                                None,
                            );
                        }
                    } else {
                        // Cross (X)

                        // Diagonal 1
                        let mut pb = PathBuilder::new();
                        pb.move_to(point.x as f32 - half_size, point.y as f32 - half_size);
                        pb.line_to(point.x as f32 + half_size, point.y as f32 + half_size);
                        if let Some(path) = pb.finish() {
                            self.pixmap.stroke_path(
                                &path,
                                &paint,
                                &stroke,
                                self.get_skia_transform(),
                                None,
                            );
                        }

                        // Diagonal 2
                        let mut pb = PathBuilder::new();
                        pb.move_to(point.x as f32 - half_size, point.y as f32 + half_size);
                        pb.line_to(point.x as f32 + half_size, point.y as f32 - half_size);
                        if let Some(path) = pb.finish() {
                            self.pixmap.stroke_path(
                                &path,
                                &paint,
                                &stroke,
                                self.get_skia_transform(),
                                None,
                            );
                        }
                    }
                }
                MarkerStyle::None => {}
                _ => {
                    // Other marker styles not yet implemented for raster backend
                    // Fall back to simple circle
                    let mut path_builder = PathBuilder::new();
                    path_builder.push_circle(
                        point.x as f32,
                        point.y as f32,
                        marker_size / 2.0,
                    );

                    if let Some(path) = path_builder.finish() {
                        self.pixmap.fill_path(
                            &path,
                            &paint,
                            FillRule::Winding,
                            self.get_skia_transform(),
                            None,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn draw_arrow(&mut self, start: Point2D, end: Point2D, options: &PlotOptions) -> Result<()> {
        // Draw the line
        let mut path_builder = PathBuilder::new();
        path_builder.move_to(start.x as f32, start.y as f32);
        path_builder.line_to(end.x as f32, end.y as f32);

        if let Some(path) = path_builder.finish() {
            let paint = Self::create_stroke_paint(options);
            let stroke = Self::create_stroke(options);
            self.pixmap.stroke_path(&path, &paint, &stroke, self.get_skia_transform(), None);
        }

        // Draw arrowhead
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let angle = dy.atan2(dx);
        let arrow_size = 10.0;

        let angle1 = angle + std::f64::consts::PI * 0.75;
        let angle2 = angle - std::f64::consts::PI * 0.75;

        let p1 = Point2D::new(
            end.x + arrow_size * angle1.cos(),
            end.y + arrow_size * angle1.sin(),
        );
        let p2 = Point2D::new(
            end.x + arrow_size * angle2.cos(),
            end.y + arrow_size * angle2.sin(),
        );

        // Draw arrowhead triangle
        let mut path_builder = PathBuilder::new();
        path_builder.move_to(end.x as f32, end.y as f32);
        path_builder.line_to(p1.x as f32, p1.y as f32);
        path_builder.line_to(p2.x as f32, p2.y as f32);
        path_builder.close();

        if let Some(path) = path_builder.finish() {
            let paint = Self::create_fill_paint(options);
            self.pixmap.fill_path(&path, &paint, FillRule::Winding, self.get_skia_transform(), None);
        }

        Ok(())
    }

    fn begin_path(&mut self) -> Result<()> {
        self.path_builder = Some(PathBuilder::new());
        Ok(())
    }

    fn move_to(&mut self, point: Point2D) -> Result<()> {
        if let Some(builder) = &mut self.path_builder {
            builder.move_to(point.x as f32, point.y as f32);
            Ok(())
        } else {
            Err(PlotError::RenderError(
                "begin_path() must be called before move_to()".to_string(),
            ))
        }
    }

    fn line_to(&mut self, point: Point2D) -> Result<()> {
        if let Some(builder) = &mut self.path_builder {
            builder.line_to(point.x as f32, point.y as f32);
            Ok(())
        } else {
            Err(PlotError::RenderError(
                "begin_path() must be called before line_to()".to_string(),
            ))
        }
    }

    fn close_path(&mut self) -> Result<()> {
        if let Some(builder) = &mut self.path_builder {
            builder.close();
            Ok(())
        } else {
            Err(PlotError::RenderError(
                "begin_path() must be called before close_path()".to_string(),
            ))
        }
    }

    fn stroke(&mut self, options: &PlotOptions) -> Result<()> {
        if let Some(builder) = self.path_builder.take() {
            if let Some(path) = builder.finish() {
                let paint = Self::create_stroke_paint(options);
                let stroke = Self::create_stroke(options);
                self.pixmap.stroke_path(&path, &paint, &stroke, self.get_skia_transform(), None);
            }
            Ok(())
        } else {
            Err(PlotError::RenderError("No path to stroke".to_string()))
        }
    }

    fn fill(&mut self, options: &PlotOptions) -> Result<()> {
        if let Some(builder) = self.path_builder.take() {
            if let Some(path) = builder.finish() {
                let paint = Self::create_fill_paint(options);
                self.pixmap.fill_path(&path, &paint, FillRule::Winding, self.get_skia_transform(), None);
            }
            Ok(())
        } else {
            Err(PlotError::RenderError("No path to fill".to_string()))
        }
    }

    fn get_transform(&self) -> Transform2D {
        self.transform
    }

    fn set_transform(&mut self, transform: Transform2D) {
        self.transform = transform;
    }

    fn finalize(&mut self) -> Result<Vec<u8>> {
        self.to_png()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raster_backend_creation() {
        let backend = RasterBackend::new(800, 600);
        assert!(backend.is_ok());
        let backend = backend.unwrap();
        assert_eq!(backend.width, 800);
        assert_eq!(backend.height, 600);
    }

    #[test]
    fn test_raster_draw_line() {
        let mut backend = RasterBackend::new(800, 600).unwrap();
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(100.0, 100.0)];
        let result = backend.draw_line(&points, &PlotOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_raster_draw_circle() {
        let mut backend = RasterBackend::new(800, 600).unwrap();
        let result = backend.draw_circle(Point2D::new(50.0, 50.0), 25.0, &PlotOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_raster_to_png() {
        let mut backend = RasterBackend::new(800, 600).unwrap();
        backend.set_background(Color::white());
        let result = backend.to_png();
        assert!(result.is_ok());
        let png_data = result.unwrap();
        assert!(!png_data.is_empty());

        // Check that it starts with PNG magic bytes
        assert_eq!(&png_data[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }
}
