//! Animation support for RustMath plots
//!
//! Provides functionality to create animated sequences of graphics,
//! similar to SageMath's sage.plot.animate module.
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_plot::{Animation, Graphics};
//! use rustmath_plot::primitives::circle;
//!
//! // Create an animation of a moving circle
//! let frames: Vec<Graphics> = (0..10)
//!     .map(|i| {
//!         let mut g = Graphics::new();
//!         let x = i as f64 / 10.0;
//!         g.add(circle((x, 0.0), 0.1, None));
//!         g
//!     })
//!     .collect();
//!
//! let anim = Animation::new(frames, 30);
//! anim.save("animation.gif", None)?;
//! ```

use crate::{Graphics, PlotError, Result};
use std::path::Path;

/// Animation containing a sequence of graphics frames
///
/// Based on SageMath's Animate class from sage.plot.animate
pub struct Animation {
    /// Sequence of graphics frames
    frames: Vec<Graphics>,
    /// Frames per second for playback
    fps: u32,
    /// Whether to loop the animation
    loop_animation: bool,
}

impl Animation {
    /// Create a new animation from a sequence of graphics
    ///
    /// # Arguments
    ///
    /// * `frames` - Vector of graphics objects to animate
    /// * `fps` - Frames per second (default: 30)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_plot::{Animation, Graphics};
    ///
    /// let frames = vec![Graphics::new(), Graphics::new()];
    /// let anim = Animation::new(frames, 30);
    /// ```
    pub fn new(frames: Vec<Graphics>, fps: u32) -> Self {
        Self {
            frames,
            fps,
            loop_animation: true,
        }
    }

    /// Create an animation by generating frames from a function
    ///
    /// # Arguments
    ///
    /// * `generator` - Function that takes a parameter and returns a Graphics object
    /// * `times` - Vector of time values to generate frames
    /// * `fps` - Frames per second
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_plot::Animation;
    ///
    /// let anim = Animation::from_generator(
    ///     |t| {
    ///         let mut g = Graphics::new();
    ///         // Create graphics based on t
    ///         g
    ///     },
    ///     vec![0.0, 0.1, 0.2, 0.3],
    ///     30,
    /// );
    /// ```
    pub fn from_generator<F>(generator: F, times: Vec<f64>, fps: u32) -> Self
    where
        F: Fn(f64) -> Graphics,
    {
        let frames = times.into_iter().map(generator).collect();
        Self::new(frames, fps)
    }

    /// Get the number of frames in the animation
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get the frames per second
    pub fn fps(&self) -> u32 {
        self.fps
    }

    /// Set the frames per second
    pub fn set_fps(&mut self, fps: u32) {
        self.fps = fps;
    }

    /// Get whether the animation loops
    pub fn loops(&self) -> bool {
        self.loop_animation
    }

    /// Set whether the animation loops
    pub fn set_loop(&mut self, loop_animation: bool) {
        self.loop_animation = loop_animation;
    }

    /// Get the duration of the animation in seconds
    pub fn duration(&self) -> f64 {
        self.frames.len() as f64 / self.fps as f64
    }

    /// Get a specific frame by index
    pub fn frame(&self, index: usize) -> Option<&Graphics> {
        self.frames.get(index)
    }

    /// Get a mutable reference to a specific frame
    pub fn frame_mut(&mut self, index: usize) -> Option<&mut Graphics> {
        self.frames.get_mut(index)
    }

    /// Get all frames
    pub fn frames(&self) -> &[Graphics] {
        &self.frames
    }

    /// Add a frame to the animation
    pub fn add_frame(&mut self, frame: Graphics) {
        self.frames.push(frame);
    }

    /// Insert a frame at a specific position
    pub fn insert_frame(&mut self, index: usize, frame: Graphics) {
        if index <= self.frames.len() {
            self.frames.insert(index, frame);
        }
    }

    /// Remove a frame at a specific position
    pub fn remove_frame(&mut self, index: usize) -> Option<Graphics> {
        if index < self.frames.len() {
            Some(self.frames.remove(index))
        } else {
            None
        }
    }

    /// Save the animation to a file
    ///
    /// Supported formats:
    /// - GIF (animated)
    /// - APNG (animated PNG)
    /// - MP4 (video, requires ffmpeg)
    /// - WebM (video, requires ffmpeg)
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `options` - Optional animation export options
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_plot::Animation;
    ///
    /// let anim = Animation::new(vec![], 30);
    /// anim.save("output.gif", None)?;
    /// ```
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        options: Option<AnimationOptions>,
    ) -> Result<()> {
        let path = path.as_ref();
        let opts = options.unwrap_or_default();

        // Determine format from extension
        let format = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("gif");

        match format {
            "gif" => self.save_gif(path, &opts),
            "apng" => self.save_apng(path, &opts),
            "mp4" => self.save_video(path, "mp4", &opts),
            "webm" => self.save_video(path, "webm", &opts),
            _ => Err(PlotError::UnsupportedFormat(format.to_string())),
        }
    }

    /// Save as animated GIF
    fn save_gif(&self, _path: &Path, _options: &AnimationOptions) -> Result<()> {
        // TODO: Implement GIF export using the `gif` crate
        // This requires:
        // 1. Render each frame to a raster image
        // 2. Encode as GIF with timing information
        // 3. Write to file
        Err(PlotError::NotImplemented(
            "GIF export not yet implemented".to_string(),
        ))
    }

    /// Save as animated PNG (APNG)
    fn save_apng(&self, _path: &Path, _options: &AnimationOptions) -> Result<()> {
        // TODO: Implement APNG export
        Err(PlotError::NotImplemented(
            "APNG export not yet implemented".to_string(),
        ))
    }

    /// Save as video file (MP4/WebM)
    fn save_video(&self, _path: &Path, format: &str, _options: &AnimationOptions) -> Result<()> {
        // TODO: Implement video export by:
        // 1. Render each frame as PNG to temp directory
        // 2. Call ffmpeg to encode as video
        // 3. Clean up temp files
        Err(PlotError::NotImplemented(
            format!("{} export not yet implemented", format.to_uppercase()),
        ))
    }

    /// Show the animation (interactive display)
    ///
    /// This will display the animation in an interactive viewer.
    /// The implementation depends on the available backend.
    pub fn show(&self) -> Result<()> {
        // TODO: Implement interactive display
        // This could use:
        // - Web browser (export to HTML with JavaScript player)
        // - Native GUI window (using egui or similar)
        // - Jupyter notebook integration
        Err(PlotError::NotImplemented(
            "Interactive animation display not yet implemented".to_string(),
        ))
    }

    /// Export frames as individual image files
    ///
    /// # Arguments
    ///
    /// * `directory` - Output directory for frames
    /// * `prefix` - Filename prefix (e.g., "frame" -> frame_0001.png, frame_0002.png, ...)
    /// * `format` - Image format (png, svg, jpeg)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rustmath_plot::Animation;
    ///
    /// let anim = Animation::new(vec![], 30);
    /// anim.export_frames("output_dir", "frame", "png")?;
    /// ```
    pub fn export_frames<P: AsRef<Path>>(
        &self,
        directory: P,
        prefix: &str,
        format: &str,
    ) -> Result<()> {
        use crate::RenderFormat;
        use std::fs;

        let dir = directory.as_ref();
        fs::create_dir_all(dir).map_err(|e| PlotError::IoError(e.to_string()))?;

        let render_format = match format {
            "png" => RenderFormat::PNG,
            "svg" => RenderFormat::SVG,
            "jpeg" | "jpg" => RenderFormat::JPEG,
            _ => return Err(PlotError::UnsupportedFormat(format.to_string())),
        };

        // Calculate number of digits needed for frame numbering
        let num_digits = (self.frames.len() as f64).log10().ceil() as usize + 1;

        for (i, frame) in self.frames.iter().enumerate() {
            let filename = format!("{}_{:0width$}.{}", prefix, i, format, width = num_digits);
            let filepath = dir.join(filename);
            frame.save(&filepath, render_format)?;
        }

        Ok(())
    }

    // NOTE: The following methods are commented out because they require Graphics to be Clone,
    // which is not possible with the current trait object design (Box<dyn GraphicPrimitive>).
    // To enable these methods, GraphicPrimitive would need to support cloning.

    // /// Concatenate two animations
    // pub fn concat(&self, other: &Animation) -> Animation {
    //     let mut frames = self.frames.clone();
    //     frames.extend(other.frames.clone());
    //     Animation {
    //         frames,
    //         fps: self.fps, // Keep fps from first animation
    //         loop_animation: self.loop_animation,
    //     }
    // }

    // /// Reverse the animation
    // pub fn reverse(&self) -> Animation {
    //     let mut frames = self.frames.clone();
    //     frames.reverse();
    //     Animation {
    //         frames,
    //         fps: self.fps,
    //         loop_animation: self.loop_animation,
    //     }
    // }

    // /// Create a ping-pong animation (forward then backward)
    // pub fn ping_pong(&self) -> Animation {
    //     let forward = self.clone();
    //     let backward = self.reverse();
    //     forward.concat(&backward)
    // }

    // /// Slice the animation to a range of frames
    // pub fn slice(&self, start: usize, end: usize) -> Animation {
    //     let frames = self.frames[start..end].to_vec();
    //     Animation {
    //         frames,
    //         fps: self.fps,
    //         loop_animation: self.loop_animation,
    //     }
    // }

    // /// Repeat the animation n times
    // pub fn repeat(&self, n: usize) -> Animation {
    //     let mut frames = Vec::new();
    //     for _ in 0..n {
    //         frames.extend(self.frames.clone());
    //     }
    //     Animation {
    //         frames,
    //         fps: self.fps,
    //         loop_animation: self.loop_animation,
    //     }
    // }
}

/// Options for animation export
#[derive(Debug, Clone)]
pub struct AnimationOptions {
    /// Width in pixels (for raster formats)
    pub width: Option<usize>,
    /// Height in pixels (for raster formats)
    pub height: Option<usize>,
    /// Quality setting (0-100, for lossy formats)
    pub quality: Option<u8>,
    /// Delay between frames in milliseconds (overrides fps)
    pub delay: Option<u32>,
    /// Number of times to loop (None = infinite)
    pub loop_count: Option<u32>,
}

impl Default for AnimationOptions {
    fn default() -> Self {
        Self {
            width: Some(800),
            height: Some(600),
            quality: Some(90),
            delay: None,
            loop_count: None,
        }
    }
}

/// Helper function to create an animation from a generator function
///
/// # Examples
///
/// ```ignore
/// use rustmath_plot::animate;
/// use rustmath_plot::primitives::circle;
///
/// let anim = animate(
///     |t| {
///         let mut g = Graphics::new();
///         g.add(circle((t, 0.0), 0.1, None));
///         g
///     },
///     vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
///     Some(30),
/// );
/// ```
pub fn animate<F>(generator: F, times: Vec<f64>, fps: Option<u32>) -> Animation
where
    F: Fn(f64) -> Graphics,
{
    Animation::from_generator(generator, times, fps.unwrap_or(30))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_animation_creation() {
        let frames = vec![Graphics::new(), Graphics::new(), Graphics::new()];
        let anim = Animation::new(frames, 30);
        assert_eq!(anim.num_frames(), 3);
        assert_eq!(anim.fps(), 30);
    }

    #[test]
    fn test_animation_from_generator() {
        let anim = Animation::from_generator(
            |t| {
                let mut g = Graphics::new();
                g.set_title(&format!("t = {}", t));
                g
            },
            vec![0.0, 0.5, 1.0],
            30,
        );
        assert_eq!(anim.num_frames(), 3);
        assert_eq!(anim.frame(1).unwrap().options().title, Some("t = 0.5".to_string()));
    }

    #[test]
    fn test_animation_duration() {
        let mut frames = Vec::new();
        for _ in 0..60 {
            frames.push(Graphics::new());
        }
        let anim = Animation::new(frames, 30);
        assert_eq!(anim.duration(), 2.0);
    }

    #[test]
    fn test_animation_add_frame() {
        let mut anim = Animation::new(vec![], 30);
        anim.add_frame(Graphics::new());
        anim.add_frame(Graphics::new());
        assert_eq!(anim.num_frames(), 2);
    }

    #[test]
    fn test_animation_remove_frame() {
        let mut anim = Animation::new(vec![Graphics::new(), Graphics::new()], 30);
        let removed = anim.remove_frame(0);
        assert!(removed.is_some());
        assert_eq!(anim.num_frames(), 1);
    }

    // NOTE: These tests are commented out because the methods they test require Graphics to be Clone

    // #[test]
    // fn test_animation_concat() {
    //     let anim1 = Animation::new(vec![Graphics::new(), Graphics::new()], 30);
    //     let anim2 = Animation::new(vec![Graphics::new()], 30);
    //     let combined = anim1.concat(&anim2);
    //     assert_eq!(combined.num_frames(), 3);
    // }

    // #[test]
    // fn test_animation_reverse() {
    //     let mut anim = Animation::new(vec![], 30);
    //     for i in 0..3 {
    //         let mut g = Graphics::new();
    //         g.set_title(&format!("Frame {}", i));
    //         anim.add_frame(g);
    //     }
    //     let reversed = anim.reverse();
    //     assert_eq!(
    //         reversed.frame(0).unwrap().options().title,
    //         Some("Frame 2".to_string())
    //     );
    // }

    // #[test]
    // fn test_animation_ping_pong() {
    //     let anim = Animation::new(vec![Graphics::new(), Graphics::new()], 30);
    //     let pp = anim.ping_pong();
    //     assert_eq!(pp.num_frames(), 4);
    // }

    // #[test]
    // fn test_animation_slice() {
    //     let frames = vec![
    //         Graphics::new(),
    //         Graphics::new(),
    //         Graphics::new(),
    //         Graphics::new(),
    //     ];
    //     let anim = Animation::new(frames, 30);
    //     let sliced = anim.slice(1, 3);
    //     assert_eq!(sliced.num_frames(), 2);
    // }

    // #[test]
    // fn test_animation_repeat() {
    //     let anim = Animation::new(vec![Graphics::new(), Graphics::new()], 30);
    //     let repeated = anim.repeat(3);
    //     assert_eq!(repeated.num_frames(), 6);
    // }

    #[test]
    fn test_animate_helper() {
        let anim = animate(|t| Graphics::new(), vec![0.0, 1.0, 2.0], Some(24));
        assert_eq!(anim.num_frames(), 3);
        assert_eq!(anim.fps(), 24);
    }

    #[test]
    fn test_animate_helper_default_fps() {
        let anim = animate(|t| Graphics::new(), vec![0.0, 1.0], None);
        assert_eq!(anim.fps(), 30);
    }
}
