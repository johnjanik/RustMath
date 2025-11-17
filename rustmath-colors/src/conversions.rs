//! Color space conversion functions
//!
//! Based on SageMath's color conversion algorithms from sage.plot.colors

/// Helper function to keep values in [0, 1] range with modulo 1
pub fn mod_one(x: f64) -> f64 {
    let result = x - x.floor();
    if result < 0.0 {
        result + 1.0
    } else {
        result
    }
}

/// Convert RGB to HSL color space
///
/// # Arguments
/// * `r` - Red component (0.0 to 1.0)
/// * `g` - Green component (0.0 to 1.0)
/// * `b` - Blue component (0.0 to 1.0)
///
/// # Returns
/// Tuple of (hue, saturation, lightness) all in [0.0, 1.0]
///
/// Based on the algorithm from sage.plot.colors.rgb_to_hls
pub fn rgb_to_hsl(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max_c = r.max(g).max(b);
    let min_c = r.min(g).min(b);
    let l = (max_c + min_c) / 2.0;

    if (max_c - min_c).abs() < 1e-10 {
        // Achromatic (gray)
        return (0.0, 0.0, l);
    }

    let delta = max_c - min_c;

    let s = if l > 0.5 {
        delta / (2.0 - max_c - min_c)
    } else {
        delta / (max_c + min_c)
    };

    let h = if (r - max_c).abs() < 1e-10 {
        // Red is max
        mod_one((g - b) / delta / 6.0)
    } else if (g - max_c).abs() < 1e-10 {
        // Green is max
        mod_one((b - r) / delta / 6.0 + 1.0 / 3.0)
    } else {
        // Blue is max
        mod_one((r - g) / delta / 6.0 + 2.0 / 3.0)
    };

    (h, s, l)
}

/// Convert HSL to RGB color space
///
/// # Arguments
/// * `h` - Hue (0.0 to 1.0, wraps around)
/// * `s` - Saturation (0.0 to 1.0)
/// * `l` - Lightness (0.0 to 1.0)
///
/// # Returns
/// Tuple of (red, green, blue) all in [0.0, 1.0]
///
/// Based on the algorithm from sage.plot.colors.hls_to_rgb
pub fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    let h = mod_one(h);

    if s.abs() < 1e-10 {
        // Achromatic (gray)
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };

    let p = 2.0 * l - q;

    let hue_to_rgb = |p: f64, q: f64, t: f64| -> f64 {
        let t = mod_one(t);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

/// Convert RGB to HSV color space
///
/// # Arguments
/// * `r` - Red component (0.0 to 1.0)
/// * `g` - Green component (0.0 to 1.0)
/// * `b` - Blue component (0.0 to 1.0)
///
/// # Returns
/// Tuple of (hue, saturation, value) all in [0.0, 1.0]
pub fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max_c = r.max(g).max(b);
    let min_c = r.min(g).min(b);
    let v = max_c;

    if (max_c - min_c).abs() < 1e-10 {
        // Achromatic (gray)
        return (0.0, 0.0, v);
    }

    let delta = max_c - min_c;
    let s = delta / max_c;

    let h = if (r - max_c).abs() < 1e-10 {
        // Red is max
        mod_one((g - b) / delta / 6.0)
    } else if (g - max_c).abs() < 1e-10 {
        // Green is max
        mod_one((b - r) / delta / 6.0 + 1.0 / 3.0)
    } else {
        // Blue is max
        mod_one((r - g) / delta / 6.0 + 2.0 / 3.0)
    };

    (h, s, v)
}

/// Convert HSV to RGB color space
///
/// # Arguments
/// * `h` - Hue (0.0 to 1.0, wraps around)
/// * `s` - Saturation (0.0 to 1.0)
/// * `v` - Value/brightness (0.0 to 1.0)
///
/// # Returns
/// Tuple of (red, green, blue) all in [0.0, 1.0]
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let h = mod_one(h);

    if s.abs() < 1e-10 {
        // Achromatic (gray)
        return (v, v, v);
    }

    let h_i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - h_i as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    match h_i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_one() {
        assert!((mod_one(0.5) - 0.5).abs() < 1e-10);
        assert!((mod_one(1.5) - 0.5).abs() < 1e-10);
        assert!((mod_one(-0.5) - 0.5).abs() < 1e-10);
        assert!((mod_one(0.0) - 0.0).abs() < 1e-10);
        assert!((mod_one(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rgb_to_hsl_pure_colors() {
        // Red
        let (h, s, l) = rgb_to_hsl(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1e-10);
        assert!((s - 1.0).abs() < 1e-10);
        assert!((l - 0.5).abs() < 1e-10);

        // Green
        let (h, s, l) = rgb_to_hsl(0.0, 1.0, 0.0);
        assert!((h - 1.0 / 3.0).abs() < 1e-10);
        assert!((s - 1.0).abs() < 1e-10);
        assert!((l - 0.5).abs() < 1e-10);

        // Blue
        let (h, s, l) = rgb_to_hsl(0.0, 0.0, 1.0);
        assert!((h - 2.0 / 3.0).abs() < 1e-10);
        assert!((s - 1.0).abs() < 1e-10);
        assert!((l - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rgb_to_hsl_gray() {
        let (h, s, l) = rgb_to_hsl(0.5, 0.5, 0.5);
        assert!((s - 0.0).abs() < 1e-10);
        assert!((l - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hsl_to_rgb_pure_colors() {
        // Red
        let (r, g, b) = hsl_to_rgb(0.0, 1.0, 0.5);
        assert!((r - 1.0).abs() < 1e-10);
        assert!((g - 0.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);

        // Green
        let (r, g, b) = hsl_to_rgb(1.0 / 3.0, 1.0, 0.5);
        assert!((r - 0.0).abs() < 1e-10);
        assert!((g - 1.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);

        // Blue
        let (r, g, b) = hsl_to_rgb(2.0 / 3.0, 1.0, 0.5);
        assert!((r - 0.0).abs() < 1e-10);
        assert!((g - 0.0).abs() < 1e-10);
        assert!((b - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rgb_hsl_roundtrip() {
        let test_colors = vec![
            (0.2, 0.3, 0.4),
            (0.7, 0.1, 0.9),
            (0.5, 0.5, 0.5),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ];

        for (r1, g1, b1) in test_colors {
            let (h, s, l) = rgb_to_hsl(r1, g1, b1);
            let (r2, g2, b2) = hsl_to_rgb(h, s, l);
            assert!((r1 - r2).abs() < 1e-10, "Red mismatch for {:?}", (r1, g1, b1));
            assert!((g1 - g2).abs() < 1e-10, "Green mismatch for {:?}", (r1, g1, b1));
            assert!((b1 - b2).abs() < 1e-10, "Blue mismatch for {:?}", (r1, g1, b1));
        }
    }

    #[test]
    fn test_rgb_to_hsv_pure_colors() {
        // Red
        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1e-10);
        assert!((s - 1.0).abs() < 1e-10);
        assert!((v - 1.0).abs() < 1e-10);

        // Green
        let (h, s, v) = rgb_to_hsv(0.0, 1.0, 0.0);
        assert!((h - 1.0 / 3.0).abs() < 1e-10);
        assert!((s - 1.0).abs() < 1e-10);
        assert!((v - 1.0).abs() < 1e-10);

        // Blue
        let (h, s, v) = rgb_to_hsv(0.0, 0.0, 1.0);
        assert!((h - 2.0 / 3.0).abs() < 1e-10);
        assert!((s - 1.0).abs() < 1e-10);
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hsv_to_rgb_pure_colors() {
        // Red
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 1e-10);
        assert!((g - 0.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);

        // Green
        let (r, g, b) = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!((r - 0.0).abs() < 1e-10);
        assert!((g - 1.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);

        // Blue
        let (r, g, b) = hsv_to_rgb(2.0 / 3.0, 1.0, 1.0);
        assert!((r - 0.0).abs() < 1e-10);
        assert!((g - 0.0).abs() < 1e-10);
        assert!((b - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let test_colors = vec![
            (0.2, 0.3, 0.4),
            (0.7, 0.1, 0.9),
            (0.5, 0.5, 0.5),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ];

        for (r1, g1, b1) in test_colors {
            let (h, s, v) = rgb_to_hsv(r1, g1, b1);
            let (r2, g2, b2) = hsv_to_rgb(h, s, v);
            assert!((r1 - r2).abs() < 1e-10, "Red mismatch for {:?}", (r1, g1, b1));
            assert!((g1 - g2).abs() < 1e-10, "Green mismatch for {:?}", (r1, g1, b1));
            assert!((b1 - b2).abs() < 1e-10, "Blue mismatch for {:?}", (r1, g1, b1));
        }
    }
}
