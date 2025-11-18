//! Named color constants
//!
//! Based on common web/CSS color names

use crate::color::Color;

/// Get a color by name
///
/// Returns None if the color name is not recognized
pub fn get_named_color(name: &str) -> Option<Color> {
    let name_lower = name.to_lowercase();
    match name_lower.as_str() {
        // Basic colors
        "black" => Some(Color::black()),
        "white" => Some(Color::white()),
        "red" => Some(Color::red_color()),
        "green" | "lime" => Some(Color::green_color()),
        "blue" => Some(Color::blue_color()),
        "yellow" => Some(Color::yellow_color()),
        "cyan" | "aqua" => Some(Color::cyan_color()),
        "magenta" | "fuchsia" => Some(Color::magenta_color()),
        "gray" | "grey" => Some(Color::gray_color()),

        // Extended colors
        "darkred" => Some(Color::rgb(0.545, 0.0, 0.0)),
        "darkgreen" => Some(Color::rgb(0.0, 0.392, 0.0)),
        "darkblue" => Some(Color::rgb(0.0, 0.0, 0.545)),
        "orange" => Some(Color::rgb(1.0, 0.647, 0.0)),
        "purple" => Some(Color::rgb(0.502, 0.0, 0.502)),
        "brown" => Some(Color::rgb(0.647, 0.165, 0.165)),
        "pink" => Some(Color::rgb(1.0, 0.753, 0.796)),
        "gold" => Some(Color::rgb(1.0, 0.843, 0.0)),
        "silver" => Some(Color::rgb(0.753, 0.753, 0.753)),
        "navy" => Some(Color::rgb(0.0, 0.0, 0.502)),
        "teal" => Some(Color::rgb(0.0, 0.502, 0.502)),
        "olive" => Some(Color::rgb(0.502, 0.502, 0.0)),
        "maroon" => Some(Color::rgb(0.502, 0.0, 0.0)),
        "indigo" => Some(Color::rgb(0.294, 0.0, 0.510)),
        "violet" => Some(Color::rgb(0.933, 0.510, 0.933)),
        "turquoise" => Some(Color::rgb(0.251, 0.878, 0.816)),
        "coral" => Some(Color::rgb(1.0, 0.498, 0.314)),
        "salmon" => Some(Color::rgb(0.980, 0.502, 0.447)),
        "khaki" => Some(Color::rgb(0.941, 0.902, 0.549)),
        "tan" => Some(Color::rgb(0.824, 0.706, 0.549)),
        "beige" => Some(Color::rgb(0.961, 0.961, 0.863)),
        "ivory" => Some(Color::rgb(1.0, 1.0, 0.941)),
        "wheat" => Some(Color::rgb(0.961, 0.871, 0.702)),
        "lavender" => Some(Color::rgb(0.902, 0.902, 0.980)),
        "azure" => Some(Color::rgb(0.941, 1.0, 1.0)),
        "crimson" => Some(Color::rgb(0.863, 0.078, 0.235)),
        "plum" => Some(Color::rgb(0.867, 0.627, 0.867)),
        "orchid" => Some(Color::rgb(0.855, 0.439, 0.839)),
        "chartreuse" => Some(Color::rgb(0.498, 1.0, 0.0)),
        "peru" => Some(Color::rgb(0.804, 0.522, 0.247)),
        "chocolate" => Some(Color::rgb(0.824, 0.412, 0.118)),
        "sienna" => Some(Color::rgb(0.627, 0.322, 0.176)),
        "sandybrown" => Some(Color::rgb(0.957, 0.643, 0.376)),
        "goldenrod" => Some(Color::rgb(0.855, 0.647, 0.125)),

        // Shades of gray
        "lightgray" | "lightgrey" => Some(Color::rgb(0.827, 0.827, 0.827)),
        "darkgray" | "darkgrey" => Some(Color::rgb(0.663, 0.663, 0.663)),
        "dimgray" | "dimgrey" => Some(Color::rgb(0.412, 0.412, 0.412)),
        "slategray" | "slategrey" => Some(Color::rgb(0.439, 0.502, 0.565)),

        _ => None,
    }
}

/// Get a list of all named color names
pub fn color_names() -> Vec<&'static str> {
    vec![
        "black", "white", "red", "green", "blue", "yellow", "cyan", "magenta", "gray",
        "darkred", "darkgreen", "darkblue", "orange", "purple", "brown", "pink", "gold",
        "silver", "navy", "teal", "olive", "maroon", "indigo", "violet", "turquoise",
        "coral", "salmon", "khaki", "tan", "beige", "ivory", "wheat", "lavender",
        "azure", "crimson", "plum", "orchid", "chartreuse", "peru", "chocolate",
        "sienna", "sandybrown", "goldenrod", "lightgray", "darkgray", "dimgray", "slategray",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_colors() {
        assert_eq!(get_named_color("black"), Some(Color::black()));
        assert_eq!(get_named_color("white"), Some(Color::white()));
        assert_eq!(get_named_color("red"), Some(Color::red_color()));
        assert_eq!(get_named_color("green"), Some(Color::green_color()));
        assert_eq!(get_named_color("blue"), Some(Color::blue_color()));
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(get_named_color("RED"), Some(Color::red_color()));
        assert_eq!(get_named_color("Red"), Some(Color::red_color()));
        assert_eq!(get_named_color("rEd"), Some(Color::red_color()));
    }

    #[test]
    fn test_aliases() {
        assert_eq!(get_named_color("lime"), Some(Color::green_color()));
        assert_eq!(get_named_color("aqua"), Some(Color::cyan_color()));
        assert_eq!(get_named_color("fuchsia"), Some(Color::magenta_color()));
        assert_eq!(get_named_color("grey"), Some(Color::gray_color()));
    }

    #[test]
    fn test_unknown_color() {
        assert_eq!(get_named_color("notacolor"), None);
        assert_eq!(get_named_color(""), None);
    }

    #[test]
    fn test_extended_colors() {
        assert!(get_named_color("orange").is_some());
        assert!(get_named_color("purple").is_some());
        assert!(get_named_color("brown").is_some());
        assert!(get_named_color("pink").is_some());
    }

    #[test]
    fn test_color_names_list() {
        let names = color_names();
        assert!(names.contains(&"black"));
        assert!(names.contains(&"white"));
        assert!(names.contains(&"orange"));
    }
}
