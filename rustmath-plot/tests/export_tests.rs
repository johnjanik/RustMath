//! Integration tests for SVG and PNG export functionality

use rustmath_plot::{Graphics, RenderFormat};
use std::fs;
use std::path::PathBuf;

#[test]
fn test_save_empty_graphics_svg() {
    let g = Graphics::new();
    let path = PathBuf::from("/tmp/test_empty.svg");

    let result = g.save(&path, RenderFormat::SVG);
    assert!(result.is_ok(), "Failed to save empty SVG: {:?}", result.err());

    // Verify file was created
    assert!(path.exists(), "SVG file was not created");

    // Read the file and verify it's valid SVG
    let content = fs::read_to_string(&path).unwrap();
    assert!(content.contains("<svg"), "SVG file does not contain <svg tag");
    // Check for either self-closing tag or separate closing tag
    assert!(
        content.contains("</svg>") || content.contains("/>"),
        "SVG file does not contain closing tag"
    );

    // Clean up
    let _ = fs::remove_file(path);
}

#[test]
fn test_save_empty_graphics_png() {
    let g = Graphics::new();
    let path = PathBuf::from("/tmp/test_empty.png");

    let result = g.save(&path, RenderFormat::PNG);
    assert!(result.is_ok(), "Failed to save empty PNG: {:?}", result.err());

    // Verify file was created
    assert!(path.exists(), "PNG file was not created");

    // Read the file and verify it's a valid PNG (check magic bytes)
    let content = fs::read(&path).unwrap();
    assert!(content.len() > 8, "PNG file is too small");
    assert_eq!(&content[0..8], &[137, 80, 78, 71, 13, 10, 26, 10], "PNG magic bytes are incorrect");

    // Clean up
    let _ = fs::remove_file(path);
}

#[test]
fn test_save_graphics_with_options_svg() {
    let mut g = Graphics::new();
    g.set_title("Test Plot");
    g.set_labels("X Axis", "Y Axis");
    g.set_figsize(800, 600);

    let path = PathBuf::from("/tmp/test_with_options.svg");
    let result = g.save(&path, RenderFormat::SVG);
    assert!(result.is_ok(), "Failed to save SVG with options: {:?}", result.err());

    // Verify file exists
    assert!(path.exists(), "SVG file was not created");

    // Verify it's valid SVG
    let content = fs::read_to_string(&path).unwrap();
    assert!(content.contains("width=\"800\""), "SVG width not set correctly");
    assert!(content.contains("height=\"600\""), "SVG height not set correctly");

    // Clean up
    let _ = fs::remove_file(path);
}

#[test]
fn test_save_graphics_with_options_png() {
    let mut g = Graphics::new();
    g.set_title("Test Plot");
    g.set_figsize(640, 480);

    let path = PathBuf::from("/tmp/test_with_options.png");
    let result = g.save(&path, RenderFormat::PNG);
    assert!(result.is_ok(), "Failed to save PNG with options: {:?}", result.err());

    // Verify file exists and is a valid PNG
    assert!(path.exists(), "PNG file was not created");
    let content = fs::read(&path).unwrap();
    assert_eq!(&content[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // Clean up
    let _ = fs::remove_file(path);
}

#[test]
fn test_save_jpeg_format() {
    let g = Graphics::new();
    let path = PathBuf::from("/tmp/test.jpg");

    let result = g.save(&path, RenderFormat::JPEG);
    assert!(result.is_ok(), "Failed to save JPEG: {:?}", result.err());

    // Verify file exists and starts with JPEG magic bytes
    assert!(path.exists(), "JPEG file was not created");
    let content = fs::read(&path).unwrap();
    assert!(content.len() > 2, "JPEG file is too small");
    assert_eq!(&content[0..2], &[0xFF, 0xD8], "JPEG magic bytes are incorrect");

    // Clean up
    let _ = fs::remove_file(path);
}

#[test]
fn test_unsupported_format_pdf() {
    let g = Graphics::new();
    let path = PathBuf::from("/tmp/test.pdf");

    let result = g.save(&path, RenderFormat::PDF);
    assert!(result.is_err(), "PDF format should not be supported yet");

    // Verify error message
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not yet implemented"), "Error message should mention not implemented");
}

#[test]
fn test_unsupported_format_eps() {
    let g = Graphics::new();
    let path = PathBuf::from("/tmp/test.eps");

    let result = g.save(&path, RenderFormat::EPS);
    assert!(result.is_err(), "EPS format should not be supported yet");

    // Verify error message
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not yet implemented"), "Error message should mention not implemented");
}

#[test]
fn test_save_creates_parent_directories() {
    // This test verifies that save fails gracefully if parent directory doesn't exist
    let path = PathBuf::from("/tmp/nonexistent_dir_12345/test.svg");
    let g = Graphics::new();

    let result = g.save(&path, RenderFormat::SVG);
    // Should fail because parent directory doesn't exist
    assert!(result.is_err(), "Should fail when parent directory doesn't exist");
}
