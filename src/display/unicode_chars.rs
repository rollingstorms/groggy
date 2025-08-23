/*!
Unicode box-drawing characters and display symbols for professional formatting.
Direct Rust port of the Python unicode_chars.py module.
*/

/// Unicode box-drawing characters for professional table display
pub struct BoxChars;

impl BoxChars {
    // Corners
    pub const TOP_LEFT: &'static str = "╭";
    pub const TOP_RIGHT: &'static str = "╮";
    pub const BOTTOM_LEFT: &'static str = "╰";
    pub const BOTTOM_RIGHT: &'static str = "╯";

    // Lines
    pub const HORIZONTAL: &'static str = "─";
    pub const VERTICAL: &'static str = "│";

    // Intersections
    pub const CROSS: &'static str = "┼";
    pub const T_TOP: &'static str = "┬";
    pub const T_BOTTOM: &'static str = "┴";
    pub const T_LEFT: &'static str = "├";
    pub const T_RIGHT: &'static str = "┤";

    // Double lines for emphasis
    pub const HORIZONTAL_DOUBLE: &'static str = "═";
    pub const VERTICAL_DOUBLE: &'static str = "║";
}

/// Special symbols for data display
pub struct Symbols;

impl Symbols {
    pub const ELLIPSIS: &'static str = "…"; // For truncated content
    pub const DOT_SEPARATOR: &'static str = "•"; // For summary statistics
    pub const NULL_DISPLAY: &'static str = "NaN"; // For null/missing values
    pub const TRUNCATION_INDICATOR: &'static str = "⋯"; // For matrix truncation
    pub const HEADER_PREFIX: &'static str = "⊖⊖"; // For section headers
}

/// ANSI color codes for enhanced display
pub struct Colors;

impl Colors {
    pub const RESET: &'static str = "\x1b[0m";
    pub const BOLD: &'static str = "\x1b[1m";
    pub const DIM: &'static str = "\x1b[2m";

    // Text colors
    pub const RED: &'static str = "\x1b[31m";
    pub const GREEN: &'static str = "\x1b[32m";
    pub const YELLOW: &'static str = "\x1b[33m";
    pub const BLUE: &'static str = "\x1b[34m";
    pub const MAGENTA: &'static str = "\x1b[35m";
    pub const CYAN: &'static str = "\x1b[36m";
    pub const WHITE: &'static str = "\x1b[37m";
    pub const GRAY: &'static str = "\x1b[90m";
}

/// Apply color formatting to text if color is supported
pub fn colorize(text: &str, color: Option<&str>, bold: bool, dim: bool, use_color: bool) -> String {
    if !use_color {
        return text.to_string();
    }

    let mut result = String::new();

    if bold {
        result.push_str(Colors::BOLD);
    }
    if dim {
        result.push_str(Colors::DIM);
    }
    if let Some(color_code) = color {
        result.push_str(color_code);
    }

    result.push_str(text);

    if bold || dim || color.is_some() {
        result.push_str(Colors::RESET);
    }

    result
}

/// Simple colorize for bold text
pub fn bold(text: &str, use_color: bool) -> String {
    colorize(text, None, true, false, use_color)
}

/// Simple colorize for dim text
pub fn dim(text: &str, use_color: bool) -> String {
    colorize(text, None, false, true, use_color)
}
