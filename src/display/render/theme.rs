//! Theme system for unified display styling
//!
//! Provides consistent theming across Unicode, HTML, and future interactive modes.

use std::collections::HashMap;

/// Theme system managing all built-in and custom themes
#[derive(Clone, Debug)]
pub struct ThemeSystem {
    themes: HashMap<String, Theme>,
}

impl ThemeSystem {
    pub fn new() -> Self {
        let mut themes = HashMap::new();

        // Built-in themes
        themes.insert("sleek".to_string(), Theme::sleek());
        themes.insert("light".to_string(), Theme::light());
        themes.insert("dark".to_string(), Theme::dark());
        themes.insert("publication".to_string(), Theme::publication());
        themes.insert("minimal".to_string(), Theme::minimal());

        Self { themes }
    }

    /// Get theme by name, fallback to light theme
    pub fn get_theme(&self, name: &str) -> &Theme {
        self.themes.get(name).unwrap_or(&self.themes["sleek"])
    }

    /// Register a custom theme
    pub fn register_theme(&mut self, name: String, theme: Theme) {
        self.themes.insert(name, theme);
    }

    /// Get all available theme names
    pub fn theme_names(&self) -> Vec<&String> {
        self.themes.keys().collect()
    }
}

impl Default for ThemeSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete theme definition
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub display_name: String,
    pub description: String,

    // HTML styling
    pub table_class: String,
    pub css: String,

    // Unicode styling (future: could include colors via ANSI codes)
    pub unicode_style: UnicodeStyle,

    // Color palette
    pub colors: ColorPalette,
}

impl Theme {
    /// Sleek theme (ultra-clean)
    pub fn sleek() -> Self {
        Self {
            name: "sleek".to_string(),
            display_name: "Sleek".to_string(),
            description: "Ultra-clean sleek theme".to_string(),
            table_class: "theme-sleek".to_string(),
            css: include_str!("themes/sleek.css").to_string(),
            unicode_style: UnicodeStyle::default(),
            colors: ColorPalette::sleek(),
        }
    }
    /// Light theme (default)
    pub fn light() -> Self {
        Self {
            name: "light".to_string(),
            display_name: "Light".to_string(),
            description: "Clean light theme for general use".to_string(),
            table_class: "theme-light".to_string(),
            css: include_str!("themes/light.css").to_string(),
            unicode_style: UnicodeStyle::default(),
            colors: ColorPalette::light(),
        }
    }

    /// Dark theme
    pub fn dark() -> Self {
        Self {
            name: "dark".to_string(),
            display_name: "Dark".to_string(),
            description: "Professional dark theme for presentations".to_string(),
            table_class: "theme-dark".to_string(),
            css: include_str!("themes/dark.css").to_string(),
            unicode_style: UnicodeStyle::default(),
            colors: ColorPalette::dark(),
        }
    }

    /// Publication theme (black & white, print-ready)
    pub fn publication() -> Self {
        Self {
            name: "publication".to_string(),
            display_name: "Publication".to_string(),
            description: "Black and white theme for academic papers".to_string(),
            table_class: "theme-publication".to_string(),
            css: include_str!("themes/publication.css").to_string(),
            unicode_style: UnicodeStyle::default(),
            colors: ColorPalette::publication(),
        }
    }

    /// Minimal theme (ultra-clean)
    pub fn minimal() -> Self {
        Self {
            name: "minimal".to_string(),
            display_name: "Minimal".to_string(),
            description: "Ultra-clean minimal theme".to_string(),
            table_class: "theme-minimal".to_string(),
            css: include_str!("themes/minimal.css").to_string(),
            unicode_style: UnicodeStyle::default(),
            colors: ColorPalette::minimal(),
        }
    }
}

/// Unicode styling options (for future terminal color support)
#[derive(Debug, Clone)]
pub struct UnicodeStyle {
    pub use_colors: bool,
    pub header_color: Option<String>,
    pub border_color: Option<String>,
    pub data_color: Option<String>,
}

impl Default for UnicodeStyle {
    fn default() -> Self {
        Self {
            use_colors: false, // Disabled for now, will be enabled in future phases
            header_color: None,
            border_color: None,
            data_color: None,
        }
    }
}

/// Color palette for theme consistency
#[derive(Debug, Clone)]
pub struct ColorPalette {
    // Primary colors
    pub background: String,
    pub foreground: String,
    pub border: String,

    // Table-specific colors
    pub header_bg: String,
    pub header_fg: String,
    pub row_even_bg: String,
    pub row_odd_bg: String,
    pub cell_text: String,

    // Data type colors
    pub numeric_color: String,
    pub string_color: String,
    pub boolean_color: String,
    pub null_color: String,
}

impl ColorPalette {
    pub fn light() -> Self {
        Self {
            background: "#ffffff".to_string(),
            foreground: "#212529".to_string(),
            border: "#dee2e6".to_string(),
            header_bg: "#f8f9fa".to_string(),
            header_fg: "#495057".to_string(),
            row_even_bg: "#f8f9fa".to_string(),
            row_odd_bg: "#ffffff".to_string(),
            cell_text: "#212529".to_string(),
            numeric_color: "#0066cc".to_string(),
            string_color: "#212529".to_string(),
            boolean_color: "#28a745".to_string(),
            null_color: "#6c757d".to_string(),
        }
    }

    pub fn dark() -> Self {
        Self {
            background: "#1a1a1a".to_string(),
            foreground: "#e9ecef".to_string(),
            border: "#495057".to_string(),
            header_bg: "#343a40".to_string(),
            header_fg: "#e9ecef".to_string(),
            row_even_bg: "#2d3338".to_string(),
            row_odd_bg: "#1a1a1a".to_string(),
            cell_text: "#e9ecef".to_string(),
            numeric_color: "#66b3ff".to_string(),
            string_color: "#e9ecef".to_string(),
            boolean_color: "#51cf66".to_string(),
            null_color: "#adb5bd".to_string(),
        }
    }

    pub fn publication() -> Self {
        Self {
            background: "#ffffff".to_string(),
            foreground: "#000000".to_string(),
            border: "#000000".to_string(),
            header_bg: "#ffffff".to_string(),
            header_fg: "#000000".to_string(),
            row_even_bg: "#ffffff".to_string(),
            row_odd_bg: "#ffffff".to_string(),
            cell_text: "#000000".to_string(),
            numeric_color: "#000000".to_string(),
            string_color: "#000000".to_string(),
            boolean_color: "#000000".to_string(),
            null_color: "#666666".to_string(),
        }
    }

    pub fn minimal() -> Self {
        Self {
            background: "#fefefe".to_string(),
            foreground: "#333333".to_string(),
            border: "#e0e0e0".to_string(),
            header_bg: "#fafafa".to_string(),
            header_fg: "#333333".to_string(),
            row_even_bg: "#fefefe".to_string(),
            row_odd_bg: "#fefefe".to_string(),
            cell_text: "#333333".to_string(),
            numeric_color: "#333333".to_string(),
            string_color: "#333333".to_string(),
            boolean_color: "#333333".to_string(),
            null_color: "#999999".to_string(),
        }
    }

    pub fn sleek() -> Self {
        Self {
            background: "#ffffff".to_string(),
            foreground: "#1f2328".to_string(),
            border: "#eee".to_string(),
            header_bg: "#fafafa".to_string(),
            header_fg: "#1f2328".to_string(),
            row_even_bg: "#fafafa".to_string(),
            row_odd_bg: "#ffffff".to_string(),
            cell_text: "#1f2328".to_string(),
            numeric_color: "#1f2328".to_string(),
            string_color: "#1f2328".to_string(),
            boolean_color: "#1f2328".to_string(),
            null_color: "#6a737d".to_string(),
        }
    }
}

/// Built-in theme enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltInTheme {
    Sleek,
    Light,
    Dark,
    Publication,
    Minimal,
}

impl BuiltInTheme {
    pub fn name(&self) -> &'static str {
        match self {
            BuiltInTheme::Sleek => "sleek",
            BuiltInTheme::Light => "light",
            BuiltInTheme::Dark => "dark",
            BuiltInTheme::Publication => "publication",
            BuiltInTheme::Minimal => "minimal",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            BuiltInTheme::Sleek,
            BuiltInTheme::Light,
            BuiltInTheme::Dark,
            BuiltInTheme::Publication,
            BuiltInTheme::Minimal,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_system_creation() {
        let theme_system = ThemeSystem::new();
        let theme_names = theme_system.theme_names();

        assert!(theme_names.contains(&&"sleek".to_string()));
        assert!(theme_names.contains(&&"light".to_string()));
        assert!(theme_names.contains(&&"dark".to_string()));
        assert!(theme_names.contains(&&"publication".to_string()));
        assert!(theme_names.contains(&&"minimal".to_string()));
    }

    #[test]
    fn test_get_theme() {
        let theme_system = ThemeSystem::new();

        let sleek_theme = theme_system.get_theme("sleek");
        assert_eq!(sleek_theme.name, "sleek");

        let light_theme = theme_system.get_theme("light");
        assert_eq!(light_theme.name, "light");

        let dark_theme = theme_system.get_theme("dark");
        assert_eq!(dark_theme.name, "dark");

        // Fallback to sleek theme for unknown theme
        let unknown_theme = theme_system.get_theme("nonexistent");
        assert_eq!(unknown_theme.name, "sleek");
    }

    #[test]
    fn test_theme_properties() {
        let light_theme = Theme::light();
        assert_eq!(light_theme.name, "light");
        assert_eq!(light_theme.table_class, "theme-light");
        assert!(!light_theme.css.is_empty());

        let dark_theme = Theme::dark();
        assert_eq!(dark_theme.name, "dark");
        assert_eq!(dark_theme.table_class, "theme-dark");
    }

    #[test]
    fn test_color_palettes() {
        let light_colors = ColorPalette::light();
        let dark_colors = ColorPalette::dark();
        let pub_colors = ColorPalette::publication();

        assert_eq!(light_colors.background, "#ffffff");
        assert_eq!(dark_colors.background, "#1a1a1a");
        assert_eq!(pub_colors.background, "#ffffff");

        // All palettes should have all required colors
        assert!(!light_colors.numeric_color.is_empty());
        assert!(!dark_colors.numeric_color.is_empty());
        assert!(!pub_colors.numeric_color.is_empty());
    }

    #[test]
    fn test_builtin_theme_enum() {
        let all_themes = BuiltInTheme::all();
        assert_eq!(all_themes.len(), 5);

        assert_eq!(BuiltInTheme::Sleek.name(), "sleek");
        assert_eq!(BuiltInTheme::Light.name(), "light");
        assert_eq!(BuiltInTheme::Dark.name(), "dark");
        assert_eq!(BuiltInTheme::Publication.name(), "publication");
        assert_eq!(BuiltInTheme::Minimal.name(), "minimal");
    }

    #[test]
    fn test_custom_theme_registration() {
        let mut theme_system = ThemeSystem::new();

        let custom_theme = Theme {
            name: "custom".to_string(),
            display_name: "Custom".to_string(),
            description: "Custom theme".to_string(),
            table_class: "theme-custom".to_string(),
            css: ".custom { color: red; }".to_string(),
            unicode_style: UnicodeStyle::default(),
            colors: ColorPalette::light(),
        };

        theme_system.register_theme("custom".to_string(), custom_theme);

        let retrieved = theme_system.get_theme("custom");
        assert_eq!(retrieved.name, "custom");
        assert_eq!(retrieved.css, ".custom { color: red; }");
    }
}
