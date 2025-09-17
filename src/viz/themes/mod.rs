//! Theme system for graph visualizations

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Visual theme for graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizTheme {
    pub name: String,
    pub background_color: String,
    pub node_color: String,
    pub edge_color: String,
    pub text_color: String,
    pub highlight_color: String,
    pub selection_color: String,
}

impl VizTheme {
    /// Get a built-in theme by name
    pub fn get_builtin(name: &str) -> Option<Self> {
        match name {
            "light" => Some(Self::light()),
            "dark" => Some(Self::dark()),
            "publication" => Some(Self::publication()),
            "minimal" => Some(Self::minimal()),
            _ => None,
        }
    }
    
    /// Light theme
    pub fn light() -> Self {
        Self {
            name: "light".to_string(),
            background_color: "#ffffff".to_string(),
            node_color: "#4a90e2".to_string(),
            edge_color: "#999999".to_string(),
            text_color: "#333333".to_string(),
            highlight_color: "#ff6b6b".to_string(),
            selection_color: "#ffa500".to_string(),
        }
    }
    
    /// Dark theme
    pub fn dark() -> Self {
        Self {
            name: "dark".to_string(),
            background_color: "#1a1a1a".to_string(),
            node_color: "#64b5f6".to_string(),
            edge_color: "#666666".to_string(),
            text_color: "#ffffff".to_string(),
            highlight_color: "#ff5722".to_string(),
            selection_color: "#ffb74d".to_string(),
        }
    }
    
    /// Publication theme (black and white)
    pub fn publication() -> Self {
        Self {
            name: "publication".to_string(),
            background_color: "#ffffff".to_string(),
            node_color: "#000000".to_string(),
            edge_color: "#666666".to_string(),
            text_color: "#000000".to_string(),
            highlight_color: "#000000".to_string(),
            selection_color: "#999999".to_string(),
        }
    }
    
    /// Minimal theme
    pub fn minimal() -> Self {
        Self {
            name: "minimal".to_string(),
            background_color: "#fafafa".to_string(),
            node_color: "#607d8b".to_string(),
            edge_color: "#bdbdbd".to_string(),
            text_color: "#424242".to_string(),
            highlight_color: "#37474f".to_string(),
            selection_color: "#78909c".to_string(),
        }
    }
}

/// Theme manager for handling multiple themes
pub struct ThemeManager {
    themes: HashMap<String, VizTheme>,
    current_theme: String,
}

impl ThemeManager {
    /// Create a new theme manager with built-in themes
    pub fn new() -> Self {
        let mut themes = HashMap::new();
        
        // Add built-in themes
        for theme_name in &["light", "dark", "publication", "minimal"] {
            if let Some(theme) = VizTheme::get_builtin(theme_name) {
                themes.insert(theme_name.to_string(), theme);
            }
        }
        
        Self {
            themes,
            current_theme: "light".to_string(),
        }
    }
    
    /// Get the current theme
    pub fn current_theme(&self) -> Option<&VizTheme> {
        self.themes.get(&self.current_theme)
    }
    
    /// Set the current theme
    pub fn set_theme(&mut self, name: &str) -> bool {
        if self.themes.contains_key(name) {
            self.current_theme = name.to_string();
            true
        } else {
            false
        }
    }
    
    /// Add a custom theme
    pub fn add_theme(&mut self, theme: VizTheme) {
        self.themes.insert(theme.name.clone(), theme);
    }
    
    /// List available theme names
    pub fn available_themes(&self) -> Vec<String> {
        self.themes.keys().cloned().collect()
    }
}

impl Default for ThemeManager {
    fn default() -> Self {
        Self::new()
    }
}