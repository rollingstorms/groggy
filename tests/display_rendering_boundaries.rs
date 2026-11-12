//! Renderer controls must not split UTF-8 or overflow exact integer values.
use groggy::viz::display::{
    ColumnSchema, DataSchema, DataType, DataWindow, DisplayConfig, HtmlRenderer, Theme,
};

fn one_cell(value: &str, dtype: DataType) -> DataWindow {
    DataWindow::new(
        vec!["value".into()],
        vec![vec![value.into()]],
        DataSchema::new(vec![ColumnSchema {
            name: "value".into(),
            data_type: dtype,
        }]),
    )
}

#[test]
fn html_json_truncation_handles_every_unicode_boundary_and_zero_width() {
    let value = r#"{"消息":"猫😺<&"}"#;
    let data = one_cell(value, DataType::Json);
    for width in 0..=value.chars().count() + 1 {
        let config = DisplayConfig {
            max_cell_width: width,
            ..Default::default()
        };
        let html = HtmlRenderer::new().render_semantic_table(&data, &Theme::sleek(), &config);
        assert!(html.contains("<table"));
        assert!(!html.contains("猫😺<&")); // text/title both remain escaped
        let cell = html
            .split("<td")
            .nth(1)
            .unwrap()
            .split_once('>')
            .unwrap()
            .1
            .split("</td>")
            .next()
            .unwrap();
        let decoded = cell
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&");
        assert!(decoded.chars().count() <= width);
        if width == 0 {
            assert!(decoded.is_empty());
        }
        if width >= value.chars().count() {
            assert_eq!(decoded, value);
        }
    }
}

#[test]
fn legacy_zero_limits_and_native_compatibility_aliases_keep_one_type_owner() {
    use groggy::display::truncation::{truncate_columns, truncate_rows, truncate_string};
    assert_eq!(truncate_string("猫😺", 0), "");
    assert_eq!(truncate_string("猫😺", 1), "…");
    assert_eq!(truncate_string("猫😺", 2), "猫😺");
    assert_eq!(truncate_rows(vec![1, 2, 3], 0), (vec![], true));
    assert_eq!(
        truncate_columns(vec!["a", "b"], vec![vec!["1", "2"]], 0),
        (vec![], vec![vec![]], true)
    );
    assert_eq!(
        std::any::TypeId::of::<DisplayConfig>(),
        std::any::TypeId::of::<groggy::display::render::DisplayConfig>()
    );
    assert_eq!(
        std::any::TypeId::of::<groggy::core::DisplayEngine>(),
        std::any::TypeId::of::<groggy::display::render::DisplayEngine>()
    );
    let data = one_cell("plain", DataType::String);
    let engine = groggy::display::render::DisplayEngine::with_config(DisplayConfig::default());
    assert_eq!(
        engine.format_unicode(&data),
        groggy::viz::display::DisplayEngine::default().format_unicode(&data)
    );
}

#[test]
fn shared_text_truncation_preserves_short_values_and_bounds_large_unicode_prefixes() {
    use groggy::display::truncation::truncate_string;
    for value in ["", "ascii", "猫😺", "e\u{301}"] {
        assert_eq!(truncate_string(value, usize::MAX), value);
        for width in 0..=value.chars().count() + 1 {
            let truncated = truncate_string(value, width);
            assert!(truncated.chars().count() <= width);
            if value.chars().count() <= width {
                assert_eq!(truncated, value);
            }
        }
    }
    let large = "猫😺".repeat(500_000);
    assert_eq!(truncate_string(&large, 4), "猫😺猫…");
}

#[test]
fn html_integer_formatting_accepts_the_full_signed_domain() {
    let data = one_cell(&i64::MIN.to_string(), DataType::Integer);
    let html = HtmlRenderer::new().render_semantic_table(
        &data,
        &Theme::sleek(),
        &DisplayConfig::default(),
    );
    assert!(html.contains("-9,223,372,036,854,775,808"));
}
