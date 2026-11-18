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

#[test]
fn interactive_selection_is_bounded_exact_and_uses_typed_omissions() {
    use groggy::display::render::interactive::positions;
    for length in 0..40 {
        for budget in 0..40 {
            let plan = positions(length, budget);
            assert_eq!(plan.len(), length.min(budget));
            let indices: Vec<_> = plan.iter().flatten().copied().collect();
            assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(indices.iter().all(|&i| i < length));
            assert_eq!(plan.contains(&None), length > budget && budget >= 3);
            if length > budget && budget >= 3 {
                assert_eq!(plan[budget / 2], None);
                assert_eq!(plan[0], Some(0));
                if budget > 3 {
                    assert_eq!(plan.last(), Some(&Some(length - 1)));
                }
            }
        }
    }
    assert_eq!(
        positions(usize::MAX, 3),
        vec![Some(0), None, Some(usize::MAX - 1)]
    );
    assert!(positions(0, usize::MAX).is_empty());
}

#[test]
fn interactive_clipping_matches_custom_ellipsis_and_unicode_budgets() {
    use groggy::display::render::interactive::clip;
    for value in ["", "abcde", "猫😺e\u{301}<&"] {
        for ellipsis in ["", "…", "--", "猫😺"] {
            for width in 0..=value.chars().count() + 2 {
                let output = clip(value, width, ellipsis);
                assert!(output.chars().count() <= width);
                if value.chars().count() <= width {
                    assert_eq!(output, value);
                } else if width >= ellipsis.chars().count() {
                    assert!(output.ends_with(ellipsis));
                }
            }
        }
    }
    assert_eq!(clip("猫😺xxx", 3, "--"), "猫--");
}

#[test]
fn legacy_dictionary_renderers_clip_unicode_and_obey_row_limits() {
    use groggy::display::{format_data_structure, DisplayConfig};
    use std::collections::HashMap;
    let config = DisplayConfig {
        max_rows: 1,
        use_color: false,
        ..Default::default()
    };
    for kind in ["array", "matrix"] {
        let text = "猫😺".repeat(12);
        let data = if kind == "array" {
            serde_json::json!([text, "hidden-row"])
        } else {
            serde_json::json!([[text], ["hidden-row"]])
        };
        let fields = HashMap::from([
            ("data".into(), data),
            ("shape".into(), serde_json::json!([2, 1])),
        ]);
        let output = format_data_structure(fields, Some(kind), &config);
        assert!(!output.contains("hidden-row"));
        assert!(output.contains("猫😺"));
    }
}

#[test]
fn interactive_layout_keeps_scalar_spellings_and_matches_type_headers() {
    use groggy::display::render::interactive::{array, matrix, table};
    let cells = vec![
        (Some(0), "…".into(), false),
        (None, "…".into(), false),
        (Some(99), "True".into(), true),
    ];
    let rendered = array(&cells, 100, "values", "object", "(100,)");
    assert!(rendered.contains("│  0 │ …"));
    assert!(rendered.contains("│ 99 │   True │"));
    assert_eq!(
        array(&[], 0, "a", "f64", "(0,)"),
        "⊖⊖ gr.array (empty) • shape: (0,) • dtype: f64"
    );
    let rows = vec![vec!["猫😺".into(), "-0.00".into()]];
    let rendered = matrix(&rows, false, "(1, 2)", "f64");
    assert!(rendered.contains("-0.00"));
    let plain = table(
        &["#".into(), "last".into()],
        &["".into(), "f64".into()],
        &rows,
        false,
        "summary",
        40,
        false,
    );
    let colored = table(
        &["#".into(), "last".into()],
        &["".into(), "f64".into()],
        &rows,
        false,
        "summary",
        40,
        true,
    );
    assert!(plain.contains("f64"));
    assert!(!plain.contains('\u{1b}'));
    assert!(colored.contains("\u{1b}[1m"));
    assert!(colored.contains("\u{1b}[2m"));
}
