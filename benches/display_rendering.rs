//! Matched display refactoring receipt. ASCII output fingerprints must stay exact.
use groggy::viz::display::{
    ColumnSchema, DataSchema, DataType, DataWindow, DisplayConfig, DisplayEngine,
};
use serde_json::json;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for rows in [0, 1, 10, 100] {
        let data = DataWindow::new(
            vec!["name".into(), "count".into(), "score".into(), "json".into()],
            (0..rows)
                .map(|id| {
                    vec![
                        format!("name <&> {id}"),
                        format!("{}", 10000 + id),
                        format!("{}.125", id),
                        format!(r#"{{"id":{id},"tag":"value"}}"#),
                    ]
                })
                .collect(),
            DataSchema::new(
                [
                    DataType::String,
                    DataType::Integer,
                    DataType::Float,
                    DataType::Json,
                ]
                .into_iter()
                .enumerate()
                .map(|(i, data_type)| ColumnSchema {
                    name: format!("c{i}"),
                    data_type,
                })
                .collect(),
            ),
        );
        for mode in ["compact", "full", "html"] {
            let config = DisplayConfig {
                compact_mode: mode != "full",
                ..Default::default()
            };
            let engine = DisplayEngine::with_config(config);
            let render = || {
                if mode == "html" {
                    engine.format_html(&data)
                } else {
                    engine.format_unicode(&data)
                }
            };
            let output = render();
            let mut hasher = DefaultHasher::new();
            output.hash(&mut hasher);
            let hash = format!("{:016x}", hasher.finish());
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                for _ in 0..100 {
                    std::hint::black_box(render());
                }
                samples.push(start.elapsed().as_secs_f64() / 100.0);
            }
            cases.push(json!({"rows":rows,"mode":mode,"seconds_per_render":samples,"ascii_output_hash":hash,"output_bytes":output.len()}));
        }
    }
    let mut text_prefix = Vec::new();
    for count in [1_000, 1_000_000] {
        let text = "猫😺".repeat(count / 2);
        let mut old = Vec::new();
        let mut bounded = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..100 {
                std::hint::black_box(previous_prefix(std::hint::black_box(&text), 20));
            }
            old.push(start.elapsed().as_secs_f64() / 100.0);
            let start = Instant::now();
            for _ in 0..100 {
                std::hint::black_box(groggy::display::truncation::truncate_string(
                    std::hint::black_box(&text),
                    20,
                ));
            }
            bounded.push(start.elapsed().as_secs_f64() / 100.0);
        }
        assert_eq!(
            previous_prefix(&text, 20),
            groggy::display::truncation::truncate_string(&text, 20)
        );
        text_prefix.push(json!({"input_characters":count,"budget_characters":20,"prior_count_all_seconds":old,"bounded_prefix_seconds":bounded}));
    }
    let output =
        std::env::var("GROGGY_DISPLAY_PROFILE").unwrap_or_else(|_| "display_rendering.json".into());
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(
            &json!({"scope":"Native release ASCII display corpus: compact/full Unicode and themed semantic HTML. Five samples of 100 renders; fixture/config construction excluded, rendering allocations included but not counted. Fingerprints use DefaultHasher only for a same-toolchain matched refactor comparison, not a stable external hash. Text-prefix cases compare the prior safe helper from 97705924 against the bounded helper in the same binary; budget 20; construction excluded and outputs validated outside timing. Whole HTML still escapes full tooltip content. No allocation/RSS/GIL/production claim.","cases":cases,"text_prefix":text_prefix}),
        )?,
    )?;
    println!("Wrote display receipt to {output}");
    Ok(())
}

// Positive-budget reference from 97705924 src/display/truncation.rs.
fn previous_prefix(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        text.to_owned()
    } else {
        format!(
            "{}…",
            text.chars()
                .take(width.saturating_sub(1))
                .collect::<String>()
        )
    }
}
