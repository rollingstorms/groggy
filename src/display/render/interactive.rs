//! Layouts for the historical Python dictionary presentation.
//!
//! Python supplies its own scalar spellings (including arbitrary objects and
//! significant-digit formatting). Selection, widths, clipping, padding, borders
//! and ANSI layout have one native owner; no Python object crosses this boundary.
use crate::display::{
    truncation,
    unicode_chars::{colorize, BoxChars, Symbols},
};

/// Visible positions, with `None` denoting an omitted middle segment.
/// Tiny budgets select a prefix. Work and storage are bounded by visible length.
pub fn positions(length: usize, budget: usize) -> Vec<Option<usize>> {
    if length <= budget {
        return (0..length).map(Some).collect();
    }
    if budget < 3 {
        return (0..budget).map(Some).collect();
    }
    let first = budget / 2;
    let last = budget - first - 1;
    (0..first)
        .map(Some)
        .chain(std::iter::once(None))
        .chain((length - last..length).map(Some))
        .collect()
}

/// Python-compatible clipping with an explicit custom ellipsis.
pub fn clip(text: &str, width: usize, ellipsis: &str) -> String {
    if ellipsis == Symbols::ELLIPSIS {
        return truncation::truncate_string(text, width);
    }
    if !truncation::exceeds_character_budget(text, width) {
        return text.to_owned();
    }
    let ellipsis_width = ellipsis.chars().count();
    if width < ellipsis_width {
        text.chars().take(width).collect()
    } else {
        text.chars()
            .take(width - ellipsis_width)
            .chain(ellipsis.chars())
            .collect()
    }
}

/// Minimal historical widths, proportionally reduced to the total budget.
/// The historical three-character readable minimum can exceed a tiny budget.
pub fn widths(headers: &[String], rows: &[Vec<String>], budget: usize) -> Vec<usize> {
    let n = headers.len();
    if n == 0 {
        return Vec::new();
    }
    let minimum: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            rows.iter()
                .filter_map(|r| r.get(i))
                .map(|s| s.chars().count())
                .chain(std::iter::once(h.chars().count()))
                .max()
                .unwrap_or(0)
                .max(4)
        })
        .collect();
    let overhead = n.saturating_mul(3).saturating_add(1);
    let available = budget.saturating_sub(overhead);
    let total: usize = minimum.iter().sum();
    if total <= available {
        return minimum;
    }
    if available < n.saturating_mul(3) {
        return vec![3; n];
    }
    let scale = available as f64 / total as f64;
    let mut result: Vec<usize> = minimum
        .iter()
        .map(|&w| ((w as f64 * scale) as usize).max(3))
        .collect();
    let actual: usize = result.iter().sum();
    for i in 0..actual.abs_diff(available) {
        let cell = &mut result[i % n];
        if actual < available {
            *cell += 1;
        } else if *cell > 3 {
            *cell -= 1;
        }
    }
    result
}

fn pad(text: &str, width: usize, alignment: u8) -> String {
    let extra = width.saturating_sub(text.chars().count());
    let left = match alignment {
        1 => extra,
        2 => extra / 2 + usize::from(extra % 2 == 1 && width % 2 == 1),
        _ => 0,
    };
    format!("{}{}{}", " ".repeat(left), text, " ".repeat(extra - left))
}

pub(crate) fn border(widths: &[usize], position: u8) -> String {
    let (left, middle, right) = match position {
        0 => (BoxChars::TOP_LEFT, BoxChars::T_TOP, BoxChars::TOP_RIGHT),
        1 => (BoxChars::T_LEFT, BoxChars::CROSS, BoxChars::T_RIGHT),
        _ => (
            BoxChars::BOTTOM_LEFT,
            BoxChars::T_BOTTOM,
            BoxChars::BOTTOM_RIGHT,
        ),
    };
    format!(
        "{}{}{}",
        left,
        widths
            .iter()
            .map(|w| BoxChars::HORIZONTAL.repeat(w.saturating_add(2)))
            .collect::<Vec<_>>()
            .join(middle),
        right
    )
}

pub(crate) fn row(
    cells: &[String],
    widths: &[usize],
    alignment: impl Fn(usize, &str) -> u8,
    truncate: bool,
    style: u8,
    color: bool,
) -> String {
    let cells: Vec<_> = cells
        .iter()
        .zip(widths)
        .enumerate()
        .map(|(i, (cell, &width))| {
            let cell = if truncate {
                clip(cell, width, Symbols::ELLIPSIS)
            } else {
                cell.clone()
            };
            let padded = pad(&cell, width, alignment(i, &cell));
            format!(
                " {} ",
                colorize(&padded, None, style == 1, style == 2, color)
            )
        })
        .collect();
    format!(
        "{}{}{}",
        BoxChars::VERTICAL,
        cells.join(BoxChars::VERTICAL),
        BoxChars::VERTICAL
    )
}

/// Render preformatted visible array cells `(original index, text, numeric)`.
/// An absent index is an actual omission marker, never a magic user value.
pub fn array(
    cells: &[(Option<usize>, String, bool)],
    length: usize,
    name: &str,
    dtype: &str,
    shape: &str,
) -> String {
    if length == 0 {
        return format!(
            "{} gr.array (empty) • shape: {} • dtype: {}",
            Symbols::HEADER_PREFIX,
            shape,
            dtype
        );
    }
    let index_width = length.saturating_sub(1).to_string().len().max(1);
    let value_width = cells
        .iter()
        .map(|(_, s, _)| s.chars().count())
        .max()
        .unwrap_or(6)
        .max(name.chars().count())
        .min(20);
    let widths = [index_width, value_width];
    let mut lines = vec![
        format!("{} gr.array", Symbols::HEADER_PREFIX),
        String::new(),
        border(&widths, 0),
    ];
    lines.push(row(
        &["#".into(), clip(name, value_width, Symbols::ELLIPSIS)],
        &widths,
        |i, _| if i == 0 { 2 } else { 0 },
        false,
        0,
        false,
    ));
    lines.push(row(
        &[String::new(), dtype.into()],
        &widths,
        |_, _| 0,
        false,
        0,
        false,
    ));
    lines.push(border(&widths, 1));
    for (index, text, numeric) in cells {
        let values = [
            index.map_or_else(|| Symbols::ELLIPSIS.into(), |i| i.to_string()),
            text.clone(),
        ];
        lines.push(row(
            &values,
            &widths,
            |i, _| {
                if index.is_none() {
                    2
                } else if i == 0 || *numeric {
                    1
                } else {
                    0
                }
            },
            false,
            0,
            false,
        ));
    }
    lines.extend([border(&widths, 2), format!("shape: {}", shape)]);
    lines.join("\n")
}

/// Render the historical matrix layout from already selected scalar spellings.
pub fn matrix(rows: &[Vec<String>], empty: bool, shape: &str, dtype: &str) -> String {
    if empty {
        return format!(
            "{} gr.matrix (empty) • shape: {} • dtype: {}",
            Symbols::HEADER_PREFIX,
            shape,
            dtype
        );
    }
    let columns = rows.first().map_or(0, Vec::len);
    let widths: Vec<_> = (0..columns)
        .map(|i| {
            rows.iter()
                .filter_map(|r| r.get(i))
                .map(|s| s.chars().count())
                .max()
                .unwrap_or(6)
                .clamp(6, 12)
        })
        .collect();
    let mut lines = vec![
        format!("{} gr.matrix", Symbols::HEADER_PREFIX),
        border(&widths, 0),
    ];
    lines.extend(rows.iter().map(|r| {
        row(
            r,
            &widths,
            |_, s| {
                if s == Symbols::TRUNCATION_INDICATOR {
                    2
                } else {
                    1
                }
            },
            false,
            0,
            false,
        )
    }));
    lines.extend([
        border(&widths, 2),
        format!("shape: {} • dtype: {}", shape, dtype),
    ]);
    lines.join("\n")
}

/// Render the historical table layout; header and dtype selection are paired.
pub fn table(
    headers: &[String],
    types: &[String],
    rows: &[Vec<String>],
    empty: bool,
    summary: &str,
    budget: usize,
    color: bool,
) -> String {
    if empty {
        return format!("{} gr.table (empty)", Symbols::HEADER_PREFIX);
    }
    let widths = widths(headers, rows, budget);
    let align = |i, _: &str| if i == 0 { 1 } else { 0 };
    let mut lines = vec![
        format!("{} gr.table", Symbols::HEADER_PREFIX),
        border(&widths, 0),
        row(headers, &widths, align, true, 1, color),
        row(types, &widths, align, true, 2, color),
        border(&widths, 1),
    ];
    lines.extend(rows.iter().map(|r| row(r, &widths, align, true, 0, color)));
    lines.extend([border(&widths, 2), summary.into()]);
    lines.join("\n")
}
