/*!
Text truncation utilities for display formatting.
Direct Rust port of the Python truncation.py module.
*/

use std::cmp;

/// Truncate to a Unicode scalar-character budget, adding ellipsis if needed.
/// A zero budget returns empty text. This is not terminal/grapheme display width.
#[inline]
pub fn truncate_string(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if text.as_bytes()[..text.len().min(max_width.saturating_add(1))].is_ascii() {
        if text.len() <= max_width {
            return text.to_owned();
        }
        // Only a bounded ASCII prefix is inspected, including the cut position.
        return format!(
            "{}{}",
            &text[..max_width - 1],
            super::unicode_chars::Symbols::ELLIPSIS
        );
    }
    let mut chars = text.char_indices();
    let Some((end, _)) = chars.nth(max_width - 1) else {
        return text.to_owned();
    };
    if chars.next().is_none() {
        return text.to_owned();
    }
    // end comes from this immutable str's char_indices, so slicing cannot split
    // UTF-8. Only the visible budget plus one character is inspected.
    format!(
        "{}{}",
        &text[..end],
        super::unicode_chars::Symbols::ELLIPSIS
    )
}

#[inline]
pub(crate) fn exceeds_character_budget(text: &str, max_width: usize) -> bool {
    if text.as_bytes()[..text.len().min(max_width.saturating_add(1))].is_ascii() {
        text.len() > max_width
    } else {
        text.chars().nth(max_width).is_some()
    }
}

#[inline]
pub(crate) fn character_width_up_to(text: &str, limit: usize) -> usize {
    if text.as_bytes()[..text.len().min(limit)].is_ascii() {
        text.len().min(limit)
    } else {
        text.chars().take(limit).count()
    }
}

/// Truncate rows to maximum display count
pub fn truncate_rows<T: Clone>(data: Vec<T>, max_rows: usize) -> (Vec<T>, bool) {
    if data.len() <= max_rows {
        (data, false)
    } else {
        let show_first = max_rows / 2;
        let show_last = max_rows.saturating_sub(show_first + 1); // reserve ellipsis row

        let mut result = Vec::new();

        // Add first rows
        result.extend(data.iter().take(show_first).cloned());

        // Add ellipsis placeholder (represented as special marker)
        // We'll handle this in the calling code

        // Add last rows
        if show_last > 0 {
            result.extend(data.iter().skip(data.len() - show_last).cloned());
        }

        (result, true)
    }
}

/// Truncate columns to maximum display count
pub fn truncate_columns<T: Clone>(
    headers: Vec<T>,
    data: Vec<Vec<T>>,
    max_cols: usize,
) -> (Vec<T>, Vec<Vec<T>>, bool) {
    if headers.len() <= max_cols {
        (headers, data, false)
    } else {
        let show_first = max_cols / 2;
        let show_last = max_cols.saturating_sub(show_first + 1); // reserve ellipsis column

        // Truncate headers
        let mut truncated_headers = Vec::new();
        truncated_headers.extend(headers.iter().take(show_first).cloned());
        // Ellipsis header will be added by calling code
        if show_last > 0 {
            truncated_headers.extend(headers.iter().skip(headers.len() - show_last).cloned());
        }

        // Truncate data rows
        let truncated_data: Vec<Vec<T>> = data
            .into_iter()
            .map(|row| {
                let mut truncated_row = Vec::new();
                truncated_row.extend(row.iter().take(show_first).cloned());
                // Ellipsis cell will be added by calling code
                if show_last > 0 {
                    truncated_row.extend(row.iter().skip(row.len() - show_last).cloned());
                }
                truncated_row
            })
            .collect();

        (truncated_headers, truncated_data, true)
    }
}

/// Calculate optimal column widths for display
pub fn calculate_column_widths(
    headers: &[String],
    data: &[Vec<String>],
    max_total_width: usize,
) -> Vec<usize> {
    let num_cols = headers.len();
    if num_cols == 0 {
        return Vec::new();
    }

    // Calculate minimum required width for each column
    let mut min_widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();

    for row in data {
        for (i, cell) in row.iter().enumerate() {
            if i < min_widths.len() {
                min_widths[i] = cmp::max(min_widths[i], cell.chars().count());
            }
        }
    }

    // Calculate available width (accounting for borders and padding)
    // Each column needs: | padding content padding |
    // So we need: num_cols * 3 + 1 characters for borders/padding
    let border_overhead = num_cols * 3 + 1;
    let available_width = max_total_width.saturating_sub(border_overhead);

    let total_min_width: usize = min_widths.iter().sum();

    if total_min_width <= available_width {
        // We have extra space - distribute it proportionally
        let extra_space = available_width - total_min_width;
        let extra_per_col = extra_space / num_cols;
        let remainder = extra_space % num_cols;

        for (i, width) in min_widths.iter_mut().enumerate() {
            *width += extra_per_col;
            if i < remainder {
                *width += 1; // Distribute remainder to first few columns
            }
        }
    } else {
        // Not enough space - we need to truncate some columns
        // For now, just use minimum widths (truncation will happen in formatting)
    }

    min_widths
}
