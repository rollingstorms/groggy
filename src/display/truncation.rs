/*!
Text truncation utilities for display formatting.
Direct Rust port of the Python truncation.py module.
*/

use std::cmp;

/// Truncate a string to specified width, adding ellipsis if needed
pub fn truncate_string(text: &str, max_width: usize) -> String {
    if text.chars().count() <= max_width {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_width.saturating_sub(1)).collect();
        format!("{}{}", truncated, super::unicode_chars::Symbols::ELLIPSIS)
    }
}

/// Truncate rows to maximum display count
pub fn truncate_rows<T: Clone>(data: Vec<T>, max_rows: usize) -> (Vec<T>, bool) {
    if data.len() <= max_rows {
        (data, false)
    } else {
        let show_first = max_rows / 2;
        let show_last = max_rows - show_first - 1; // -1 for ellipsis row

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
        let show_last = max_cols - show_first - 1; // -1 for ellipsis column

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
