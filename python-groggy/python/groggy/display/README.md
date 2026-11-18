# Groggy Rich Display Module

Beautiful, professional display formatting for GraphTable, GraphMatrix, and GraphArray data structures using Unicode box-drawing characters and smart truncation.

## Features

- **Professional Unicode formatting** with box-drawing characters (`╭─╮│├┤╰─╯`)
- **Smart truncation** for large datasets showing first/last rows and columns
- **Type annotations** showing data types (str[8], cat(12), f32, date, etc.)
- **Summary statistics** including row counts, column counts, null counts
- **Color support** (optional) for enhanced terminal display
- **Consistent API** across all Groggy data structures

## Example Outputs

### GraphTable Display
```
⊖⊖ gr.table
╭──────┬─────────┬───────────┬──────┬───────┬────────────╮
│    # │ name    │ city      │ age  │ score │ joined     │
│      │ str[8]  │ cat(12)   │ i64  │ f32   │ date       │
├──────┼─────────┼───────────┼──────┼───────┼────────────┤
│    0 │ Alice   │ NYC       │ 25   │ 91.50 │ 2024-02-15 │
│    1 │ Bob     │ Paris     │ 30   │ 87.00 │ 2023-11-20 │
│    … │ …       │ …         │ …    │ …     │ …          │
│   11 │ Liam    │ Amsterdam │ 30   │ 91.90 │ 2023-07-16 │
╰──────┴─────────┴───────────┴──────┴───────┴────────────╯
rows: 1,000 • cols: 5 • nulls: score=12 • index: int64
```

### GraphMatrix Display
```
⊖⊖ gr.matrix
╭────────┬────────┬────────┬────────┬────────┬────────╮
│   0.12 │  -1.50 │   2.00 │   ⋯    │   7.77 │   8.88 │
│   3.14 │   0.00 │    NaN │   ⋯    │  -2.10 │   1.25 │
│   ⋯    │   ⋯    │   ⋯    │   ⋯    │   ⋯    │   ⋯    │
│  -0.01 │   4.50 │  -2.30 │   ⋯    │   9.99 │   8.00 │
╰────────┴────────┴────────┴────────┴────────┴────────╯
shape: (500, 200) • dtype: f32
```

### GraphArray Display
```
⊖⊖ gr.array

╭───┬────────╮
│ # │ values │
│   │ f32    │
├───┼────────┤
│ 0 │  0.125 │
│ 1 │  3.142 │
│ 2 │ NaN    │
│ 3 │  -2.75 │
│ 4 │      8 │
╰───┴────────╯
shape: (5,)
```

## Usage

### Direct Formatting Functions

```python
from groggy.display import format_table, format_matrix, format_array

# Format a table
table_data = {
    'columns': ['name', 'age'],
    'dtypes': {'name': 'string', 'age': 'int64'},
    'data': [['Alice', 25], ['Bob', 30]],
    'shape': (2, 2),
    'nulls': {},
    'index_type': 'int64'
}
print(format_table(table_data))

# Format a matrix
matrix_data = {
    'data': [[1.0, 2.0], [3.0, 4.0]],
    'shape': (2, 2),
    'dtype': 'f32'
}
print(format_matrix(matrix_data))

# Format an array
array_data = {
    'data': [1.0, 2.0, 3.0],
    'dtype': 'f32',
    'shape': (3,),
    'name': 'values'
}
print(format_array(array_data))
```

### Class Integration

```python
from groggy.display import format_table

class MyDataStructure:
    def __init__(self, data):
        self.data = data
    
    def _get_display_data(self):
        return {
            'columns': ['col1', 'col2'],
            'dtypes': {'col1': 'int64', 'col2': 'float32'},
            'data': self.data,
            'shape': (len(self.data), 2),
            'nulls': {},
            'index_type': 'int64'
        }
    
    def __repr__(self):
        return format_table(self._get_display_data())
```

### Configuration

```python
from groggy.display import configure_display

# Configure global display settings
configure_display(
    max_rows=20,       # Show up to 20 rows
    max_cols=10,       # Show up to 10 columns  
    max_width=150,     # Maximum terminal width
    precision=3        # Floating point precision
)
```

## Module Structure

```
groggy/display/
├── __init__.py          # Public API and configuration
├── formatters.py        # Main formatting functions
├── table_display.py     # GraphTable formatter
├── matrix_display.py    # GraphMatrix formatter
├── array_display.py     # GraphArray formatter
├── unicode_chars.py     # Box-drawing characters
├── truncation.py        # Smart truncation algorithms
├── demo.py             # Demonstration script
└── integration_example.py # Integration examples
```

## Data Format Requirements

### GraphTable Data Format
```python
{
    'columns': ['col1', 'col2'],           # Column names
    'dtypes': {'col1': 'int64', 'col2': 'f32'}, # Column data types
    'data': [[1, 2.0], [3, 4.0]], # Row data
    'shape': (2, 2),                      # Dimensions
    'nulls': {'col1': 0},              # Null counts per column
    'index_type': 'int64'                       # Index data type
}
```

### GraphMatrix Data Format
```python
{
    'data': [[1, 2], [3, 4]],    # 2D matrix data
    'shape': (2, 2),       # Dimensions
    'dtype': 'f32'               # Element data type
}
```

### GraphArray Data Format
```python
{
    'data': [1, 2, 3, 4],    # 1D array data
    'dtype': 'f32',          # Element data type
    'shape': (4,),      # Dimensions
    'name': 'column_name'    # Optional column name
}
```

## Implementation Notes

- **Performance**: Display formatting is optimized for interactive use with reasonable defaults
- **Validation**: Invalid configuration and unrepresentable text raise explicit errors; callers choose their own fallback.
- **Terminal Support**: Automatically detects color support and adjusts formatting accordingly
- **Unicode Safety**: Uses standard Unicode box-drawing characters supported by modern terminals
- **Visible cells**: Dictionary adapters select rows/columns before invoking Python scalar formatting. Native rendering owns no Python objects. This does not cap memory already held by the input.

## Running the Demo

```bash
cd python-groggy/python
python -m groggy.display.demo
```

This will show example outputs for all three data structure types with various sizes and content types.


## Native ownership and compatibility

Layouts, row/column plans, text clipping and dtype spellings are owned by
`groggy::display` / `display::render` in Rust. The Python modules retain object
projection, Python scalar spellings, ordered summaries and terminal detection.
The public imports and normal default layouts remain available.

`configure_display` now affects new dictionary formatters; explicit options win
and existing formatter instances retain their settings. It does not reconfigure
the separate typed graph/table HTML renderer. Zero limits are valid; negative or
noninteger configuration rejects before changing defaults. `max_width` is a table
allocation hint, not a hard terminal width promise. Array/matrix retain that
setting without assigning it new semantics.

The migration corrects tiny-budget indices, literal ellipsis handling, truncated
type headers, duplicate date ellipses and native UTF-8/row-limit defects. Full
policies and executable examples: `docs/api/display-foundation.md` in the source
repository. Performance observations distinguish Python traced memory from native
allocation and RSS; no production timing guarantee is implied.
