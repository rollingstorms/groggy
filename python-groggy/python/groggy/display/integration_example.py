"""
Integration example showing how to hook the display module into existing Groggy classes.

This is a mockup showing how the display system would integrate with 
PyGraphTable, PyGraphMatrix, and PyGraphArray classes.
"""

from typing import Dict, Any, List
from groggy.display import format_table, format_matrix, format_array

class PyGraphTableWithDisplay:
    """Example PyGraphTable class with rich display integration."""
    
    def __init__(self, data: List[List[Any]], columns: List[str], dtypes: Dict[str, str]):
        self.data = data
        self.columns = columns
        self.dtypes = dtypes
    
    def _get_display_data(self) -> Dict[str, Any]:
        """Extract data in format expected by display module."""
        return {
            'columns': self.columns,
            'dtypes': self.dtypes,
            'data': self.data,
            'shape': (len(self.data), len(self.columns)),
            'nulls': self._count_nulls(),
            'index_type': 'int64'
        }
    
    def _count_nulls(self) -> Dict[str, int]:
        """Count null values per column."""
        nulls = {}
        for col_idx, col_name in enumerate(self.columns):
            null_count = sum(1 for row in self.data if row[col_idx] is None)
            if null_count > 0:
                nulls[col_name] = null_count
        return nulls
    
    def __repr__(self) -> str:
        """Rich display representation."""
        try:
            return format_table(self._get_display_data())
        except Exception as e:
            # Fallback to simple representation
            return f"PyGraphTable(shape={len(self.data), len(self.columns)}, columns={self.columns})"
    
    def __str__(self) -> str:
        """String representation (same as repr for rich display)."""
        return self.__repr__()

class PyGraphMatrixWithDisplay:
    """Example PyGraphMatrix class with rich display integration."""
    
    def __init__(self, data: List[List[Any]], dtype: str = 'f32'):
        self.data = data
        self.dtype = dtype
    
    def _get_display_data(self) -> Dict[str, Any]:
        """Extract data in format expected by display module."""
        return {
            'data': self.data,
            'shape': (len(self.data), len(self.data[0]) if self.data else 0),
            'dtype': self.dtype
        }
    
    def __repr__(self) -> str:
        """Rich display representation."""
        try:
            return format_matrix(self._get_display_data())
        except Exception as e:
            # Fallback to simple representation
            shape = (len(self.data), len(self.data[0]) if self.data else 0)
            return f"PyGraphMatrix(shape={shape}, dtype={self.dtype})"
    
    def __str__(self) -> str:
        """String representation (same as repr for rich display)."""
        return self.__repr__()

class PyGraphArrayWithDisplay:
    """Example PyGraphArray class with rich display integration."""
    
    def __init__(self, data: List[Any], dtype: str = 'f32', name: str = 'col1'):
        self.data = data
        self.dtype = dtype
        self.name = name
    
    def _get_display_data(self) -> Dict[str, Any]:
        """Extract data in format expected by display module."""
        return {
            'data': self.data,
            'dtype': self.dtype,
            'shape': (len(self.data),),
            'name': self.name
        }
    
    def __repr__(self) -> str:
        """Rich display representation."""
        try:
            return format_array(self._get_display_data())
        except Exception as e:
            # Fallback to simple representation
            return f"PyGraphArray(shape=({len(self.data)},), dtype={self.dtype})"
    
    def __str__(self) -> str:
        """String representation (same as repr for rich display)."""
        return self.__repr__()

# Demo the integration
def demo_integration():
    """Demonstrate the integrated display system."""
    print("Groggy Display Integration Demo")
    print("==============================\n")
    
    # Demo GraphTable with display
    table = PyGraphTableWithDisplay(
        data=[
            ['Alice', 'NYC', 25, 91.5],
            ['Bob', 'Paris', 30, 87.0],
            ['Charlie', 'Tokyo', 35, None]
        ],
        columns=['name', 'city', 'age', 'score'],
        dtypes={'name': 'string', 'city': 'category', 'age': 'int64', 'score': 'float32'}
    )
    
    print("GraphTable with Rich Display:")
    print(table)
    print()
    
    # Demo GraphMatrix with display
    matrix = PyGraphMatrixWithDisplay(
        data=[
            [0.12, -1.50, 2.00],
            [3.14, 0.00, 1.25],
            [-0.01, 4.50, 8.00]
        ],
        dtype='f32'
    )
    
    print("GraphMatrix with Rich Display:")
    print(matrix)
    print()
    
    # Demo GraphArray with display  
    array = PyGraphArrayWithDisplay(
        data=[0.125, 3.1416, float('nan'), -2.75, 8.0],
        dtype='f32',
        name='values'
    )
    
    print("GraphArray with Rich Display:")
    print(array)
    print()

if __name__ == '__main__':
    demo_integration()
