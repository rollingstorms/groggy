from typing import Union, List, Dict, Optional, Tuple
from enum import Enum

# Type aliases matching Rust
NodeId = int
EdgeId = int  
AttrName = str
StateId = int
BranchName = str

class AttrValue:
    """Python representation of Rust AttrValue enum"""
    
    def __init__(self, value: Union[int, float, str, bool, List[float], bytes]):
        self._value = value
        self._type = self._determine_type(value)
    
    def _determine_type(self, value):
        if isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "text"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, list) and all(isinstance(x, float) for x in value):
            return "float_vec"
        elif isinstance(value, bytes):
            return "bytes"
        else:
            raise ValueError(f"Unsupported attribute value type: {type(value)}")
    
    @property
    def value(self):
        return self._value
    
    @property 
    def type_name(self) -> str:
        return self._type
    
    def __repr__(self) -> str:
        return f"AttrValue({self._value!r})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, AttrValue):
            return self._value == other._value
        return False

# Query filter types
class AttributeFilter:
    def __init__(self, filter_type: str, value: AttrValue, **kwargs):
        self.filter_type = filter_type
        self.value = value
        self.kwargs = kwargs

class NodeFilter:
    def __init__(self, filter_type: str, **kwargs):
        self.filter_type = filter_type
        self.kwargs = kwargs

class EdgeFilter:
    def __init__(self, filter_type: str, **kwargs):
        self.filter_type = filter_type
        self.kwargs = kwargs
