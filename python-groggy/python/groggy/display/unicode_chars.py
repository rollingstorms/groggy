"""
Unicode box-drawing characters and display symbols for rich formatting.
"""

# Box drawing characters for table borders
class BoxChars:
    """Unicode box-drawing characters for professional table display."""
    
    # Corners
    TOP_LEFT = '╭'
    TOP_RIGHT = '╮'
    BOTTOM_LEFT = '╰'
    BOTTOM_RIGHT = '╯'
    
    # Lines
    HORIZONTAL = '─'
    VERTICAL = '│'
    
    # Intersections
    CROSS = '┼'
    T_TOP = '┬'
    T_BOTTOM = '┴'
    T_LEFT = '├'
    T_RIGHT = '┤'
    
    # Double lines for emphasis
    HORIZONTAL_DOUBLE = '═'
    VERTICAL_DOUBLE = '║'

# Display symbols  
class Symbols:
    """Special symbols for data display."""
    
    ELLIPSIS = '…'          # For truncated content
    DOT_SEPARATOR = '•'     # For summary statistics  
    NULL_DISPLAY = 'NaN'    # For null/missing values
    TRUNCATION_INDICATOR = '⋯'  # For matrix truncation
    HEADER_PREFIX = '⊖⊖'    # For section headers

# Color codes (if terminal supports color)
class Colors:
    """ANSI color codes for enhanced display."""
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Text colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

def has_color_support():
    """Check if the current terminal supports color output."""
    import os
    import sys
    
    # Check for common indicators of color support
    if os.getenv('NO_COLOR'):
        return False
    if os.getenv('FORCE_COLOR'):
        return True
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    
    term = os.getenv('TERM', '').lower()
    return any(color_term in term for color_term in ['color', 'xterm', 'screen', 'tmux'])

def colorize(text, color=None, bold=False, dim=False):
    """Apply color formatting to text if color is supported."""
    if not has_color_support():
        return text
    
    result = ''
    if bold:
        result += Colors.BOLD
    if dim:
        result += Colors.DIM
    if color:
        result += color
    
    result += text
    if result != text:  # Only add reset if we added formatting
        result += Colors.RESET
        
    return result
