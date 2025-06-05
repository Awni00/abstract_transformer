# This file defines color constants based on your LaTeX presentation scheme

from manim import *

# Define colors based on your LaTeX color scheme
SENSORY_COLOR = "#8B0000"      # Maroon equivalent
RELATIONAL_COLOR = "#4169E1"   # RoyalBlue equivalent  
ATTENTION_COLOR = "#CCCCFF"    # Periwinkle equivalent
SYMBOLIC_COLOR = "#228B22"     # ForestGreen equivalent
MIXED_COLOR = "#DDA0DD"        # Plum equivalent
ALERT_COLOR = "#FF1493"        # WildStrawberry equivalent

# Additional useful colors
BACKGROUND_COLOR = "#2E3440"
TEXT_COLOR = WHITE
SUBTITLE_COLOR = GRAY_B
HIGHLIGHT_COLOR = YELLOW

class Colors:
    PRIMARY = "#1f77b4"  # Blue
    SECONDARY = "#ff7f0e"  # Orange
    TERTIARY = "#2ca02c"  # Green
    QUATERNARY = "#d62728"  # Red
    QUINARY = "#9467bd"  # Purple
    SENARY = "#8c564b"  # Brown
    SEVENTH = "#e377c2"  # Pink
    EIGHTH = "#7f7f7f"  # Gray
    NINTH = "#bcbd22"  # Olive
    TENTH = "#17becf"  # Cyan

    @classmethod
    def get_color(cls, color_name):
        return getattr(cls, color_name.upper(), None)