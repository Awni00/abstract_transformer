# This file contains reusable animation functions that can be used across different scenes.

from manim import *

def fade_in(obj, duration=1):
    """Fade in an object over a specified duration."""
    return FadeIn(obj, duration=duration)

def slide_in_from_left(obj, duration=1):
    """Slide in an object from the left over a specified duration."""
    return obj.shift(LEFT * config.frame_width).animate.shift(RIGHT * config.frame_width).set_duration(duration)

def write_text(text, duration=1):
    """Write text on the screen over a specified duration."""
    text_mobject = Text(text)
    return FadeIn(text_mobject, duration=duration)

def draw_circle(radius=1, color=WHITE):
    """Draw a circle with a specified radius and color."""
    circle = Circle(radius=radius, color=color)
    return circle

def create_rectangle(width=2, height=1, color=WHITE):
    """Create a rectangle with specified width, height, and color."""
    rectangle = Rectangle(width=width, height=height, color=color)
    return rectangle

def animate_mobject(mobject, animation, **kwargs):
    """Animate a given mobject with a specified animation."""
    return animation(mobject, **kwargs)