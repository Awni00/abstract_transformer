# Manim-Slides Presentation: Disentangling and Integrating Relational and Sensory Information in Transformer Architectures

This directory contains a Manim-based presentation that translates the LaTeX beamer slides to an interactive manim-slides presentation.

## Structure

```
manim-slides/
├── src/
│   ├── main.py                 # Main presentation file
│   └── utils/
│       ├── colors.py           # Color constants matching LaTeX theme
│       └── latex_macros.py     # LaTeX macro translations
├── config/
│   └── manim.cfg              # Manim configuration
├── requirements.txt           # Python dependencies
├── run_presentation.py        # Helper script to run presentation
└── README.md                  # This file
```

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using the helper script
```bash
python run_presentation.py
```

### Option 2: Manual commands
```bash
# Render the presentation
manim-slides render src/main.py Main

# Run the presentation
manim-slides Main
```

### Option 3: Development mode (high quality)
```bash
manim-slides render src/main.py Main --quality high_quality
manim-slides Main
```

## Usage Guidelines

- Each scene is designed to correspond to a specific section of the original presentation. You can modify individual scene files to update the content as needed.
- The `utils/latex_macros.py` file contains functions that replicate the LaTeX macros defined in `mymathstyle.sty` and `paper_commands.sty`, ensuring consistency in mathematical notation and formatting.
- The `utils/colors.py` file allows you to define and use color constants throughout the presentation for a cohesive visual style.