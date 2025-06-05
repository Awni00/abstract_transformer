#!/usr/bin/env python3
"""
Simple test script to render and view the manim-slides presentation
"""

import subprocess
import sys
import os

def main():
    # Change to the correct directory
    os.chdir(r"c:\Users\awnya\Documents\project-code\abstract_transformer\manim-slides")
    
    # Render the presentation
    print("Rendering presentation...")
    try:
        subprocess.run([
            "manim-slides", "render", 
            "src/main.py", "Main",
            "--quality", "medium_quality"
        ], check=True)
        
        print("Presentation rendered successfully!")
        
        # Launch the presentation
        print("Launching presentation...")
        subprocess.run(["manim-slides", "Main"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return 1
    except FileNotFoundError:
        print("Error: manim-slides not found. Please install it with: pip install manim-slides")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
