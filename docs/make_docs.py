"""
PyGreedy Documentation Builder
===========================

Script to build PyGreedy documentation.

Created by: devhliu
Created at: 2025-02-18 05:27:54 UTC
"""

import os
import shutil
import subprocess
from pathlib import Path

def clean_build():
    """Clean build directory."""
    build_dir = Path('_build')
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("Cleaned build directory")

def build_html():
    """Build HTML documentation."""
    try:
        subprocess.run(['make', 'html'], check=True)
        print("Built HTML documentation")
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        return False
    return True

def build_pdf():
    """Build PDF documentation."""
    try:
        subprocess.run(['make', 'latexpdf'], check=True)
        print("Built PDF documentation")
    except subprocess.CalledProcessError as e:
        print(f"Error building PDF: {e}")
        return False
    return True

def main():
    """Main documentation build function."""
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Clean previous build
    clean_build()
    
    # Build documentation
    success = build_html() and build_pdf()
    
    if success:
        print("\nDocumentation built successfully!")
        print("HTML: _build/html/index.html")
        print("PDF:  _build/latex/pygreedy.pdf")
    else:
        print("\nDocumentation build failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())