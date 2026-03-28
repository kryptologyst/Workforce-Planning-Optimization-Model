#!/usr/bin/env python3
"""Script to run the workforce planning demo."""

import subprocess
import sys
import os

def main():
    """Run the Streamlit demo."""
    demo_path = os.path.join(os.path.dirname(__file__), "demo", "app.py")
    
    if not os.path.exists(demo_path):
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", demo_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
