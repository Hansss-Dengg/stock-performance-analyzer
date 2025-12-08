"""
Launch script for Stock Performance Analyzer Streamlit app.

Run this file to start the web application:
    python run_app.py
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    # Get the path to the app module
    app_path = Path(__file__).parent / "src" / "spa" / "app.py"
    
    if not app_path.exists():
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
    
    # Launch Streamlit
    print("Starting Stock Performance Analyzer...")
    print(f"App will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")
    except FileNotFoundError:
        print("\nError: Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
