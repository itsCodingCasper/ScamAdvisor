"""
SEBI Fraud Detection System - Web Application Launcher
Run this script to start the web application

Usage:
    python run_web_app.py

The application will be available at: http://localhost:8000
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import jinja2
        import pandas
        import numpy
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install -r web_requirements.txt")
        return False

def check_environment():
    """Check environment setup"""
    env_file = Path(".env")
    tech_env_file = Path("tech/.env")
    
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        if tech_env_file.exists():
            print("✅ Found API key in tech/.env")
            # Copy the API key from tech/.env to current environment
            with open(tech_env_file, 'r') as f:
                for line in f:
                    if 'GEMINI_API_KEY' in line:
                        key_value = line.strip().split('=')[1]
                        os.environ['GEMINI_API_KEY'] = key_value
                        print("✅ API key loaded from tech/.env")
                        break
        else:
            print("⚠️  No Gemini API key found. AI features will be limited.")
            print("To enable full AI capabilities:")
            print("1. Get API key from: https://aistudio.google.com/app/apikey")
            print("2. Set environment variable: GEMINI_API_KEY=your_api_key")
    else:
        print("✅ Gemini API key found")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/regulatory_db",
        "templates",
        "static"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main launcher function"""
    print("🛡️  SEBI Fraud Detection System - Web Application Launcher")
    print("=" * 70)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Create directories
    create_directories()
    
    # Start the web application
    print("\n🚀 Starting SEBI Fraud Detection Web Application...")
    print("📱 Application Features:")
    print("   - Content Scanner: AI-powered fraud detection")
    print("   - Advisor Verification: SEBI credential validation")
    print("   - Social Media Monitor: Platform fraud monitoring")
    print("   - Dashboard: Real-time analytics and reporting")
    print("   - Google-inspired UI: Modern, accessible design")
    
    print("\n🌐 Access the application at:")
    print("   Main Interface: http://localhost:8000")
    print("   API Documentation: http://localhost:8000/api/docs")
    print("   Dashboard: http://localhost:8000/dashboard")
    
    print("\n📊 Press Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        # Run with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "web_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting server: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install -r web_requirements.txt")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
