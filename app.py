"""
Securities Market Fraud Detection System - Main Application
SEBI Safe Space Initiative

This is the main application file that provides a command-line interface
for the fraud detection system.
"""

import argparse
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from data_generator import main as generate_data
from train_models import main as train_models

def setup_environment():
    """Set up the environment for the fraud detection system"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    Config.create_directories()
    
    print("✅ Environment setup complete!")

def run_data_generation():
    """Run data generation process"""
    print("📊 Starting data generation...")
    generate_data()
    print("✅ Data generation complete!")

def run_model_training():
    """Run model training process"""
    print("🤖 Starting model training...")
    train_models()
    print("✅ Model training complete!")

def run_fraud_detection_demo():
    """Run a demonstration of the fraud detection system"""
    print("🛡️ Running fraud detection demonstration...")
    
    # This would typically load the trained models and run detection
    # For now, we'll show a simple demo message
    
    demo_texts = [
        "GUARANTEED 500% returns in 30 days! Join our exclusive group NOW!",
        "Consider investing in diversified mutual funds for long-term growth.",
        "BREAKING: Secret government scheme! Make ₹10 lakhs in one week!",
        "Market analysis suggests careful portfolio diversification."
    ]
    
    print("\n🔍 Analyzing sample texts for fraud indicators:")
    print("-" * 60)
    
    for i, text in enumerate(demo_texts, 1):
        # Simple fraud scoring based on keywords
        fraud_keywords = ['guaranteed', 'secret', 'exclusive', 'breaking']
        score = sum(1 for keyword in fraud_keywords if keyword.lower() in text.lower()) / len(fraud_keywords)
        
        risk_level = "High" if score > 0.5 else "Medium" if score > 0.2 else "Low"
        
        print(f"\n{i}. Text: {text[:50]}...")
        print(f"   Risk Score: {score:.2f}")
        print(f"   Risk Level: {risk_level}")

def show_system_status():
    """Show the current status of the fraud detection system"""
    print("📊 System Status Report")
    print("=" * 50)
    
    # Check if data exists
    data_files = [
        'social_media_posts.csv',
        'corporate_announcements.csv',
        'market_data.csv'
    ]
    
    print("\n📁 Data Files:")
    for file in data_files:
        file_path = Config.get_data_path(file)
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"   {exists} {file}")
    
    # Check if models exist
    model_files = [
        'random_forest_text.pkl',
        'logistic_regression_text.pkl',
        'anomaly_detector.pkl'
    ]
    
    print("\n🤖 Model Files:")
    for file in model_files:
        file_path = Config.get_model_path(file.replace('.pkl', ''))
        exists = "✅" if os.path.exists(file_path) else "❌"
        print(f"   {exists} {file}")
    
    print(f"\n⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏗️  Project: {Config.PROJECT_NAME}")
    print(f"📦 Version: {Config.VERSION}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Securities Market Fraud Detection System - SEBI Safe Space Initiative",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py setup                    # Set up environment
  python app.py generate-data           # Generate sample data
  python app.py train                   # Train ML models
  python app.py demo                    # Run fraud detection demo
  python app.py status                  # Show system status
  python app.py full-pipeline          # Run complete pipeline
        """
    )
    
    parser.add_argument(
        'command',
        choices=['setup', 'generate-data', 'train', 'demo', 'status', 'full-pipeline'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🛡️ " + "=" * 60)
    print(f"   {Config.PROJECT_NAME}")
    print(f"   {Config.DESCRIPTION}")
    print(f"   Version: {Config.VERSION}")
    print("🛡️ " + "=" * 60)
    
    try:
        if args.command == 'setup':
            setup_environment()
        
        elif args.command == 'generate-data':
            setup_environment()
            run_data_generation()
        
        elif args.command == 'train':
            setup_environment()
            if not os.path.exists(Config.get_data_path('social_media_posts.csv')):
                print("⚠️  No training data found. Generating sample data first...")
                run_data_generation()
            run_model_training()
        
        elif args.command == 'demo':
            run_fraud_detection_demo()
        
        elif args.command == 'status':
            show_system_status()
        
        elif args.command == 'full-pipeline':
            print("🚀 Running full fraud detection pipeline...")
            setup_environment()
            run_data_generation()
            run_model_training()
            run_fraud_detection_demo()
            show_system_status()
            print("\n🎉 Full pipeline completed successfully!")
        
        print(f"\n✅ Command '{args.command}' completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
