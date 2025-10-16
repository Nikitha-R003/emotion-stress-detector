#!/usr/bin/env python3
"""
Run emotion model training
This script will train the custom emotion detection model
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'nltk'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("All packages installed!")
    else:
        print("All required packages are available!")

def main():
    """Main function to run training"""
    print("Starting Custom Emotion Model Training")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Run training
    try:
        from train_emotion_model import main as train_main
        model, vocab, label_encoder, accuracy = train_main()
        
        print(f"\nTraining completed successfully!")
        print(f"Final accuracy: {accuracy:.4f}")
        
        if accuracy >= 0.90:
            print("Target accuracy of 90%+ achieved!")
        else:
            print(f"Accuracy below 90% target. Consider:")
            print("   - Increasing training data")
            print("   - Adjusting model architecture")
            print("   - Tuning hyperparameters")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
