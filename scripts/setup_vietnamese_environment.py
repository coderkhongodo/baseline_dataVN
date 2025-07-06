#!/usr/bin/env python3
"""
Setup Vietnamese Environment for Clickbait Classification
Cài đặt và kiểm tra môi trường tiếng Việt
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Kiểm tra Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version}")
    return True

def install_vietnamese_packages():
    """Cài đặt packages cho tiếng Việt"""
    print("\n📦 Installing Vietnamese NLP packages...")
    
    packages = [
        "underthesea",
        "pyvi", 
        "transformers",
        "torch",
        "datasets"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def test_vietnamese_nlp():
    """Test Vietnamese NLP libraries"""
    print("\n🧪 Testing Vietnamese NLP libraries...")
    
    try:
        # Test underthesea
        import underthesea
        test_text = "Đây là một tiêu đề tin tức để kiểm tra."
        segmented = underthesea.word_tokenize(test_text, format="text")
        print(f"✅ Underthesea word segmentation: {segmented}")
        
        # Test pyvi
        import pyvi
        from pyvi import ViTokenizer
        tokenized = ViTokenizer.tokenize(test_text)
        print(f"✅ PyVi tokenization: {tokenized}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vietnamese NLP test failed: {e}")
        return False

def download_vietnamese_models():
    """Download Vietnamese pre-trained models"""
    print("\n🤖 Downloading Vietnamese models...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        models_to_download = [
            "vinai/phobert-base",
            "xlm-roberta-base"
        ]
        
        for model_name in models_to_download:
            print(f"Downloading {model_name}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                print(f"✅ {model_name} downloaded successfully")
            except Exception as e:
                print(f"⚠️ {model_name} download failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

def test_vietnamese_model():
    """Test Vietnamese model inference"""
    print("\n🔬 Testing Vietnamese model...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Test PhoBERT
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = AutoModel.from_pretrained("vinai/phobert-base")
        
        test_text = "Bạn sẽ không tin được điều này!"
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model(**inputs)
        
        print("✅ Vietnamese model test successful")
        print(f"   Input: {test_text}")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vietnamese model test failed: {e}")
        return False

def create_vietnamese_directories():
    """Tạo thư mục cho Vietnamese data"""
    print("\n📁 Creating Vietnamese directories...")
    
    directories = [
        "data_vietnamese/train",
        "data_vietnamese/val", 
        "data_vietnamese/test",
        "outputs_vietnamese",
        "logs_vietnamese"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True

def check_gpu():
    """Kiểm tra GPU availability"""
    print("\n🖥️ Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:
                print("⚠️ Warning: GPU has less than 8GB memory")
                print("   Consider using smaller batch sizes")
            
            return True
        else:
            print("❌ No GPU available - will use CPU (slower)")
            return False
            
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🇻🇳 VIETNAMESE CLICKBAIT CLASSIFICATION - ENVIRONMENT SETUP")
    print("=" * 70)
    
    success = True
    
    # Check Python version
    success &= check_python_version()
    
    # Install packages
    success &= install_vietnamese_packages()
    
    # Test Vietnamese NLP
    success &= test_vietnamese_nlp()
    
    # Download models
    success &= download_vietnamese_models()
    
    # Test model
    success &= test_vietnamese_model()
    
    # Create directories
    success &= create_vietnamese_directories()
    
    # Check GPU
    check_gpu()  # Don't fail if no GPU
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 VIETNAMESE ENVIRONMENT SETUP COMPLETED!")
        print("\nNext steps:")
        print("1. Prepare your Vietnamese dataset")
        print("2. Run: python scripts/preprocess_vietnamese_data.py")
        print("3. Run: python scripts/train_vietnamese_models.py")
    else:
        print("❌ Setup failed. Please check errors above.")
    
    return success

if __name__ == "__main__":
    main() 