#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script để kiểm tra RTX A5000 setup
Chạy test training nhanh với PhoBERT-base (10-15 phút)
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    print("🚀 RTX A5000 SETUP TEST")
    print("=" * 60)
    print("Kiểm tra setup và chạy training test với PhoBERT-base")
    print("Estimated time: 10-15 phút")
    print("=" * 60)

def check_gpu():
    print("\n🖥️  CHECKING GPU...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return False
        
        gpu_count = torch.cuda.device_count()
        print(f"✅ Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed!")
        return False

def check_data():
    print("\n📊 CHECKING DATA...")
    
    required_files = [
        "data/train_dtVN/training_data.jsonl",
        "data/val_dtVN/training_data.jsonl", 
        "data/test_dtVN/training_data.jsonl"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return False
    
    return True

def run_quick_test():
    print("\n🎯 RUNNING QUICK TRAINING TEST...")
    print("Model: PhoBERT-base")
    print("Strategy: fast (reduced epochs)")
    print("Expected time: 10-15 phút")
    print("-" * 40)
    
    # Create a simple test config for very quick training
    cmd = [
        sys.executable, "scripts/train_vietnamese_rtx_a5000.py",
        "--model-type", "phobert-base",
        "--strategy", "fast",
        "--test-samples"
    ]
    
    print(f"🚀 Executing: {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("\n✅ QUICK TEST PASSED!")
        print("🎉 RTX A5000 setup is working correctly!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ QUICK TEST FAILED: {e}")
        return False

def main():
    print_header()
    
    # Step 1: Check GPU
    if not check_gpu():
        print("\n❌ GPU check failed. Please ensure:")
        print("  1. CUDA is installed")
        print("  2. PyTorch with CUDA support is installed")
        print("  3. RTX A5000 GPUs are detected")
        return
    
    # Step 2: Check data
    if not check_data():
        print("\n❌ Data check failed. Please run:")
        print("  python scripts/prepare_training_data.py")
        return
    
    # Step 3: Confirm test
    print(f"\n📋 READY TO RUN QUICK TEST:")
    print(f"  Model: PhoBERT-base")
    print(f"  Time: ~10-15 phút")
    print(f"  Purpose: Verify RTX A5000 setup")
    
    confirm = input(f"\n🚀 Start quick test? [Y/n]: ")
    if confirm.lower() in ['', 'y', 'yes']:
        # Step 4: Run test
        if run_quick_test():
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! RTX A5000 setup is ready for production training.")
            print("=" * 60)
            print("\n🔥 NEXT STEPS:")
            print("1. Train PhoBERT-large (best Vietnamese): 2-3 hours")
            print("   python scripts/run_rtx_a5000_training.py --models phobert-large")
            print()
            print("2. Compare multiple models:")
            print("   python scripts/run_rtx_a5000_training.py --models phobert-base phobert-large xlm-roberta-large")
            print()
            print("3. State-of-the-art với LLM:")
            print("   python scripts/run_rtx_a5000_training.py --models vistral-7b")
            print()
            print("📖 See RTX_A5000_SETUP.md for full documentation")
        else:
            print("\n❌ Quick test failed. Check logs for details.")
    else:
        print("❌ Test cancelled by user")

if __name__ == "__main__":
    main() 