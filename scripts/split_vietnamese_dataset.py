#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để phân tích và chia dữ liệu clickbait tiếng Việt
Chia theo tỉ lệ 7:1.5:1.5 (70%:15%:15%) với stratification
"""

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse
from pathlib import Path

def load_and_analyze_data(file_path):
    """
    Load dữ liệu và phân tích cấu trúc
    """
    print(f"🔍 Đang phân tích dữ liệu từ: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
                if line_num <= 5:  # In ra 5 dòng đầu để kiểm tra format
                    print(f"Dòng {line_num}: {item}")
            except json.JSONDecodeError as e:
                print(f"❌ Lỗi JSON ở dòng {line_num}: {e}")
                continue
    
    print(f"\n📊 Tổng số mẫu: {len(data)}")
    
    # Kiểm tra các trường dữ liệu
    if data:
        keys = data[0].keys()
        print(f"🔑 Các trường dữ liệu: {list(keys)}")
        
        # Xác định trường label (có thể là 'label', 'clickbait', 'target', v.v.)
        possible_label_fields = ['label', 'clickbait', 'target', 'class', 'category']
        label_field = None
        
        for field in possible_label_fields:
            if field in keys:
                label_field = field
                break
        
        if not label_field:
            print("⚠️  Không tìm thấy trường label. Các trường có sẵn:", list(keys))
            return None, None
        
        print(f"🏷️  Trường label được sử dụng: '{label_field}'")
        
        # Phân tích phân phối label
        labels = [item[label_field] for item in data]
        label_counts = Counter(labels)
        print(f"\n📈 Phân phối label:")
        for label, count in label_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        return data, label_field
    
    return None, None

def split_data_stratified(data, label_field, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Chia dữ liệu với stratification
    """
    print(f"\n🔀 Chia dữ liệu theo tỉ lệ {train_ratio*100}%:{val_ratio*100}%:{test_ratio*100}%")
    
    # Tạo DataFrame để dễ xử lý
    df = pd.DataFrame(data)
    
    # Chia train vs temp (val+test)
    train_data, temp_data = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        stratify=df[label_field], 
        random_state=42
    )
    
    # Chia temp thành val và test
    # Tỉ lệ val:test trong temp = val_ratio:(val_ratio+test_ratio) = 0.15:0.15 = 1:1
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=relative_test_size, 
        stratify=temp_data[label_field], 
        random_state=42
    )
    
    print(f"✅ Kết quả chia:")
    print(f"  Train: {len(train_data)} mẫu ({len(train_data)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_data)} mẫu ({len(val_data)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_data)} mẫu ({len(test_data)/len(df)*100:.1f}%)")
    
    # Kiểm tra phân phối label sau khi chia
    print(f"\n📊 Phân phối label sau khi chia:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        label_counts = Counter(split_data[label_field])
        print(f"  {split_name}:")
        for label, count in label_counts.items():
            percentage = (count / len(split_data)) * 100
            print(f"    {label}: {count} ({percentage:.1f}%)")
    
    return train_data, val_data, test_data

def save_split_data(train_data, val_data, test_data, output_dir="data"):
    """
    Lưu dữ liệu đã chia vào các folder
    """
    print(f"\n💾 Lưu dữ liệu vào {output_dir}/")
    
    # Tạo các folder
    folders = {
        "train_dtVN": train_data,
        "val_dtVN": val_data, 
        "test_dtVN": test_data
    }
    
    for folder_name, data in folders.items():
        folder_path = Path(output_dir) / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Lưu data.jsonl
        data_file = folder_path / "data.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            for item in data.to_dict('records'):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  ✅ {folder_name}/data.jsonl: {len(data)} mẫu")
        
        # Tạo truth.jsonl (chỉ chứa labels)
        truth_file = folder_path / "truth.jsonl"
        with open(truth_file, 'w', encoding='utf-8') as f:
            for item in data.to_dict('records'):
                # Tìm trường label
                label_field = None
                for field in ['label', 'clickbait', 'target', 'class', 'category']:
                    if field in item:
                        label_field = field
                        break
                
                if label_field:
                    f.write(json.dumps({"truth": item[label_field]}, ensure_ascii=False) + '\n')
        
        print(f"  ✅ {folder_name}/truth.jsonl: {len(data)} labels")

def create_data_demo(train_data, val_data, test_data, output_dir="data", demo_size=100):
    """
    Tạo file data_demo.jsonl với mẫu nhỏ để test
    """
    print(f"\n🎯 Tạo file demo với {demo_size} mẫu mỗi split")
    
    folders = {
        "train_dtVN": train_data,
        "val_dtVN": val_data,
        "test_dtVN": test_data
    }
    
    for folder_name, data in folders.items():
        folder_path = Path(output_dir) / folder_name
        
        # Lấy mẫu demo (nhỏ hơn nếu không đủ)
        demo_data = data.sample(n=min(demo_size, len(data)), random_state=42)
        
        # Lưu data_demo.jsonl
        demo_file = folder_path / "data_demo.jsonl"
        with open(demo_file, 'w', encoding='utf-8') as f:
            for item in demo_data.to_dict('records'):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"  ✅ {folder_name}/data_demo.jsonl: {len(demo_data)} mẫu")

def main():
    parser = argparse.ArgumentParser(description="Chia dữ liệu clickbait tiếng Việt")
    parser.add_argument("--input", "-i", default="data/clickbait_dataset_vietnamese.jsonl",
                       help="Đường dẫn file dữ liệu đầu vào")
    parser.add_argument("--output", "-o", default="data",
                       help="Thư mục đầu ra")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Tỉ lệ tập train (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Tỉ lệ tập validation (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Tỉ lệ tập test (default: 0.15)")
    parser.add_argument("--demo-size", type=int, default=100,
                       help="Số mẫu trong file demo (default: 100)")
    
    args = parser.parse_args()
    
    # Kiểm tra tỉ lệ
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Tổng tỉ lệ phải bằng 1.0, hiện tại: {total_ratio}")
        return
    
    print("🚀 Bắt đầu chia dữ liệu clickbait tiếng Việt")
    print("=" * 60)
    
    # 1. Load và phân tích dữ liệu
    data, label_field = load_and_analyze_data(args.input)
    if data is None:
        print("❌ Không thể load dữ liệu")
        return
    
    # 2. Chia dữ liệu
    train_data, val_data, test_data = split_data_stratified(
        data, label_field, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    # 3. Lưu dữ liệu
    save_split_data(train_data, val_data, test_data, args.output)
    
    # 4. Tạo file demo
    create_data_demo(train_data, val_data, test_data, args.output, args.demo_size)
    
    print("\n🎉 Hoàn thành chia dữ liệu!")
    print("=" * 60)
    print("📁 Cấu trúc thư mục được tạo:")
    print("data/")
    print("├── train_dtVN/")
    print("│   ├── data.jsonl")
    print("│   ├── data_demo.jsonl")
    print("│   └── truth.jsonl")
    print("├── val_dtVN/")
    print("│   ├── data.jsonl") 
    print("│   ├── data_demo.jsonl")
    print("│   └── truth.jsonl")
    print("└── test_dtVN/")
    print("    ├── data.jsonl")
    print("    ├── data_demo.jsonl")
    print("    └── truth.jsonl")

if __name__ == "__main__":
    main() 