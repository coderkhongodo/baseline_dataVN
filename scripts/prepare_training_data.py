#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để chuẩn bị dữ liệu training với chỉ các cột cần thiết: id, title, label
Phù hợp cho việc huấn luyện model phân loại clickbait tiếng Việt
"""

import json
import os
import argparse
from pathlib import Path
import pandas as pd
from collections import Counter

def extract_training_data(input_file, output_file):
    """
    Extract chỉ các cột cần thiết từ file dữ liệu gốc
    """
    print(f"🔍 Đang xử lý file: {input_file}")
    
    training_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # Chỉ giữ lại các cột cần thiết
                training_item = {
                    "id": item.get("id", f"item_{line_num}"),
                    "title": item.get("title", ""),
                    "label": item.get("label", "non-clickbait")
                }
                
                # Kiểm tra dữ liệu hợp lệ
                if training_item["title"].strip():
                    training_data.append(training_item)
                else:
                    print(f"⚠️  Dòng {line_num}: Title rỗng, bỏ qua")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Lỗi JSON ở dòng {line_num}: {e}")
                continue
    
    # Lưu dữ liệu đã xử lý
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Đã lưu {len(training_data)} mẫu vào {output_file}")
    
    # Thống kê
    labels = [item["label"] for item in training_data]
    label_counts = Counter(labels)
    print(f"📊 Phân phối label:")
    for label, count in label_counts.items():
        percentage = (count / len(training_data)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    return training_data

def check_data_quality(data_file):
    """
    Kiểm tra chất lượng dữ liệu
    """
    print(f"\n🔍 Kiểm tra chất lượng dữ liệu: {data_file}")
    
    issues = []
    valid_count = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # Kiểm tra các trường bắt buộc
                if not item.get("id"):
                    issues.append(f"Dòng {line_num}: Thiếu ID")
                    continue
                    
                if not item.get("title", "").strip():
                    issues.append(f"Dòng {line_num}: Title rỗng")
                    continue
                    
                if item.get("label") not in ["clickbait", "non-clickbait"]:
                    issues.append(f"Dòng {line_num}: Label không hợp lệ '{item.get('label')}'")
                    continue
                
                # Kiểm tra độ dài title
                title_len = len(item["title"])
                if title_len < 10:
                    issues.append(f"Dòng {line_num}: Title quá ngắn ({title_len} ký tự)")
                elif title_len > 200:
                    issues.append(f"Dòng {line_num}: Title quá dài ({title_len} ký tự)")
                
                valid_count += 1
                
            except json.JSONDecodeError:
                issues.append(f"Dòng {line_num}: Lỗi JSON")
    
    print(f"✅ Mẫu hợp lệ: {valid_count}")
    if issues:
        print(f"⚠️  Phát hiện {len(issues)} vấn đề:")
        for issue in issues[:10]:  # Chỉ hiển thị 10 vấn đề đầu
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... và {len(issues) - 10} vấn đề khác")
    else:
        print("🎉 Dữ liệu hoàn toàn sạch!")

def create_vocab_analysis(data_files):
    """
    Phân tích từ vựng để setup tokenizer
    """
    print(f"\n📝 Phân tích từ vựng...")
    
    all_titles = []
    total_chars = 0
    max_length = 0
    min_length = float('inf')
    
    for data_file in data_files:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    title = item.get("title", "")
                    if title:
                        all_titles.append(title)
                        title_len = len(title)
                        total_chars += title_len
                        max_length = max(max_length, title_len)
                        min_length = min(min_length, title_len)
                except:
                    continue
    
    if all_titles:
        avg_length = total_chars / len(all_titles)
        
        print(f"📊 Thống kê độ dài title:")
        print(f"  Tổng số: {len(all_titles)} titles")
        print(f"  Độ dài trung bình: {avg_length:.1f} ký tự")
        print(f"  Độ dài min: {min_length} ký tự")
        print(f"  Độ dài max: {max_length} ký tự")
        
        # Phân tích phân phối
        lengths = [len(title) for title in all_titles]
        lengths.sort()
        
        percentiles = [50, 75, 90, 95, 99]
        print(f"  Phân phối độ dài:")
        for p in percentiles:
            idx = int(len(lengths) * p / 100)
            print(f"    {p}%: {lengths[idx]} ký tự")
        
        # Đề xuất max_length cho tokenizer
        recommended_length = lengths[int(len(lengths) * 0.95)]  # 95th percentile
        token_length = int(recommended_length * 1.5)  # Ước lượng số token
        
        print(f"\n💡 Đề xuất cấu hình:")
        print(f"  max_length cho tokenizer: {min(512, token_length)} tokens")
        print(f"  Sẽ cover ~95% dữ liệu")

def process_all_splits(data_dir="data"):
    """
    Xử lý tất cả các split dữ liệu
    """
    print("🚀 Bắt đầu chuẩn bị dữ liệu training")
    print("=" * 60)
    
    splits = ["train_dtVN", "val_dtVN", "test_dtVN"]
    processed_files = []
    
    for split in splits:
        input_file = Path(data_dir) / split / "data.jsonl"
        output_file = Path(data_dir) / split / "training_data.jsonl"
        
        if input_file.exists():
            print(f"\n📁 Xử lý {split}:")
            extract_training_data(input_file, output_file)
            check_data_quality(output_file)
            processed_files.append(output_file)
        else:
            print(f"⚠️  Không tìm thấy file: {input_file}")
    
    # Phân tích từ vựng
    if processed_files:
        create_vocab_analysis(processed_files)
    
    print("\n🎉 Hoàn thành chuẩn bị dữ liệu!")
    print("=" * 60)
    print("📁 Files đã tạo:")
    for split in splits:
        training_file = Path(data_dir) / split / "training_data.jsonl"
        if training_file.exists():
            print(f"  ✅ {training_file}")
    
    print(f"\n📋 Cấu trúc dữ liệu training:")
    print(f"  {{")
    print(f"    \"id\": \"article_xxxx\",")
    print(f"    \"title\": \"Tiêu đề bài báo tiếng Việt\",")
    print(f"    \"label\": \"clickbait\" hoặc \"non-clickbait\"")
    print(f"  }}")

def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu training clickbait")
    parser.add_argument("--data-dir", "-d", default="data",
                       help="Thư mục chứa dữ liệu (default: data)")
    parser.add_argument("--single-file", "-f", 
                       help="Xử lý một file duy nhất")
    parser.add_argument("--output", "-o",
                       help="File output (chỉ khi dùng --single-file)")
    
    args = parser.parse_args()
    
    if args.single_file:
        if not args.output:
            args.output = args.single_file.replace('.jsonl', '_training.jsonl')
        
        print(f"🔍 Xử lý file đơn: {args.single_file}")
        extract_training_data(args.single_file, args.output)
        check_data_quality(args.output)
        create_vocab_analysis([args.output])
    else:
        process_all_splits(args.data_dir)

if __name__ == "__main__":
    main() 