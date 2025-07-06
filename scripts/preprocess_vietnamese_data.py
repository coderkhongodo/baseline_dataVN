#!/usr/bin/env python3
"""
Vietnamese Data Preprocessing for Clickbait Classification
Xử lý dữ liệu tiếng Việt cho training clickbait classification
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import underthesea
from pyvi import ViTokenizer

class VietnameseDataPreprocessor:
    """Preprocessor cho dữ liệu clickbait tiếng Việt"""
    
    def __init__(self, input_data_path: str = None, output_dir: str = "data_vietnamese"):
        self.input_data_path = input_data_path
        self.output_dir = output_dir
        self.stats = {}
        
    def normalize_vietnamese_text(self, text: str) -> str:
        """Chuẩn hóa text tiếng Việt"""
        
        # 1. Unicode normalization (quan trọng cho tiếng Việt)
        text = unicodedata.normalize('NFC', text)
        
        # 2. Remove URLs, emails, phone numbers
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'(\+84|0)[0-9]{8,10}', '', text)
        
        # 3. Remove special characters nhưng giữ dấu câu Vietnamese
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:()\-"""''']', '', text)
        
        # 4. Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 5. Handle Vietnamese-specific cleaning
        # Remove redundant punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text
    
    def segment_vietnamese_text(self, text: str, method: str = "underthesea") -> str:
        """Word segmentation cho tiếng Việt"""
        try:
            if method == "underthesea":
                segmented = underthesea.word_tokenize(text, format="text")
            elif method == "pyvi":
                segmented = ViTokenizer.tokenize(text)
            else:
                segmented = text  # No segmentation
                
            return segmented
        except Exception as e:
            print(f"⚠️ Segmentation failed for: {text[:50]}... Error: {e}")
            return text
    
    def detect_clickbait_patterns(self, text: str) -> Dict[str, bool]:
        """Phát hiện patterns clickbait trong tiếng Việt"""
        
        text_lower = text.lower()
        
        patterns = {
            # Emotional keywords
            'emotional_words': any(word in text_lower for word in [
                'sốc', 'choáng', 'kinh hoàng', 'bất ngờ', 'không thể tin',
                'tuyệt vời', 'tuyệt đỉnh', 'hoàn hảo', 'xuất sắc',
                'đáng sợ', 'khủng khiếp', 'ghê gớm'
            ]),
            
            # Curiosity gap words
            'curiosity_words': any(word in text_lower for word in [
                'bí mật', 'bí quyết', 'cách', 'mẹo', 'thủ thuật',
                'không ai biết', 'chưa từng', 'lần đầu tiên'
            ]),
            
            # Clickbait phrases
            'clickbait_phrases': any(phrase in text_lower for phrase in [
                'bạn sẽ không tin', 'điều xảy ra tiếp theo',
                'ai cũng phải', 'chắc chắn bạn', 'hãy xem điều',
                'thật không thể tin', 'bạn không thể bỏ lỡ'
            ]),
            
            # Numbers and percentages
            'has_percentage': bool(re.search(r'\d+%', text)),
            'has_numbers': bool(re.search(r'\d+', text)),
            
            # Question marks and exclamations
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'multiple_punctuation': bool(re.search(r'[!?]{2,}', text)),
            
            # Vague language
            'vague_words': any(word in text_lower for word in [
                'điều này', 'điều đó', 'cái này', 'cái đó',
                'thứ này', 'thứ đó', 'việc này', 'việc đó'
            ])
        }
        
        return patterns
    
    def load_vietnamese_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Vietnamese dataset từ file"""
        
        print(f"📂 Loading Vietnamese dataset from: {file_path}")
        
        data = []
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    
            elif file_ext == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = [json.loads(line) for line in f if line.strip()]
                    
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                raw_data = df.to_dict('records')
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Convert to standard format
            for item in raw_data:
                processed_item = self.standardize_item_format(item)
                if processed_item:
                    data.append(processed_item)
            
            print(f"✅ Loaded {len(data)} samples")
            return data
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
    
    def standardize_item_format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Chuẩn hóa format của mỗi item"""
        
        # Detect possible text fields
        text_fields = ['text', 'title', 'headline', 'content', 'description']
        text = None
        
        for field in text_fields:
            if field in item and item[field]:
                text = str(item[field])
                break
        
        if not text:
            return None
        
        # Detect possible label fields
        label_fields = ['label', 'class', 'category', 'target', 'y']
        label = None
        
        for field in label_fields:
            if field in item:
                label_value = item[field]
                # Convert various label formats to 0/1
                if isinstance(label_value, str):
                    if label_value.lower() in ['clickbait', 'click-bait', '1', 'true', 'yes']:
                        label = 1
                    elif label_value.lower() in ['no-clickbait', 'not-clickbait', '0', 'false', 'no']:
                        label = 0
                elif isinstance(label_value, (int, float)):
                    label = int(label_value)
                break
        
        if label is None:
            print(f"⚠️ No valid label found for item: {item}")
            return None
        
        # Generate ID if not exists
        item_id = item.get('id', item.get('idx', len(str(hash(text)))))
        
        return {
            'id': str(item_id),
            'text': text,
            'label': label,
            'language': 'vietnamese'
        }
    
    def split_dataset(self, data: List[Dict[str, Any]], 
                     test_size: float = 0.2, 
                     val_size: float = 0.1,
                     random_state: int = 42) -> Tuple[List, List, List]:
        """Chia dataset thành train/val/test"""
        
        print(f"📊 Splitting dataset: {len(data)} total samples")
        
        # Extract texts and labels for stratified split
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        # First split: train+val vs test
        train_val_texts, test_texts, train_val_labels, test_labels, train_val_indices, test_indices = train_test_split(
            texts, labels, range(len(data)), 
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels, train_indices, val_indices = train_test_split(
            train_val_texts, train_val_labels, train_val_indices,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        # Create datasets
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        print(f"✅ Split completed:")
        print(f"   Train: {len(train_data)} samples")
        print(f"   Val:   {len(val_data)} samples")
        print(f"   Test:  {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def preprocess_dataset(self, data: List[Dict[str, Any]], 
                          segment_method: str = "underthesea") -> List[Dict[str, Any]]:
        """Preprocess toàn bộ dataset"""
        
        print(f"🔄 Preprocessing {len(data)} samples...")
        
        processed_data = []
        skipped = 0
        
        for i, item in enumerate(data):
            try:
                # Normalize text
                normalized_text = self.normalize_vietnamese_text(item['text'])
                
                # Skip if text too short
                if len(normalized_text.strip()) < 10:
                    skipped += 1
                    continue
                
                # Word segmentation
                segmented_text = self.segment_vietnamese_text(normalized_text, segment_method)
                
                # Detect patterns
                patterns = self.detect_clickbait_patterns(normalized_text)
                
                # Create processed item
                processed_item = {
                    'id': item['id'],
                    'text': segmented_text,
                    'original_text': item['text'],
                    'label': item['label'],
                    'language': 'vietnamese',
                    'patterns': patterns,
                    'text_length': len(segmented_text),
                    'word_count': len(segmented_text.split())
                }
                
                processed_data.append(processed_item)
                
                if (i + 1) % 1000 == 0:
                    print(f"   Processed {i + 1}/{len(data)} samples")
                    
            except Exception as e:
                print(f"⚠️ Error processing item {i}: {e}")
                skipped += 1
                continue
        
        print(f"✅ Preprocessing completed: {len(processed_data)} samples, {skipped} skipped")
        return processed_data
    
    def save_processed_data(self, train_data: List, val_data: List, test_data: List):
        """Lưu processed data"""
        
        print(f"💾 Saving processed data to {self.output_dir}/")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in datasets.items():
            # Create split directory
            split_dir = Path(self.output_dir) / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Save main data file
            output_file = split_dir / "data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✅ Saved {split_name}: {len(split_data)} samples to {output_file}")
            
            # Save demo version (first 100 samples)
            demo_file = split_dir / "data_demo.jsonl"
            demo_data = split_data[:100] if len(split_data) > 100 else split_data
            with open(demo_file, 'w', encoding='utf-8') as f:
                for item in demo_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✅ Saved {split_name} demo: {len(demo_data)} samples to {demo_file}")
    
    def generate_statistics(self, train_data: List, val_data: List, test_data: List) -> Dict:
        """Tạo thống kê dataset"""
        
        print("📊 Generating dataset statistics...")
        
        all_data = train_data + val_data + test_data
        
        stats = {
            'total_samples': len(all_data),
            'splits': {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            },
            'label_distribution': {
                'clickbait': sum(1 for item in all_data if item['label'] == 1),
                'no_clickbait': sum(1 for item in all_data if item['label'] == 0)
            },
            'text_statistics': {
                'avg_length': sum(item['text_length'] for item in all_data) / len(all_data),
                'avg_words': sum(item['word_count'] for item in all_data) / len(all_data),
                'min_length': min(item['text_length'] for item in all_data),
                'max_length': max(item['text_length'] for item in all_data)
            }
        }
        
        # Pattern analysis
        pattern_stats = {}
        for pattern_name in all_data[0]['patterns'].keys():
            pattern_stats[pattern_name] = {
                'total': sum(1 for item in all_data if item['patterns'][pattern_name]),
                'clickbait': sum(1 for item in all_data if item['label'] == 1 and item['patterns'][pattern_name]),
                'no_clickbait': sum(1 for item in all_data if item['label'] == 0 and item['patterns'][pattern_name])
            }
        
        stats['pattern_analysis'] = pattern_stats
        
        # Save statistics
        stats_file = Path(self.output_dir) / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Statistics saved to {stats_file}")
        
        # Print summary
        print(f"\n📈 DATASET SUMMARY:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Clickbait: {stats['label_distribution']['clickbait']} ({stats['label_distribution']['clickbait']/stats['total_samples']*100:.1f}%)")
        print(f"   No-clickbait: {stats['label_distribution']['no_clickbait']} ({stats['label_distribution']['no_clickbait']/stats['total_samples']*100:.1f}%)")
        print(f"   Avg length: {stats['text_statistics']['avg_length']:.1f} chars")
        print(f"   Avg words: {stats['text_statistics']['avg_words']:.1f} words")
        
        return stats
    
    def process_full_pipeline(self, input_file: str = None, 
                             segment_method: str = "underthesea",
                             test_size: float = 0.2,
                             val_size: float = 0.1):
        """Chạy toàn bộ pipeline preprocessing"""
        
        print("🇻🇳 VIETNAMESE DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Use provided file or class attribute
        input_file = input_file or self.input_data_path
        if not input_file:
            raise ValueError("Please provide input_data_path")
        
        # Step 1: Load data
        raw_data = self.load_vietnamese_dataset(input_file)
        if not raw_data:
            return False
        
        # Step 2: Split dataset
        train_data, val_data, test_data = self.split_dataset(raw_data, test_size, val_size)
        
        # Step 3: Preprocess each split
        train_processed = self.preprocess_dataset(train_data, segment_method)
        val_processed = self.preprocess_dataset(val_data, segment_method)
        test_processed = self.preprocess_dataset(test_data, segment_method)
        
        # Step 4: Save processed data
        self.save_processed_data(train_processed, val_processed, test_processed)
        
        # Step 5: Generate statistics
        self.generate_statistics(train_processed, val_processed, test_processed)
        
        print("\n🎉 PREPROCESSING COMPLETED!")
        print(f"📁 Data saved to: {self.output_dir}/")
        print("\nNext steps:")
        print("1. Review dataset statistics")
        print("2. Run: python scripts/train_vietnamese_bert.py")
        print("3. Run: python scripts/evaluate_vietnamese_model.py")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese Data Preprocessing")
    parser.add_argument("--input", required=True, help="Path to input Vietnamese dataset")
    parser.add_argument("--output", default="data_vietnamese", help="Output directory")
    parser.add_argument("--segment", default="underthesea", choices=["underthesea", "pyvi", "none"], help="Segmentation method")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = VietnameseDataPreprocessor(
        input_data_path=args.input,
        output_dir=args.output
    )
    
    # Run pipeline
    success = preprocessor.process_full_pipeline(
        segment_method=args.segment,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    if success:
        print("✅ Preprocessing successful!")
    else:
        print("❌ Preprocessing failed!")

if __name__ == "__main__":
    main() 