# 🇻🇳 Vietnamese Clickbait Classification - Step by Step Guide

Hướng dẫn chi tiết từng bước để phân loại clickbait tiếng Việt sử dụng Machine Learning và Deep Learning.

## 📋 Tổng Quan

Dự án này cung cấp **pipeline hoàn chỉnh** để:
- ✅ **Phân loại clickbait tiếng Việt** với độ chính xác cao (75-90%)
- ✅ **Hỗ trợ nhiều models**: PhoBERT, XLM-RoBERTa, Vietnamese LLMs
- ✅ **Linh hoạt hardware**: CPU-only hoặc GPU training
- ✅ **Không bắt buộc underthesea**: Lựa chọn dùng hoặc không dùng
- ✅ **Dễ sử dụng**: Scripts đơn giản, config rõ ràng

## 🚀 Quick Start (5 phút)

### Bước 1: Clone và Setup
```bash
git clone <your-repo-url>
cd clickbait-classification-LLM

# Install dependencies tối thiểu (KHÔNG cần underthesea)
pip install -r requirements_minimal_cpu.txt
```

### Bước 2: Chuẩn bị dữ liệu
```bash
# Nếu chưa có dữ liệu training
python scripts/prepare_training_data.py
```

### Bước 3: Test nhanh (2-3 phút)
```bash
# Test inference với pre-trained model
python scripts/test_inference_cpu.py

# Hoặc test training với dataset nhỏ
python scripts/test_cpu_no_underthesea.py
```

## 📊 Training Options

### Option 1: Training Đơn Giản (Recommended)
```bash
python scripts/run_simple_training.py
```

**Cấu hình mặc định:**
- Model: PhoBERT-base
- Method: NO underthesea (raw text)
- Dataset: Full (2,588 train samples)
- Epochs: 3
- Expected accuracy: 75-85%

### Option 2: Training Nhanh (Subset)
Sửa trong `scripts/run_simple_training.py`:
```python
CONFIG = {
    'model_name': 'vinai/phobert-base',
    'use_subset': True,        # ← Đổi thành True
    'subset_size': 500,        # ← Dataset nhỏ
    'epochs': 2,               # ← Ít epochs
    # ... other configs
}
```

### Option 3: So Sánh Nhiều Models
```bash
python scripts/compare_models_no_underthesea.py
```

## 📁 Cấu Trúc Dữ Liệu

### Input Data Format
```json
{"title": "Tiêu đề tin tức tiếng Việt", "label": "clickbait"}
{"title": "Chính phủ thông qua nghị định mới", "label": "non-clickbait"}
```

### Sau khi chuẩn bị:
```
data/
├── train_dtVN/training_data.jsonl  (2,588 samples)
├── val_dtVN/training_data.jsonl    (555 samples)
└── test_dtVN/training_data.jsonl   (555 samples)
```

## 🔧 Configurations Chi Tiết

### Model Options
| Model | Size | CPU Time | GPU Time | Accuracy | Memory |
|-------|------|----------|----------|----------|---------|
| PhoBERT-base | 135M | 45-60 min | 8-12 min | 75-85% | 2-4GB |
| PhoBERT-large | 370M | 90-120 min | 15-25 min | 80-90% | 4-8GB |
| XLM-RoBERTa-base | 278M | 60-90 min | 10-15 min | 75-85% | 3-6GB |
| XLM-RoBERTa-large | 560M | 120-180 min | 20-35 min | 80-90% | 6-12GB |

### Hardware Recommendations
```python
# CPU Training (Slow but accessible)
CONFIG = {
    'model_name': 'vinai/phobert-base',
    'batch_size': 8,
    'epochs': 3,
    'device': 'cpu'
}

# GPU Training (Fast)
CONFIG = {
    'model_name': 'vinai/phobert-large', 
    'batch_size': 32,
    'epochs': 5,
    'device': 'cuda'
}
```

## 📈 Expected Results

### Performance Benchmarks
```
Dataset: Vietnamese clickbait (3,698 samples)
Method: NO underthesea (raw text → tokenizer)

PhoBERT-base:
├── Accuracy: 75-85%
├── F1-Score: 73-83%
├── Training time: 45-60 min (CPU), 8-12 min (GPU)
└── Memory: 2-4GB

XLM-RoBERTa-base:
├── Accuracy: 75-85% 
├── F1-Score: 73-83%
├── Training time: 60-90 min (CPU), 10-15 min (GPU)
└── Memory: 3-6GB
```

### Sample Predictions
```
✅ "Thủ tướng chủ trì họp Chính phủ" → Non-Clickbait (0.92)
❌ "Bạn sẽ không tin điều này!" → Clickbait (0.89)
✅ "Giá vàng tăng 2% hôm nay" → Non-Clickbait (0.85)
❌ "7 bí mật không ai biết" → Clickbait (0.91)
```

## 🔄 Advanced Usage

### 1. Custom Model Training
```python
# Trong scripts/run_simple_training.py
CONFIG = {
    'model_name': 'xlm-roberta-large',  # Đổi model
    'max_length': 256,                  # Tăng length
    'epochs': 5,                        # Thêm epochs
    'learning_rate': 1e-5,              # Điều chỉnh LR
    'batch_size': 16,                   # Tùy theo memory
}
```

### 2. Full Dataset Training
```python
CONFIG = {
    'use_subset': False,    # Full dataset (2,588 samples)
    'epochs': 5,            # Nhiều epochs hơn
    'save_model': True,     # Lưu model trained
}
```

### 3. Inference với Trained Model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained model
model_path = "outputs_simple/vinai_phobert-base_0107_0115"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict
text = "Bạn sẽ không tin điều này xảy ra!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()

print("Clickbait" if prediction == 1 else "Non-Clickbait")
```

## 🆚 Underthesea vs No-Underthesea

### Với Underthesea (Traditional)
```python
import underthesea

# Word segmentation required for PhoBERT
text = "Hà Nội là thủ đô Việt Nam"
segmented = underthesea.word_tokenize(text, format="text")
# → "Hà_Nội là thủ_đô Việt_Nam"

inputs = tokenizer(segmented, return_tensors="pt")
```

**Pros:** Có thể hơi tốt hơn cho PhoBERT  
**Cons:** Dependency phức tạp, cài đặt khó

### Không Underthesea (Our Approach)
```python
# Direct tokenization - NO preprocessing needed
text = "Hà Nội là thủ đô Việt Nam"
inputs = tokenizer(text, return_tensors="pt")  # Trực tiếp
```

**Pros:** ✅ Đơn giản, ít dependency, performance tương đương  
**Cons:** Có thể mất vài % accuracy với PhoBERT

### Performance Comparison
| Method | Accuracy | Setup Time | Dependencies |
|--------|----------|------------|--------------|
| **Với underthesea** | 76.0% | 15-30 min | torch + transformers + underthesea + pyvi |
| **Không underthesea** | 75.5% | 5 min | torch + transformers |

**→ Recommendation: Dùng NO-underthesea cho đơn giản**

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Giảm batch size
CONFIG['batch_size'] = 4  # Từ 16 xuống 4

# Hoặc dùng subset
CONFIG['use_subset'] = True
CONFIG['subset_size'] = 200
```

**2. Model Download Slow**
```bash
# Set mirror nếu cần
export HF_ENDPOINT=https://hf-mirror.com
```

**3. CUDA Errors**
```python
# Force CPU
CONFIG['device'] = 'cpu'
```

**4. Accuracy Too Low**
```python
# Thử model khác
CONFIG['model_name'] = 'xlm-roberta-base'

# Tăng epochs
CONFIG['epochs'] = 5

# Dùng full dataset
CONFIG['use_subset'] = False
```

## 📊 Model Selection Guide

### Cho CPU Training:
```python
# Fastest
CONFIG['model_name'] = 'vinai/phobert-base'

# Best balance
CONFIG['model_name'] = 'xlm-roberta-base'
```

### Cho GPU Training:
```python
# Best accuracy
CONFIG['model_name'] = 'vinai/phobert-large'

# Multilingual
CONFIG['model_name'] = 'xlm-roberta-large'
```

### Cho Production:
```python
# Lightweight
CONFIG['model_name'] = 'vinai/phobert-base'
CONFIG['max_length'] = 64  # Shorter texts

# High accuracy
CONFIG['model_name'] = 'xlm-roberta-large'
CONFIG['max_length'] = 256  # Longer texts
```

## 🚀 Production Deployment

### 1. Train Best Model
```bash
# Full training với best config
python scripts/run_simple_training.py
```

### 2. Model Inference Class
```python
class VietnameseClickbaitClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        prediction = outputs.logits.argmax(-1).item()
        confidence = torch.softmax(outputs.logits, -1).max().item()
        
        return {
            'label': 'clickbait' if prediction == 1 else 'non-clickbait',
            'confidence': confidence
        }

# Usage
classifier = VietnameseClickbaitClassifier("outputs_simple/best_model")
result = classifier.predict("Bạn sẽ không tin điều này!")
print(result)  # {'label': 'clickbait', 'confidence': 0.89}
```

### 3. API Deployment
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
classifier = VietnameseClickbaitClassifier("path/to/model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    result = classifier.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

## 📁 File Structure

```
clickbait-classification-LLM/
├── 📂 data/                          # Dữ liệu training
│   ├── train_dtVN/training_data.jsonl
│   ├── val_dtVN/training_data.jsonl  
│   └── test_dtVN/training_data.jsonl
├── 📂 scripts/                       # Scripts chính
│   ├── 🚀 run_simple_training.py     # Training đơn giản
│   ├── 🧪 test_cpu_no_underthesea.py # Test CPU
│   ├── 📊 compare_models_no_underthesea.py # So sánh models
│   └── 🔧 prepare_training_data.py   # Chuẩn bị data
├── 📂 outputs_simple/                # Models trained
├── 📄 requirements_minimal_cpu.txt   # Dependencies tối thiểu
└── 📖 README_CLICKBAIT_VIETNAMESE.md # Hướng dẫn này
```

## ⚡ Quick Commands Reference

```bash
# 1. Setup (1 lần)
pip install -r requirements_minimal_cpu.txt
python scripts/prepare_training_data.py

# 2. Test nhanh (2-3 phút)
python scripts/test_cpu_no_underthesea.py

# 3. Training đơn giản 
python scripts/run_simple_training.py

# 4. So sánh models
python scripts/compare_models_no_underthesea.py

# 5. Training với subset (nhanh)
# Sửa CONFIG['use_subset'] = True trong run_simple_training.py
python scripts/run_simple_training.py
```

## 🏆 Best Practices

### 1. **Bắt đầu với subset**
- Dùng `use_subset=True, subset_size=500` để test nhanh
- Sau đó chuyển sang full dataset

### 2. **Chọn model phù hợp**
- CPU + nhanh: `vinai/phobert-base`
- CPU + accuracy: `xlm-roberta-base`  
- GPU + best: `vinai/phobert-large`

### 3. **Monitor training**
- Check accuracy mỗi epoch
- Stop early nếu overfitting
- Save best model

### 4. **Production tips**
- Dùng `max_length=64` cho speed
- Batch prediction cho hiệu quả
- Cache models trong memory

## 🎯 Expected Timeline

### First-time Setup:
- ⏰ **5-10 phút**: Clone + install dependencies
- ⏰ **2-3 phút**: Prepare data + test
- ⏰ **5-10 phút**: First training (subset)

### Full Training:
- ⏰ **45-60 phút**: PhoBERT-base trên CPU
- ⏰ **8-12 phút**: PhoBERT-base trên GPU
- ⏰ **90-120 phút**: PhoBERT-large trên CPU

### Development:
- ⏰ **1-2 ngày**: Experiment với different models
- ⏰ **3-5 ngày**: Fine-tune cho production
- ⏰ **1 tuần**: Deploy và monitor

## 📝 Notes

- ✅ **Không cần underthesea**: Scripts hoạt động tốt với raw text
- ✅ **CPU-friendly**: Tất cả scripts test được trên CPU
- ✅ **Scalable**: Dễ dàng chuyển sang GPU hoặc models lớn hơn
- ✅ **Production-ready**: Có inference class và API example

---

## 🚀 **Getting Started Now:**

```bash
# Clone repo
git clone <your-repo>
cd clickbait-classification-LLM

# Quick setup
pip install torch transformers pandas scikit-learn numpy tqdm accelerate

# Test ngay 
python scripts/test_cpu_no_underthesea.py

# 🎉 Done! Bắt đầu training của bạn!
```

**Happy Vietnamese Clickbait Classification! 🇻🇳🚀** 