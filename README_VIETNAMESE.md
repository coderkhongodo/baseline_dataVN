# 🇻🇳 Vietnamese Clickbait Classification using LLM Fine-tuning

## Tổng Quan

Dự án này triển khai hệ thống phân loại clickbait tiếng Việt sử dụng Large Language Models (LLMs) và BERT family models. Hệ thống được tối ưu hóa đặc biệt cho tiếng Việt với các kỹ thuật preprocessing, word segmentation và prompting phù hợp.

### ✨ Tính Năng Chính

- **🇻🇳 Tối ưu cho tiếng Việt**: Hỗ trợ PhoBERT, XLM-RoBERTa và các Vietnamese LLMs
- **📊 Word Segmentation**: Tích hợp Underthesea và PyVi cho tiếng Việt
- **🎯 Multiple Approaches**: Fine-tuning BERT + Prompting với LLMs
- **📈 Comprehensive Evaluation**: Metrics chi tiết và visualization
- **🔧 Hardware Optimized**: Cấu hình cho RTX 4090/3080/3060

### 🏆 Models Được Hỗ Trợ

#### BERT Family
- **PhoBERT-base/large**: BERT được train trên corpus tiếng Việt
- **XLM-RoBERTa-base/large**: Multilingual model hỗ trợ tiếng Việt
- **mBERT**: BERT multilingual chuẩn

#### Vietnamese LLMs (với LoRA)
- **Vistral-7B**: Vietnamese adaptation của Mistral 7B
- **VinaLLaMA-7B**: Vietnamese version của LLaMA 7B
- **SeaLLM-7B**: Southeast Asian LLM
- **Gemma-7B**: Google's multilingual model

## 🚀 Quick Start

### 1. Cài Đặt Môi Trường

```bash
# Clone repository
git clone <your-repo>
cd clickbait-classification-LLM

# Cài đặt dependencies cho Vietnamese
pip install -r requirements_vietnamese.txt

# Setup Vietnamese environment
python scripts/setup_vietnamese_environment.py
```

### 2. Chuẩn Bị Dữ Liệu Vietnamese

```bash
# Preprocessing dữ liệu tiếng Việt
python scripts/preprocess_vietnamese_data.py \
    --input your_vietnamese_data.jsonl \
    --output data_vietnamese \
    --segment underthesea
```

**Format dữ liệu đầu vào:**
```json
{"text": "Tiêu đề tin tức tiếng Việt", "label": 0}
{"text": "Bạn sẽ không tin được điều này!", "label": 1}
```

### 3. Training Models

#### Option A: Chạy Pipeline Hoàn Chỉnh
```bash
python scripts/run_vietnamese_pipeline.py \
    --input_data your_vietnamese_data.jsonl \
    --models phobert-base xlm-roberta-base \
    --hardware rtx_4090
```

#### Option B: Training Từng Bước

**Train BERT Models:**
```bash
python scripts/train_vietnamese_bert.py \
    --model phobert-base \
    --hardware rtx_4090
```

**Prompting Evaluation:**
```bash
python scripts/vietnamese_prompting.py
```

### 4. Evaluation

```bash
# Đánh giá tất cả models
python scripts/evaluate_vietnamese_models.py --compare_all

# Đánh giá model cụ thể
python scripts/evaluate_vietnamese_models.py \
    --model_path outputs_vietnamese/phobert-base-vietnamese
```

## 📊 Cấu Trúc Dữ Liệu

```
data_vietnamese/
├── train/
│   ├── data.jsonl          # Full training data
│   └── data_demo.jsonl     # Demo subset
├── val/
│   ├── data.jsonl          # Validation data
│   └── data_demo.jsonl
└── test/
    ├── data.jsonl          # Test data
    └── data_demo.jsonl
```

## 🤖 Models Configuration

### Hardware Requirements

| Hardware | Memory | Recommended Models | Batch Size |
|----------|---------|-------------------|------------|
| RTX 4090 | 24GB   | PhoBERT-large, Vistral-7B | Full |
| RTX 3080 | 10GB   | PhoBERT-base, XLM-RoBERTa | Medium |
| RTX 3060 | 8GB    | PhoBERT-base | Small |
| CPU Only | 32GB+  | PhoBERT-base | Very Small |

### Model-Specific Settings

```python
# PhoBERT - Cần word segmentation
{
    "preprocessing": "word_segmentation",
    "batch_size": 32,
    "learning_rate": 2e-5,
    "max_length": 256
}

# XLM-RoBERTa - Không cần word segmentation
{
    "preprocessing": "none", 
    "batch_size": 24,
    "learning_rate": 2e-5,
    "max_length": 256
}
```

## 🎯 Prompting Methods

### Zero-Shot
```python
prompt = """Phân loại tiêu đề tin tức tiếng Việt:
- 0: Không phải clickbait (thông tin rõ ràng, khách quan)
- 1: Clickbait (giật tít, gây tò mò, phóng đại)

Tiêu đề: "{title}"
Trả lời: [0/1]"""
```

### Few-Shot
```python
prompt = """Dựa trên các ví dụ:
"Thủ tướng ký nghị định mới" → 0
"Bạn sẽ sốc khi biết điều này!" → 1

Tiêu đề: "{title}"
Trả lời: [0/1]"""
```

### Chain-of-Thought
```python
prompt = """Phân tích từng bước:
1. Từ khóa cảm xúc
2. Tính cụ thể của thông tin  
3. Khoảng trống tò mò
4. Kết luận: [0/1]"""
```

## 📈 Evaluation Metrics

- **Accuracy**: Độ chính xác tổng thể
- **F1-Score**: F1 weighted và per-class
- **Precision/Recall**: Cho từng class
- **ROC AUC**: Area under curve
- **Confusion Matrix**: Ma trận nhầm lẫn

## 🔧 Customization

### Thêm Vietnamese Patterns

```python
# Trong preprocess_vietnamese_data.py
def detect_clickbait_patterns(self, text: str):
    vietnamese_patterns = {
        'emotional_words': any(word in text.lower() for word in [
            'sốc', 'choáng', 'kinh hoàng', 'bất ngờ',
            'tuyệt vời', 'hoàn hảo', 'xuất sắc'
        ]),
        'curiosity_phrases': any(phrase in text.lower() for phrase in [
            'bạn sẽ không tin', 'điều xảy ra tiếp theo',
            'bí mật', 'cách', 'thủ thuật'
        ])
    }
    return vietnamese_patterns
```

### Custom Model Config

```python
# Trong configs/vietnamese_models.py
CUSTOM_MODEL = {
    "model_name": "your-vietnamese-model",
    "batch_size": 16,
    "learning_rate": 2e-5,
    "preprocessing": "word_segmentation",
    "language": "vietnamese"
}
```

## 📋 Examples

### Inference với Trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import underthesea

# Load model
model_path = "outputs_vietnamese/phobert-base-vietnamese"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocess Vietnamese text
text = "Bạn sẽ không tin được điều này!"
segmented = underthesea.word_tokenize(text, format="text")

# Predict
inputs = tokenizer(segmented, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()

print(f"Prediction: {'Clickbait' if prediction == 1 else 'Không clickbait'}")
```

### Batch Prediction

```python
vietnamese_headlines = [
    "Chính phủ thông qua nghị định mới về thuế",
    "7 bí mật mà bác sĩ không muốn bạn biết!",
    "Giá vàng tăng 2% trong phiên hôm nay",
    "Cách làm giàu 99% người Việt chưa biết"
]

predictions = []
for headline in vietnamese_headlines:
    # Preprocess and predict
    segmented = underthesea.word_tokenize(headline, format="text")
    inputs = tokenizer(segmented, return_tensors="pt")
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    predictions.append(pred)

print("Predictions:", predictions)
```

## 📚 Vietnamese NLP Resources

### Libraries
- **Underthesea**: Vietnamese NLP toolkit
- **PyVi**: Vietnamese text processing
- **VnCoreNLP**: Vietnamese core NLP toolkit

### Models
- **PhoBERT**: Vietnamese BERT (VinAI Research)
- **XLM-RoBERTa**: Facebook's multilingual model
- **Vistral**: Vietnamese Mistral adaptation

### Datasets
- **VietNewsCorpus**: Vietnamese news corpus
- **VLSP**: Vietnamese Language and Speech Processing
- **UIT-ViSum**: Vietnamese text summarization

## 🐛 Troubleshooting

### Common Issues

**1. Word Segmentation Error**
```bash
# Fix: Reinstall underthesea
pip uninstall underthesea
pip install underthesea
```

**2. CUDA Out of Memory**
```bash
# Solution: Reduce batch size
--hardware rtx_3060  # Automatically reduces batch size
```

**3. Model Loading Error**
```bash
# Fix: Check model path
ls outputs_vietnamese/  # List available models
```

**4. Vietnamese Character Encoding**
```python
# Ensure UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as f:
    data = f.read()
```

## 📝 Vietnamese Text Preprocessing

### Word Segmentation Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Underthesea | Medium | High | PhoBERT, research |
| PyVi | Fast | Medium | Quick processing |
| None | Fastest | N/A | XLM-RoBERTa |

### Text Normalization

```python
# Vietnamese-specific normalization
def normalize_vietnamese(text):
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove tone marks if needed
    text = unidecode(text)  # Optional
    
    # Clean punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    return text
```

## 🏁 Results & Performance

### Expected Performance (Vietnamese)

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| PhoBERT-base | 0.85-0.90 | 0.84-0.89 | 2-3 hours |
| XLM-RoBERTa | 0.82-0.87 | 0.81-0.86 | 3-4 hours |
| Prompting (GPT-4) | 0.80-0.85 | 0.79-0.84 | Few minutes |

### Vietnamese-Specific Challenges

1. **Word Segmentation**: Tiếng Việt không có space giữa từ ghép
2. **Tone Marks**: 5 dấu thanh khác nhau
3. **Slang & Internet Language**: Ngôn ngữ mạng xã hội
4. **Regional Variations**: Khác biệt miền Bắc/Nam

## 🤝 Contributing

### Add New Vietnamese Model

1. Thêm config vào `configs/vietnamese_models.py`
2. Update training script
3. Test với Vietnamese data
4. Add evaluation

### Improve Vietnamese Preprocessing

1. Fork repository
2. Enhance `preprocess_vietnamese_data.py`
3. Test với multiple Vietnamese datasets
4. Submit pull request

## 📜 License & Citation

```bibtex
@article{vietnamese_clickbait_2024,
  title={Vietnamese Clickbait Classification using LLM Fine-tuning},
  author={Your Name},
  year={2024},
  journal={Vietnamese NLP Research}
}
```

## 🔗 Resources

- [PhoBERT Paper](https://arxiv.org/abs/2003.00744)
- [Underthesea Documentation](https://underthesea.readthedocs.io/)
- [Vietnamese NLP Resources](https://github.com/vietnlp/vietnlp)
- [VinAI Research](https://www.vinai.io/)

---

**🇻🇳 Chúc bạn thành công với Vietnamese Clickbait Classification!** 