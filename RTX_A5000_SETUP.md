# 🚀 Vietnamese Clickbait Classification - RTX A5000 Setup

## 📋 Tổng quan

Setup hoàn chỉnh cho huấn luyện Vietnamese Clickbait Classification trên **2x RTX A5000** (48GB VRAM tổng cộng).

### ✅ Đã hoàn thành:
- ✅ **Dữ liệu**: 3,698 samples tiếng Việt đã được chia stratified (70:15:15)
- ✅ **Preprocessing**: Chỉ giữ lại các cột `id`, `title`, `label` 
- ✅ **Configs**: Tối ưu cho 2x RTX A5000 với multi-GPU training
- ✅ **Models**: PhoBERT, XLM-RoBERTa, Vietnamese LLMs (Vistral, VinaLlama, SeaLLM)
- ✅ **Pipeline**: Script tự động từ setup đến evaluation

## 🖥️ Hardware Requirements

```
✅ GPU Memory: 48GB - Đủ mạnh cho mọi model
✅ Compute: 8.6 - Hỗ trợ mixed precision
✅ Multi-GPU: NCCL backend với 2 cards
```

### 📊 Capacity Estimates:
- **PhoBERT-base**: 48 batch/GPU, ~8GB VRAM
- **PhoBERT-large**: 24 batch/GPU, ~12GB VRAM  
- **XLM-RoBERTa-base**: 32 batch/GPU, ~10GB VRAM
- **XLM-RoBERTa-large**: 16 batch/GPU, ~16GB VRAM
- **Vietnamese 7B LLMs**: 4 batch/GPU, ~20GB VRAM

## 🎯 Training Strategies

| Strategy | Model | Time | Expected Accuracy | Use Case |
|----------|-------|------|-------------------|----------|
| **QUICK_TEST** | PhoBERT-base | 30 phút | 85-88% | Test setup nhanh |
| **BEST_VIETNAMESE** | PhoBERT-large | 2-3 giờ | 88-91% | Tốt nhất cho tiếng Việt |
| **MULTILINGUAL** | XLM-RoBERTa-large | 3-4 giờ | 85-89% | Multilingual support |
| **LLM_LORA** | Vistral-7B | 4-6 giờ | 89-92% | State-of-the-art |

## 🚀 Quick Start

### 1. Kiểm tra dữ liệu đã chuẩn bị
```bash
# Dữ liệu đã được chia và chuẩn bị
ls data/train_dtVN/training_data.jsonl
ls data/val_dtVN/training_data.jsonl  
ls data/test_dtVN/training_data.jsonl
```

### 2. Test nhanh với PhoBERT-base (30 phút)
```bash
python scripts/run_rtx_a5000_training.py \\
    --models phobert-base \\
    --strategy fast
```

### 3. Training tốt nhất cho tiếng Việt (2-3 giờ)
```bash
python scripts/run_rtx_a5000_training.py \\
    --models phobert-large \\
    --strategy balanced
```

### 4. Multi-model comparison 
```bash
python scripts/run_rtx_a5000_training.py \\
    --models phobert-base phobert-large xlm-roberta-large \\
    --strategy balanced
```

### 5. State-of-the-art với LLM (4-6 giờ)
```bash
python scripts/run_rtx_a5000_training.py \\
    --models vistral-7b \\
    --strategy thorough
```

## 🔧 Manual Training

### Chạy từng bước:

#### 1. Chuẩn bị dữ liệu (đã xong)
```bash
python scripts/prepare_training_data.py
```

#### 2. Training riêng lẻ
```bash
# PhoBERT-base
python scripts/train_vietnamese_rtx_a5000.py \\
    --model-type phobert-base \\
    --strategy balanced \\
    --test-samples

# PhoBERT-large  
python scripts/train_vietnamese_rtx_a5000.py \\
    --model-type phobert-large \\
    --strategy balanced \\
    --test-samples

# XLM-RoBERTa-large
python scripts/train_vietnamese_rtx_a5000.py \\
    --model-type xlm-roberta-large \\
    --strategy balanced \\
    --test-samples

# Vistral-7B với LoRA
python scripts/train_vietnamese_rtx_a5000.py \\
    --model-type vistral-7b \\
    --strategy balanced \\
    --test-samples
```

## 📊 Data Overview

### Dữ liệu đã được chuẩn bị:
```
📊 Phân phối label:
  Train: clickbait 922 (35.6%), non-clickbait 1666 (64.4%)
  Val:   clickbait 197 (35.5%), non-clickbait 358 (64.5%)  
  Test:  clickbait 198 (35.7%), non-clickbait 357 (64.3%)

📝 Thống kê độ dài title:
  Trung bình: 66.6 ký tự
  95%: 103 ký tự
  Max: 194 ký tự
  Đề xuất max_length: 154 tokens
```

### Cấu trúc dữ liệu:
```json
{
  "id": "article_xxxx",
  "title": "Tiêu đề bài báo tiếng Việt",
  "label": "clickbait" hoặc "non-clickbait"
}
```

## ⚡ Performance Optimizations

### Đã được cấu hình:
- ✅ **Mixed Precision (FP16)**: Tăng tốc 1.5-2x
- ✅ **Gradient Checkpointing**: Cho large models
- ✅ **Multi-GPU Training**: NCCL backend
- ✅ **Optimized Batch Sizes**: Phù hợp với từng model
- ✅ **Learning Rate Scheduling**: Linear/Cosine
- ✅ **Early Stopping**: Tránh overfitting

## 📁 Output Structure

```
models/vietnamese_clickbait/
├── phobert-base_20250623_143022_balanced/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── training_config.json
│   ├── final_results.json
│   └── logs/
├── phobert-large_20250623_150145_balanced/
└── training_summary_20250623.json
```

## 🧪 Testing Models

### Test với sample predictions:
```bash
python scripts/train_vietnamese_rtx_a5000.py \\
    --model-type phobert-base \\
    --strategy fast \\
    --test-samples
```

### Sample test cases:
```
1. "Bí mật kinh hoàng mà bác sĩ không bao giờ tiết lộ!" -> clickbait
2. "Nghiên cứu mới về tác động của caffeine đến sức khỏe" -> non-clickbait  
3. "Cách làm giàu nhanh trong 30 ngày mà ai cũng có thể làm" -> clickbait
4. "Báo cáo kinh tế quý II năm 2025 của Ngân hàng Thế giới" -> non-clickbait
5. "Sốc: Phát hiện bí mật động trời về sao Việt nổi tiếng!" -> clickbait
```

## 🔍 Monitoring & Logs

### Training logs:
- `training.log`: Chi tiết quá trình training
- `rtx_a5000_training.log`: Pipeline logs
- `models/.../logs/`: TensorBoard logs

### TensorBoard:
```bash
tensorboard --logdir models/vietnamese_clickbait/phobert-base_*/logs/
```

## 🚨 Troubleshooting

### Common Issues:

#### 1. CUDA Out of Memory
```bash
# Giảm batch size trong configs/rtx_a5000_configs.py
# Hoặc sử dụng strategy="fast"
```

#### 2. Multi-GPU không hoạt động
```bash
# Kiểm tra:
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Nếu cần, chạy single GPU:
CUDA_VISIBLE_DEVICES=0 python scripts/train_vietnamese_rtx_a5000.py ...
```

#### 3. Dependencies missing
```bash
pip install -r requirements_vietnamese.txt
```

## 🎯 Expected Results

### Performance Benchmarks:

| Model | Accuracy | F1-Score | Training Time | VRAM Usage |
|-------|----------|----------|---------------|------------|
| PhoBERT-base | 85-88% | 0.83-0.86 | 30-45 min | ~8GB/GPU |
| PhoBERT-large | 88-91% | 0.86-0.90 | 2-3 hours | ~12GB/GPU |
| XLM-RoBERTa-large | 85-89% | 0.83-0.87 | 3-4 hours | ~16GB/GPU |
| Vistral-7B (LoRA) | 89-92% | 0.87-0.91 | 4-6 hours | ~20GB/GPU |

### Success Criteria:
- ✅ F1-Score > 0.85 cho production
- ✅ Accuracy > 87% cho competitive performance  
- ✅ Training time < 6 hours
- ✅ Memory usage < 24GB/GPU

## 📞 Support

### Quick Commands:
```bash
# Show hardware info
python scripts/run_rtx_a5000_training.py --show-hardware

# Show strategies
python scripts/run_rtx_a5000_training.py --show-strategies

# Check data
python scripts/prepare_training_data.py --data-dir data

# Test config
python configs/rtx_a5000_configs.py
```

---

## 🎉 Ready to Train!

Hệ thống đã được setup hoàn chỉnh và tối ưu cho 2x RTX A5000. Bạn có thể bắt đầu training ngay:

```bash
# Test nhanh 30 phút:
python scripts/run_rtx_a5000_training.py --models phobert-base --strategy fast

# Hoặc training full với PhoBERT-large (2-3 giờ):
python scripts/run_rtx_a5000_training.py --models phobert-large --strategy balanced
``` 