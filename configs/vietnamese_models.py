#!/usr/bin/env python3
"""
Vietnamese Model Configurations for Clickbait Classification
Cấu hình các mô hình tiếng Việt cho phân loại clickbait
"""

# Vietnamese BERT Family Models
VIETNAMESE_BERT_CONFIGS = {
    "phobert-base": {
        "model_name": "vinai/phobert-base",
        "tokenizer_name": "vinai/phobert-base",
        "batch_size": 32,  # PhoBERT handle tốt batch size lớn
        "learning_rate": 2e-5,
        "epochs": 4,
        "max_length": 256,  # Tiếng Việt thường dài hơn
        "fp16": True,
        "gradient_accumulation_steps": 1,
        "preprocessing": "word_segmentation",  # PhoBERT cần word segmentation
        "language": "vietnamese",
        "description": "PhoBERT-base - BERT model pre-trained on Vietnamese corpus"
    },
    
    "phobert-large": {
        "model_name": "vinai/phobert-large",
        "tokenizer_name": "vinai/phobert-large", 
        "batch_size": 16,  # Larger model
        "learning_rate": 1e-5,
        "epochs": 3,
        "max_length": 256,
        "fp16": True,
        "gradient_accumulation_steps": 2,
        "preprocessing": "word_segmentation",
        "language": "vietnamese",
        "description": "PhoBERT-large - Larger Vietnamese BERT model"
    },
    
    "xlm-roberta-base": {
        "model_name": "xlm-roberta-base",
        "tokenizer_name": "xlm-roberta-base",
        "batch_size": 24,
        "learning_rate": 2e-5,
        "epochs": 4,
        "max_length": 256,
        "fp16": True,
        "gradient_accumulation_steps": 1,
        "preprocessing": "none",  # XLM-RoBERTa không cần word segmentation
        "language": "multilingual",
        "description": "XLM-RoBERTa-base - Multilingual model supporting Vietnamese"
    },
    
    "xlm-roberta-large": {
        "model_name": "xlm-roberta-large",
        "tokenizer_name": "xlm-roberta-large",
        "batch_size": 12,
        "learning_rate": 1e-5,
        "epochs": 3,
        "max_length": 256,
        "fp16": True,
        "gradient_accumulation_steps": 3,
        "preprocessing": "none",
        "language": "multilingual",
        "description": "XLM-RoBERTa-large - Large multilingual model"
    },
    
    "bert-multilingual": {
        "model_name": "bert-base-multilingual-cased",
        "tokenizer_name": "bert-base-multilingual-cased",
        "batch_size": 28,
        "learning_rate": 2e-5,
        "epochs": 4,
        "max_length": 256,
        "fp16": True,
        "gradient_accumulation_steps": 1,
        "preprocessing": "none",
        "language": "multilingual",
        "description": "BERT multilingual - Supports Vietnamese and many other languages"
    }
}

# Vietnamese Large Language Models with LoRA
VIETNAMESE_LLM_CONFIGS = {
    "vistral-7b": {
        "model_name": "Viet-Mistral/Vistral-7B-Chat",
        "tokenizer_name": "Viet-Mistral/Vistral-7B-Chat",
        "batch_size": 6,  # Vietnamese LLM cần batch nhỏ hơn
        "learning_rate": 4e-6,
        "epochs": 3,
        "max_length": 512,  # Context dài hơn cho Vietnamese
        "quantization": "4bit",
        "lora_r": 16,  # Tăng rank cho Vietnamese
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "gradient_accumulation_steps": 8,  # Effective batch = 48
        "preprocessing": "none",
        "language": "vietnamese",
        "description": "Vistral-7B - Vietnamese adaptation of Mistral 7B"
    },
    
    "vinallama-7b": {
        "model_name": "vilm/vinallama-7b-chat",
        "tokenizer_name": "vilm/vinallama-7b-chat",
        "batch_size": 6,
        "learning_rate": 4e-6,
        "epochs": 3,
        "max_length": 512,
        "quantization": "4bit",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "gradient_accumulation_steps": 8,
        "preprocessing": "none",
        "language": "vietnamese", 
        "description": "VinaLLaMA-7B - Vietnamese version of LLaMA 7B"
    },
    
    "seallm-7b": {
        "model_name": "SeaLLMs/SeaLLM-7B-v2.5",
        "tokenizer_name": "SeaLLMs/SeaLLM-7B-v2.5",
        "batch_size": 6,
        "learning_rate": 5e-6,
        "epochs": 3,
        "max_length": 512,
        "quantization": "4bit",
        "lora_r": 12,
        "lora_alpha": 24,
        "lora_dropout": 0.1,
        "gradient_accumulation_steps": 8,
        "preprocessing": "none",
        "language": "southeast_asian",
        "description": "SeaLLM-7B - Southeast Asian LLM supporting Vietnamese"
    },
    
    "gemma-7b-vietnamese": {
        "model_name": "google/gemma-7b",
        "tokenizer_name": "google/gemma-7b",
        "batch_size": 6,
        "learning_rate": 3e-6,
        "epochs": 3,
        "max_length": 512,
        "quantization": "4bit",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "gradient_accumulation_steps": 8,
        "preprocessing": "none",
        "language": "multilingual",
        "description": "Gemma-7B - Google's multilingual model with Vietnamese support"
    }
}

# Ensemble configurations 
VIETNAMESE_ENSEMBLE_CONFIGS = {
    "phobert_xlm_ensemble": {
        "models": ["phobert-base", "xlm-roberta-base"],
        "weights": [0.6, 0.4],  # PhoBERT có weight cao hơn cho Vietnamese
        "description": "Ensemble of PhoBERT and XLM-RoBERTa"
    },
    
    "vistral_seallm_ensemble": {
        "models": ["vistral-7b", "seallm-7b"],
        "weights": [0.7, 0.3],  # Vistral chuyên Vietnamese
        "description": "Ensemble of Vietnamese LLMs"
    }
}

# Training configurations cho hardware khác nhau
HARDWARE_SPECIFIC_CONFIGS = {
    "rtx_4090": {
        "memory_gb": 24,
        "recommended_models": ["phobert-large", "vistral-7b"],
        "batch_multiplier": 1.0
    },
    
    "rtx_3080": {
        "memory_gb": 10,
        "recommended_models": ["phobert-base", "xlm-roberta-base"],
        "batch_multiplier": 0.5
    },
    
    "rtx_3060": {
        "memory_gb": 8,
        "recommended_models": ["phobert-base"],
        "batch_multiplier": 0.3
    },
    
    "cpu_only": {
        "memory_gb": 32,
        "recommended_models": ["phobert-base"],
        "batch_multiplier": 0.1
    }
}

def get_vietnamese_model_config(model_name: str, hardware: str = "rtx_4090") -> dict:
    """Lấy config cho Vietnamese model dựa trên hardware"""
    
    # Tìm model trong BERT configs
    if model_name in VIETNAMESE_BERT_CONFIGS:
        config = VIETNAMESE_BERT_CONFIGS[model_name].copy()
        model_type = "bert"
    # Tìm model trong LLM configs
    elif model_name in VIETNAMESE_LLM_CONFIGS:
        config = VIETNAMESE_LLM_CONFIGS[model_name].copy()
        model_type = "llm"
    else:
        raise ValueError(f"Model {model_name} not found in Vietnamese configs")
    
    # Adjust cho hardware
    if hardware in HARDWARE_SPECIFIC_CONFIGS:
        hw_config = HARDWARE_SPECIFIC_CONFIGS[hardware]
        multiplier = hw_config["batch_multiplier"]
        
        # Adjust batch size
        if model_type == "bert":
            config["batch_size"] = max(1, int(config["batch_size"] * multiplier))
        else:  # llm
            config["batch_size"] = max(1, int(config["batch_size"] * multiplier))
            # Tăng gradient accumulation để maintain effective batch size
            if multiplier < 1.0:
                config["gradient_accumulation_steps"] = int(config["gradient_accumulation_steps"] / multiplier)
    
    return config

def list_vietnamese_models() -> dict:
    """List tất cả Vietnamese models available"""
    
    return {
        "bert_family": list(VIETNAMESE_BERT_CONFIGS.keys()),
        "llm_family": list(VIETNAMESE_LLM_CONFIGS.keys()),
        "ensembles": list(VIETNAMESE_ENSEMBLE_CONFIGS.keys())
    }

def get_recommended_models(hardware: str = "rtx_4090", task_type: str = "classification") -> list:
    """Lấy danh sách models được recommend cho hardware cụ thể"""
    
    if hardware in HARDWARE_SPECIFIC_CONFIGS:
        return HARDWARE_SPECIFIC_CONFIGS[hardware]["recommended_models"]
    else:
        # Default recommendations
        return ["phobert-base", "xlm-roberta-base"]

def print_vietnamese_models_summary():
    """In summary của tất cả Vietnamese models"""
    
    print("🇻🇳 VIETNAMESE MODELS FOR CLICKBAIT CLASSIFICATION")
    print("=" * 70)
    
    print("\n📚 BERT FAMILY MODELS:")
    for name, config in VIETNAMESE_BERT_CONFIGS.items():
        print(f"  {name:20} - {config['description']}")
        print(f"  {'':20}   Batch: {config['batch_size']}, Max Length: {config['max_length']}")
    
    print("\n🚀 LARGE LANGUAGE MODELS:")
    for name, config in VIETNAMESE_LLM_CONFIGS.items():
        print(f"  {name:20} - {config['description']}")
        print(f"  {'':20}   Batch: {config['batch_size']}, Max Length: {config['max_length']}, LoRA rank: {config['lora_r']}")
    
    print("\n🔧 HARDWARE RECOMMENDATIONS:")
    for hw, config in HARDWARE_SPECIFIC_CONFIGS.items():
        models = ", ".join(config["recommended_models"])
        print(f"  {hw:15} ({config['memory_gb']}GB) - {models}")

if __name__ == "__main__":
    print_vietnamese_models_summary() 