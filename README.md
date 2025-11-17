# Punjabi-English Mobile Translation Pipeline

A complete end-to-end pipeline for training lightweight Punjabi ↔ English translation models optimized for mobile deployment (Android/iOS).

## Project Overview

This project implements a neural machine translation system that:
- Trains on 2-3M Punjabi-English parallel sentences
- Produces models < 50MB for mobile deployment
- Supports bidirectional translation (PA→EN and EN→PA)
- Converts to ONNX and TensorFlow Lite formats
- Targets <500ms inference on mobile CPUs

## Documentation

- **[Translation Pipeline Plan](TRANSLATION_PIPELINE_PLAN.md)** - Complete 8-week implementation plan
- **[Phase 1 README](PHASE1_README.md)** - Data preparation pipeline guide

## Project Status

### ✅ Phase 1: Data Preparation (IMPLEMENTED)
Complete implementation for data collection and preparation:
- [x] Dataset download from multiple sources (Samanantar, OPUS, FLORES)
- [x] Data cleaning and quality filtering
- [x] Train/validation/test splitting
- [x] SentencePiece tokenizer training
- [x] Comprehensive logging and statistics

**See [PHASE1_README.md](PHASE1_README.md) for detailed usage instructions**

### 🔲 Phase 2: Model Training (TODO)
- [ ] Training infrastructure setup
- [ ] MarianMT model implementation
- [ ] Training loop with evaluation
- [ ] Hyperparameter tuning

### 🔲 Phase 3: Model Optimization (TODO)
- [ ] Model quantization
- [ ] ONNX conversion
- [ ] TensorFlow Lite conversion
- [ ] Mobile performance validation

### 🔲 Phase 4: Mobile Deployment (TODO)
- [ ] iOS demo app (Core ML)
- [ ] Android demo app (TFLite)
- [ ] Performance benchmarking

## Quick Start

### Phase 1: Prepare Data

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure pipeline (optional)
# Edit configs/data_config.yaml

# 3. Run data pipeline
python scripts/run_data_pipeline.py

# Output: Clean data splits + trained tokenizer
```

See [PHASE1_README.md](PHASE1_README.md) for detailed instructions.

## Directory Structure

```
text_trans_1/
├── configs/              # Configuration files
│   └── data_config.yaml
├── data/                 # Data files (gitignored)
│   ├── raw/             # Downloaded datasets
│   ├── processed/       # Cleaned data
│   └── splits/          # Train/val/test splits
├── models/              # Model files
│   ├── tokenizer/       # Trained tokenizers
│   ├── checkpoints/     # Training checkpoints
│   └── final/          # Final models
├── scripts/             # Pipeline scripts
│   ├── 01_download_data.py
│   ├── 02_clean_data.py
│   ├── 03_split_data.py
│   ├── 04_train_tokenizer.py
│   └── run_data_pipeline.py
├── utils/               # Utility functions
├── logs/                # Execution logs
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── TRANSLATION_PIPELINE_PLAN.md
└── PHASE1_README.md
```

## Key Features

### Data Pipeline (Phase 1)
- Multi-source dataset collection (2-3M pairs)
- Robust cleaning with script validation
- Stratified train/val/test splitting
- SentencePiece BPE tokenization (32K vocab)

### Planned Features
- Lightweight transformer architecture (~60-80M params)
- Mobile-optimized inference (INT8 quantization)
- Cross-platform deployment (iOS + Android)
- Offline translation capability

## Requirements

### Development Environment
- Python 3.10+
- 8-16 GB RAM
- 10 GB disk space (for data)

### Training Infrastructure (Phase 2)
- NVIDIA GPU (A100/RTX 4090)
- 64 GB RAM
- CUDA 11.8+

## Technology Stack

- **ML Frameworks**: PyTorch, HuggingFace Transformers
- **Tokenization**: SentencePiece
- **Conversion**: ONNX, TensorFlow Lite
- **Mobile**: Core ML (iOS), TFLite (Android)

## Contributing

This is a learning/research project for building mobile-optimized NMT systems.

## License

See individual dataset licenses:
- Samanantar: CC0
- OPUS: Various (check individual corpora)
- FLORES: CC-BY-SA

## References

- [Samanantar Dataset](https://ai4bharat.org/samanantar)
- [OPUS Parallel Corpora](https://opus.nlpl.eu/)
- [FLORES-200](https://github.com/facebookresearch/flores)
- [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)
- [SentencePiece](https://github.com/google/sentencepiece)
