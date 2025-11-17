# Phase 1: Data Preparation Pipeline

This directory contains the complete implementation of Phase 1 of the Punjabi-English translation pipeline: **Data Collection and Preparation**.

## Overview

Phase 1 prepares high-quality parallel data for training translation models. The pipeline includes:

1. **Data Download**: Collect parallel corpora from multiple sources
2. **Data Cleaning**: Filter and normalize parallel text
3. **Data Splitting**: Create train/validation/test splits
4. **Tokenizer Training**: Train SentencePiece BPE tokenizer

## Directory Structure

```
text_trans_1/
├── configs/
│   └── data_config.yaml          # Configuration for all data pipeline settings
├── data/
│   ├── raw/                      # Downloaded raw datasets (gitignored)
│   ├── processed/                # Cleaned and filtered data (gitignored)
│   └── splits/                   # Train/val/test splits (gitignored)
├── models/
│   └── tokenizer/                # Trained tokenizer models
├── scripts/
│   ├── 01_download_data.py       # Download datasets from sources
│   ├── 02_clean_data.py          # Clean and filter parallel data
│   ├── 03_split_data.py          # Split into train/val/test
│   ├── 04_train_tokenizer.py     # Train SentencePiece tokenizer
│   └── run_data_pipeline.py      # Main orchestration script
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Logging utilities
│   ├── config.py                 # Configuration management
│   └── stats.py                  # Statistics tracking
├── logs/                         # Pipeline logs
├── requirements.txt              # Python dependencies
└── PHASE1_README.md             # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Pipeline

Edit `configs/data_config.yaml` to customize:
- Data sources to download
- Cleaning parameters
- Split ratios
- Tokenizer settings

### 3. Run Complete Pipeline

```bash
# Run all steps
python scripts/run_data_pipeline.py

# Or run with selective steps
python scripts/run_data_pipeline.py --skip-download --skip-clean
```

### 4. Run Individual Steps

```bash
# Step 1: Download data
python scripts/01_download_data.py

# Step 2: Clean data
python scripts/02_clean_data.py

# Step 3: Split data
python scripts/03_split_data.py

# Step 4: Train tokenizer
python scripts/04_train_tokenizer.py
```

## Data Sources

### Primary Sources

#### 1. **Samanantar** (AI4Bharat)
- **Size**: ~2.7M Punjabi-English parallel sentences
- **Quality**: High quality, diverse domains
- **URL**: https://ai4bharat.org/samanantar
- **Format**: TSV files

#### 2. **OPUS Corpora**
- **Ubuntu**: Localization data
- **GNOME**: Software translations
- **KDE**: KDE translations
- **Tatoeba**: User-contributed sentences
- **URL**: https://opus.nlpl.eu/

#### 3. **FLORES-200**
- **Size**: ~1000 sentences (for evaluation)
- **Quality**: Professional translations
- **URL**: https://github.com/facebookresearch/flores

### Expected Total Dataset Size

- **Training**: 2-3M sentence pairs
- **Validation**: 5K sentence pairs
- **Test**: 5K sentence pairs

## Data Cleaning Pipeline

The cleaning process applies multiple filters to ensure data quality:

### 1. **Text Normalization**
- Unicode normalization (NFKC)
- Whitespace normalization
- Remove zero-width characters

### 2. **Content Filtering**
- Remove HTML tags
- Remove URLs
- Remove email addresses

### 3. **Script Validation**
- Punjabi text must contain Gurmukhi characters (U+0A00-U+0A7F)
- English text must contain Latin characters

### 4. **Length Filtering**
- Minimum: 3 words per sentence
- Maximum: 100 words per sentence
- Length ratio: 0.5 < source/target < 2.0

### 5. **Quality Filtering**
- Maximum digit ratio: 50%
- Maximum punctuation ratio: 30%
- Remove identical source/target pairs
- Deduplication

## Data Splitting Strategy

### Split Ratios (Configurable)
- **Train**: 90% (~2.5M pairs)
- **Validation**: 5% (~5K pairs)
- **Test**: 5% (~5K pairs)

### Stratification
Optional stratification by sentence length to ensure balanced distribution:
- Bin 0: 0-10 words
- Bin 1: 11-20 words
- Bin 2: 21-40 words
- Bin 3: 41-80 words
- Bin 4: 81+ words

### Output Formats
For each split (train/val/test):
- `{split}.json.gz` - Complete data with metadata
- `{split}.pa` - Source language (Punjabi) only
- `{split}.en` - Target language (English) only

## Tokenizer Training

### SentencePiece BPE Configuration

```yaml
Model Type: BPE (Byte Pair Encoding)
Vocabulary Size: 32,000 tokens
Character Coverage: 0.9995
Normalization: NFKC
```

### Vocabulary Options

#### Option 1: Shared Vocabulary (Default)
- Single tokenizer for both languages
- Vocabulary size: 32K total
- Better for related languages
- More parameter sharing in model

#### Option 2: Separate Vocabularies
- Two tokenizers (Punjabi + English)
- Vocabulary size: 16K each
- Better language-specific coverage
- More flexible

### Special Tokens
- `<pad>` (ID: 0) - Padding token
- `<unk>` (ID: 1) - Unknown token
- `<s>` (ID: 2) - Begin of sentence
- `</s>` (ID: 3) - End of sentence

## Configuration Reference

### Key Configuration Options (`configs/data_config.yaml`)

```yaml
# Data sources
data_sources:
  samanantar:
    enabled: true
  opus:
    enabled: true
    corpora: ["Ubuntu", "GNOME", "KDE", "Tatoeba"]

# Cleaning parameters
cleaning:
  min_length: 3
  max_length: 100
  min_length_ratio: 0.5
  max_length_ratio: 2.0
  remove_duplicates: true

# Tokenizer settings
tokenizer:
  type: "sentencepiece"
  model_type: "bpe"
  vocab_size: 32000
  shared_vocab: true

# Data splits
splits:
  train: 0.90
  validation: 0.05
  test: 0.05
  seed: 42
```

## Output Files

### After Step 1 (Download)
```
data/raw/
├── samanantar/
│   └── samanantar-pa.zip (extracted)
├── opus/
│   ├── ubuntu/
│   ├── gnome/
│   └── kde/
└── flores/
    └── dev/devtest sets
```

### After Step 2 (Clean)
```
data/processed/
├── cleaned_parallel_data.json.gz  # Cleaned data
└── cleaning_statistics.json       # Processing stats
```

### After Step 3 (Split)
```
data/splits/
├── train.json.gz, train.pa, train.en
├── val.json.gz, val.pa, val.en
├── test.json.gz, test.pa, test.en
└── split_metadata.json
```

### After Step 4 (Tokenizer)
```
models/tokenizer/
├── tokenizer_shared.model         # Tokenizer model
├── tokenizer_shared.vocab         # Vocabulary
└── combined_train.txt            # Training data
```

## Monitoring and Logging

### Logs
All pipeline steps create detailed logs in `logs/`:
- `data_download_YYYYMMDD_HHMMSS.log`
- `data_cleaning_YYYYMMDD_HHMMSS.log`
- `data_splitting_YYYYMMDD_HHMMSS.log`
- `tokenizer_training_YYYYMMDD_HHMMSS.log`

### Statistics
Track data quality and pipeline performance:
- Cleaning statistics: Removal reasons, retention rate
- Split statistics: Size distribution, length statistics
- Tokenizer statistics: Vocabulary coverage, sample tokenizations

## Troubleshooting

### Common Issues

#### 1. Download Failures
**Problem**: Network errors or unavailable datasets

**Solution**:
```bash
# Retry download
python scripts/01_download_data.py

# Or skip problematic sources in config
```

#### 2. Memory Issues
**Problem**: Out of memory during cleaning

**Solution**:
- Reduce `processing.batch_size` in config
- Process data sources individually
- Use streaming processing for large files

#### 3. Low Data Retention
**Problem**: Too much data filtered out

**Solution**:
- Review `cleaning_statistics.json`
- Relax filtering parameters in config
- Check data source quality

#### 4. Tokenizer Training Slow
**Problem**: Training takes too long

**Solution**:
- Reduce training data sample size
- Adjust `num_threads` in tokenizer config
- Use smaller vocabulary size

## Performance Expectations

### Resource Requirements
- **Disk Space**: 5-10 GB
- **RAM**: 8-16 GB
- **Time**: 2-4 hours total

### Step-by-Step Timing
1. Download: 30-60 minutes (network dependent)
2. Cleaning: 30-60 minutes
3. Splitting: 5-10 minutes
4. Tokenizer: 15-30 minutes

## Validation Checklist

Before proceeding to Phase 2 (Model Training):

- [ ] Downloaded datasets exist in `data/raw/`
- [ ] Cleaned data has >100K pairs in `data/processed/`
- [ ] Train/val/test splits created in `data/splits/`
- [ ] Train set has >100K pairs
- [ ] Val/test sets have >5K pairs each
- [ ] Tokenizer model trained successfully
- [ ] Tokenizer vocabulary size is 32K
- [ ] Sample tokenizations look correct
- [ ] No critical errors in log files
- [ ] Statistics files generated

## Next Steps

After completing Phase 1:

1. **Review Data Quality**
   - Check cleaning statistics
   - Inspect sample sentences
   - Verify language distribution

2. **Validate Tokenizer**
   - Test on sample sentences
   - Check vocabulary coverage
   - Verify special tokens work

3. **Proceed to Phase 2**
   - Model training setup
   - Define model architecture
   - Configure training parameters

## Support and Resources

### Documentation
- Main Plan: `TRANSLATION_PIPELINE_PLAN.md`
- Configuration: `configs/data_config.yaml`
- Logs: `logs/` directory

### Datasets
- Samanantar: https://ai4bharat.org/samanantar
- OPUS: https://opus.nlpl.eu/
- FLORES: https://github.com/facebookresearch/flores

### Tools
- SentencePiece: https://github.com/google/sentencepiece
- HuggingFace Datasets: https://huggingface.co/docs/datasets

## License and Attribution

This pipeline uses publicly available datasets. Please cite the original sources:

```bibtex
@inproceedings{samanantar2021,
    title={Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages},
    author={Ramesh, Gowtham and others},
    year={2021}
}
```

---

**Phase 1 Implementation Status**: ✅ Complete

**Ready for Phase 2**: After running pipeline and validation
