# Mobile Text Translation Pipeline Plan
## Punjabi ↔ English Translation Model

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Selection](#architecture-selection)
3. [Dataset Collection & Preparation](#dataset-collection--preparation)
4. [Training Infrastructure](#training-infrastructure)
5. [Model Training Pipeline](#model-training-pipeline)
6. [Model Optimization & Conversion](#model-optimization--conversion)
7. [Mobile Deployment](#mobile-deployment)
8. [Punjabi-English Recipe](#punjabi-english-recipe)
9. [Evaluation Strategy](#evaluation-strategy)
10. [Timeline & Milestones](#timeline--milestones)

---

## Project Overview

### Objective
Build a lightweight neural machine translation (NMT) system for Punjabi ↔ English translation that can run efficiently on mobile devices (Android/iOS).

### Key Requirements
- **Model Size**: < 50MB for mobile deployment
- **Inference Speed**: < 500ms per sentence on mobile CPU
- **Quality**: BLEU score > 25 for both directions
- **Formats**: ONNX and TensorFlow Lite
- **Bidirectional**: Separate models for PA→EN and EN→PA

---

## Architecture Selection

### Recommended Architecture: MarianMT (Transformer-based)
**Why MarianMT:**
- Proven track record for low-resource language pairs
- Compact architecture (6-layer encoder/decoder)
- Good balance between quality and size
- Easy integration with HuggingFace ecosystem
- Strong community support

### Alternative Options
1. **T5-Small/T5-Base** (Text-to-Text Transfer Transformer)
   - Pros: Unified architecture, pre-trained
   - Cons: Larger model size (~250MB)

2. **MBart-50** (Multilingual BART)
   - Pros: Pre-trained on 50 languages
   - Cons: Larger size, may need heavy pruning

3. **Custom Lightweight Transformer**
   - Encoder: 4 layers, 512 hidden, 8 attention heads
   - Decoder: 4 layers, 512 hidden, 8 attention heads
   - Pros: Fully customizable, optimized for mobile
   - Cons: No pre-training benefits

### Final Recommendation
**Start with MarianMT base**, then explore custom lightweight transformer if size constraints are too tight.

**Target Model Specifications:**
- Vocabulary: 32k BPE tokens (16k per language)
- Parameters: ~60-80M (compresses to ~30-40MB)
- Hidden size: 512
- Layers: 6 encoder + 6 decoder
- Attention heads: 8

---

## Dataset Collection & Preparation

### Primary Data Sources

#### 1. **Open-Source Parallel Corpora**
- **OPUS Project** (opus.nlpl.eu)
  - Ubuntu localization files
  - GNOME/KDE translations
  - OpenSubtitles (if available)
  - Tatoeba sentences

- **PMIndia (Prime Minister India) Corpus**
  - Government speeches and press releases
  - High quality, domain-specific

- **AI4Bharat IndicNLP**
  - Samanantar dataset (largest public parallel corpus for Indic languages)
  - Expected: 2.7M+ PA-EN sentence pairs

- **FLORES-200**
  - High-quality benchmark dataset
  - ~1000 sentences for evaluation

#### 2. **Web-Scraped Data**
- Wikipedia parallel articles (PA-EN)
- News websites with bilingual content
- Government websites (.gov.in)
- Religious texts (freely available translations)

#### 3. **Augmentation Sources**
- Back-translation using existing models
- Synthetic data generation
- Monolingual data for language model pre-training

### Data Quality Pipeline

```
Raw Data → Cleaning → Filtering → Deduplication → Train/Val/Test Split
```

#### Cleaning Steps
1. **Encoding normalization** (UTF-8)
2. **Script detection** (Gurmukhi for Punjabi, Latin for English)
3. **Length filtering** (3-100 tokens)
4. **Length ratio** (0.5 < len(src)/len(tgt) < 2.0)
5. **Language detection** (fastText LID)
6. **Deduplication** (exact and fuzzy matching)
7. **Quality scoring** (remove noisy pairs)

#### Target Dataset Size
- **Training**: 2-3M parallel sentences
- **Validation**: 5K sentences
- **Test**: 5K sentences (held-out domains)

### Preprocessing
1. **Tokenization**: SentencePiece BPE
2. **Vocabulary size**: 32K (shared or separate)
3. **Max sequence length**: 128 tokens
4. **Normalization**: Unicode NFKC, punctuation normalization

---

## Training Infrastructure

### Hardware Requirements

#### GPU Training Machine
- **GPU**: NVIDIA A100 (40GB) or RTX 4090 (24GB)
- **RAM**: 64GB+
- **Storage**: 500GB SSD
- **Alternative**: Cloud GPU (Google Colab Pro+, AWS p3.2xlarge, Lambda Labs)

#### Estimated Training Time
- **MarianMT (6-6)**: 12-24 hours on A100
- **Custom Lightweight (4-4)**: 6-12 hours on A100

### Software Stack
- **Framework**: PyTorch 2.0+ with HuggingFace Transformers
- **Training**: HuggingFace Trainer API or PyTorch Lightning
- **Data**: Datasets library (HuggingFace)
- **Tokenization**: SentencePiece or HuggingFace Tokenizers
- **Experiment Tracking**: Weights & Biases or TensorBoard
- **Version Control**: Git + DVC (for datasets)

---

## Model Training Pipeline

### Phase 1: Data Preparation
```bash
1. Download datasets from sources
2. Merge and clean parallel corpora
3. Split into train/val/test (90/5/5)
4. Train SentencePiece tokenizer
5. Preprocess and cache datasets
```

### Phase 2: Model Training

#### Training Configuration (PA→EN)
```yaml
model:
  architecture: MarianMT
  encoder_layers: 6
  decoder_layers: 6
  hidden_size: 512
  attention_heads: 8
  ffn_dim: 2048
  dropout: 0.1

training:
  batch_size: 32 (per GPU)
  gradient_accumulation: 4
  effective_batch: 128
  learning_rate: 5e-4
  warmup_steps: 4000
  max_steps: 100000
  optimizer: AdamW
  scheduler: inverse_sqrt
  mixed_precision: fp16

regularization:
  label_smoothing: 0.1
  dropout: 0.1
  weight_decay: 0.01
```

#### Training Strategy
1. **Curriculum Learning**: Start with shorter sentences, gradually increase
2. **Domain Mixing**: Balance domains in batches
3. **Regular Checkpointing**: Every 5K steps
4. **Early Stopping**: Patience of 10 evaluations
5. **Gradient Clipping**: Max norm 1.0

### Phase 3: Bidirectional Training
- Train two separate models: PA→EN and EN→PA
- Use same hyperparameters for consistency
- Can share tokenizer or use separate ones

### Phase 4: Fine-tuning (Optional)
- Domain-specific fine-tuning (if target domain known)
- Knowledge distillation from larger models
- Quantization-aware training

---

## Model Optimization & Conversion

### Optimization Pipeline
```
Trained Model → Pruning → Quantization → Format Conversion → Validation
```

### Step 1: Model Pruning (Optional)
- **Magnitude-based pruning**: Remove 20-30% of smallest weights
- **Structured pruning**: Remove entire attention heads or layers
- **Tools**: PyTorch pruning utilities, Neural Network Intelligence (NNI)

### Step 2: Quantization

#### Dynamic Quantization
```python
# Post-training quantization (PTQ)
- Weight precision: INT8
- Activation precision: INT8
- Expected size reduction: 4x
- Quality loss: < 1 BLEU point
```

#### Quantization-Aware Training (QAT)
- Better quality preservation
- Longer training time
- Recommended for production

### Step 3: ONNX Conversion

```python
# Export to ONNX
- Opset version: 14+
- Optimization level: All
- Operator simplification: Yes
- Constant folding: Yes
```

**Tools:**
- `torch.onnx.export()`
- ONNX Runtime for validation
- ONNX Simplifier for graph optimization

**Target Size**: 30-40MB per model (quantized)

### Step 4: TensorFlow Lite Conversion

```python
# For Android/iOS
- Conversion: PyTorch → ONNX → TensorFlow → TFLite
- Or: Direct PyTorch → TFLite (using AI Edge Torch)
- Quantization: Post-training dynamic range quantization
- Optimization: Default optimizations enabled
```

**Alternative Path:**
```
PyTorch → ONNX → TFLite (using onnx-tf + TFLiteConverter)
```

**Target Size**: 25-35MB per model (quantized)

### Step 5: Validation
- Compare outputs: Original vs ONNX vs TFLite
- Max acceptable deviation: 0.01
- BLEU score difference: < 0.5 points
- Latency benchmarking on target devices

---

## Mobile Deployment

### iOS Deployment

#### Option 1: Core ML
```
ONNX → Core ML (using onnx-coreml)
```
- **Advantages**: Native iOS integration, optimized
- **Framework**: Core ML, MLKit
- **Deployment size**: 25-30MB

#### Option 2: ONNX Runtime Mobile
- Cross-platform consistency
- Smaller binary size
- Good performance

### Android Deployment

#### Option 1: TensorFlow Lite
```
Primary recommendation for Android
```
- **Advantages**: Native support, NNAPI acceleration
- **Framework**: TFLite Interpreter
- **Deployment size**: 25-35MB
- **Delegates**: GPU, NNAPI, Hexagon DSP

#### Option 2: ONNX Runtime Mobile
- Consistent with iOS
- Good CPU performance

### Performance Optimization

#### Mobile-Specific Optimizations
1. **Beam Search**: Reduce beam size to 1-3
2. **Sequence Length**: Cap at 100 tokens
3. **Batching**: Process single sentences
4. **Thread Optimization**: Use 2-4 threads
5. **Memory**: Preallocate buffers

#### Expected Performance
- **Inference Time**: 200-500ms per sentence (CPU)
- **GPU Acceleration**: 50-150ms (if available)
- **Memory Usage**: 100-200MB RAM
- **Battery Impact**: Minimal for occasional use

### App Integration Architecture
```
User Input → Tokenization → Model Inference → Detokenization → Display
              ↓                    ↓                   ↓
          Cache Layer      Model Selection       Post-processing
                          (PA→EN / EN→PA)
```

---

## Punjabi-English Recipe

### Complete Step-by-Step Recipe

#### Step 1: Environment Setup
```bash
# Create conda environment
conda create -n pa-en-translation python=3.10
conda activate pa-en-translation

# Install dependencies
pip install torch transformers datasets sentencepiece
pip install onnx onnxruntime tensorflow
pip install sacrebleu wandb

# GPU setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Step 2: Dataset Collection
```bash
# Download Samanantar (AI4Bharat)
wget https://ai4bharat-public-projects.s3.amazonaws.com/samanantar/samanantar-pa.zip
unzip samanantar-pa.zip

# Download OPUS datasets
opus_get -s en -t pa -p Ubuntu,GNOME,KDE

# Download PMIndia
# (Manual download from CDAC website)

# Expected total: 2-3M sentence pairs
```

#### Step 3: Data Cleaning & Preprocessing
```python
# Clean and filter data
- Remove duplicates
- Filter by length (3-100 words)
- Language detection (remove misaligned pairs)
- Quality scoring (remove noisy pairs)
- Normalize punctuation and Unicode

# Split data
- Train: 2.5M pairs (90%)
- Validation: 5K pairs
- Test: 5K pairs (FLORES + custom)
```

#### Step 4: Tokenizer Training
```python
# Train SentencePiece BPE tokenizer
- Vocabulary size: 32K
- Character coverage: 0.9995
- Model type: BPE
- Special tokens: <pad>, <s>, </s>, <unk>

# Option 1: Shared vocabulary
# Option 2: Separate vocabularies (16K each)
```

#### Step 5: Model Training (PA→EN)
```bash
# Training script
python train_translation.py \
  --model_type marian \
  --train_file data/train.pa-en.json \
  --val_file data/val.pa-en.json \
  --source_lang pa \
  --target_lang en \
  --encoder_layers 6 \
  --decoder_layers 6 \
  --hidden_size 512 \
  --num_attention_heads 8 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --warmup_steps 4000 \
  --max_steps 100000 \
  --fp16 \
  --output_dir models/pa-en

# Expected training time: 18-24 hours on A100
```

#### Step 6: Model Training (EN→PA)
```bash
# Reverse direction - same config
python train_translation.py \
  --model_type marian \
  --train_file data/train.en-pa.json \
  --val_file data/val.en-pa.json \
  --source_lang en \
  --target_lang pa \
  [same params as above] \
  --output_dir models/en-pa

# Expected training time: 18-24 hours on A100
```

#### Step 7: Model Evaluation
```bash
# Evaluate on test set
python evaluate.py \
  --model_dir models/pa-en \
  --test_file data/test.pa-en.json \
  --metrics bleu,chrf,ter

# Expected BLEU: 25-35 (depends on data quality)

# Evaluate EN→PA
python evaluate.py \
  --model_dir models/en-pa \
  --test_file data/test.en-pa.json \
  --metrics bleu,chrf,ter
```

#### Step 8: Model Optimization
```python
# Quantization
python quantize_model.py \
  --model_path models/pa-en \
  --quantization_mode dynamic \
  --output_path models/pa-en-quantized

# Size reduction: ~75MB → ~30MB

# Repeat for EN→PA
```

#### Step 9: ONNX Conversion
```bash
# Convert PA→EN to ONNX
python convert_to_onnx.py \
  --model_path models/pa-en-quantized \
  --output_path models/pa-en.onnx \
  --opset_version 14

# Optimize ONNX
python -m onnxruntime.transformers.optimizer \
  --input models/pa-en.onnx \
  --output models/pa-en-optimized.onnx

# Repeat for EN→PA
```

#### Step 10: TensorFlow Lite Conversion
```bash
# Convert ONNX → TFLite
python convert_to_tflite.py \
  --onnx_path models/pa-en-optimized.onnx \
  --output_path models/pa-en.tflite \
  --quantize

# Final size: ~25-30MB

# Repeat for EN→PA
```

#### Step 11: Mobile Validation
```bash
# Test ONNX model
python test_onnx.py \
  --model_path models/pa-en-optimized.onnx \
  --test_sentences test_inputs.txt

# Test TFLite model
python test_tflite.py \
  --model_path models/pa-en.tflite \
  --test_sentences test_inputs.txt

# Benchmark inference speed
python benchmark.py \
  --model_path models/pa-en.tflite \
  --device cpu \
  --num_threads 4
```

#### Step 12: Mobile App Integration
```
# iOS
- Use Core ML or ONNX Runtime Mobile
- Bundle models in app or download on demand
- Implement caching for tokenization

# Android
- Use TFLite with NNAPI delegate
- Implement model loading and inference
- Add language detection and text preprocessing
```

---

## Evaluation Strategy

### Automatic Metrics
1. **BLEU** (Primary metric)
   - Target: > 25 for both directions

2. **chrF** (Character n-gram F-score)
   - Better for morphologically rich languages

3. **TER** (Translation Error Rate)
   - Measures post-editing effort

4. **COMET** (Neural evaluation metric)
   - Correlates better with human judgment

### Human Evaluation
- **Adequacy**: Does translation preserve meaning?
- **Fluency**: Is translation grammatically correct?
- **Sample Size**: 500 sentences per direction
- **Evaluators**: Native Punjabi speakers

### Test Sets
1. **FLORES-200**: Standard benchmark
2. **Custom domain-specific**: News, conversational, technical
3. **Edge cases**: Idioms, names, numbers, dates

### Mobile-Specific Testing
- **Latency**: < 500ms per sentence
- **Memory**: < 200MB RAM usage
- **Battery**: < 5% per 100 translations
- **Accuracy**: < 1% degradation from original model

---

## Timeline & Milestones

### Phase 1: Setup & Data Preparation (Week 1-2)
- [ ] Set up development environment
- [ ] Collect and merge datasets
- [ ] Clean and preprocess data
- [ ] Train tokenizer
- [ ] Create train/val/test splits
- **Deliverable**: Clean dataset ready for training

### Phase 2: Model Training (Week 3-4)
- [ ] Train PA→EN model
- [ ] Train EN→PA model
- [ ] Hyperparameter tuning
- [ ] Evaluate on validation set
- **Deliverable**: Two trained PyTorch models

### Phase 3: Optimization & Conversion (Week 5)
- [ ] Apply quantization
- [ ] Convert to ONNX
- [ ] Convert to TFLite
- [ ] Validate conversions
- **Deliverable**: ONNX and TFLite models

### Phase 4: Mobile Integration (Week 6-7)
- [ ] Create iOS demo app (Core ML)
- [ ] Create Android demo app (TFLite)
- [ ] Performance benchmarking
- [ ] User testing
- **Deliverable**: Working mobile apps

### Phase 5: Evaluation & Iteration (Week 8)
- [ ] Comprehensive evaluation
- [ ] Human evaluation
- [ ] Identify weaknesses
- [ ] Plan improvements
- **Deliverable**: Evaluation report and roadmap

---

## Risk Mitigation

### Potential Challenges

1. **Limited Punjabi Data**
   - **Mitigation**: Use back-translation, synthetic data, multilingual pre-training

2. **Model Size Constraints**
   - **Mitigation**: Start with 4-layer transformer, aggressive quantization, knowledge distillation

3. **Quality vs Size Trade-off**
   - **Mitigation**: Iterative optimization, A/B testing different architectures

4. **Domain Mismatch**
   - **Mitigation**: Diverse training data, domain adaptation techniques

5. **Mobile Performance**
   - **Mitigation**: Beam size = 1, operator fusion, hardware acceleration

---

## Future Enhancements

1. **Multilingual Support**: Add Hindi, Urdu, other Indic languages
2. **Offline Speech Translation**: Integrate with ASR/TTS
3. **Context-Aware Translation**: Multi-sentence context
4. **User Personalization**: On-device fine-tuning
5. **Federated Learning**: Privacy-preserving improvements
6. **Real-time Streaming**: Word-by-word translation

---

## Resources & References

### Datasets
- Samanantar: https://ai4bharat.org/samanantar
- OPUS: https://opus.nlpl.eu/
- FLORES-200: https://github.com/facebookresearch/flores

### Models
- MarianMT: https://huggingface.co/Helsinki-NLP
- AI4Bharat IndicTrans: https://github.com/AI4Bharat/indicTrans

### Tools
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- ONNX Runtime: https://onnxruntime.ai/
- TensorFlow Lite: https://www.tensorflow.org/lite

### Papers
- "Attention Is All You Need" (Transformer architecture)
- "MarianNMT: Fast Neural Machine Translation in C++"
- "Samanantar: The Largest Publicly Available Parallel Corpora Collection"

---

## Success Criteria

### Technical Metrics
- ✅ Model size < 50MB (quantized)
- ✅ BLEU score > 25 (both directions)
- ✅ Inference time < 500ms on mobile CPU
- ✅ ONNX and TFLite formats working

### Business Metrics
- ✅ Usable in offline scenarios
- ✅ Battery efficient
- ✅ User satisfaction > 4/5
- ✅ Covers common use cases (90%+)

---

## Conclusion

This plan provides a comprehensive roadmap for building a production-ready Punjabi↔English translation system for mobile devices. The modular approach allows for iterative development and continuous improvement. Key success factors include:

1. **Quality data** (2-3M parallel sentences)
2. **Efficient architecture** (MarianMT or custom lightweight)
3. **Aggressive optimization** (quantization, pruning)
4. **Thorough testing** (automatic and human evaluation)
5. **Mobile-first design** (performance and size constraints)

By following this plan, you should be able to deliver a high-quality, mobile-optimized translation system within 8 weeks.
