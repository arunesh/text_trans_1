# Phase 2: Model Training

This directory contains the complete implementation of Phase 2 of the Punjabi-English translation pipeline: **Model Training**.

## Overview

Phase 2 trains lightweight transformer models for bidirectional Punjabi↔English translation. The implementation includes:

1. **Custom Transformer Architecture**: Mobile-optimized encoder-decoder model
2. **Training Infrastructure**: Complete training loop with mixed precision
3. **Evaluation Framework**: BLEU, chrF, and TER metrics
4. **Checkpoint Management**: Automatic saving and resuming
5. **Monitoring**: TensorBoard logging and progress tracking

## Directory Structure

```
text_trans_1/
├── configs/
│   └── train_config.yaml         # Training configuration
├── training/
│   ├── models/
│   │   └── transformer.py        # Transformer model implementation
│   ├── data/
│   │   └── dataset.py            # Dataset and data loaders
│   ├── evaluation/
│   │   ├── metrics.py            # BLEU, chrF, TER metrics
│   │   └── evaluator.py          # Model evaluation
│   ├── trainer.py                # Main training logic
│   └── __init__.py
├── scripts/
│   ├── 05_train_model.py         # Train a translation model
│   └── 06_evaluate_model.py      # Evaluate trained model
└── PHASE2_README.md             # This file
```

## Quick Start

### Prerequisites

Ensure Phase 1 is complete:
- ✅ Data splits created in `data/splits/`
- ✅ Tokenizer trained in `models/tokenizer/`

### Train a Model

```bash
# Train Punjabi → English
python scripts/05_train_model.py --direction pa-en

# Train English → Punjabi
python scripts/05_train_model.py --direction en-pa

# Resume from checkpoint
python scripts/05_train_model.py --direction pa-en --resume models/checkpoints/checkpoint-10000.pt

# Custom configuration
python scripts/05_train_model.py --config configs/my_config.yaml --direction pa-en
```

### Evaluate a Model

```bash
# Evaluate on test set
python scripts/06_evaluate_model.py \
    --checkpoint models/checkpoints/pa-en-42/best_model.pt \
    --source-file data/splits/test.pa \
    --reference-file data/splits/test.en
```

## Model Architecture

### Lightweight Transformer

The implementation uses a standard Transformer encoder-decoder architecture optimized for mobile deployment:

**Default Configuration:**
```yaml
Encoder layers: 6
Decoder layers: 6
Hidden size: 512
Attention heads: 8
FFN size: 2048
Dropout: 0.1
Vocabulary: 32K (shared)
Parameters: ~74M
Size (fp32): ~296 MB
Size (int8): ~74 MB (after quantization)
```

**Key Features:**
- Sinusoidal positional encoding
- Multi-head self-attention
- Layer normalization
- Tied input/output embeddings
- Greedy and beam search decoding

### Model Variants

You can adjust model size in `configs/train_config.yaml`:

**Small (Mobile-optimized):**
```yaml
encoder_layers: 4
decoder_layers: 4
hidden_size: 256
num_attention_heads: 8
intermediate_size: 1024
# ~18M parameters, ~72 MB (fp32)
```

**Base (Recommended):**
```yaml
encoder_layers: 6
decoder_layers: 6
hidden_size: 512
num_attention_heads: 8
intermediate_size: 2048
# ~74M parameters, ~296 MB (fp32)
```

**Large (Best quality):**
```yaml
encoder_layers: 6
decoder_layers: 6
hidden_size: 768
num_attention_heads: 12
intermediate_size: 3072
# ~165M parameters, ~660 MB (fp32)
```

## Training Configuration

### Key Parameters

Edit `configs/train_config.yaml` to customize training:

```yaml
training:
  # Batch size and accumulation
  batch_size: 32                      # Per-GPU batch size
  gradient_accumulation_steps: 4      # Effective batch = 128

  # Learning rate
  learning_rate: 5.0e-4               # Peak learning rate
  warmup_steps: 4000                  # Warmup steps
  lr_scheduler_type: "inverse_sqrt"   # Scheduler type

  # Training duration
  num_train_epochs: 10                # Number of epochs
  max_steps: 100000                   # Max steps (overrides epochs)

  # Evaluation and checkpointing
  eval_steps: 2000                    # Evaluate every N steps
  save_steps: 5000                    # Save checkpoint every N steps
  logging_steps: 100                  # Log every N steps

  # Early stopping
  early_stopping_patience: 10         # Stop if no improvement

  # Optimization
  optimizer: "adamw"
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Mixed precision (faster on GPU)
  fp16: true
```

### Learning Rate Schedule

**Inverse Square Root (Recommended):**
- Linear warmup for first `warmup_steps`
- Then decays as `1/sqrt(step)`
- Commonly used in transformer training

**Formula:**
```
LR(step) = base_lr * min(step^-0.5, step * warmup_steps^-1.5)
```

## Training Process

### Step-by-Step

1. **Data Loading**
   - Loads train/val splits from Phase 1
   - Creates PyTorch DataLoaders with batching and padding
   - Uses SentencePiece tokenizer from Phase 1

2. **Model Initialization**
   - Creates TranslationTransformer model
   - Initializes with Xavier uniform
   - Moves to GPU if available

3. **Training Loop**
   - Iterates through batches
   - Forward pass with teacher forcing
   - Backward pass with gradient accumulation
   - Optimizer step with gradient clipping
   - Learning rate schedule update

4. **Evaluation**
   - Every `eval_steps`, generates translations on validation set
   - Computes BLEU, chrF, TER metrics
   - Saves best model based on BLEU score
   - Early stopping if no improvement

5. **Checkpointing**
   - Saves model every `save_steps`
   - Keeps only best N checkpoints
   - Saves final model at end

### Mixed Precision Training

Enabled by default (`fp16: true`):
- Uses FP16 for forward/backward pass
- Uses FP32 for optimizer state
- 2x faster training on modern GPUs
- Uses less memory

### Gradient Accumulation

Simulates larger batch sizes:
- `effective_batch = batch_size * gradient_accumulation_steps`
- Useful when GPU memory is limited
- Default: 32 * 4 = 128 effective batch size

## Evaluation Metrics

### Automatic Metrics

**BLEU (Bilingual Evaluation Understudy)**
- Measures n-gram overlap with reference
- Range: 0-100 (higher is better)
- Target: > 25 for both directions

**chrF (Character n-gram F-score)**
- Character-level metric
- Better for morphologically rich languages
- Range: 0-100 (higher is better)

**TER (Translation Error Rate)**
- Measures edit distance
- Range: 0-100 (lower is better)

### Evaluation During Training

Validation metrics computed every `eval_steps`:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- Overall BLEU score
- chrF score
- TER score

Results logged to:
- Console output
- TensorBoard
- Training logs

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training:

```bash
tensorboard --logdir runs/
```

**Metrics Tracked:**
- Training loss
- Learning rate
- Validation BLEU/chrF/TER
- Gradient norms
- Step timing

### Console Output

Real-time training progress:
```
Epoch 1/10
Step 100 - Loss: 4.2341, LR: 1.25e-05
Step 200 - Loss: 3.8932, LR: 2.50e-05
...
Step 2000 - BLEU: 12.34, chrF: 35.67, TER: 65.43
```

### Log Files

Detailed logs saved to `logs/trainer_TIMESTAMP.log`:
- Full configuration
- Data loading info
- Training progress
- Evaluation results
- Checkpoint saves

## Output Files

### During Training

```
models/checkpoints/{direction}-{seed}/
├── checkpoint-5000.pt          # Periodic checkpoint
├── checkpoint-10000.pt
├── best_model.pt               # Best model (highest BLEU)
└── final_model.pt              # Final model (end of training)
```

### Checkpoint Contents

Each `.pt` file contains:
```python
{
    'model_state_dict': ...,      # Model weights
    'optimizer_state_dict': ...,  # Optimizer state
    'scheduler_state_dict': ...,  # LR scheduler state
    'global_step': ...,           # Training step
    'epoch': ...,                 # Epoch number
    'best_metric': ...,           # Best BLEU score
    'config': ...                 # Full configuration
}
```

## Expected Results

### Training Time

**GPU (NVIDIA A100):**
- ~12-18 hours for 100K steps
- ~1.5-2.5 hours per epoch

**GPU (RTX 3090/4090):**
- ~18-24 hours for 100K steps
- ~2.5-3.5 hours per epoch

**CPU:**
- Not recommended (100x slower)
- Use for testing only

### Model Quality

**Expected BLEU Scores (after full training):**

| Direction | BLEU | chrF | Notes |
|-----------|------|------|-------|
| PA → EN   | 25-35 | 45-55 | Depends on data quality |
| EN → PA   | 20-30 | 40-50 | Gurmukhi script harder |

**Factors Affecting Quality:**
- Training data size (more is better)
- Data quality (clean parallel pairs)
- Model size (larger = better, slower)
- Training duration (longer = better)

### Convergence

Typical training curve:
- **Steps 0-4000**: Warmup, loss decreases rapidly
- **Steps 4000-20000**: Rapid improvement, BLEU increases quickly
- **Steps 20000-60000**: Steady improvement
- **Steps 60000-100000**: Slow improvement, approaching plateau

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Problem**: CUDA out of memory error

**Solutions**:
```yaml
# Reduce batch size
batch_size: 16  # or 8

# Increase gradient accumulation
gradient_accumulation_steps: 8

# Enable gradient checkpointing
gradient_checkpointing: true

# Reduce sequence length
max_source_length: 64
max_target_length: 64
```

#### 2. Slow Training

**Problem**: Training is very slow

**Solutions**:
- Enable FP16: `fp16: true`
- Increase num_workers: `num_workers: 8`
- Use larger batch size if memory allows
- Check GPU utilization (`nvidia-smi`)

#### 3. Poor Quality

**Problem**: BLEU score is very low (<10)

**Diagnosis**:
```bash
# Check data quality
python scripts/02_clean_data.py  # Review statistics

# Inspect model predictions
python scripts/06_evaluate_model.py --checkpoint ... --source-file ... --reference-file ...
```

**Solutions**:
- Train longer (more epochs or steps)
- Use larger model
- Improve data quality (more cleaning)
- Check tokenizer is working correctly

#### 4. NaN Loss

**Problem**: Loss becomes NaN during training

**Solutions**:
```yaml
# Reduce learning rate
learning_rate: 1.0e-4

# Enable gradient clipping
max_grad_norm: 0.5

# Disable FP16
fp16: false

# Increase warmup
warmup_steps: 8000
```

## Advanced Usage

### Resume Training

```bash
python scripts/05_train_model.py \
    --direction pa-en \
    --resume models/checkpoints/pa-en-42/checkpoint-50000.pt
```

### Custom Configuration

Create custom config file:

```yaml
# configs/my_config.yaml
model:
  encoder_layers: 4
  decoder_layers: 4
  hidden_size: 384
  # ... other settings
```

Run with custom config:

```bash
python scripts/05_train_model.py \
    --config configs/my_config.yaml \
    --direction pa-en
```

### Multi-GPU Training

Not implemented in current version. To add:
- Use `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel`
- Set `distributed: true` in config
- Launch with `torchrun`

## Model Export

After training, models can be exported for mobile deployment (Phase 3):

```python
# Load checkpoint
checkpoint = torch.load('models/checkpoints/pa-en-42/best_model.pt')

# Extract model
model.load_state_dict(checkpoint['model_state_dict'])

# Export to ONNX (Phase 3)
# Export to TFLite (Phase 3)
```

## Validation Checklist

Before proceeding to Phase 3 (Model Optimization):

- [ ] Training completed without errors
- [ ] BLEU score > 20 on validation set
- [ ] Best model checkpoint saved
- [ ] Evaluation metrics logged
- [ ] TensorBoard shows learning curve
- [ ] Manual inspection of translations looks reasonable
- [ ] Both PA→EN and EN→PA models trained

## Next Steps

After completing Phase 2:

1. **Evaluate Both Directions**
   - Train PA→EN model
   - Train EN→PA model
   - Compare quality

2. **Select Best Model**
   - Choose based on BLEU score
   - Consider use case requirements

3. **Proceed to Phase 3**
   - Model optimization
   - Quantization
   - ONNX/TFLite conversion

## Performance Tips

### For Faster Training

1. **Use FP16**: `fp16: true`
2. **Larger Batch Size**: If GPU memory allows
3. **More Workers**: `num_workers: 8`
4. **Reduce Eval Frequency**: `eval_steps: 5000`

### For Better Quality

1. **More Data**: Use all available parallel corpora
2. **Longer Training**: `max_steps: 200000`
3. **Larger Model**: Increase hidden_size and layers
4. **More Beams**: `num_beams: 5` for evaluation

### For Smaller Models

1. **Reduce Layers**: `encoder_layers: 4`, `decoder_layers: 4`
2. **Reduce Hidden Size**: `hidden_size: 256`
3. **Shared Embeddings**: `tie_embeddings: true`
4. **Knowledge Distillation**: Train from larger model (advanced)

## Support and Resources

### Documentation
- Main Plan: `TRANSLATION_PIPELINE_PLAN.md`
- Phase 1: `PHASE1_README.md`
- Configuration: `configs/train_config.yaml`
- Logs: `logs/trainer_*.log`

### Code References
- Model: `training/models/transformer.py`
- Trainer: `training/trainer.py`
- Dataset: `training/data/dataset.py`
- Metrics: `training/evaluation/metrics.py`

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "MarianNMT: Fast Neural Machine Translation in C++"
- "On the Relationship between Self-Attention and Convolutional Layers"

---

**Phase 2 Implementation Status**: ✅ Complete

**Ready for Phase 3**: After training and evaluation
