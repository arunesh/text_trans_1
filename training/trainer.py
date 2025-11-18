"""
Trainer for translation models.

Handles training loop, evaluation, checkpointing, and logging.
"""

import sys
import time
import math
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from training.evaluation import TranslationEvaluator, format_metrics


class InverseSqrtScheduler:
    """Inverse square root learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps: int, initial_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * self.current_step / self.warmup_steps
        else:
            # Inverse square root decay
            return self.initial_lr * math.sqrt(self.warmup_steps / self.current_step)


class TranslationTrainer:
    """Trainer for translation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config: Dict,
        output_dir: str,
        device: torch.device
    ):
        """
        Initialize trainer.

        Args:
            model: Translation model
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Tokenizer
            config: Training configuration
            output_dir: Output directory for checkpoints
            device: Device to use
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger('trainer', config.get('logging', {}).get('logs_dir', 'logs'))

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup evaluator
        eval_config = config.get('evaluation', {})
        self.evaluator = TranslationEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=eval_config.get('max_length', 128),
            num_beams=eval_config.get('num_beams', 4)
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('-inf') if config['output']['greater_is_better'] else float('inf')
        self.patience_counter = 0

        # Setup tensorboard
        if config.get('logging', {}).get('tensorboard', {}).get('enabled', True):
            log_dir = config.get('logging', {}).get('tensorboard', {}).get('log_dir', 'runs')
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Mixed precision training
        self.use_fp16 = config['training'].get('fp16', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None

        # Gradient accumulation
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

        # Training parameters
        self.max_steps = config['training'].get('max_steps')
        self.num_epochs = config['training'].get('num_train_epochs', 10)
        self.eval_steps = config['training'].get('eval_steps', 1000)
        self.save_steps = config['training'].get('save_steps', 5000)
        self.logging_steps = config['training'].get('logging_steps', 100)
        self.max_grad_norm = config['training'].get('max_grad_norm', 1.0)

        # Early stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 10)
        self.metric_for_best = config['output'].get('metric_for_best_model', 'bleu')

        self.logger.info("Trainer initialized")
        self.logger.info(f"  Model parameters: {self.model.get_num_parameters():,}")
        self.logger.info(f"  Training samples: {len(train_loader.dataset):,}")
        self.logger.info(f"  Validation samples: {len(val_loader.dataset):,}")
        self.logger.info(f"  Batch size: {train_loader.batch_size}")
        self.logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch size: {train_loader.batch_size * self.gradient_accumulation_steps}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config['training']

        optimizer_type = opt_config.get('optimizer', 'adamw').lower()

        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                betas=(opt_config.get('adam_beta1', 0.9), opt_config.get('adam_beta2', 0.98)),
                eps=opt_config.get('adam_epsilon', 1e-9),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                betas=(opt_config.get('adam_beta1', 0.9), opt_config.get('adam_beta2', 0.98)),
                eps=opt_config.get('adam_epsilon', 1e-9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        opt_config = self.config['training']
        scheduler_type = opt_config.get('lr_scheduler_type', 'inverse_sqrt')

        if scheduler_type == 'inverse_sqrt':
            scheduler = InverseSqrtScheduler(
                self.optimizer,
                warmup_steps=opt_config.get('warmup_steps', 4000),
                initial_lr=opt_config['learning_rate']
            )
        else:
            # Fallback to constant LR
            scheduler = None

        return scheduler

    def train(self):
        """Run training loop."""
        self.logger.info("Starting training")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train one epoch
            train_loss = self.train_epoch()

            self.logger.info(f"Epoch {epoch + 1} - Train loss: {train_loss:.4f}")

            # Evaluate
            metrics = self.evaluate()
            self.logger.info(format_metrics(metrics, f"Epoch {epoch + 1} Validation"))

            # Check for early stopping
            if self._should_stop_early(metrics):
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Check if max steps reached
            if self.max_steps and self.global_step >= self.max_steps:
                self.logger.info(f"Max steps ({self.max_steps}) reached")
                break

        elapsed_time = time.time() - start_time
        self.logger.info(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")

        # Save final model
        self.save_checkpoint("final_model")

        if self.writer:
            self.writer.close()

    def train_epoch(self) -> float:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    _, loss = self.model(**batch)
                    loss = loss / self.gradient_accumulation_steps
            else:
                _, loss = self.model(**batch)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                if self.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f"Step {self.global_step} - Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, LR: {current_lr:.2e}"
                    )

                    if self.writer:
                        self.writer.add_scalar('train/loss', loss.item() * self.gradient_accumulation_steps, self.global_step)
                        self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)

                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    metrics = self.evaluate()
                    self.logger.info(format_metrics(metrics, f"Step {self.global_step}"))

                # Checkpointing
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss / num_batches})

            # Check max steps
            if self.max_steps and self.global_step >= self.max_steps:
                break

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        metrics = self.evaluator.evaluate(self.val_loader, max_samples=5000)

        # Log metrics
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'eval/{key}', value, self.global_step)

        self.model.train()

        return metrics

    def _should_stop_early(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        if self.metric_for_best not in metrics:
            return False

        current_metric = metrics[self.metric_for_best]
        improved = False

        if self.config['output']['greater_is_better']:
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                improved = True
        else:
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                improved = True

        if improved:
            self.patience_counter = 0
            self.save_checkpoint("best_model")
            self.logger.info(f"New best model! {self.metric_for_best}: {self.best_metric:.4f}")
        else:
            self.patience_counter += 1
            self.logger.info(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")

        return self.patience_counter >= self.early_stopping_patience

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"{name}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.__dict__ if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.__dict__.update(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"  Resuming from step {self.global_step}, epoch {self.current_epoch}")
