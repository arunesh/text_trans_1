"""
Lightweight Transformer model for neural machine translation.

Optimized for mobile deployment with configurable size.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with added positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )

        self.layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            src: Source sequence (batch_size, src_len, d_model)
            src_key_padding_mask: Mask for padding tokens (batch_size, src_len)

        Returns:
            Encoded sequence
        """
        return self.layers(src, src_key_padding_mask=src_key_padding_mask)


class TransformerDecoder(nn.Module):
    """Transformer decoder."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )

        self.layers = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            tgt: Target sequence (batch_size, tgt_len, d_model)
            memory: Encoder output (batch_size, src_len, d_model)
            tgt_mask: Causal mask for target
            tgt_key_padding_mask: Padding mask for target
            memory_key_padding_mask: Padding mask for encoder output

        Returns:
            Decoded sequence
        """
        return self.layers(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )


class TranslationTransformer(nn.Module):
    """
    Transformer model for neural machine translation.

    Lightweight architecture optimized for mobile deployment.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        tie_embeddings: bool = True
    ):
        """
        Initialize translation transformer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            pad_token_id: Padding token ID
            bos_token_id: Begin of sentence token ID
            eos_token_id: End of sentence token ID
            tie_embeddings: Whether to tie input and output embeddings
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Encoder and decoder
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings to output projection
        if tie_embeddings:
            self.output_projection.weight = self.embedding.weight

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Source token IDs (batch_size, src_len)
            attention_mask: Source attention mask (batch_size, src_len)
            labels: Target token IDs for training (batch_size, tgt_len)
            decoder_input_ids: Target input IDs for inference

        Returns:
            Tuple of (logits, loss)
        """
        # Encode source
        src_embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        src_embeddings = self.pos_encoding(src_embeddings)

        # Create source padding mask (True for padding)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        memory = self.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)

        # Prepare decoder input
        if decoder_input_ids is None:
            if labels is not None:
                # Training: shift labels right
                decoder_input_ids = self._shift_right(labels)
            else:
                raise ValueError("Either labels or decoder_input_ids must be provided")

        # Decode target
        tgt_embeddings = self.embedding(decoder_input_ids) * math.sqrt(self.d_model)
        tgt_embeddings = self.pos_encoding(tgt_embeddings)

        # Create causal mask for decoder
        tgt_len = decoder_input_ids.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(decoder_input_ids.device)

        # Create target padding mask
        tgt_key_padding_mask = (decoder_input_ids == self.pad_token_id)

        # Decode
        decoder_output = self.decoder(
            tgt_embeddings,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Shift labels to the right for teacher forcing.

        Args:
            labels: Target labels (batch_size, tgt_len)

        Returns:
            Shifted input (batch_size, tgt_len)
        """
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.bos_token_id

        # Replace -100 (padding in labels) with pad_token_id
        shifted[shifted == -100] = self.pad_token_id

        return shifted

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive decoding.

        Args:
            sz: Sequence length

        Returns:
            Causal mask of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """
        Generate translations using greedy or beam search.

        Args:
            input_ids: Source token IDs (batch_size, src_len)
            attention_mask: Source attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Generated token IDs
        """
        if num_beams > 1:
            return self._beam_search(input_ids, attention_mask, max_length, num_beams)
        else:
            return self._greedy_search(input_ids, attention_mask, max_length, temperature, top_k, top_p)

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Greedy decoding."""
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Encode source
        src_embeddings = self.embedding(input_ids) * math.sqrt(self.d_model)
        src_embeddings = self.pos_encoding(src_embeddings)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        memory = self.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)

        # Initialize decoder input with BOS token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )

        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Get logits
            tgt_embeddings = self.embedding(decoder_input_ids) * math.sqrt(self.d_model)
            tgt_embeddings = self.pos_encoding(tgt_embeddings)

            tgt_len = decoder_input_ids.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)

            decoder_output = self.decoder(
                tgt_embeddings,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            next_token_logits = self.output_projection(decoder_output[:, -1, :])

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Get next token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if (next_token == self.eos_token_id).all():
                break

        return decoder_input_ids

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_length: int,
        num_beams: int
    ) -> torch.Tensor:
        """
        Beam search decoding.

        Simplified implementation for demonstration.
        For production, use HuggingFace's beam search.
        """
        # For simplicity, fallback to greedy for now
        # A full beam search implementation would be more complex
        return self._greedy_search(input_ids, attention_mask, max_length, 1.0, 50, 1.0)

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing TranslationTransformer...")

    model = TranslationTransformer(
        vocab_size=32000,
        d_model=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    )

    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Model size: ~{model.get_num_parameters() * 4 / 1024 / 1024:.1f} MB (fp32)")

    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 15

    input_ids = torch.randint(0, 32000, (batch_size, src_len))
    labels = torch.randint(0, 32000, (batch_size, tgt_len))
    attention_mask = torch.ones_like(input_ids)

    logits, loss = model(input_ids, attention_mask, labels)

    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    outputs = model.generate(input_ids, attention_mask, max_length=30)
    print(f"\nGeneration:")
    print(f"  Generated shape: {outputs.shape}")
