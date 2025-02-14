import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import math
from typing import Optional, Dict, Any, Union, Callable, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TransformerConfig:
    """Configuration class for transformer models."""

    d_model: int
    num_heads: int
    num_encoder_layers: int
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 5000
    layer_norm_eps: float = 1e-5

    def __init__(self, d_model: int, num_heads: int, num_encoder_layers: int, d_ff: int = 2048, dropout: float = 0.1, max_seq_length: int = 5000, layer_norm_eps: float = 1e-5) -> None:     
        """
        Configuration class for transformer models.
        
        Args:
            d_model (int): The number of expected features in the input.
            num_heads (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            d_ff (int, optional): The dimension of the feedforward network model. Defaults to 2048.
            dropout (float, optional): The dropout value. Defaults to 0.1.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 5000.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-5.
        """
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.layer_norm_eps = layer_norm_eps

        self.__post_init__()

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads."
    

class BaseEmbedding(nn.Module, ABC):
    """Abstract base class for custom embeddings."""

    @abstractmethod
    def forward(self, x: Any) -> torch.Tensor:
        pass

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, config: TransformerConfig):
        """
        Multi-head attention mechanism.

        Args:
            config (TransformerConfig): Configuration class for transformer models.
        """

        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = self.d_model // self.num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model)  # Query 
        self.W_k = nn.Linear(self.d_model, self.d_model)  # Key
        self.W_v = nn.Linear(self.d_model, self.d_model)  # Value
        self.W_o = nn.Linear(self.d_model, self.d_model)  # Output

        self.dropout = nn.Dropout(config.dropout)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k). Used for multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor.
        """
        
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism. As input, it takes a query, key, and value tensor. Returns the attention-weighted value tensor.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor], optional): Mask tensor. Defaults to None.
        
        Returns:
            torch.Tensor: Attention-weighted value tensor.
        """

        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:  # For masked attention
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = torch.matmul(self.dropout(attention_weights), V)

        attention_weights = attention_weights.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attention_weights)
    
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models. Accepts continuous values as input."""
    
    def __init__(self, config: TransformerConfig):
        """
        Positional encoding for transformer models.

        Args:
            config (TransformerConfig): Configuration class for transformer models.
        """
        
        super().__init__()

        self.d_model = config.d_model

        self.div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Positional encoding tensor.
        """

        batch_size, seq_len = x.size()
        
        positions = x.unsqueeze(-1)
        
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=positions.device)
        pe[:, :, 0::2] = torch.sin(positions * self.div_term)
        pe[:, :, 1::2] = torch.cos(positions * self.div_term)

        return pe

class EncoderLayer(nn.Module):
    """Single encoder layer in the transformer model."""

    def __init__(self, config: TransformerConfig):
        """
        Single encoder layer in the transformer model.

        Args:
            config (TransformerConfig): Configuration class for transformer models.
        """

        super().__init__()

        self.self_attn = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            mask (Optional[torch.Tensor], optional): Mask tensor. Defaults to None.
        
        Returns:
            torch.Tensor: Output tensor.
        """

        attention_output = x + self.dropout(self.self_attn(x, x, x, mask))
        attention_output_norm = self.layer_norm1(attention_output)

        ff = attention_output_norm + self.feedforward(attention_output_norm)
        ff_norm = self.layer_norm2(ff)

        return ff_norm

