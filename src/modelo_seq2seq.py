# src/modelo_seq2seq.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import logging

# Configura logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal para entradas de un Transformer.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model (int): Dimensión del embedding.
            max_len (int): Longitud máxima de la secuencia.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Suma la codificación posicional a la entrada.

        Args:
            x (Tensor): Tensor de entrada [batch_size, seq_len, d_model].

        Returns:
            Tensor: Entrada con codificación posicional sumada.
        """
        return x + self.pe[:, :x.size(1)]


class TransformerSeq2Seq(nn.Module):
    """
    Modelo Transformer Seq2Seq para traducción automática.
    """

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Inicializa el modelo Transformer Seq2Seq.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            d_model (int): Dimensión del embedding.
            nhead (int): Número de cabeceras de atención.
            num_encoder_layers (int): Cantidad de capas encoder.
            num_decoder_layers (int): Cantidad de capas decoder.
            dim_feedforward (int): Dimensión de la capa feedforward.
            dropout (float): Tasa de dropout.
        """
        super().__init__()

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer completo
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)

        # Proyección final a vocabulario
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Parámetros
        self.d_model = d_model

        logger.info("Transformer Seq2Seq inicializado con %d capas encoder/decoder", num_encoder_layers)

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None,
               src_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Codifica la secuencia de entrada.

        Args:
            src (Tensor): Secuencia fuente [batch_size, seq_len].
            src_mask (Tensor, opcional): Máscara de atención.
            src_padding_mask (Tensor, opcional): Máscara de padding.

        Returns:
            Tensor: Representación codificada [batch_size, seq_len, d_model].
        """
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        embedded = self.positional_encoding(embedded)
        memory = self.transformer.encoder(embedded,
                                          mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        return memory

    def decode(self, tgt: Tensor, memory: Tensor,
               tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Decodifica la secuencia destino a partir de la memoria del encoder.

        Args:
            tgt (Tensor): Secuencia objetivo [batch_size, seq_len].
            memory (Tensor): Memoria generada por el encoder.
            tgt_mask (Tensor, opcional): Máscara de atención.
            tgt_padding_mask (Tensor, opcional): Máscara de padding.

        Returns:
            Tensor: Logits [batch_size, seq_len, vocab_size].
        """ 
        embedded = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        embedded = self.positional_encoding(embedded)
        output = self.transformer.decoder(embedded, memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(output)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                src_padding_mask: Optional[Tensor] = None, tgt_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward completo del modelo Seq2Seq.

        Args:
            src (Tensor): Secuencia fuente.
            tgt (Tensor): Secuencia objetivo.
            src_mask (Tensor, opcional): Máscara fuente.
            tgt_mask (Tensor, opcional): Máscara objetivo.
            src_padding_mask (Tensor, opcional): Padding fuente.
            tgt_padding_mask (Tensor, opcional): Padding objetivo.

        Returns:
            Tensor: Logits [batch_size, seq_len, vocab_size].
        """
        memory = self.encode(src, src_mask, src_padding_mask)
        output = self.decode(tgt, memory, tgt_mask, tgt_padding_mask)
        return output
