import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def generar_mask_causal(size: int) -> Tensor:
    """
    Crea una máscara triangular inferior para el decoder (máscara causal)

    Args:
        size (int): longitud de la secuencia

    Returns:
        Tensor: máscara [size, size] booleana (True para posiciones enmascaradas)
    """
    return torch.triu(torch.ones(size, size), diagonal=1).bool()

class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal para entradas de un Transformer
    """

    def __init__(self, d_model: int, max_len: int = 5000):
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
        Suma la codificación posicional a la entrada

        Args:
            x (Tensor): Entrada [batch_size, seq_len, d_model]

        Returns:
            Tensor: Entrada con codificación posicional
        """
        return x + self.pe[:, :x.size(1)]


class TransformerSeq2Seq(nn.Module):
    """
    Modelo Transformer Seq2Seq para traducción automática
    """

    def __init__(self, config: dict):
        """
        Inicializa el modelo con configuración externa

        Args:
            config (dict): Diccionario con hiperparámetros del modelo
        """
        super().__init__()

        self.sos_token = config["sos_token"]
        self.eos_token = config["eos_token"]
        self.d_model = config["embed_dim"]

        # Escalado precomputado (√d_model)
        self.scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # Embedding y codificación posicional
        self.embedding = nn.Embedding(config["vocab_size"], self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, config["max_len"])

        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=config["num_heads"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"],
            dim_feedforward=config["ff_dim"],
            dropout=config["dropout"],
            batch_first=True
        )

        # Proyección final al vocabulario
        self.fc_out = nn.Linear(self.d_model, config["vocab_size"])

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
        embedded = self.embedding(src) * self.scale
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
        embedded = self.embedding(tgt) * self.scale 
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
        if tgt_mask is None:
            tgt_mask = generar_mask_causal(tgt.size(1)).to(tgt.device)

        memory = self.encode(src, src_mask, src_padding_mask)
        output = self.decode(tgt, memory, tgt_mask, tgt_padding_mask)
        return output