import torch
from torch import Tensor
from typing import List
import logging

logger = logging.getLogger(__name__)

class TopPSampler:
    """
    Decodificador Top-p (nucleus sampling) para modelos Seq2Seq

    Selecciona aleatoriamente entre los tokens cuya probabilidad acumulada alcanza al menos p
    """

    def __init__(self, model, sos_token: int, eos_token: int, p: float = 0.9, max_len: int = 50):
        """
        Inicializa el decodificador

        Args:
            model: Modelo Seq2Seq ya cargado
            sos_token (int): Token <SOS>
            eos_token (int): Token <EOS>
            p (float): Umbral de probabilidad acumulada
            max_len (int): Longitud máxima de la secuencia generada
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.p = p
        self.max_len = max_len

        logger.info("TopPSampler inicializado con p = %.2f y max_len = %d", p, max_len)

    def decode(self, src: Tensor) -> List[int]:
        """
        Genera una secuencia usando nucleus sampling (top-p)

        Args:
            src (Tensor): Secuencia fuente [batch_size, seq_len]

        Returns:
            List[int]: Lista de tokens generados
        """
        device = src.device
        batch_size = src.size(0)

        with torch.no_grad():
            memory = self.model.encode(src)

        generated = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)

        for _ in range(self.max_len):
            with torch.no_grad():
                output = self.model.decode(generated, memory)
                logits = output[:, -1, :]
                probs = torch.softmax(logits, dim=-1)

                sorted_probs, sorted_tokens = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Seleccionar tokens cuya probabilidad acumulada sea <= p
                cutoff = cumulative_probs >= self.p
                if torch.any(cutoff):
                    last_idx = torch.nonzero(cutoff, as_tuple=True)[1][0] + 1
                    sorted_probs = sorted_probs[:, :last_idx]
                    sorted_tokens = sorted_tokens[:, :last_idx]

                # Sampling entre los tokens seleccionados
                next_token = sorted_tokens.gather(
                    -1,
                    torch.multinomial(sorted_probs, num_samples=1)
                )

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == self.eos_token).all():
                break

        return generated[0].tolist()[1:]  # Excluir <SOS>
