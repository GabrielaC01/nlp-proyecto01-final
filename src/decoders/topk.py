import torch
from torch import Tensor
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TopKSampler:
    """
    Decodificador Top-k Sampling para modelos Seq2Seq

    En cada paso selecciona aleatoriamente uno de los k tokens más probables
    """

    def __init__(self, model, sos_token: int, eos_token: int, k: int = 5, max_len: int = 50, seed: Optional[int] = None):
        """
        Inicializa el decodificador

        Args:
            model: Modelo Seq2Seq ya cargado
            sos_token (int): Token <SOS>
            eos_token (int): Token <EOS>
            k (int): Número de tokens candidatos
            max_len (int): Longitud máxima de la secuencia generada
            seed (int, opcional): Semilla para aleatoriedad
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.k = k
        self.max_len = max_len
        self.seed = seed

        logger.info("TopKSampler inicializado con k = %d y max_len = %d", k, max_len)

    def decode(self, src: Tensor) -> List[int]:
        """
        Genera una secuencia usando sampling con top-k

        Args:
            src (Tensor): Secuencia fuente [batch_size, seq_len]

        Returns:
            List[int]: Lista de tokens generados
        """

        if self.seed is not None:
            torch.manual_seed(self.seed)

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

                topk_probs, topk_tokens = torch.topk(probs, self.k, dim=-1)

                # Sampling entre los k tokens más probables
                next_token = topk_tokens.gather(
                    -1,
                    torch.multinomial(topk_probs, num_samples=1)
                )

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == self.eos_token).all():
                break

        return generated[0].tolist()[1:]  # Excluir <SOS>
