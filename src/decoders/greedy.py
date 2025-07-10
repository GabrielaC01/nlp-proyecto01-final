import torch
from torch import Tensor
from typing import List
import logging

logger = logging.getLogger(__name__)


class GreedyDecoder:
    """
    Decodificador Greedy para modelos Seq2Seq.

    En cada paso de generación, selecciona el token con mayor probabilidad.
    """

    def __init__(self, model, sos_token: int, eos_token: int, max_len: int = 50):
        """
        Inicializa el decodificador.

        Args:
            model: Modelo Seq2Seq ya cargado.
            sos_token (int): Token <SOS> (start-of-sequence).
            eos_token (int): Token <EOS> (end-of-sequence).
            max_len (int): Longitud máxima de la secuencia generada.
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_len = max_len

        logger.info("GreedyDecoder inicializado con max_len = %d", max_len)

    def decode(self, src: Tensor) -> List[int]:
        """
        Genera una secuencia de salida a partir de la entrada fuente.

        Args:
            src (Tensor): Secuencia fuente [batch_size, seq_len].

        Returns:
            List[int]: Lista de tokens generados.
        """
        device = src.device
        batch_size = src.size(0)

        # Codificar la entrada
        with torch.no_grad():
            memory = self.model.encode(src)

        # Inicializar con <SOS>
        generated = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)

        for _ in range(self.max_len):
            with torch.no_grad():
                output = self.model.decode(generated, memory)
                logits = output[:, -1, :]  # Último token
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy

            generated = torch.cat([generated, next_token], dim=1)

            # Detener si se genera <EOS>
            if (next_token == self.eos_token).all():
                break

        # Convertir a lista y excluir el token <SOS>
        return generated[0].tolist()[1:]
