import torch
from torch import Tensor
from typing import List, Tuple
import heapq
import logging

logger = logging.getLogger(__name__)

class DiverseBeamDecoder:
    """
    Decodificador Diverse Beam Search para modelos Seq2Seq

    Introduce diversidad penalizando secuencias similares dentro del beam
    """

    def __init__(self, model, sos_token: int, eos_token: int,
                 beam_width: int = 3, diversity_strength: float = 0.5, max_len: int = 50):
        """
        Inicializa el decodificador

        Args:
            model: Modelo Seq2Seq ya cargado
            sos_token (int): Token <SOS>
            eos_token (int): Token <EOS>
            beam_width (int): Tamaño del haz
            diversity_strength (float): Penalización por repetición
            max_len (int): Longitud máxima de la secuencia generada
        """
        self.model = model
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.beam_width = beam_width
        self.diversity_strength = diversity_strength
        self.max_len = max_len

        logger.info("DiverseBeamDecoder inicializado con beam_width = %d, diversity_strength = %.2f, max_len = %d",
                    beam_width, diversity_strength, max_len)

    def decode(self, src: Tensor) -> List[int]:
        """
        Genera una secuencia usando diverse beam search

        Args:
            src (Tensor): Secuencia fuente [batch_size, seq_len]

        Returns:
            List[int]: Mejor secuencia generada (tokens)
        """
        device = src.device
        batch_size = src.size(0)

        with torch.no_grad():
            memory = self.model.encode(src)

        # Beam: lista de (score, secuencia generada)
        beams: List[Tuple[float, List[int]]] = [(0.0, [self.sos_token])]

        for _ in range(self.max_len):
            all_candidates = []

            for score, seq in beams:
                if seq[-1] == self.eos_token:
                    all_candidates.append((score, seq))
                    continue

                input_seq = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

                with torch.no_grad():
                    output = self.model.decode(input_seq, memory)
                    logits = output[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)

                topk = torch.topk(log_probs, self.beam_width, dim=-1)
                topk_scores = topk.values.squeeze(0)
                topk_tokens = topk.indices.squeeze(0)

                # Penalizar tokens que ya hayan aparecido en la secuencia
                for i in range(self.beam_width):
                    next_token = topk_tokens[i].item()
                    repetition_penalty = self.diversity_strength * seq.count(next_token)
                    next_score = score + topk_scores[i].item() - repetition_penalty
                    candidate = (next_score, seq + [next_token])
                    all_candidates.append(candidate)

            # Mantener los mejores beams
            beams = heapq.nlargest(self.beam_width, all_candidates, key=lambda x: x[0])

            if all(seq[-1] == self.eos_token for _, seq in beams):
                break

        # Retornar la mejor secuencia (sin <SOS>)
        best_seq = beams[0][1]
        return best_seq[1:]
