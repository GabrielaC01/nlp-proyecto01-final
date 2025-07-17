from datasets import load_dataset
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

def tokenizar(texto: str) -> List[str]:
    """
    Limpia y tokeniza el texto respetando tildes y signos importantes

    Args:
        texto (str): texto original

    Returns:
        List[str]: lista de tokens limpios
    """
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"\s+", " ", texto)  # normaliza espacios
    tokens = re.findall(r"\w+|[¿¡?.!,;:]", texto)  # palabras y signos
    return tokens

def cargar_dataset(num_ejemplos: int = 5000, offset: int = 0) -> List[Tuple[str, str]]:
    """
    Carga un subconjunto del dataset opus_books EN-ES

    Args:
        num_ejemplos (int): número de pares a cargar

    Returns:
        List[Tuple[str, str]]: lista de pares (inglés, español)
    """
    logger.info(f"Cargando {num_ejemplos} ejemplos del dataset opus_books EN-ES")
    dataset = load_dataset("opus_books", "en-es")
    datos = dataset["train"].select(range(offset, offset + num_ejemplos))
    pares = [(ej["translation"]["en"], ej["translation"]["es"]) for ej in datos]
    logger.info(f"Se cargaron {len(pares)} pares EN-ES")
    return pares


def construir_vocabulario(pares: List[Tuple[str, str]], max_vocab: int = 10000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Construye el vocabulario y su versión invertida a partir de los pares de oraciones

    Args:
        pares (List[Tuple[str, str]]): oraciones EN-ES
        max_vocab (int): tamaño máximo del vocabulario

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: (vocab, vocab_inv)
    """
    contador = Counter()

    for en, es in pares:
        for oracion in [en, es]:
            contador.update(tokenizar(oracion))

    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for idx, (palabra, _) in enumerate(contador.most_common(max_vocab - len(vocab)), start=4):
        vocab[palabra] = idx

    vocab_inv = {idx: token for token, idx in vocab.items()}

    logger.info(f"Vocabulario construido con {len(vocab)} palabras")
    return vocab, vocab_inv


def codificar(texto: str, vocab: Dict[str, int], max_len: int = 256) -> torch.Tensor:
    """
    Convierte una oración en un tensor de índices

    Args:
        texto (str): oración original
        vocab (Dict[str, int]): vocabulario palabra → índice
        max_len (int): longitud máxima permitida

    Returns:
        Tensor: secuencia de índices incluyendo <SOS> y <EOS>
    """
    tokens = tokenizar(texto)
    if len(tokens) + 2 > max_len:
        tokens = tokens[:max_len - 2]  # dejar espacio para <SOS> y <EOS>
    indices = [vocab.get("<SOS>")] + [vocab.get(p, vocab.get("<UNK>")) for p in tokens] + [vocab.get("<EOS>")]
    return torch.tensor(indices, dtype=torch.long)


class ParesDataset(Dataset):
    """
    Dataset de pares codificados de oraciones EN-ES

    Args:
        pares (List[Tuple[str, str]]): lista de pares (inglés, español)
        vocab (Dict[str, int]): vocabulario palabra → índice
    """

    def __init__(self, pares: List[Tuple[str, str]], vocab: Dict[str, int], max_len: int = 256):
        self.vocab = vocab
        self.max_len = max_len
        self.pares = [(en, es) for en, es in pares if self._es_valido(en) and self._es_valido(es)]

    def _es_valido(self, texto: str) -> bool:
        """
        Verifica que la longitud no supere el máximo
        """
        return len(tokenizar(texto)) + 2 <= self.max_len

    def __len__(self):
        """
        Devuelve el número de ejemplos en el dataset
        """
        return len(self.pares)

    def __getitem__(self, idx):
        """
        Devuelve el par (src, tgt) codificado como tensores

        Args:
            idx (int): índice del ejemplo

        Returns:
            Tuple[Tensor, Tensor]: (src_tensor, tgt_tensor)
        """
        en, es = self.pares[idx]
        src_ids = codificar(en, self.vocab, self.max_len)
        tgt_ids = codificar(es, self.vocab, self.max_len)
        return src_ids, tgt_ids


def collate_batch(batch, pad_idx: int = 0):
    """
    Aplica padding a los tensores del batch

    Args:
        batch (List[Tuple[Tensor, Tensor]]): lote de pares (src, tgt)
        pad_idx (int): índice del token de padding

    Returns:
        Tuple[Tensor, Tensor]: (src_batch, tgt_batch) con padding aplicado
    """
    src_batch, tgt_batch = zip(*batch)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_pad, tgt_pad


def crear_dataloader(pares: List[Tuple[str, str]], vocab: Dict[str, int], batch_size: int = 32, max_len: int = 256) -> DataLoader:
    """
    Crea un DataLoader para entrenamiento

    Args:
        pares (List[Tuple[str, str]]): lista de oraciones EN-ES
        vocab (Dict[str, int]): vocabulario palabra → índice
        batch_size (int): tamaño del batch
        max_len (int): longitud máxima permitida

    Returns:
        DataLoader: generador de lotes para entrenamiento
    """
    dataset = ParesDataset(pares, vocab, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=lambda b: collate_batch(b, pad_idx=0))
