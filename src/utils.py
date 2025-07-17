import pickle
import os

def guardar_vocabulario(vocab, vocab_inv, ruta="checkpoints"):
    """
    Guarda el vocabulario y su versión invertida como archivos pickle

    Args:
        vocab (dict): Mapeo palabra → índice
        vocab_inv (dict): Mapeo índice → palabra
        ruta (str): Carpeta destino
    """
    os.makedirs(ruta, exist_ok=True)
    with open(f"{ruta}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{ruta}/vocab_inv.pkl", "wb") as f:
        pickle.dump(vocab_inv, f)

def cargar_vocabulario(vocab_path="checkpoints/vocab.pkl", vocab_inv_path="checkpoints/vocab_inv.pkl"):
    """
    Carga el vocabulario y su versión invertida desde disco

    Args:
        vocab_path (str): ruta del archivo pickle del vocabulario
        vocab_inv_path (str): ruta del archivo pickle del vocabulario invertido

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: vocab y vocab_inv
    """
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(vocab_inv_path, "rb") as f:
        vocab_inv = pickle.load(f)
    return vocab, vocab_inv
