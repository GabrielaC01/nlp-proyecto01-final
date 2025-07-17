import time
import logging
import psutil
from typing import Callable, Tuple, List
from torchmetrics.text.bleu import BLEUScore

logger = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """
    Decorador para medir el tiempo de ejecución de una función

    Args:
        func (Callable): función a cronometrar

    Returns:
        Callable: función decorada
    """
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        duracion = fin - inicio
        #logger.info(f"Tiempo de ejecución de {func.__name__}: {duracion:.4f} segundos")
        return resultado, duracion
    return wrapper


def memory_usage() -> float:
    """
    Obtiene el uso de memoria actual del proceso en MB

    Returns:
        float: memoria en megabytes
    """
    proceso = psutil.Process()
    memoria_bytes = proceso.memory_info().rss
    memoria_mb = memoria_bytes / (1024 ** 2)
    logger.info(f"Uso de memoria: {memoria_mb:.2f} MB")
    return memoria_mb


def calcular_bleu(hypotheses: List[str], referencias: List[str]) -> float:
    """
    Calcula el BLEU score promedio entre hipótesis y referencias

    Args:
        hypotheses (List[str]): oraciones generadas
        referencias (List[str]): oraciones de referencia

    Returns:
        float: BLEU promedio
    """
    bleu = BLEUScore(n_gram=4, smooth=True) 

    score = bleu(hypotheses, [[ref] for ref in referencias]).item()    
    logger.info(f"BLEU: {score:.4f}")

    return score
