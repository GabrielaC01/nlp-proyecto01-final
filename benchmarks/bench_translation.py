import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import logging
import pandas as pd
from src.datos import cargar_dataset, codificar
from src.modelo_seq2seq import TransformerSeq2Seq
from src.config import config
from src.decoders.greedy import GreedyDecoder
from src.decoders.beam import BeamSearchDecoder
from src.decoders.topk import TopKSampler
from src.decoders.topp import TopPSampler
from src.decoders.diverse_beam import DiverseBeamDecoder
from src.evaluation import calcular_bleu, measure_time, memory_usage
from src.config import config
from src.utils import cargar_vocabulario

logging.basicConfig(level=logging.INFO)

# Preparar datos
pares = cargar_dataset(num_ejemplos=config["num_ejemplos"], offset=config["num_ejemplos_entrenamiento"])

# Cargar vocabulario
vocab, vocab_inv = cargar_vocabulario()

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo
modelo_config = config.copy()
modelo_config["vocab_size"] = len(vocab) + 1
modelo = TransformerSeq2Seq(modelo_config).to(dispositivo)
modelo.load_state_dict(torch.load(config["modelo_guardado"], map_location=dispositivo))
modelo.eval()

# Instanciar decodificadores
decoders = {
    "greedy": GreedyDecoder(modelo, sos_token=config["sos_token"], eos_token=config["eos_token"]),
    "beam": BeamSearchDecoder(modelo, sos_token=config["sos_token"], eos_token=config["eos_token"], beam_width=3),
    "topk": TopKSampler(modelo, sos_token=config["sos_token"], eos_token=config["eos_token"], k=5),
    "topp": TopPSampler(modelo, sos_token=config["sos_token"], eos_token=config["eos_token"], p=0.9),
    "diverse_beam": DiverseBeamDecoder(modelo, sos_token=config["sos_token"], eos_token=config["eos_token"], beam_width=3, diversity_strength=0.5)
}

# Evaluaciones
resultados_bleu = []
resultados_memoria = []

for nombre, decodificador in decoders.items():
    logging.info(f"Ejecutando benchmark para {nombre}")
    predicciones = []
    referencias = []
    total_tiempo = 0.0

    for en, es in pares:
        entrada = codificar(en, vocab, config["max_len"]).unsqueeze(0).to(dispositivo)

        @measure_time
        def generar():
            return decodificador.decode(entrada)

        salida_tokens, duracion = generar()
        salida = " ".join([vocab_inv.get(tok, "<UNK>") for tok in salida_tokens])
        predicciones.append(salida)
        referencias.append(es)
        total_tiempo += duracion

    bleu = calcular_bleu(predicciones, referencias)
    mem = memory_usage()

    resultados_bleu.append({
        "decodificador": nombre,
        "BLEU": round(bleu, 4),
        "latencia_promedio_s": round(total_tiempo / len(pares), 4)
    })

    for en, _ in pares:
        resultados_memoria.append({
            "decodificador": nombre,
            "longitud": len(en.split()),
            "memoria_MB": round(mem, 2)
        })

# Guardar CSVs
df_bleu = pd.DataFrame(resultados_bleu)
df_bleu.to_csv(config["csv_bleu_vs_latencia"], index=False)

df_mem = pd.DataFrame(resultados_memoria)
df_mem.to_csv(config["csv_memoria_vs_len"], index=False)
