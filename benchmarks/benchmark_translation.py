import time
import torch
from datasets import load_dataset
from torchmetrics.text.bleu import BLEUScore

from src.modelo_seq2seq import TransformerSeq2Seq
from src.decoders.greedy import GreedyDecoder
from src.decoders.beam import BeamSearchDecoder


# Cargar el dataset real Tatoeba EN-ES desde Hugging Face
dataset = load_dataset("opus_books", "en-es")
pares = dataset["train"][:50]  # Se usan solo 50 pares para prueba rápida

# Tokenización simple
def tokenizar(texto: str) -> list:
    return texto.lower().split()

# Construir vocabulario
vocabulario = {"<SOS>": 1, "<EOS>": 2}
indice_actual = 3
for par in pares["translation"]:
    for oracion in [par["en"], par["es"]]:
        for palabra in tokenizar(oracion):
            if palabra not in vocabulario:
                vocabulario[palabra] = indice_actual
                indice_actual += 1

# Función para convertir una oración a índices y añadir tokens especiales
def codificar(texto: str) -> list:
    return [vocabulario.get(palabra, 0) for palabra in tokenizar(texto)] + [2]  # 2 es <EOS>

# Parámetros generales
VOCAB_SIZE = len(vocabulario) + 1
SOS_TOKEN = 1
EOS_TOKEN = 2

# Crear modelo base
model = TransformerSeq2Seq(vocab_size=VOCAB_SIZE, d_model=32, nhead=4,
                           num_encoder_layers=2, num_decoder_layers=2,
                           dim_feedforward=64)

# Configurar decodificadores
decoders = {
    "greedy": GreedyDecoder(model=model, sos_token=SOS_TOKEN, eos_token=EOS_TOKEN, max_len=20),
    "beam": BeamSearchDecoder(model=model, sos_token=SOS_TOKEN, eos_token=EOS_TOKEN, beam_width=3, max_len=20)
}

# Inicializar métrica BLEU
bleu = BLEUScore(n_gram=4)

# Guardar resultados
results = {}

# Ejecutar benchmark
for decoder_name, decoder in decoders.items():
    total_bleu = 0.0
    start_time = time.time()

    for par in pares["translation"]:
        src_seq = codificar(par["en"])
        ref_seq = codificar(par["es"])

        src_tensor = torch.tensor([src_seq], dtype=torch.long)

        # Generar traducción
        pred_tokens = decoder.decode(src_tensor)

        # Convertir secuencias a texto para BLEU (igual que en clase)
        pred_sentence = [" ".join([str(tok) for tok in pred_tokens])]
        ref_sentence = [[" ".join([str(tok) for tok in ref_seq])]]

        # Calcular BLEU
        score = bleu(pred_sentence, ref_sentence)
        total_bleu += score.item()

    elapsed_time = time.time() - start_time
    avg_bleu = total_bleu / len(pares)

    results[decoder_name] = {
        "BLEU": round(avg_bleu, 4),
        "Tiempo (s)": round(elapsed_time, 4)
    }

# Mostrar resultados en consola
for decoder, metrics in results.items():
    print(f"[{decoder}] BLEU: {metrics['BLEU']}, Tiempo: {metrics['Tiempo (s)']}")
