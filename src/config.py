"""
Parámetros de configuración
"""

config = {
    "vocab_size": 10000,         # Tamaño del vocabulario
    "embed_dim": 128,           # Tamaño de los embeddings
    "num_heads": 4,             # Número de cabeceras de atención
    "num_layers": 4,            # Capas del encoder y decoder
    "ff_dim": 256,              # Dimensión del feedforward 
    "dropout": 0.1,             # Dropout regularización
    "max_len": 128,            # Longitud máxima de las secuencias
    "lr": 3e-4,                 # Learning rate
    "batch_size": 8,           # Tamaño de batch
    "epochs": 30,               # Número de épocas de entrenamiento
    "sos_token": 1,             # Token <SOS>
    "eos_token": 2,             # Token <EOS>
    "num_ejemplos_entrenamiento": 10000,         # Entrenamiento
    "num_ejemplos": 1000,       
    "modelo_guardado": "checkpoints/best_modelo.pth",  # Ruta para guardar el modelo
    "csv_bleu_vs_latencia": "benchmarks/results/bleu_vs_latency.csv",
    "csv_memoria_vs_len": "benchmarks/results/memoria_vs_len.csv"
}
