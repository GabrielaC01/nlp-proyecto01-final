"""
Configuración del modelo Transformer Seq2Seq
"""

config = {
    "vocab_size": 5000,         # Tamaño del vocabulario
    "embed_dim": 128,           # Tamaño de los embeddings
    "num_heads": 4,             # Número de cabeceras de atención
    "num_layers": 2,            # Capas del encoder y decoder
    "ff_dim": 256,              # Dimensión del feedforward
    "dropout": 0.1,             # Dropout regularización
    "max_len": 100,             # Longitud máxima de las secuencias
    "lr": 1e-3,                 # Learning rate
    "batch_size": 32,           # Tamaño de batch
    "epochs": 5,                # Número de épocas de entrenamiento
    "sos_token": 1,             # Token <SOS>
    "eos_token": 2              # Token <EOS>
}
