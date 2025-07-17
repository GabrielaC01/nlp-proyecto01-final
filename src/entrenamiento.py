import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import guardar_vocabulario

from datos import cargar_dataset, construir_vocabulario, crear_dataloader
from modelo_seq2seq import TransformerSeq2Seq
from config import config

def evaluar(modelo, val_loader, loss_fn, dispositivo):
    """
    Evalúa el modelo sobre el conjunto de validación    

    Args:
        modelo: modelo Seq2Seq a evaluar
        val_loader: dataloader con ejemplos de validación
        loss_fn: función de pérdida
        dispositivo: CPU o GPU

    Returns:
        Pérdida promedio sobre el conjunto de validación
    """
    modelo.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(dispositivo), tgt.to(dispositivo)
            logits = modelo(src, tgt[:, :-1])
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar datos y construir vocabulario
    total = config["num_ejemplos_entrenamiento"] + config["num_ejemplos"]
    pares = cargar_dataset(num_ejemplos=total)

    vocab, vocab_inv = construir_vocabulario(pares)
    vocab_size = len(vocab) + 1

    # GUARDAR vocab y vocab_inv
    guardar_vocabulario(vocab, vocab_inv)
    print("Vocabulario guardado en checkpoints")

    # Separar el dataset 
    train_pares = pares[:config["num_ejemplos_entrenamiento"]]
    val_pares = pares[config["num_ejemplos_entrenamiento"]:]
    train_loader = crear_dataloader(train_pares, vocab, batch_size=config["batch_size"], max_len=config["max_len"])
    val_loader = crear_dataloader(val_pares, vocab, batch_size=config["batch_size"], max_len=config["max_len"])

    # Inicializar modelo
    modelo_config = config.copy()
    modelo_config["vocab_size"] = vocab_size
    modelo = TransformerSeq2Seq(modelo_config).to(dispositivo)

    # Definir optimizador y función de pérdida
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    criterio = nn.CrossEntropyLoss(ignore_index=0)

    mejor_val_loss = float("inf")

    # Entrenamiento por épocas
    for epoca in range(config["epochs"]):
        modelo.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(dispositivo), tgt.to(dispositivo)

            # Separar entrada y objetivo
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Calcular logits y pérdida
            logits = modelo(src, tgt_input)
            loss = criterio(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))

            # Propagación hacia atrás y actualización de pesos
            optimizador.zero_grad()
            loss.backward()
            optimizador.step()

            total_loss += loss.item()

        # Calcular métricas en validación
        loss_promedio = total_loss / len(train_loader)
        val_loss = evaluar(modelo, val_loader, criterio, dispositivo)
        val_perplexity = torch.exp(torch.tensor(val_loss))

        print(f"Época {epoca + 1}/{config['epochs']} - Loss entrenamiento: {loss_promedio:.4f} - Loss validación: {val_loss:.4f} - Perplejidad: {val_perplexity:.2f}")

        # Guardar el mejor modelo según validación
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save(modelo.state_dict(), config["modelo_guardado"])
            print(f"Mejor modelo guardado en {config['modelo_guardado']} durante la época {epoca + 1} con PPL={val_perplexity:.2f}")
