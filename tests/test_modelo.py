import torch
from src.modelo_seq2seq import TransformerSeq2Seq


def test_modelo_forward_shape():
    """
    Verifica que el modelo TransformerSeq2Seq produce salidas
    con la forma esperada cuando recibe entradas dummy
    """
    vocab_size = 100
    batch_size = 2
    seq_len = 5

    model = TransformerSeq2Seq(vocab_size=vocab_size, d_model=32, nhead=4,
                                num_encoder_layers=2, num_decoder_layers=2,
                                dim_feedforward=64)

    # Secuencias fuente y destino dummy
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward
    output = model(src, tgt)

    # Verificar dimensiones [batch_size, seq_len, vocab_size]
    assert output.shape == (batch_size, seq_len, vocab_size), \
        f"La forma de salida {output.shape} no es la esperada {(batch_size, seq_len, vocab_size)}"


def test_modelo_encode_decode_independiente():
    """
    Verifica que las funciones encode y decode funcionan por separado
    """
    vocab_size = 100
    batch_size = 1
    seq_len = 4

    model = TransformerSeq2Seq(vocab_size=vocab_size, d_model=32, nhead=4,
                                num_encoder_layers=2, num_decoder_layers=2,
                                dim_feedforward=64)

    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))

    memory = model.encode(src)
    output = model.decode(tgt, memory)

    # Verificar dimensiones [batch_size, seq_len, vocab_size]
    assert output.shape == (batch_size, seq_len, vocab_size), \
        f"La forma de decode {output.shape} no es la esperada {(batch_size, seq_len, vocab_size)}"
