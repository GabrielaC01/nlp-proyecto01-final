import torch
from src.modelo_seq2seq import TransformerSeq2Seq
from src.decoders.beam import BeamSearchDecoder


def test_beam_decoder_output_shape():
    """
    Verifica que BeamSearchDecoder genera una lista
    de longitud esperada cuando el modelo no está entrenado
    """
    vocab_size = 100
    sos_token = 1
    eos_token = 2
    max_len = 5

    model = TransformerSeq2Seq(vocab_size=vocab_size, d_model=32, nhead=4,
                                num_encoder_layers=2, num_decoder_layers=2,
                                dim_feedforward=64)

    src = torch.tensor([[3, 4, 5]])

    decoder = BeamSearchDecoder(model=model, sos_token=sos_token, eos_token=eos_token,
                                 beam_width=3, max_len=max_len)

    output = decoder.decode(src)

    assert isinstance(output, list), "La salida no es una lista"
    assert len(output) <= max_len, "La secuencia generada supera la longitud máxima"
    assert all(isinstance(token, int) for token in output), "La secuencia contiene elementos no enteros"


def test_beam_decoder_stops_on_eos():
    """
    Verifica que BeamSearchDecoder detiene la generación si genera <EOS>
    """

    class DummyModel(TransformerSeq2Seq):
        def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
            # Devuelve logits donde siempre <EOS> (token 2) tiene máxima probabilidad
            batch_size = tgt.size(0)
            vocab_size = 100
            logits = torch.zeros(batch_size, tgt.size(1), vocab_size)
            logits[:, -1, 2] = 1.0  # Token <EOS> con máxima probabilidad
            return logits

        def encode(self, src, src_mask=None, src_padding_mask=None):
            return torch.zeros_like(src, dtype=torch.float32).unsqueeze(-1)

    sos_token = 1
    eos_token = 2

    model = DummyModel(vocab_size=100)

    src = torch.tensor([[3, 4, 5]])

    decoder = BeamSearchDecoder(model=model, sos_token=sos_token, eos_token=eos_token,
                                 beam_width=2, max_len=10)

    output = decoder.decode(src)

    # Como siempre genera EOS, debe detenerse rápido
    assert len(output) == 1, "No se detuvo tras generar <EOS>"
    assert output[0] == eos_token, "No generó el token <EOS>"
