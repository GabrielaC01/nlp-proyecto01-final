import torch
from src.decoders.beam import BeamSearchDecoder
from src.decoders.greedy import GreedyDecoder

def test_output_shape(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que BeamSearchDecoder genera una lista
    de longitud esperada cuando el modelo no está entrenado
    """

    # Configuración básica
    beam_width = 3
    max_len = 20

    # Decoder
    decoder = BeamSearchDecoder(
        model=modelo_ligero,
        sos_token=sos_token,
        eos_token=eos_token,
        beam_width=beam_width,
        max_len=max_len
    )

    # Ejecutar decodificación
    output = decoder.decode(dummy_src)

    assert isinstance(output, list), "La salida no es una lista"
    assert len(output) <= max_len, "La secuencia generada supera la longitud máxima"
    assert all(isinstance(token, int) for token in output), "La secuencia contiene elementos no enteros"

def test_stops_on_eos():
    """
    Verifica que BeamSearchDecoder detiene la generación si genera <EOS>
    """

    class DummyModel(torch.nn.Module):
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

    model = DummyModel()
    src = torch.tensor([[3, 4, 5]])

    decoder = BeamSearchDecoder(model=model, sos_token=sos_token, eos_token=eos_token,
                                 beam_width=2, max_len=10)

    output = decoder.decode(src)

    assert len(output) == 1, "No se detuvo tras generar <EOS>"
    assert output[0] == eos_token, "No generó el token <EOS>"

def test_determinismo(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que BeamSearchDecoder produce salidas deterministas con la misma seed
    """
    decoder = BeamSearchDecoder(model=modelo_ligero, sos_token=sos_token, eos_token=eos_token,
                                 beam_width=2, max_len=10)

    torch.manual_seed(0)
    salida1 = decoder.decode(dummy_src)

    torch.manual_seed(0)
    salida2 = decoder.decode(dummy_src)

    assert salida1 == salida2, "El decoder no es determinista con la seed 0"

def test_beam_equivale_a_greedy(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que BeamSearchDecoder con beam_width=1 es equivalente a GreedyDecoder
    """
    greedy = GreedyDecoder(modelo_ligero, sos_token, eos_token, max_len=10)
    beam = BeamSearchDecoder(modelo_ligero, sos_token, eos_token, beam_width=1, max_len=10)

    torch.manual_seed(0)
    salida_greedy = greedy.decode(dummy_src)

    torch.manual_seed(0)
    salida_beam = beam.decode(dummy_src)

    assert salida_greedy == salida_beam, "Beam width=1 no es equivalente a Greedy"
