import torch
from src.decoders.greedy import GreedyDecoder

def test_output_shape(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que GreedyDecoder genera una lista de longitud esperada
    cuando el modelo no está entrenado
    """

    # Configuración básica
    max_len = 5

    # Decoder
    decoder = GreedyDecoder(
        model=modelo_ligero,
        sos_token=sos_token,
        eos_token=eos_token,
        max_len=max_len
    )

    # Ejecutar decodificación
    output = decoder.decode(dummy_src)

    # Comprobar que la salida es una lista de tokens y que no excede el max_len
    assert isinstance(output, list), "La salida no es una lista"
    assert len(output) <= max_len, "La secuencia generada supera la longitud máxima"
    assert all(isinstance(token, int) for token in output), "La secuencia contiene elementos no enteros"

def test_stops_on_eos():
    """
    Verifica que el GreedyDecoder detiene la generación si produce un <EOS>
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

    decoder = GreedyDecoder(model=model, sos_token=sos_token, eos_token=eos_token, max_len=10)

    output = decoder.decode(src)
    
    assert len(output) == 1, "No se detuvo tras generar <EOS>"
    assert output[0] == eos_token, "No generó el token <EOS>"

def test_determinismo(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que GreedyDecoder produce salidas deterministas con la misma seed
    """
    decoder = GreedyDecoder(
        model=modelo_ligero,
        sos_token=sos_token,
        eos_token=eos_token,
        max_len=10
    )

    torch.manual_seed(0)
    salida1 = decoder.decode(dummy_src)

    torch.manual_seed(0)
    salida2 = decoder.decode(dummy_src)

    assert salida1 == salida2, "El decoder no es determinista con la seed 0"