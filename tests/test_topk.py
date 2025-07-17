import torch
from src.decoders.topk import TopKSampler

def test_output_shape(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que TopKSampler genera una lista de longitud esperada
    cuando el modelo no está entrenado
    """

    # Configuración básica
    max_len = 5

    # Decoder
    decoder = TopKSampler(
        model=modelo_ligero,
        sos_token=sos_token,
        eos_token=eos_token,
        k=5,
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
    Verifica que TopKSampler detiene la generación si genera <EOS>
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
            return torch.zeros((src.size(0), src.size(1), 32))  # Simula memoria del encoder

    sos_token = 1
    eos_token = 2

    model = DummyModel()
    src = torch.tensor([[3, 4, 5]])

    decoder = TopKSampler(
        model=model,
        sos_token=sos_token,
        eos_token=eos_token,
        k=5,
        max_len=10
    )

    output = decoder.decode(src)

    # Verifica que se generó <EOS> y que se detuvo luego de él
    assert eos_token in output, "No generó el token <EOS>"
    assert output[-1] == eos_token, "No se detuvo tras generar <EOS>"
    

def test_determinismo(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que TopKSampler produce salidas deterministas con la misma seed
    """
    decoder = TopKSampler(
        model=modelo_ligero,
        sos_token=sos_token,
        eos_token=eos_token,
        k=5,
        max_len=10
    )
    torch.manual_seed(0)
    salida1 = decoder.decode(dummy_src)

    torch.manual_seed(0)
    salida2 = decoder.decode(dummy_src)

    assert salida1 == salida2, "El decoder no es determinista con la seed 0"

def test_con_seed(modelo_ligero, dummy_src, sos_token, eos_token):
    """
    Verifica que TopKSampler produce salidas deterministas cuando se le pasa un seed explícito
    """
    decoder = TopKSampler(
        model=modelo_ligero,
        sos_token=sos_token,
        eos_token=eos_token,
        k=5,
        max_len=10,
        seed=123
    )

    salida1 = decoder.decode(dummy_src)
    salida2 = decoder.decode(dummy_src)

    assert salida1 == salida2, "El decoder no es determinista con el seed explícito"
