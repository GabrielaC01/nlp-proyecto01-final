import pytest
import torch
from src.config import config
from src.modelo_seq2seq import TransformerSeq2Seq

@pytest.fixture
def modelo_ligero():
    return TransformerSeq2Seq(config)

@pytest.fixture
def dummy_src():
    return torch.tensor([[3, 4, 5]])

@pytest.fixture
def sos_token():
    return config["sos_token"]

@pytest.fixture
def eos_token():
    return config["eos_token"]
