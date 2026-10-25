import pathlib

import pytest
import torch

from src.cells import CIFGCell, LSTMPCell, RecurrentLayer, VanillaLSTMCell
from src.model import SentimentClassifier, count_params

D, H, P, B = 16, 32, 8, 5


def make(name):
    return {"vanilla": VanillaLSTMCell(D, H), "cifg": CIFGCell(D, H), "lstmp": LSTMPCell(D, H, P)}[name]


@pytest.mark.parametrize("name,h_dim", [("vanilla", H), ("cifg", H), ("lstmp", P)])
def test_output_shapes(name, h_dim):
    cell = make(name)
    x = torch.randn(B, D)
    out, (h, c) = cell(x, cell.init_state(B, x.device))
    assert out.shape == (B, h_dim) and h.shape == (B, h_dim) and c.shape == (B, H)
    assert torch.equal(out, h)


@pytest.mark.parametrize("name,expected", [
    ("vanilla", 4 * H * (D + H + 1)),           # 4 gates: W_x, W_h, one bias each
    ("cifg", 3 * H * (D + H + 1)),              # no input gate
    ("lstmp", 4 * H * (D + P + 1) + P * H + P),  # recurrent weights act on P-dim h, plus W_p, b_p
])
def test_param_count(name, expected):
    assert count_params(make(name)) == expected


def test_cifg_input_gate_is_one_minus_forget():
    cell = CIFGCell(D, H)
    x = torch.randn(B, D)
    _, _, gates = cell(x, (torch.randn(B, H), torch.randn(B, H)), return_gates=True)
    assert torch.allclose(gates["i"], 1 - gates["f"])


def test_vanilla_matches_equations():
    cell = VanillaLSTMCell(D, H)
    x, h0, c0 = torch.randn(B, D), torch.randn(B, H), torch.randn(B, H)
    z = cell.x2h(x) + cell.h2h(h0)
    i, f, g, o = z.chunk(4, -1)
    c = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)
    h = torch.sigmoid(o) * torch.tanh(c)
    out, (_, c_out) = cell(x, (h0, c0))
    assert torch.allclose(out, h) and torch.allclose(c_out, c)


@pytest.mark.parametrize("cell_type", ["vanilla", "cifg", "lstmp"])
def test_padding_uses_true_last_token(cell_type):
    torch.manual_seed(0)
    layer = RecurrentLayer(cell_type, D, H, num_layers=2, proj_size=P).eval()
    x = torch.randn(1, 4, D)
    _, last_short = layer(x[:, :2], torch.tensor([2]))
    padded = torch.cat([x[:, :2], torch.randn(1, 5, D)], dim=1)
    _, last_padded = layer(padded, torch.tensor([2]))
    assert torch.allclose(last_short, last_padded, atol=1e-6)


def test_model_forward_backward():
    m = SentimentClassifier(100, "lstmp", embed_dim=D, hidden_size=H, proj_size=P, num_layers=2)
    x = torch.randint(1, 100, (B, 7))
    logits = m(x, torch.tensor([7, 3, 5, 1, 2]))
    assert logits.shape == (B, 2)
    logits.sum().backward()


def test_no_builtin_recurrent_modules():
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "nn.LSTM" not in text, f
        assert "nn.RNN" not in text, f
        assert "nn.GRU" not in text, f
