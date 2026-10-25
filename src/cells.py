"""LSTM cell variants implemented from scratch with nn.Linear (no built-in recurrent modules).

Every cell follows the same interface:
    h_t, (h_t, c_t) = cell(x_t, (h_prev, c_prev))

For speed, gate pre-activations are fused: one nn.Linear maps x_t to all gate blocks and one
(bias-free) nn.Linear maps h_{t-1} to all gate blocks, so each gate has exactly one bias as in
the equations. ``step`` takes the already-projected input so that RecurrentLayer can compute
W_x x_t for all time steps with a single matmul; the math is identical.
"""
import torch
import torch.nn as nn


class _GatedCell(nn.Module):
    n_gates = 4

    def __init__(self, input_size, hidden_size, state_size=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # size of c_t
        self.state_size = state_size or hidden_size  # size of h_t (recurrent/output state)
        self.x2h = nn.Linear(input_size, self.n_gates * hidden_size)
        self.h2h = nn.Linear(self.state_size, self.n_gates * hidden_size, bias=False)

    @property
    def output_size(self):
        return self.state_size

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.state_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c

    def reset_parameters(self, scheme="xavier", forget_bias=1.0):
        """scheme: 'xavier' (all weights Xavier uniform) or 'orthogonal' (recurrent weights
        orthogonal per gate block, input weights Xavier uniform). Biases are zero except the
        forget gate bias."""
        H = self.hidden_size
        nn.init.xavier_uniform_(self.x2h.weight)
        if scheme == "orthogonal":
            for k in range(self.n_gates):
                nn.init.orthogonal_(self.h2h.weight.data[k * H:(k + 1) * H])
        else:
            nn.init.xavier_uniform_(self.h2h.weight)
        nn.init.zeros_(self.x2h.bias)
        self.x2h.bias.data[self._forget_slice()] = forget_bias

    def forward(self, x, state, return_gates=False):
        return self.step(self.x2h(x), state, return_gates)


class VanillaLSTMCell(_GatedCell):
    """Eq. (1)-(6). Gate order in the fused projections: i, f, g, o."""
    n_gates = 4

    def _forget_slice(self):
        return slice(self.hidden_size, 2 * self.hidden_size)

    def step(self, xg, state, return_gates=False):
        h_prev, c_prev = state
        z = xg + self.h2h(h_prev)
        zi, zf, zg, zo = z.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(zi), torch.sigmoid(zf), torch.sigmoid(zo)
        g = torch.tanh(zg)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        if return_gates:
            return h, (h, c), {"i": i, "f": f, "o": o, "g": g}
        return h, (h, c)


class CIFGCell(_GatedCell):
    """Coupled Input-Forget Gate, Eq. (13)-(18): i_t = 1 - f_t. Gate order: f, g, o."""
    n_gates = 3

    def _forget_slice(self):
        return slice(0, self.hidden_size)

    def step(self, xg, state, return_gates=False):
        h_prev, c_prev = state
        z = xg + self.h2h(h_prev)
        zf, zg, zo = z.chunk(3, dim=-1)
        f, o = torch.sigmoid(zf), torch.sigmoid(zo)
        i = 1.0 - f
        g = torch.tanh(zg)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        if return_gates:
            return h, (h, c), {"i": i, "f": f, "o": o, "g": g}
        return h, (h, c)


class LSTMPCell(VanillaLSTMCell):
    """LSTM with recurrent projection, Eq. (19)-(25).
    c_t, g_t, m_t in R^H; h_t = W_p m_t + b_p in R^P is both the output and the recurrent state."""

    def __init__(self, input_size, hidden_size, proj_size=128):
        assert proj_size < hidden_size, "LSTMP expects P < H"
        super().__init__(input_size, hidden_size, state_size=proj_size)
        self.proj_size = proj_size
        self.proj = nn.Linear(hidden_size, proj_size)

    def reset_parameters(self, scheme="xavier", forget_bias=1.0):
        super().reset_parameters(scheme, forget_bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def step(self, xg, state, return_gates=False):
        h_prev, c_prev = state
        z = xg + self.h2h(h_prev)
        zi, zf, zg, zo = z.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(zi), torch.sigmoid(zf), torch.sigmoid(zo)
        g = torch.tanh(zg)
        c = f * c_prev + i * g
        m = o * torch.tanh(c)
        h = self.proj(m)
        if return_gates:
            return h, (h, c), {"i": i, "f": f, "o": o, "g": g, "m": m}
        return h, (h, c)


CELLS = {"vanilla": VanillaLSTMCell, "cifg": CIFGCell, "lstmp": LSTMPCell}


def make_cell(cell_type, input_size, hidden_size, proj_size=128):
    if cell_type == "lstmp":
        return LSTMPCell(input_size, hidden_size, proj_size)
    return CELLS[cell_type](input_size, hidden_size)


class RecurrentLayer(nn.Module):
    """Unrolls a (stack of) cell(s) over time.

    Padding: at time t, sequences with t >= length keep their previous (h, c), so the final state
    of every sequence is exactly its state at its TRUE last token (not at the padded position).
    """

    def __init__(self, cell_type, input_size, hidden_size, num_layers=1, dropout=0.0, proj_size=128):
        super().__init__()
        self.cells = nn.ModuleList()
        in_size = input_size
        for _ in range(num_layers):
            cell = make_cell(cell_type, in_size, hidden_size, proj_size)
            self.cells.append(cell)
            in_size = cell.output_size
        self.dropout = nn.Dropout(dropout)
        self.output_size = in_size

    def reset_parameters(self, scheme="xavier"):
        for cell in self.cells:
            cell.reset_parameters(scheme)

    def forward(self, x, lengths):
        """x: (B, T, D), lengths: (B,). Returns (outputs (B, T, out), last_hidden (B, out))."""
        B, T, _ = x.shape
        lengths = lengths.to(x.device)
        # mask[t, b] = 1 if position t is a real token of sequence b
        mask = (torch.arange(T, device=x.device)[:, None] < lengths[None, :]).unsqueeze(-1).float()
        layer_in = x
        for li, cell in enumerate(self.cells):
            if li > 0:
                layer_in = self.dropout(layer_in)
            xg = cell.x2h(layer_in).transpose(0, 1)  # (T, B, G*H): W_x x_t for all t at once
            h, c = cell.init_state(B, x.device)
            outs = []
            for t in range(T):
                h_new, (_, c_new) = cell.step(xg[t], (h, c))
                m = mask[t]
                h = m * h_new + (1 - m) * h
                c = m * c_new + (1 - m) * c
                outs.append(h)
            layer_in = torch.stack(outs, dim=1)
        return layer_in, h
