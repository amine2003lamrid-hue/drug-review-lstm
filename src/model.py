import torch.nn as nn

from .cells import RecurrentLayer


class SentimentClassifier(nn.Module):
    """Embedding -> RecurrentLayer(cell_type) -> dropout -> Linear -> 2 logits."""

    def __init__(self, vocab_size, cell_type="vanilla", embed_dim=128, hidden_size=256, num_layers=1,
                 proj_size=128, dropout=0.3, num_classes=2, init="xavier"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = RecurrentLayer(cell_type, embed_dim, hidden_size, num_layers, dropout, proj_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.rnn.output_size, num_classes)
        self.rnn.reset_parameters(init)

    def forward(self, x, lengths):
        _, last = self.rnn(self.embedding(x), lengths)
        return self.fc(self.dropout(last))


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
