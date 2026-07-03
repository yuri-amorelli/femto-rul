"""The sequence model.

This is the PyTorch piece — kept small on purpose so the mechanics are legible.
An LSTM/GRU reads the window of health-indicator vectors, and we regress RUL
from the final hidden state (the network's summary of "where are we in the
bearing's life"). A tiny MLP head maps that summary to a single number.

PyTorch mechanics worth internalising (this is your learning target):
  - nn.Module subclasses hold layers in __init__ and define the forward pass.
  - nn.LSTM/nn.GRU with batch_first=True expect input (batch, time, features)
    and return (output_seq, hidden_state). We only need the last step.
  - Softplus on the output keeps predictions >= 0, which is physically correct
    for a remaining-life quantity (RUL can never be negative).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RULRegressorRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 rnn_type: str = "lstm", dropout: float = 0.2,
                 bidirectional: bool = False):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )
        self.activation = nn.Softplus()  # enforce RUL >= 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, input_size)
        out, _ = self.rnn(x)
        last = out[:, -1, :]          # summary at the final timestep
        rul = self.head(last).squeeze(-1)
        return self.activation(rul)
