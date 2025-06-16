import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, bidirectional: bool = False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, input_size) # For Bi-LSTM, hidden size will be doubled
        else:
            self.fc = nn.Linear(hidden_size, input_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1, :]
        x = self.tanh(x)
        x = self.fc(x)
        return x

class TransformerTSF(nn.Module):
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.positional_encoding = nn.Parameter(torch.randn(1, 1024, d_model))  # max length 1024
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        B, T, D = x.shape
        assert T <= 1024, f"Maximum allowed Sequence length is 1024 < {T}"
        x = x + self.positional_encoding[:, :T, :]  # Add positional encoding
        x = self.transformer_encoder(x)  # (B, T, D)
        x = x[:, -1, :]  # (B, D)
        x = self.tanh(x)
        x = self.output_layer(x)  # (B, D)
        return x  # (B, 1, D)