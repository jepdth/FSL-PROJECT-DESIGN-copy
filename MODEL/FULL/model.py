import torch
import torch.nn as nn

class ModifiedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.35, use_layernorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_layernorm = use_layernorm

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size,
                    hidden_size, batch_first=True, dropout=0.0)
            for i in range(num_layers)
        ])

        if use_layernorm:
            self.layernorms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, reset_mask=None):
        out = x
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)
            if self.use_layernorm:
                out = self.layernorms[i](out)
            out = self.act(out)
            out = self.drop(out)
            if reset_mask is not None:
                out = out * reset_mask.unsqueeze(-1)
        out = out.mean(dim=1)
        return self.fc(out)

def build_modified_lstm(num_classes,
                        input_size=188,
                        hidden_size=256,
                        num_layers=2,
                        dropout=0.35,
                        use_layernorm=True):
    model = ModifiedLSTM(input_size, hidden_size, num_layers, num_classes,
                         dropout=dropout, use_layernorm=use_layernorm)
    return model
