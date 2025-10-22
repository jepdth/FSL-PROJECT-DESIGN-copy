from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np

# ======================================================
# Flask setup
# ======================================================
app = Flask(__name__)
CORS(app)

# ======================================================
# Model definition  (same as your training)
# ======================================================
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
            self.layernorms = nn.ModuleList(
                [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
            )
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

# ======================================================
# Load model + classes
# ======================================================
MODEL_PATH = r"C:\Users\Jerome\Project Design\DONE\MODIFIEDLSTM\run24.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'Color_Black', 'Color_Blue', 'Color_Brown', 'Color_Dark', 'Color_Gray',
    'Color_Green', 'Color_Light', 'Color_Orange', 'Color_Pink', 'Color_Red',
    'Color_Violet', 'Color_White', 'Color_Yellow',
    'Family_Auntie', 'Family_Cousin', 'Family_Daughter', 'Family_Father',
    'Family_Grandfather', 'Family_Grandmother', 'Family_Mother', 'Family_Parents',
    'Family_Son', 'Family_Uncle',
    'Numbers_Eight', 'Numbers_Five', 'Numbers_Four', 'Numbers_Nine',
    'Numbers_One', 'Numbers_Seven', 'Numbers_Six', 'Numbers_Ten',
    'Numbers_Three', 'Numbers_Two'
]

INPUT_SIZE = 188
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = len(CLASSES)
SEQ_LEN = 48

model = ModifiedLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                     dropout=0.35, use_layernorm=True).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ======================================================
# Helpers
# ======================================================
def prepare_sequence(data_json):
    """
    Accept either:
      - {"sequence": [48*188 floats]} or {"sequence": [[188] * 48]}
      - {"features": [188 floats]}  (will tile to 48)
    Returns: tensor [1, 48, 188]
    """
    SEQ_LEN, FEAT_DIM = 48, 188

    if "sequence" in data_json:
        seq = np.array(data_json["sequence"], dtype=np.float32)
        if seq.ndim == 1:
            if seq.size != SEQ_LEN * FEAT_DIM:
                raise ValueError(f"sequence size {seq.size}, expected {SEQ_LEN*FEAT_DIM}")
            seq = seq.reshape(SEQ_LEN, FEAT_DIM)
        elif seq.ndim == 2:
            if seq.shape != (SEQ_LEN, FEAT_DIM):
                raise ValueError(f"sequence shape {seq.shape}, expected {(SEQ_LEN, FEAT_DIM)}")
        else:
            raise ValueError("sequence must be 1D or 2D array")
    elif "features" in data_json:
        feat = np.array(data_json["features"], dtype=np.float32)
        if feat.size != FEAT_DIM:
            raise ValueError(f"features size {feat.size}, expected {FEAT_DIM}")
        seq = np.tile(feat, (SEQ_LEN, 1))
    else:
        raise ValueError("Missing 'sequence' or 'features' field in request.")

    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

# ======================================================
# Routes
# ======================================================
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Backend is reachable ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        x = prepare_sequence(data)       # shape [1, 48, 188]
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]

            # top-k (5) predictions for debugging
            k = min(5, len(CLASSES))
            top_idx = probs.argsort()[-k:][::-1]
            top_k = [{"label": CLASSES[i], "score": float(probs[i])} for i in top_idx]

        return jsonify({"label": label, "top_k": top_k})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ======================================================
# Run server
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
