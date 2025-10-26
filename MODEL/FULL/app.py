from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import os
import random
from model import ModifiedLSTM

# ======================================================
# Flask setup
# ======================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# ======================================================
# Model setup
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
# ✅ prepare_sequence (unchanged)
# ======================================================
def prepare_sequence(data_json):
    SEQ_LEN, FEAT_DIM = 48, 188
    if "sequence" in data_json:
        seq = np.array(data_json["sequence"], dtype=np.float32)
        if seq.ndim == 1 and seq.size == SEQ_LEN * FEAT_DIM:
            seq = seq.reshape(SEQ_LEN, FEAT_DIM)
        elif seq.ndim == 2:
            if seq.shape != (SEQ_LEN, FEAT_DIM):
                raise ValueError(f"sequence shape {seq.shape}, expected {(SEQ_LEN, FEAT_DIM)}")
        else:
            raise ValueError("sequence must be 1D (flattened) or 2D array")
    elif "features" in data_json:
        feat = np.array(data_json["features"], dtype=np.float32)
        if feat.size == SEQ_LEN * FEAT_DIM:
            seq = feat.reshape(SEQ_LEN, FEAT_DIM)
        elif feat.size == FEAT_DIM:
            seq = np.tile(feat, (SEQ_LEN, 1))
        else:
            raise ValueError(f"features size {feat.size}, expected {FEAT_DIM} or {SEQ_LEN*FEAT_DIM}")
    else:
        raise ValueError("Missing 'sequence' or 'features' field in request.")
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

# ======================================================
# Helper — locate demo video automatically
# ======================================================
def get_demo_video_path(label):
    parts = label.split("_")
    if len(parts) != 2:
        print(f"[VIDEO] Bad label format: {label}")
        return None
    category = parts[0].lower()
    name = parts[1].lower()
    folder_path = os.path.join("static", "video", category)
    if not os.path.exists(folder_path):
        print(f"[VIDEO] Folder not found: {folder_path}")
        return None
    files = os.listdir(folder_path)
    candidates = [f for f in files if f.lower().startswith(name)]
    if not candidates:
        print(f"[VIDEO] No matches for '{name}' in {folder_path}")
        return None
    chosen = random.choice(candidates)
    demo_path = os.path.join(folder_path, chosen).replace("\\", "/")
    print(f"[VIDEO] Found demo: {demo_path}")
    return demo_path

# ======================================================
# ROUTES — Frontend Pages
# ======================================================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/auto')
def auto_recognition():
    return render_template('auto.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/activity')
def activity():
    return render_template("activity.html")

@app.route('/detect')
def detect():
    return render_template("detect.html")

@app.route('/results')
def results():
    return render_template("results.html")

@app.route('/tutor')
def tutor():
    return render_template("tutor.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['authenticated'] = True
        flash("Logged in successfully!", "success")
        return redirect(url_for('home'))
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    flash("Logged out successfully!", "info")
    return redirect(url_for('home'))

# ======================================================
# API Routes (backend logic)
# ======================================================
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Backend is reachable ✅"})

# --------------------------
# Normal /predict (Activity)
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if "sequence" in data:
            x = prepare_sequence({"sequence": data["sequence"]})
        elif "features" in data:
            x = prepare_sequence({"features": data["features"]})
        else:
            raise ValueError("Missing 'sequence' or 'features'")

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]

        demo_path = get_demo_video_path(label)
        response = {
            "prediction": label,
            "confidence": float(np.max(probs)),
            "demo": demo_path or f"No demo found for {label}"
        }
        print(f"[PREDICT] {label} (conf={np.max(probs):.4f}) → {demo_path}")
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

# --------------------------
# New /predict_auto (Auto Recognition Only)
# --------------------------
@app.route("/predict_auto", methods=["POST"])
def predict_auto():
    try:
        data = request.get_json(force=True)

        # Prepare input sequence (same as Activity)
        if "sequence" in data:
            x = prepare_sequence({"sequence": data["sequence"]})
        elif "features" in data:
            x = prepare_sequence({"features": data["features"]})
        else:
            raise ValueError("Missing 'sequence' or 'features'")

        # Inference
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            conf = float(np.max(probs))
            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]

        # ✅ Confidence threshold check
        THRESHOLD = 0.8
        if conf < THRESHOLD:
            # Sort all probabilities to find the top-3 closest signs
            sorted_indices = np.argsort(probs)[::-1]
            top_idx = sorted_indices[0]
            closest_label = CLASSES[top_idx]
            closest_conf = float(probs[top_idx])

            response = {
                "prediction": "Incorrect",
                "closest_sign": closest_label,
                "closest_confidence": round(closest_conf, 4),
                "confidence": conf,
                "message": f"❌ Incorrect — closest sign you performed is {closest_label.replace('_', ' ')}"
            }
            print(f"[AUTO] Incorrect (conf={conf:.4f}) → Closest: {closest_label} ({closest_conf:.4f})")
        else:
            response = {
                "prediction": label,
                "confidence": conf,
                "message": f"✅ Correct — {label.replace('_', ' ')}"
            }
            print(f"[AUTO] {label} (conf={conf:.4f}) [threshold={THRESHOLD}]")

        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Auto Prediction failed: {e}")
        return jsonify({"error": f"Auto Prediction failed: {str(e)}"}), 400

# --------------------------
# /api/assess (unchanged)
# --------------------------
@app.route("/api/assess", methods=["POST"])
def assess():
    try:
        data = request.get_json(force=True)
        x = prepare_sequence(data)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = CLASSES[pred_idx]

        demo_path = get_demo_video_path(label)
        return jsonify({
            "label": label,
            "probabilities": probs.tolist(),
            "demo": demo_path
        })
    except Exception as e:
        print(f"[ERROR] Assessment failed: {e}")
        return jsonify({"error": f"Assessment failed: {str(e)}"}), 500

# ======================================================
# Run app
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
