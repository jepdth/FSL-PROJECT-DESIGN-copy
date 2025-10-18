# %% [markdown]
# # Libraries

# %%
import os
import cv2
import time
import torch
import random
import numpy as np
from pathlib import Path
import mediapipe as mp

# %% [markdown]
# # Testing

# %%
# Constants
SEQUENCE_LENGTH = 48
APPEND_FLAGS = True
SELECT_JOINTS = [0, 4, 8, 12, 16, 20]
DERIVED_PER_JOINT = 5
DERIVED_DIM = len(SELECT_JOINTS) * 2 * DERIVED_PER_JOINT
BASE_HAND_DIM = 42 * 3
FEATURE_DIM = BASE_HAND_DIM + DERIVED_DIM + (2 if APPEND_FLAGS else 0)
FLAG_START = FEATURE_DIM - 2
FLAG_END = FEATURE_DIM
FRAME_STRIDE = 2
MAX_CARRY_FRAMES = 3
MIN_PALM_SCALE = 0.02
CLIP_COORD = 5.0
INVERT_HANDEDNESS = True  

# Face and Pose Anchors
FACE_IDXS = {
    "nose": 1,
    "forehead": 10,
    "lip_u": 13,
    "brow_r": 65,
    "brow_l": 295,
    "chin": 152
}
POSE_IDXS = {"L_SH": 11, "R_SH": 12}

def get_detector():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        refine_face_landmarks=False,   
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        enable_segmentation=False
    )

def get_lr_pts(res):
    if res is None:
        return None, None
    L = (np.array([[lm.x, lm.y, lm.z] for lm in res.left_hand_landmarks.landmark], np.float32)
         if getattr(res, "left_hand_landmarks", None) else None)
    R = (np.array([[lm.x, lm.y, lm.z] for lm in res.right_hand_landmarks.landmark], np.float32)
         if getattr(res, "right_hand_landmarks", None) else None)
    if INVERT_HANDEDNESS:
        L, R = R, L
    return L, R

def get_anchors(res):
    if res is None:
        return None
    anchors = {}

    # Pose (shoulders)
    if getattr(res, "pose_landmarks", None):
        lm = res.pose_landmarks.landmark
        anchors["L_SH"] = np.array([lm[POSE_IDXS["L_SH"]].x, lm[POSE_IDXS["L_SH"]].y, lm[POSE_IDXS["L_SH"]].z], np.float32)
        anchors["R_SH"] = np.array([lm[POSE_IDXS["R_SH"]].x, lm[POSE_IDXS["R_SH"]].y, lm[POSE_IDXS["R_SH"]].z], np.float32)
    else:
        # global norm will fall back to per-hand normalization
        pass

    # Face (used for altitude features)
    if getattr(res, "face_landmarks", None):
        lm = res.face_landmarks.landmark
        for key, idx in FACE_IDXS.items():
            anchors[key] = np.array([lm[idx].x, lm[idx].y, lm[idx].z], np.float32)

    return anchors if anchors else None

def normalize_hand(pts: np.ndarray) -> np.ndarray: # Normalize hand relative to wrist–middle distance.
    if pts is None or pts.shape != (21, 3):
        return np.zeros((21, 3), np.float32)
    wrist, mid = pts[0], pts[9]
    scale = float(np.linalg.norm(mid[:2] - wrist[:2]))
    if not np.isfinite(scale) or scale < MIN_PALM_SCALE:
        scale = MIN_PALM_SCALE
    out = (pts - wrist) / scale
    return np.clip(out, -CLIP_COORD, CLIP_COORD).astype(np.float32)

def normalize_global(L_pts, R_pts, anchors): # Normalize hands by shoulder distance
    if not anchors or ("L_SH" not in anchors) or ("R_SH" not in anchors):
        return normalize_hand(L_pts), normalize_hand(R_pts)

    C = (anchors["L_SH"] + anchors["R_SH"]) / 2.0
    scale = np.linalg.norm((anchors["L_SH"] - anchors["R_SH"])[:2])
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1e-6

    def _norm(pts):
        if pts is None or pts.shape != (21, 3):
            return np.zeros((21, 3), np.float32)
        out = (pts - C) / scale
        return np.clip(out, -CLIP_COORD, CLIP_COORD).astype(np.float32)

    return _norm(L_pts), _norm(R_pts)

# %%
# Altitude Features
def derived_altitude_features(L, R, anchors):
    out = []
    if anchors is None:
        return np.zeros((DERIVED_DIM,), np.float32)

    req = ["nose", "chin", "forehead", "lip_u", "brow_r", "brow_l"]
    if not all(k in anchors for k in req):
        return np.zeros((DERIVED_DIM,), np.float32)

    brow_y = 0.5 * (anchors["brow_r"][1] + anchors["brow_l"][1])

    for H in (L, R):
        H = np.zeros((21, 3), np.float32) if H is None else H
        for j in SELECT_JOINTS:
            p = H[j]
            out.extend([
                p[1] - anchors["chin"][1],
                p[1] - anchors["lip_u"][1],
                p[1] - brow_y,
                p[1] - anchors["forehead"][1],
                p[2] - anchors["nose"][2],
            ])
    return np.asarray(out, np.float32)

def pack_feature_with_anchors(L_pts, R_pts, lf, rf, anchors):
    Lg, Rg = normalize_global(L_pts, R_pts, anchors)
    feat = np.concatenate([Lg.reshape(-1), Rg.reshape(-1)], axis=0)
    d = derived_altitude_features(Lg, Rg, anchors)
    feat = np.concatenate([feat, d], axis=0)
    if APPEND_FLAGS:
        feat = np.concatenate([feat, np.array([lf, rf], np.float32)], axis=0)
    return feat.astype(np.float32)

# Auto detect signs from classes
DATASET_DIR = Path(r"C:\Users\Jerome\Project Design\KEYPOINTS")
CLASSES = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
print(f"[INIT] {len(CLASSES)} gesture classes detected:")
for c in CLASSES:
    print("  •", c)

print(f"\n[OK] Feature dimension: {FEATURE_DIM} (126 coords + 60 altitude + 2 flags)")

def split_coords_derived_flags(seq: np.ndarray):
    T, D = seq.shape
    coords  = seq[:, :BASE_HAND_DIM].reshape(T, 42, 3).astype(np.float32)
    derived = np.zeros((T, DERIVED_DIM), np.float32)
    flags   = np.zeros((T, 2), np.float32)
    if D >= (BASE_HAND_DIM + DERIVED_DIM + (2 if APPEND_FLAGS else 0)):
        derived = seq[:, BASE_HAND_DIM:BASE_HAND_DIM+DERIVED_DIM].astype(np.float32)
        if APPEND_FLAGS:
            flags = seq[:, -2:].astype(np.float32)
    elif APPEND_FLAGS and D >= (BASE_HAND_DIM + 2):
        flags = seq[:, -2:].astype(np.float32)
    return coords, derived, flags

def combine_coords_derived_flags(coords: np.ndarray, derived: np.ndarray, flags: np.ndarray):
    T = coords.shape[0]
    out = coords.reshape(T, BASE_HAND_DIM).astype(np.float32)
    out = np.concatenate([out, derived.astype(np.float32)], axis=1)
    if APPEND_FLAGS:
        out = np.concatenate([out, flags.astype(np.float32)], axis=1)
    return out.astype(np.float32)

def resample_to_length(coords: np.ndarray, flags: np.ndarray, target_len: int):
    T = coords.shape[0]
    if T == target_len:
        return coords.astype(np.float32), flags.astype(np.float32)
    idx = np.linspace(0, T - 1, num=target_len)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    w  = (idx - lo)[:, None, None]
    coords_out = (1 - w) * coords[lo] + w * coords[hi]
    flags_out  = flags[np.round(idx).astype(int)]
    return coords_out.astype(np.float32), flags_out.astype(np.float32)

def resample_to_length_vec(X: np.ndarray, target_len: int):
    T = X.shape[0]
    if T == target_len:
        return X.astype(np.float32)
    idx = np.linspace(0, T - 1, num=target_len)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    w  = (idx - lo)[:, None]
    X_out = (1 - w) * X[lo] + w[...]* X[hi]
    return X_out.astype(np.float32)

# %%
def temporal_fix(seq: np.ndarray, target_len: int) -> np.ndarray:
    PAD_HEAD, PAD_TAIL = 5, 5
    coords, derived, flags = split_coords_derived_flags(seq)
    T = coords.shape[0]
    if T == 0:
        return seq.astype(np.float32)

    # activity = 0.7 motion + 0.3 detection
    v = np.diff(coords, axis=0)
    motion = np.linalg.norm(v, axis=(1,2))
    motion = np.r_[motion[:1], motion]
    det = (flags.sum(axis=1) > 0.5).astype(np.float32)

    def _norm01(x, eps=1e-8):
        if x.size == 0: return x
        lo, hi = float(x.min()), float(x.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    m_norm = _norm01(motion)
    score = 0.7*m_norm + 0.3*det

    CORE_LEN = max(8, target_len - (PAD_HEAD + PAD_TAIL))

    def _sliding_best_window(s, win):
        win = max(1, min(win, len(s)))
        c = np.r_[0.0, np.cumsum(s, dtype=np.float64)]
        sums = c[win:] - c[:-win]
        j = int(np.argmax(sums))
        return j, j + win

    start, end = _sliding_best_window(score, min(CORE_LEN, T))
    win_coords  = coords[start:end]
    win_derived = derived[start:end]
    win_flags   = flags[start:end]

    # local onset alignment
    if win_coords.shape[0] >= 2:
        v_loc   = np.diff(win_coords, axis=0)
        mot_loc = np.linalg.norm(v_loc, axis=(1,2))
        mot_loc = np.r_[mot_loc[:1], mot_loc]
        m_loc   = _norm01(mot_loc)
        det_loc = (win_flags.sum(axis=1) > 0.5).astype(np.float32)

        eps_onset, k = 0.08, 2
        active = (m_loc > eps_onset) & (det_loc > 0.5)
        onset_idx, run = 0, 0
        for i, a in enumerate(active):
            run = run + 1 if a else 0
            if run >= k:
                onset_idx = i - k + 1
                break
        if 0 < onset_idx < win_coords.shape[0]-1:
            win_coords  = win_coords[onset_idx:]
            win_derived = win_derived[onset_idx:]
            win_flags   = win_flags[onset_idx:]

    # resample core
    core_coords, core_flags = resample_to_length(win_coords, win_flags, CORE_LEN)
    core_derived = resample_to_length_vec(win_derived, CORE_LEN)

    # head zeros
    head_coords  = np.zeros((PAD_HEAD, 42, 3),       np.float32)
    head_derived = np.zeros((PAD_HEAD, DERIVED_DIM), np.float32)
    head_flags   = np.zeros((PAD_HEAD, 2),           np.float32)

    # tail fade to zero, flags zero
    if PAD_TAIL > 0:
        last_c = core_coords[-1:].copy()
        last_d = core_derived[-1:].copy()
        alphas = np.linspace(1.0 - 1.0/max(1,PAD_TAIL), 0.0, num=PAD_TAIL, dtype=np.float32)
        tail_coords  = np.repeat(last_c, PAD_TAIL, axis=0) * alphas[:, None, None]
        tail_derived = np.repeat(last_d, PAD_TAIL, axis=0) * alphas[:, None]
        tail_flags   = np.zeros((PAD_TAIL, 2), np.float32)
    else:
        tail_coords  = np.empty((0,42,3), np.float32)
        tail_derived = np.empty((0,DERIVED_DIM), np.float32)
        tail_flags   = np.empty((0,2), np.float32)

    out_coords  = np.concatenate([head_coords,  core_coords,  tail_coords],  axis=0)
    out_derived = np.concatenate([head_derived, core_derived, tail_derived], axis=0)
    out_flags   = np.concatenate([head_flags,   core_flags,   tail_flags],   axis=0)

    # exact len guard
    if out_coords.shape[0] != target_len:
        need = target_len - out_coords.shape[0]
        if need > 0:
            zc = np.zeros((need, 42, 3),       np.float32)
            zd = np.zeros((need, DERIVED_DIM), np.float32)
            zf = np.zeros((need, 2),           np.float32)
            out_coords  = np.concatenate([out_coords,  zc], axis=0)
            out_derived = np.concatenate([out_derived, zd], axis=0)
            out_flags   = np.concatenate([out_flags,   zf], axis=0)
        else:
            out_coords  = out_coords[:target_len]
            out_derived = out_derived[:target_len]
            out_flags   = out_flags[:target_len]

    return combine_coords_derived_flags(out_coords, out_derived, out_flags).astype(np.float32)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModifiedLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.45, use_layernorm=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(input_size if i == 0 else hidden_size,
                          hidden_size, batch_first=True)
            for i in range(num_layers)
        ])
        self.layernorms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        ) if use_layernorm else None
        self.act = torch.nn.ReLU(inplace=True)
        self.drop = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, reset_mask=None):
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if self.layernorms:
                x = self.layernorms[i](x)
            x = self.act(x)
            x = self.drop(x)
            if reset_mask is not None:
                x = x * reset_mask.unsqueeze(-1)
        return self.fc(x.mean(dim=1))

ckpt_path = Path(r"C:\Users\Jerome\Project Design\ModifiedLSTM_best\run24.pt")  # or latest runN.pt
raw_state = torch.load(str(ckpt_path), map_location=device)
state_dict = raw_state["model_state_dict"] if "model_state_dict" in raw_state else raw_state

model = ModifiedLSTM(FEATURE_DIM, 256, 2, len(CLASSES), dropout=0.35).to(device).eval()  # HS=256, NL=2
model.load_state_dict(state_dict)
print(f"[OK] Loaded: {ckpt_path.name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
BASE_THRESH = 0.60   # relaxed a bit
EMA_ALPHA   = 0.40   # smoother
N_CONSEC    = 4      # more stable
SHOW_FPS    = True
CARRY_NOISE = 0.001
SHOW_VIS    = False

try:
    import winsound
    def _beep(freq=800, dur=120):
        try: winsound.Beep(freq, dur)
        except: pass
except Exception:
    def _beep(freq=800, dur=120): 
        pass

def softmax_np(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_threshold(class_idx):
    return BASE_THRESH

def parse_class_name(class_name):
    parts = class_name.split('_')
    if len(parts) >= 2:
        category = parts[0]
        value = '_'.join(parts[1:])
        return {'category': category, 'value': value, 'full': class_name}
    return {'category': class_name, 'value': class_name, 'full': class_name}

def get_all_categories():
    categories = set()
    for cls in CLASSES:
        parsed = parse_class_name(cls)
        categories.add(parsed['category'])
    return sorted(list(categories))

def get_values_in_category(category):
    values = []
    for cls in CLASSES:
        parsed = parse_class_name(cls)
        if parsed['category'] == category:
            values.append(parsed['value'])
    return sorted(list(set(values)))

def get_class_index(category, value):
    for i, cls in enumerate(CLASSES):
        parsed = parse_class_name(cls)
        if parsed['category'] == category and parsed['value'] == value:
            return i
    return 0

def draw_center_text(img, text, y, scale=1.0, color=(230,230,230), thick=2):
    W = img.shape[1]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (W - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_progress_bar(img, x0, y0, w, h, p, fg=(90,200,255), bg=(40,40,40)):
    p = max(0.0, min(1.0, float(p)))
    cv2.rectangle(img, (x0, y0), (x0+w, y0+h), bg, -1)
    cv2.rectangle(img, (x0, y0), (x0+int(w*p), y0+h), fg, -1)
    cv2.rectangle(img, (x0, y0), (x0+w, y0+h), (70,70,70), 1)

def check_attribute_match(target_idx, pred_idx, locked_category):
    if locked_category is None:
        return pred_idx == target_idx
    target_parsed = parse_class_name(CLASSES[target_idx])
    pred_parsed = parse_class_name(CLASSES[pred_idx])
    return target_parsed['category'] == pred_parsed['category'] == locked_category

print("[OK] Helper functions ready (thr/EMA tuned)")

# %%
os.environ["OMP_NUM_THREADS"] = "2"
cv2.setNumThreads(1)
torch.set_num_threads(2)

# States
STATE_READY   = 0
STATE_COUNTDOWN = 1
STATE_ACTIVE  = 2
STATE_RESULT  = 3

COUNTDOWN_SEC   = 2.0
RESULT_HOLD_SEC = 1.5

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Runtime state
state = STATE_READY
target_idx = None
consec_ok = 0
consec_wrong = 0

navigation_mode = 'category'   
locked_category = None
category_idx = 0
value_idx = 0

all_categories = get_all_categories()

buf = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), np.float32)
have = 0
last_feat = None
carry = 0
idx = 0

ema_logits = None
t_last = time.time()
fps_smoothed = None

countdown_end_time = None
result_end_time = None
result_ok = False
result_text = ""
result_color = (255, 255, 255)

print("\n" + "="*60)
print("FSL REAL-TIME TESTING")
print("="*60)
print("\nCONTROLS:")
print("  E/R    : Navigate ←/→")
print("  ENTER  : Lock category / Select value")
print("  T      : Back to category selection")
print("  P      : Random")
print("  SPACE  : Start countdown")
print("  Q      : Quit")
print(f"\nAVAILABLE CATEGORIES: {', '.join(all_categories)}")
print(f"TOTAL GESTURES: {len(CLASSES)}")
print(f"CONFIDENCE THRESHOLD: {BASE_THRESH}")
print("="*60 + "\n")

# %%
with get_detector() as detector:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        proc_w, proc_h = 640, 360
        small = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        if (idx % FRAME_STRIDE) != 0:
            idx += 1
            continue
        idx += 1

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            res = detector.process(rgb)
        except Exception:
            res = None

        anchors = get_anchors(res)
        L_pts, R_pts = get_lr_pts(res)
        lf = 1.0 if L_pts is not None else 0.0
        rf = 1.0 if R_pts is not None else 0.0
        detected = (lf + rf) > 0.0

        if detected:
            feat = pack_feature_with_anchors(L_pts, R_pts, lf, rf, anchors)
            last_feat = feat
            carry = 0
        else:
            if (last_feat is not None) and (carry < MAX_CARRY_FRAMES):
                feat = last_feat.copy()
                if CARRY_NOISE > 0:
                    noise = np.zeros_like(feat, dtype=np.float32)
                    noise[:126] = np.random.normal(0.0, CARRY_NOISE, size=126).astype(np.float32)
                    feat += noise
                carry += 1
            else:
                feat = np.zeros((FEATURE_DIM,), np.float32)
                
        if have < SEQUENCE_LENGTH:
            buf[have] = feat
            have += 1
        else:
            buf[:-1] = buf[1:]
            buf[-1] = feat

        hud = frame.copy()
        H, W = hud.shape[:2]
        cv2.rectangle(hud, (0, 0), (W, 100), (25, 25, 25), -1)

        # nav text
        if navigation_mode == 'category':
            mode_text = f"Mode: SELECT CATEGORY ({len(all_categories)} options)"
            current_cat = all_categories[category_idx]
            tgt_text = f"Category: {current_cat} ({category_idx + 1}/{len(all_categories)})"
            hint_text = "Press ENTER to lock this category"
        else:
            current_cat = all_categories[category_idx]
            values = get_values_in_category(current_cat)
            mode_text = f"Mode: {current_cat.upper()} VALUES ({len(values)} options)"
            current_val = values[value_idx]
            tgt_text = f"Value: {current_val} ({value_idx + 1}/{len(values)})"
            target_idx = get_class_index(current_cat, current_val)
            hint_text = f"Full: {CLASSES[target_idx]}"

        cv2.putText(hud, mode_text, (W-480, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, tgt_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(hud, hint_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # hand presence
        flags_win = buf[:, FLAG_START:FLAG_END]
        present_ratio = float((flags_win.sum(axis=1) > 0.0).mean()) if have > 0 else 0.0
        hand_color = (100, 255, 100) if present_ratio > 0.25 else (120, 120, 120)
        cv2.putText(hud, f"Hands: {present_ratio*100:0.0f}%", (W-200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2, cv2.LINE_AA)

        # buffer progress
        draw_progress_bar(hud, 10, 70, 240, 16, have/SEQUENCE_LENGTH, fg=(120, 220, 120))
        cv2.putText(hud, f"Buffer: {have}/{SEQUENCE_LENGTH}", (260, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # fps
        if SHOW_FPS:
            t_now = time.time()
            fps = 1.0 / max(1e-6, (t_now - t_last))
            t_last = t_now
            fps_smoothed = fps if fps_smoothed is None else (0.8*fps_smoothed + 0.2*fps)
            cv2.putText(hud, f"{fps_smoothed:0.1f} FPS", (W-120, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

        # prediction
        probs = None
        pred_idx = None
        pred_prob = 0.0

        if have == SEQUENCE_LENGTH:
            x = torch.from_numpy(buf[None, :, :]).to(device).float()
            rm_np = (buf[:, FLAG_START:FLAG_END].sum(axis=1) > 0).astype(np.float32)
            rm = torch.from_numpy(rm_np).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x, reset_mask=rm).squeeze(0).cpu().numpy()

            if ema_logits is None:
                ema_logits = logits.copy()
            else:
                ema_logits = EMA_ALPHA * logits + (1.0 - EMA_ALPHA) * ema_logits

            probs = softmax_np(ema_logits)
            pred_idx = int(np.argmax(probs))
            pred_prob = float(probs[pred_idx])

        # Top-3 overlay
        # draw_topk(hud, probs, CLASSES, x=10, y0=100, k=3, highlight_idx=pred_idx)

        # state machine
        now = time.time()
        if state == STATE_READY:
            if navigation_mode == 'category':
                draw_center_text(hud, "SELECT CATEGORY", H//2 - 50, 1.2, (100, 220, 255), 3)
                draw_center_text(hud, f"{all_categories[category_idx]}", H//2, 1.8, (100, 255, 100), 3)
                draw_center_text(hud, "E/R: Navigate | ENTER: Lock Category", H//2 + 50, 0.7, (200, 200, 200), 2)
            else:
                draw_center_text(hud, "SELECT VALUE", H//2 - 50, 1.0, (255, 220, 100), 2)
                values = get_values_in_category(all_categories[category_idx])
                draw_center_text(hud, f"{values[value_idx]}", H//2, 1.8, (100, 255, 100), 3)
                draw_center_text(hud, "SPACE: Start | T: Back to Categories", H//2 + 50, 0.7, (200, 200, 200), 2)

        elif state == STATE_COUNTDOWN:
            secs_left = countdown_end_time - now
            if secs_left <= 0:
                state = STATE_ACTIVE
                consec_ok = consec_wrong = 0
                buf.fill(0.0); have = 0; ema_logits = None
                _beep(950, 120)
                print(f"[STATE] → ACTIVE (Target: {CLASSES[target_idx]})")
            else:
                n = int(secs_left) + 1
                draw_center_text(hud, "Get ready...", H//2 - 50, 1.2, (255, 225, 180), 3)
                draw_center_text(hud, str(n), H//2 + 30, 3.0, (255, 235, 100), 4)

        elif state == STATE_ACTIVE:
            hands_ok = present_ratio > 0.25 and probs is not None
            thr = get_threshold(pred_idx) if pred_idx is not None else BASE_THRESH

            if hands_ok and target_idx is not None and pred_idx is not None:
                if pred_prob >= thr:
                    if check_attribute_match(target_idx, pred_idx, locked_category):
                        consec_ok += 1; consec_wrong = 0
                    else:
                        consec_wrong += 1; consec_ok = 0
                else:
                    consec_ok = consec_wrong = 0

                prog = max(consec_ok, consec_wrong) / max(1, N_CONSEC)
                draw_progress_bar(hud, (W-340)//2, 110, 340, 18, prog,
                                  fg=(0, 220, 0) if consec_ok > consec_wrong else (0, 100, 255))

                tgt_name  = parse_class_name(CLASSES[target_idx])['value']
                pred_name = parse_class_name(CLASSES[pred_idx])['value'] if pred_idx is not None else "…"
                draw_center_text(hud, f"Target: {tgt_name}  |  Pred: {pred_name} ({pred_prob:.2f})",
                                 150, 0.9, (230, 230, 230), 2)

                if consec_ok >= N_CONSEC:
                    state = STATE_RESULT
                    result_ok = True
                    result_text = "✅ CORRECT!"
                    result_color = (0, 255, 0)
                    result_end_time = now + RESULT_HOLD_SEC
                    _beep(1000, 120)
                    print(f"[RESULT] ✅ Correct: {CLASSES[target_idx]}")
                elif consec_wrong >= N_CONSEC:
                    state = STATE_RESULT
                    result_ok = False
                    result_text = "❌ WRONG"
                    result_color = (0, 100, 255)
                    result_end_time = now + RESULT_HOLD_SEC
                    _beep(450, 150)
                    print(f"[RESULT] ❌ Wrong: Expected {CLASSES[target_idx]}, got {CLASSES[pred_idx]}")
            else:
                draw_center_text(hud, "Show your sign clearly!", 150, 1.1, (255, 255, 100), 3)

        elif state == STATE_RESULT:
            draw_center_text(hud, result_text, H//2 - 20, 1.8, result_color, 4)
            if not result_ok and pred_idx is not None:
                pred_name = parse_class_name(CLASSES[pred_idx])['value']
                draw_center_text(hud, f"Detected: {pred_name}", H//2 + 40, 1.0, (200, 200, 200), 2)

            if now >= result_end_time:
                state = STATE_READY
                consec_ok = consec_wrong = 0
                buf.fill(0.0); have = 0; ema_logits = None
                print("[STATE] → READY")

        # show window
        cv2.imshow("FSL Real-time Testing", hud)

        # keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Quitting...")
            break

        if key in [ord('t'), ord('T')]:
            if navigation_mode == 'value':
                navigation_mode = 'category'
                locked_category = None
                target_idx = None
                print(f"\n[NAV] Back to CATEGORY selection")

        if key == 13:  # Enter
            if navigation_mode == 'category':
                navigation_mode = 'value'
                locked_category = all_categories[category_idx]
                value_idx = 0
                values = get_values_in_category(locked_category)
                target_idx = get_class_index(locked_category, values[value_idx])
                print(f"\n[NAV] Locked to category: {locked_category} ({len(values)} values)")

        if key in [ord('e'), ord('E')]:
            if navigation_mode == 'category':
                category_idx = (category_idx - 1) % len(all_categories)
                print(f"\n[NAV] ← Category: {all_categories[category_idx]}")
            else:
                values = get_values_in_category(all_categories[category_idx])
                value_idx = (value_idx - 1) % len(values)
                target_idx = get_class_index(all_categories[category_idx], values[value_idx])
                print(f"\n[NAV] ← Value: {values[value_idx]}")

        if key in [ord('r'), ord('R')]:
            if navigation_mode == 'category':
                category_idx = (category_idx + 1) % len(all_categories)
                print(f"\n[NAV] → Category: {all_categories[category_idx]}")
            else:
                values = get_values_in_category(all_categories[category_idx])
                value_idx = (value_idx + 1) % len(values)
                target_idx = get_class_index(all_categories[category_idx], values[value_idx])
                print(f"\n[NAV] → Value: {values[value_idx]}")

        if key in [ord('p'), ord('P')]:
            if navigation_mode == 'category':
                category_idx = random.randrange(len(all_categories))
                print(f"\n[NAV] 🎲 Random category: {all_categories[category_idx]}")
            else:
                values = get_values_in_category(all_categories[category_idx])
                value_idx = random.randrange(len(values))
                target_idx = get_class_index(all_categories[category_idx], values[value_idx])
                print(f"\n[NAV] 🎲 Random value: {values[value_idx]}")

        if key == ord(' '):
            if navigation_mode == 'value' and target_idx is not None and state == STATE_READY:
                countdown_end_time = now + COUNTDOWN_SEC
                state = STATE_COUNTDOWN
                _beep(700, 120)
                print(f"[STATE] → COUNTDOWN")

# %%
cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Testing session ended")
print("="*60)



