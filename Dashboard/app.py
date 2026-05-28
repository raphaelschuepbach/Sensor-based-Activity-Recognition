"""Sensor-Based Activity Recognition Dashboard

Nutzer laed eine ZIP-Datei vom Sensor Logger hoch -> Pipeline extrahiert die
Sensordaten, resampled auf 50Hz, bildet Sliding Windows und laesst das beste
Modell die Bewegungsart vorhersagen.

Das beste Modell wird aus ../Model_data/best_model/metadata.json gelesen.
Wenn dort nichts liegt, faellt das Dashboard auf einen Demo-Modus zurueck.
"""
from __future__ import annotations
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
BEST_MODEL_DIR = Path(__file__).resolve().parent.parent / "Modell" / "Bestes_Modell"
META_PATH = BEST_MODEL_DIR / "metadata.json"

SENSOR_LOGGER_IOS     = "https://apps.apple.com/app/sensor-logger/id1531582925"
SENSOR_LOGGER_ANDROID = "https://play.google.com/store/apps/details?id=com.kelvin.sensorapp"

CLASS_ICONS = {
    "Auto":      "🚗",
    "Velo":      "🚴",
    "Lift":      "🛗",
    "Treppe":    "🪜",
    "Zug":       "🚆",
    "Laufen":    "🚶",
    "Roundkick": "🥋",
}

SENSORS  = ["Accelerometer", "Gyroscope", "Orientation"]
SENSOR_COLS = {
    "Accelerometer": ["x", "y", "z"],
    "Gyroscope":     ["x", "y", "z"],
    "Orientation":   ["qx", "qy", "qz", "qw", "roll", "pitch", "yaw"],
}
FEATURE_COLS = [
    "x_acc", "y_acc", "z_acc",
    "x_gyr", "y_gyr", "z_gyr",
    "qx", "qy", "qz", "qw", "roll", "pitch", "yaw",
]

# Visualisierung: pro Sensor eine kohärente 3-Farben-Palette
SIGNAL_PANELS = [
    {
        "title":   "Accelerometer",
        "subtitle": "Beschleunigung [m/s²]",
        "cols":    ["x_acc", "y_acc", "z_acc"],
        "labels":  ["X", "Y", "Z"],
        "colors":  ["#60a5fa", "#22d3ee", "#34d399"],
    },
    {
        "title":   "Gyroscope",
        "subtitle": "Winkelgeschwindigkeit [rad/s]",
        "cols":    ["x_gyr", "y_gyr", "z_gyr"],
        "labels":  ["X", "Y", "Z"],
        "colors":  ["#a78bfa", "#c084fc", "#e879f9"],
    },
    {
        "title":   "Orientation",
        "subtitle": "Euler-Winkel [rad]",
        "cols":    ["roll", "pitch", "yaw"],
        "labels":  ["Roll", "Pitch", "Yaw"],
        "colors":  ["#fbbf24", "#fb923c", "#f472b6"],
    },
]

SAMPLING_HZ   = 50
TRIM_SECONDS  = 3


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Activity Recognition",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ===== Global ===== */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(99, 102, 241, 0.10) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.08) 0%, transparent 50%),
            linear-gradient(180deg, #0a0e1a 0%, #0f1424 100%);
        color: #e2e8f0;
    }

    /* Sidebar komplett verstecken */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ===== Hero ===== */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        margin-bottom: 2rem;
    }
    .hero .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #c7d2fe;
        margin-bottom: 1.2rem;
        letter-spacing: 0.02em;
    }
    .hero h1 {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin: 0 0 0.6rem 0;
        background: linear-gradient(135deg, #fff 0%, #c7d2fe 50%, #fbcfe8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.55;
    }

    /* ===== Cards ===== */
    .card {
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 1.5rem 1.7rem;
        margin: 1rem 0;
    }
    .card-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #818cf8;
        margin-bottom: 0.4rem;
    }

    /* ===== Section-Headings ===== */
    .section-heading {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2.5rem 0 1rem 0;
        letter-spacing: -0.01em;
        color: #f1f5f9;
    }

    /* ===== Anleitung-Steps ===== */
    .step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.85rem;
        margin: 0.5rem 0;
    }
    .step {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        transition: all 0.2s ease;
    }
    .step:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.7rem;
        height: 1.7rem;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
    }
    .step-title {
        font-weight: 600;
        color: #e0e7ff;
        margin-bottom: 0.3rem;
        font-size: 0.97rem;
    }
    .step-desc {
        color: #94a3b8;
        font-size: 0.86rem;
        line-height: 1.5;
    }
    .step-desc a { color: #a5b4fc; text-decoration: none; border-bottom: 1px dotted #a5b4fc; }
    .step-desc a:hover { color: #c7d2fe; }

    /* ===== Result-Hero ===== */
    .result-hero {
        position: relative;
        background:
            radial-gradient(circle at 30% 50%, rgba(99,102,241,0.18) 0%, transparent 60%),
            radial-gradient(circle at 70% 50%, rgba(236,72,153,0.14) 0%, transparent 60%),
            rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 3rem 1.5rem;
        text-align: center;
        margin: 2rem 0;
        overflow: hidden;
    }
    .result-hero .icon-large {
        font-size: 5.5rem;
        line-height: 1;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 20px rgba(99, 102, 241, 0.35));
    }
    .result-hero .activity {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.3rem 0 0.5rem 0;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .result-hero .meta {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin-top: 1rem;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    .result-hero .meta-value {
        color: #a5b4fc;
        font-weight: 600;
    }

    /* ===== Activity-Pills (im Hero) ===== */
    .activity-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 1.6rem;
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
    }
    .activity-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.95rem;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 999px;
        font-size: 0.88rem;
        color: #e0e7ff;
        font-weight: 500;
    }
    .activity-pill .pill-icon {
        font-size: 1.05rem;
        line-height: 1;
    }

    /* ===== File Uploader ===== */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(99, 102, 241, 0.05) !important;
        border: 2px dashed rgba(99, 102, 241, 0.35) !important;
        border-radius: 14px !important;
        padding: 2.2rem 1.5rem !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(99, 102, 241, 0.10) !important;
        border-color: rgba(99, 102, 241, 0.55) !important;
    }
    [data-testid="stFileUploaderDropzone"] small { color: #94a3b8 !important; }

    /* ===== Status / Expander ===== */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 12px;
    }
    [data-testid="stExpander"] summary { color: #c7d2fe; }

    /* ===== Buttons / Links ===== */
    .link-row {
        display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.7rem;
    }
    .link-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.4rem 0.85rem;
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 999px;
        color: #c7d2fe !important;
        text-decoration: none;
        font-size: 0.83rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .link-pill:hover {
        background: rgba(99, 102, 241, 0.20);
        transform: translateY(-1px);
    }

    /* ===== Streamlit Tweaks ===== */
    .stAlert > div { border-radius: 12px; }
    div[data-testid="stMarkdownContainer"] hr {
        border-color: rgba(148, 163, 184, 0.1);
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not META_PATH.exists():
        return None, None, "demo"
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model_path = BEST_MODEL_DIR / meta["model_file"]
    kind = meta.get("model_kind", "sklearn")
    if kind == "sklearn":
        import joblib
        model = joblib.load(model_path)
    elif kind == "torch":
        model = {"path": str(model_path)}
    else:
        model = None
    return meta, model, kind


# ============================================================
# DATA PIPELINE
# ============================================================
def find_csv(extract_dir: Path, sensor: str) -> Path | None:
    for p in extract_dir.rglob(f"{sensor}.csv"):
        return p
    return None


def load_and_resample(extract_dir: Path, freq: str = "20ms") -> pd.DataFrame:
    parts = []
    for sensor in SENSORS:
        path = find_csv(extract_dir, sensor)
        if path is None:
            raise FileNotFoundError(f"{sensor}.csv nicht in der ZIP gefunden")
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        elapsed = df["seconds_elapsed"] - df["seconds_elapsed"].min()
        df = df.set_index(pd.to_timedelta(elapsed, unit="s"))[SENSOR_COLS[sensor]]
        df = df.resample(freq).mean()
        suffix = {"Accelerometer": "_acc", "Gyroscope": "_gyr", "Orientation": ""}[sensor]
        if suffix:
            df = df.rename(columns={c: c + suffix for c in df.columns})
        parts.append(df)

    merged = parts[0].join(parts[1:], how="outer")
    merged = merged.interpolate(method="linear", limit_direction="both")
    merged = merged.dropna()
    return merged.reset_index(drop=True)


def trim_edges(df: pd.DataFrame, seconds: int = TRIM_SECONDS,
               hz: int = SAMPLING_HZ) -> pd.DataFrame:
    n = int(seconds * hz)
    if len(df) <= 2 * n:
        return df.iloc[0:0]
    return df.iloc[n:-n].reset_index(drop=True)


def make_windows(df: pd.DataFrame, window_size: int, step_size: int) -> np.ndarray:
    data = df[FEATURE_COLS].values
    n = len(data)
    if n < window_size:
        return np.empty((0, window_size, len(FEATURE_COLS)))
    starts = range(0, n - window_size + 1, step_size)
    return np.stack([data[s:s + window_size] for s in starts])


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Fallback (5 stats × channel). Fuer ML-Modelle siehe Hinweis im Notebook."""
    feats = []
    for w in windows:
        feats.append(np.concatenate([
            w.mean(axis=0), w.std(axis=0),
            w.min(axis=0),  w.max(axis=0),
            np.median(w, axis=0),
        ]))
    return np.vstack(feats) if feats else np.empty((0, len(FEATURE_COLS) * 5))


def predict_windows(windows: np.ndarray, model, kind: str, meta: dict):
    if kind == "demo" or model is None:
        rng = np.random.default_rng(0)
        classes = meta["classes"] if meta else list(CLASS_ICONS.keys())
        n = len(windows)
        probas = rng.dirichlet(np.ones(len(classes)), size=n)
        preds  = np.array(classes)[probas.argmax(axis=1)]
        return preds, probas, classes

    classes = meta["classes"]

    if kind == "sklearn":
        if meta["feature_input"] == "extracted_features":
            X = extract_features(windows)
        else:
            X = windows.reshape(len(windows), -1)

        if isinstance(model, dict) and "model" in model:
            est, scaler, le = model["model"], model["scaler"], model["label_encoder"]
            X_proc  = scaler.transform(X)
            classes = list(le.classes_)
            if hasattr(est, "predict_proba"):
                probas = est.predict_proba(X_proc)
            else:
                preds_idx = est.predict(X_proc)
                probas    = np.zeros((len(preds_idx), len(classes)))
                probas[np.arange(len(preds_idx)), preds_idx] = 1.0
            preds = le.inverse_transform(probas.argmax(axis=1))
        else:
            preds = model.predict(X)
            if hasattr(model, "predict_proba"):
                probas  = model.predict_proba(X)
                classes = list(model.classes_)
            else:
                probas = np.zeros((len(preds), len(classes)))
                for i, p in enumerate(preds):
                    probas[i, classes.index(p)] = 1.0
        return preds, probas, classes

    if kind == "torch":
        import torch, torch.nn as nn

        class _ResidualBlock(nn.Module):
            def __init__(self, channels, dropout=0.3):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv1d(channels, channels, 3, padding=1),
                    nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
                    nn.Conv1d(channels, channels, 3, padding=1),
                    nn.BatchNorm1d(channels),
                )
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.relu(x + self.block(x))

        class CNN1D_Plain(nn.Module):
            def __init__(self, n_features, n_classes, dropout=0.4):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(n_features, 64, 5, padding=2), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, 3, padding=1),        nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(128, 256, 3, padding=1),       nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(128, n_classes),
                )
            def forward(self, x):
                return self.classifier(self.net(x))

        class CNN1D_Residual(nn.Module):
            def __init__(self, n_features, n_classes, dropout=0.4):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(n_features, 64, 7, padding=3),
                    nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                )
                self.res1a = _ResidualBlock(64,  dropout)
                self.res1b = _ResidualBlock(64,  dropout)
                self.down1 = nn.Sequential(
                    nn.Conv1d(64, 128, 3, padding=1, stride=2),
                    nn.BatchNorm1d(128), nn.ReLU(),
                )
                self.res2a = _ResidualBlock(128, dropout)
                self.res2b = _ResidualBlock(128, dropout)
                self.down2 = nn.Sequential(
                    nn.Conv1d(128, 256, 3, padding=1, stride=2),
                    nn.BatchNorm1d(256), nn.ReLU(),
                )
                self.res3  = _ResidualBlock(256, dropout)
                self.pool  = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(dropout / 2),
                    nn.Linear(64, n_classes),
                )
            def forward(self, x):
                x = self.stem(x)
                x = self.res1a(x); x = self.res1b(x); x = self.down1(x)
                x = self.res2a(x); x = self.res2b(x); x = self.down2(x)
                x = self.res3(x);  x = self.pool(x)
                return self.classifier(x)

        class BiLSTMClassifier(nn.Module):
            def __init__(self, n_features, n_classes, hidden_size=256,
                         num_layers=2, dropout=0.3):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True,
                )
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(64, n_classes),
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.classifier(out[:, -1, :])

        ARCH_BY_TAG = {
            "CNN1D_Plain":    (CNN1D_Plain,    "conv"),
            "CNN1D_Residual": (CNN1D_Residual, "conv"),
            "BiLSTM":         (BiLSTMClassifier, "lstm"),
        }
        arch_tag = meta.get("architecture")
        if arch_tag not in ARCH_BY_TAG:
            arch_tag = "CNN1D_Residual"
            for tag in ARCH_BY_TAG:
                if tag in meta.get("model_name", ""):
                    arch_tag = tag
                    break
        arch_cls, input_format = ARCH_BY_TAG[arch_tag]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = arch_cls(windows.shape[2], len(classes)).to(device)
        net.load_state_dict(torch.load(model["path"], map_location=device))
        net.eval()

        if input_format == "conv":
            X = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)
        else:
            X = torch.tensor(windows, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = net(X)
            probas = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.array(classes)[probas.argmax(axis=1)]
        return preds, probas, classes


# ============================================================
# HERO
# ============================================================
meta, model, kind = load_model()

# Aktivitäten-Liste (Reihenfolge wie im Modell-Metadata, sonst Default-Set)
hero_classes = meta["classes"] if meta else list(CLASS_ICONS.keys())
activity_pills_html = "".join(
    f'<span class="activity-pill">'
    f'<span class="pill-icon">{CLASS_ICONS.get(c, "•")}</span>{c}'
    f'</span>'
    for c in hero_classes
)

st.markdown(
    f"""
    <div class="hero">
      <span class="badge">SENSOR-BASED ACTIVITY RECOGNITION</span>
      <h1>Welche Bewegung war das?</h1>
      <p class="subtitle">Lade eine Sensor-Logger-Aufnahme hoch und erfahre,
      welche von den 7 Aktivitäten du gemacht hast, basierend auf Beschleunigung,
      Rotation und Orientierung deines Geräts.</p>
      <div class="activity-row">{activity_pills_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ANLEITUNG
# ============================================================
st.markdown('<div class="section-heading">So funktioniert\'s</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="step-grid">
      <div class="step">
        <div class="step-number">1</div>
        <div class="step-title">Sensor Logger installieren</div>
        <div class="step-desc">
          Kostenloses Sensor-Logger-App.
          <div class="link-row">
            <a href="{SENSOR_LOGGER_IOS}" target="_blank" class="link-pill">iOS</a>
            <a href="{SENSOR_LOGGER_ANDROID}" target="_blank" class="link-pill">Android</a>
          </div>
        </div>
      </div>

      <div class="step">
        <div class="step-number">2</div>
        <div class="step-title">Sensoren aktivieren</div>
        <div class="step-desc">
          In der App unter <b>Logger</b>: <b>Accelerometer</b>,
          <b>Gyroscope</b> und <b>Orientation</b> einschalten. Alle anderen Sensoren ausschalten.
        </div>
      </div>

      <div class="step">
        <div class="step-number">3</div>
        <div class="step-title">Aufnahme machen</div>
        <div class="step-desc">
          Handy in die Hosen- oder Jackentasche stecken, egal in welcher Position (Die besten Resultate erzielt man aber wenn das Handy mit der Kamera nach unten in die Hosentasche gesteckt wird), dann <b>Aufnahmen starten</b> drücken,
          einer der 7 Aktivitäten ausführen, danach auf <b>Aufn. beenden</b> drücken. Mindestens <b>10 Sekunden</b> —
          die ersten und letzten 3 s werden automatisch weggeschnitten.
        </div>
      </div>

      <div class="step">
        <div class="step-number">4</div>
        <div class="step-title">Als ZIP exportieren</div>
        <div class="step-desc">
          Unter <b>Aufnahmen</b> die aufgenommene Datei auswählen und <b>Export</b> drücken. Format <b>CSV in Zip File</b> auswählen und an einem geeigneten Ort speichern.
          Die ZIP enthält Accelerometer.csv, Gyroscope.csv, Orientation.csv.
        </div>
      </div>

      <div class="step">
        <div class="step-number">5</div>
        <div class="step-title">Hier hochladen</div>
        <div class="step-desc">
          ZIP unten reinziehen — das Modell sagt dir, welche der
          7 Aktivitäten (Auto, Velo, Lift, Treppe, Zug, Laufen, Roundkick)
          erkannt wurde.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================
st.markdown('<div class="section-heading">Aufnahme hochladen</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "ZIP-Datei aus Sensor Logger (CSV-Format)",
    type="zip",
    label_visibility="collapsed",
)


# ============================================================
# PIPELINE
# ============================================================
def render_signal_panels(merged: pd.DataFrame):
    """3-Panel-Subplot: Accelerometer / Gyroscope / Orientation.

    Jedes Panel bekommt seine eigene horizontale Legende direkt darunter,
    mit fest weisser Schrift (unabhaengig von Streamlit's Theme).
    """
    time_axis = np.arange(len(merged)) / SAMPLING_HZ

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.20,
        subplot_titles=[
            f"<b>{p['title']}</b> &nbsp;<span style='color:#94a3b8;font-weight:400'>{p['subtitle']}</span>"
            for p in SIGNAL_PANELS
        ],
    )

    # Pro Subplot wird ein eigener Legende-Key verwendet (legend, legend2, legend3)
    LEGEND_KEYS = ["legend", "legend2", "legend3"]

    for row, panel in enumerate(SIGNAL_PANELS, start=1):
        legend_key = LEGEND_KEYS[row - 1]
        for col, label, color in zip(panel["cols"], panel["labels"], panel["colors"]):
            fig.add_trace(
                go.Scatter(
                    x=time_axis, y=merged[col],
                    mode="lines", name=label,
                    line=dict(width=1.4, color=color),
                    legend=legend_key,
                    showlegend=True,
                ),
                row=row, col=1,
            )

    # Jedes Subplot hat ein eigenes Y-Domain. Wir lesen die Positionen der
    # 3 Subplots aus und platzieren die Legenden jeweils direkt darunter.
    # Plotly nummeriert die Y-Achsen: yaxis (=1), yaxis2 (=2), yaxis3 (=3)
    # — jede hat ein `domain`-Tupel [bottom, top] in Paper-Koordinaten.
    legend_style = dict(
        orientation="h",
        x=0.5, xanchor="center",
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#ffffff", size=12),
    )

    # Bei vertical_spacing=0.20: subplot-Hoehe = (1 - 2*0.20)/3 = 0.20
    # subplot 1: y_domain = [0.80, 1.00]
    # subplot 2: y_domain = [0.40, 0.60]
    # subplot 3: y_domain = [0.00, 0.20]
    # Legenden sitzen ~0.13 unter dem jeweiligen Subplot (Platz fuer X-Achsen-
    # Beschriftung + "Zeit (s)"-Title oberhalb der Legende).
    legend_positions = {
        "legend":  0.67,    # unter subplot 1 (bottom = 0.80)
        "legend2": 0.27,    # unter subplot 2 (bottom = 0.40)
        "legend3": -0.13,   # unter subplot 3 (bottom = 0.00)
    }

    fig.update_layout(
        height=820,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#ffffff", size=12),
        legend  = dict(**legend_style, y=legend_positions["legend"]),
        legend2 = dict(**legend_style, y=legend_positions["legend2"]),
        legend3 = dict(**legend_style, y=legend_positions["legend3"]),
        margin=dict(l=10, r=20, t=50, b=70),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.10)", zeroline=False,
        showline=True, linecolor="rgba(148,163,184,0.20)",
        title_text="Zeit (s)",
        title_font=dict(color="#ffffff"),
        tickfont=dict(color="#ffffff"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)", zeroline=True,
        zerolinecolor="rgba(148,163,184,0.18)",
        showline=True, linecolor="rgba(148,163,184,0.20)",
        tickfont=dict(color="#ffffff"),
        title_font=dict(color="#ffffff"),
    )
    # Subplot-Titel links-buendig
    for ann in fig["layout"]["annotations"]:
        ann["x"] = 0
        ann["xanchor"] = "left"
        ann["font"] = dict(size=14, color="#e0e7ff")

    return fig


if uploaded is not None:

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "input.zip"
        zip_path.write_bytes(uploaded.getbuffer())

        with st.status("Verarbeite Sensordaten…", expanded=True) as status:
            st.write("ZIP entpacken")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)

            st.write("Sensoren laden & auf 50 Hz resampeln")
            try:
                merged = load_and_resample(tmp)
            except FileNotFoundError as e:
                status.update(label="Fehlende Datei in ZIP", state="error")
                st.error(str(e))
                st.stop()

            raw_duration = len(merged) / SAMPLING_HZ
            st.write(f"   Aufnahme: **{raw_duration:.1f} s**, {len(merged)} Samples")

            st.write(f"Trimmen — {TRIM_SECONDS} s an beiden Enden")
            merged = trim_edges(merged)
            if merged.empty:
                status.update(label="Aufnahme zu kurz", state="error")
                st.error(
                    f"Aufnahme zu kurz: mindestens {2*TRIM_SECONDS + 2:.0f} s "
                    f"nötig (aktuell {raw_duration:.1f} s)."
                )
                st.stop()
            duration = len(merged) / SAMPLING_HZ
            st.write(f"   Nutz-Signal: **{duration:.1f} s**, {len(merged)} Samples")

            window_size = meta["window_size"] if meta else 100
            step_size   = window_size // 2
            st.write(f"Sliding Windows (size={window_size}, step={step_size})")
            windows = make_windows(merged, window_size, step_size)
            if len(windows) == 0:
                status.update(label="Zu wenig Samples für ein Fenster", state="error")
                st.error(f"Mindestens {window_size/SAMPLING_HZ:.1f} s Nutz-Signal nötig.")
                st.stop()
            st.write(f"   {len(windows)} Fenster erzeugt")

            st.write("Modell-Inferenz")
            preds, probas, classes = predict_windows(windows, model, kind, meta)

            vote_counts = Counter(preds)
            top_class, top_count = vote_counts.most_common(1)[0]
            avg_probas = probas.mean(axis=0)
            top_proba  = avg_probas[classes.index(top_class)]

            status.update(label="Fertig", state="complete")

    # ============================================================
    # RESULT
    # ============================================================
    st.markdown(
        f"""
        <div class="result-hero">
            <div class="icon-large">{CLASS_ICONS.get(top_class, '🎯')}</div>
            <div class="activity">{top_class}</div>
            <div class="meta">
                <div>Konfidenz<br><span class="meta-value">{top_proba*100:.1f}%</span></div>
                <div>Fenster-Mehrheit<br><span class="meta-value">{top_count} / {len(preds)}</span></div>
                <div>Dauer<br><span class="meta-value">{duration:.1f} s</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Wahrscheinlichkeiten pro Klasse
    st.markdown('<div class="section-heading">Wahrscheinlichkeiten pro Klasse</div>',
                unsafe_allow_html=True)
    proba_df = (
        pd.DataFrame({"Klasse": classes, "Wahrscheinlichkeit": avg_probas})
        .sort_values("Wahrscheinlichkeit", ascending=True)
    )
    proba_df["Label"] = (
        proba_df["Klasse"].map(CLASS_ICONS).fillna("•") + "  " + proba_df["Klasse"]
    )

    fig = px.bar(
        proba_df, x="Wahrscheinlichkeit", y="Label", orientation="h",
        color="Wahrscheinlichkeit", color_continuous_scale="Plasma",
        text=proba_df["Wahrscheinlichkeit"].map(lambda x: f"{x*100:.1f}%"),
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#cbd5e1"))
    fig.update_layout(
        height=max(300, 55 * len(classes)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(range=[0, 1.08], gridcolor="rgba(148,163,184,0.10)", title=""),
        yaxis=dict(title="", tickfont=dict(size=14)),
        coloraxis_showscale=False,
        margin=dict(l=10, r=50, t=10, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sensor-Signale — alle 3 Sensoren immer sichtbar
    st.markdown('<div class="section-heading">Sensor-Signale</div>',
                unsafe_allow_html=True)
    st.plotly_chart(render_signal_panels(merged), use_container_width=True)

    # Optional: Per-Window-Timeline
    with st.expander("Vorhersage pro Fenster"):
        win_df = pd.DataFrame({
            "Fenster": np.arange(len(preds)),
            "Sekunde": np.round(np.arange(len(preds)) * (step_size / SAMPLING_HZ), 2),
            "Vorhersage": preds,
        })
        win_df["Icon"] = win_df["Vorhersage"].map(CLASS_ICONS).fillna("•")
        st.dataframe(win_df, use_container_width=True, hide_index=True)

    # Modell-Details (falls vorhanden)
    if kind != "demo":
        with st.expander("Modell-Details"):
            ci = meta["ci_95"]
            m  = meta["metrics"]
            st.markdown(
                f"""
                | Metrik | Wert |
                |---|---|
                | Modell | `{meta['model_name']}` |
                | Architektur | `{meta.get('architecture') or kind}` |
                | Window-Size | {meta['window_size']} |
                | Input | {meta['feature_input']} |
                | F1-Macro (Test) | **{m['f1_macro']:.3f}** |
                | Accuracy | {m['accuracy']:.3f} |
                | 95 %-CI | [{ci['low']:.3f}, {ci['high']:.3f}] |
                """
            )

else:
    st.info(
        "👆 Lade eine ZIP von Sensor Logger hoch um zu starten. "
        "Die Datei muss `Accelerometer.csv`, `Gyroscope.csv` und `Orientation.csv` "
        "enthalten."
    )
