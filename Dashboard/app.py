"""Sensor-Based Activity Recognition Dashboard

Nutzer laed eine ZIP-Datei vom Sensor Logger hoch -> Pipeline extrahiert die
Sensordaten, resampled auf 50Hz, bildet Sliding Windows und laesst das beste
Modell die Bewegungsart vorhersagen.

Das beste Modell wird aus ../Model_data/best_model/metadata.json gelesen.
Wenn dort nichts liegt, faellt das Dashboard auf einen Demo-Modus zurueck
(zeigt UI, gibt aber keine echten Predictions aus).
"""
from __future__ import annotations
import io
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
BEST_MODEL_DIR = Path(__file__).resolve().parent.parent / "Model_data" / "best_model"
META_PATH = BEST_MODEL_DIR / "metadata.json"

CLASS_ICONS = {
    "Auto":      "🚗",
    "Velo":      "🚴",
    "Lift":      "🛗",
    "Treppe":    "🪜",
    "Zug":       "🚆",
    "Laufen":    "🏃",
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

SAMPLING_HZ   = 50            # entspricht 20ms Resampling-Grid
TRIM_SECONDS  = 3             # passend zum Training in merge_data.ipynb


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Activity Recognition",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* === Globaler Hintergrund === */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }

    /* === Hero Banner === */
    .hero {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2.5rem 2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.25);
        text-align: center;
    }
    .hero h1 {
        color: white;
        font-size: 2.8rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.15rem;
        margin: 0.5rem 0 0 0;
    }

    /* === Karten === */
    .card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }

    /* === Ergebnis-Karte === */
    .result-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(236,72,153,0.10));
        border: 2px solid rgba(99, 102, 241, 0.4);
        border-radius: 22px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-card .icon {
        font-size: 5rem;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .result-card .activity {
        font-size: 2.6rem;
        font-weight: 800;
        color: white;
        margin: 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .result-card .confidence {
        font-size: 1.2rem;
        color: #a5b4fc;
        font-weight: 600;
    }

    /* === Pipeline-Schritte === */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.8rem 1.2rem;
        background: rgba(15, 23, 42, 0.5);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    .step-icon {
        font-size: 1.4rem;
        width: 2rem;
        text-align: center;
    }

    /* === File Uploader Styling === */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(99, 102, 241, 0.08) !important;
        border: 2px dashed rgba(99, 102, 241, 0.4) !important;
        border-radius: 14px !important;
        padding: 2rem !important;
    }

    /* === Sidebar === */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
    }
    [data-testid="stSidebar"] h2 { color: #c7d2fe; }

    /* === Section-Headings === */
    h2 {
        color: #c7d2fe;
        border-bottom: 2px solid rgba(99, 102, 241, 0.25);
        padding-bottom: 0.4rem;
    }
    h3 { color: #e0e7ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Returns (meta_dict, model_object | None, kind) where kind is 'sklearn',
    'torch' or 'demo' (no real model available)."""
    if not META_PATH.exists():
        return None, None, "demo"
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model_path = BEST_MODEL_DIR / meta["model_file"]
    kind = meta.get("model_kind", "sklearn")
    if kind == "sklearn":
        import joblib
        model = joblib.load(model_path)
    elif kind == "torch":
        # Echte Reconstruction passiert in predict(); hier nur Pfad merken
        model = {"path": str(model_path)}
    else:
        model = None
    return meta, model, kind


# ============================================================
# DATA PIPELINE
# ============================================================
def find_csv(extract_dir: Path, sensor: str) -> Path | None:
    """Find <sensor>.csv anywhere under extract_dir."""
    for p in extract_dir.rglob(f"{sensor}.csv"):
        return p
    return None


def load_and_resample(extract_dir: Path, freq: str = "20ms") -> pd.DataFrame:
    """Replicate the merge_data pipeline: read each sensor CSV, normalize time
    to zero, resample to common grid, merge.

    Default freq=20ms entspricht 50Hz, was zum 50Hz-Parquet im Repo passt.
    """
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
    """Schneidet die ersten und letzten `seconds` Sekunden weg — identisch zur
    `trim_measurement`-Logik in merge_data.ipynb."""
    n = int(seconds * hz)
    if len(df) <= 2 * n:
        return df.iloc[0:0]   # zu kurz -> leer zurueck, Aufrufer macht Fehler
    return df.iloc[n:-n].reset_index(drop=True)


def make_windows(df: pd.DataFrame, window_size: int, step_size: int) -> np.ndarray:
    """Sliding windows over feature columns. Returns (n_windows, window, n_features)."""
    data = df[FEATURE_COLS].values
    n = len(data)
    if n < window_size:
        return np.empty((0, window_size, len(FEATURE_COLS)))
    starts = range(0, n - window_size + 1, step_size)
    return np.stack([data[s:s + window_size] for s in starts])


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Klassisches Feature-Engineering: mean/std/min/max/median pro Channel.

    Muss spaeter mit der gleichen Feature-Extraction wie beim Training
    abgeglichen werden — der Notebook-Code in `Model_Notebooks/ML/` schreibt
    `features_train_*WS.npz`; idealerweise importiert man diese Funktion
    dort. Hier nur als sinnvoller Fallback.
    """
    feats = []
    for w in windows:
        f = np.concatenate([
            w.mean(axis=0), w.std(axis=0),
            w.min(axis=0),  w.max(axis=0),
            np.median(w, axis=0),
        ])
        feats.append(f)
    return np.vstack(feats) if feats else np.empty((0, len(FEATURE_COLS) * 5))


def predict_windows(windows: np.ndarray, model, kind: str, meta: dict) -> np.ndarray:
    """Returns array of (class_name, proba_dict) tuples — one per window."""
    if kind == "demo" or model is None:
        # Random Predictions, damit die UI vorzeigbar bleibt
        rng = np.random.default_rng(0)
        classes = meta["classes"] if meta else list(CLASS_ICONS.keys())
        n = len(windows)
        probas = rng.dirichlet(np.ones(len(classes)), size=n)
        preds = np.array(classes)[probas.argmax(axis=1)]
        return preds, probas, classes

    classes = meta["classes"]

    if kind == "sklearn":
        if meta["feature_input"] == "extracted_features":
            X = extract_features(windows)
        else:
            X = windows.reshape(len(windows), -1)

        # Bundle-Format: {"model","scaler","label_encoder","class_names",...}
        if isinstance(model, dict) and "model" in model:
            est     = model["model"]
            scaler  = model["scaler"]
            le      = model["label_encoder"]
            X_proc  = scaler.transform(X)
            classes = list(le.classes_)
            if hasattr(est, "predict_proba"):
                probas = est.predict_proba(X_proc)
            else:
                # Fallback: One-Hot der Predictions
                preds_idx = est.predict(X_proc)
                probas    = np.zeros((len(preds_idx), len(classes)))
                probas[np.arange(len(preds_idx)), preds_idx] = 1.0
            preds = le.inverse_transform(probas.argmax(axis=1))
        else:
            # Direkter Estimator
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
            def __init__(self, n_features, n_classes, dropout=0.3):
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

        ARCH_BY_NAME = {
            "CNN1D_Plain":    CNN1D_Plain,
            "CNN1D_Residual": CNN1D_Residual,
        }
        # Architektur aus dem Modellnamen ableiten (Default: Residual)
        arch_cls = CNN1D_Residual
        for tag, cls in ARCH_BY_NAME.items():
            if tag in meta.get("model_name", ""):
                arch_cls = cls
                break

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = arch_cls(windows.shape[2], len(classes)).to(device)
        state = torch.load(model["path"], map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        net.load_state_dict(state)
        net.eval()

        X = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)
        with torch.no_grad():
            logits = net(X)
            probas = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.array(classes)[probas.argmax(axis=1)]
        return preds, probas, classes


# ============================================================
# SIDEBAR
# ============================================================
meta, model, kind = load_model()

with st.sidebar:
    st.markdown("## ⚙️ Modell-Status")
    if kind == "demo":
        st.warning("Kein trainiertes Modell gefunden.\n\nDas Dashboard laeuft im "
                   "**Demo-Modus** — die Pipeline und UI sind voll funktional, "
                   "die Vorhersagen sind aber zufaellig.\n\nNach dem Training von "
                   "`Model_Vergleich.ipynb` wird das beste Modell automatisch "
                   "geladen.")
    else:
        st.success(f"**{meta['model_name']}**")
        st.markdown(f"""
        - **Typ:** {kind}
        - **Window-Size:** {meta['window_size']}
        - **Input:** {meta['feature_input']}
        - **F1-Macro (Test):** `{meta['metrics']['f1_macro']:.3f}`
        - **Accuracy:** `{meta['metrics']['accuracy']:.3f}`
        - **95% CI:** [{meta['ci_95']['low']:.3f}, {meta['ci_95']['high']:.3f}]
        """)

    st.markdown("## 🎯 Erkannte Klassen")
    classes_to_show = (meta["classes"] if meta else list(CLASS_ICONS.keys()))
    for c in classes_to_show:
        st.markdown(f"{CLASS_ICONS.get(c, '•')} &nbsp; {c}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 💡 Tipp")
    st.markdown("Optimal sind Aufnahmen von **mindestens 5 Sekunden** Dauer "
                "fuer eine stabile Vorhersage.")


# ============================================================
# MAIN
# ============================================================
st.markdown(
    """
    <div class="hero">
      <h1>📱 Sensor-Based Activity Recognition</h1>
      <p>Lade eine ZIP-Datei aus der <b>Sensor Logger</b>-App hoch und erfahre,
      welche Bewegungsart du gemacht hast.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 ZIP-Datei hochladen")
    uploaded = st.file_uploader(
        "Drag & Drop oder klicken zum Auswaehlen",
        type="zip",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Pipeline")
    st.markdown(
        """
        <div class="pipeline-step"><span class="step-icon">📦</span> ZIP entpacken</div>
        <div class="pipeline-step"><span class="step-icon">🌀</span> Sensoren resampeln (50 Hz)</div>
        <div class="pipeline-step"><span class="step-icon">✂️</span> Trimmen (±3 s)</div>
        <div class="pipeline-step"><span class="step-icon">📐</span> Sliding Windows</div>
        <div class="pipeline-step"><span class="step-icon">🤖</span> Modell-Inferenz</div>
        <div class="pipeline-step"><span class="step-icon">🗳️</span> Mehrheits-Voting</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PIPELINE
# ============================================================
if uploaded is not None:

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "input.zip"
        zip_path.write_bytes(uploaded.getbuffer())

        with st.status("Verarbeite Sensordaten...", expanded=True) as status:
            # Step 1: Unzip
            st.write("📦 ZIP entpacken...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)

            # Step 2: Resample + Merge
            st.write("🌀 Sensoren laden und auf 50 Hz resampeln...")
            try:
                merged = load_and_resample(tmp)
            except FileNotFoundError as e:
                status.update(label="Fehler", state="error")
                st.error(str(e))
                st.stop()

            raw_duration_s = len(merged) / SAMPLING_HZ
            st.write(f"✓ Aufnahme: **{raw_duration_s:.1f} s**, {len(merged)} Samples")

            # Step 2b: Ersten und letzten 3 Sekunden wegschneiden
            # (gleich wie trim_measurement in merge_data.ipynb)
            st.write(f"✂️  Trimmen: {TRIM_SECONDS}s am Anfang + {TRIM_SECONDS}s am Ende...")
            merged = trim_edges(merged)
            if merged.empty:
                status.update(label="Aufnahme zu kurz", state="error")
                st.error(
                    f"Aufnahme zu kurz fuer Trimming. Mindestens "
                    f"{2*TRIM_SECONDS + 2:.0f}s Dauer noetig (aktuell: {raw_duration_s:.1f}s)."
                )
                st.stop()
            duration_s = len(merged) / SAMPLING_HZ
            st.write(f"✓ Nutz-Signal: **{duration_s:.1f} s**, {len(merged)} Samples")

            # Step 3: Windowing
            window_size = meta["window_size"] if meta else 100
            step_size   = window_size // 2
            st.write(f"📐 Sliding Windows (size={window_size}, step={step_size})...")
            windows = make_windows(merged, window_size, step_size)

            if len(windows) == 0:
                status.update(label="Aufnahme zu kurz", state="error")
                st.error(f"Mindestens {window_size/50:.1f}s Aufnahme noetig.")
                st.stop()

            st.write(f"✓ {len(windows)} Windows erzeugt")

            # Step 4: Predict
            st.write("🤖 Modell-Inferenz...")
            preds, probas, classes = predict_windows(windows, model, kind, meta)

            # Step 5: Vote
            vote_counts = Counter(preds)
            top_class, top_count = vote_counts.most_common(1)[0]
            avg_probas = probas.mean(axis=0)
            top_proba  = avg_probas[classes.index(top_class)]

            status.update(label="Fertig ✓", state="complete")

    # ====================================================
    # RESULT
    # ====================================================
    st.markdown(
        f"""
        <div class="result-card">
            <div class="icon">{CLASS_ICONS.get(top_class, '🎯')}</div>
            <div class="activity">{top_class}</div>
            <div class="confidence">Konfidenz: {top_proba*100:.1f}%  &nbsp;·&nbsp;
            Mehrheit in {top_count}/{len(preds)} Windows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability bars
    st.markdown("### 📊 Wahrscheinlichkeiten pro Klasse")
    proba_df = (
        pd.DataFrame({"Klasse": classes, "Wahrscheinlichkeit": avg_probas})
        .sort_values("Wahrscheinlichkeit", ascending=True)
    )
    proba_df["Icon"] = proba_df["Klasse"].map(CLASS_ICONS).fillna("•")
    proba_df["Label"] = proba_df["Icon"] + " " + proba_df["Klasse"]

    fig = px.bar(
        proba_df,
        x="Wahrscheinlichkeit", y="Label", orientation="h",
        color="Wahrscheinlichkeit",
        color_continuous_scale="Plasma",
        text=proba_df["Wahrscheinlichkeit"].map(lambda x: f"{x*100:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(280, 50 * len(classes)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(range=[0, 1.05], gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
        margin=dict(l=10, r=40, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time-series preview
    with st.expander("📈 Sensor-Signal Vorschau"):
        signal_cols = st.multiselect(
            "Spalten auswaehlen",
            FEATURE_COLS,
            default=["x_acc", "y_acc", "z_acc"],
        )
        if signal_cols:
            time_axis = np.arange(len(merged)) / 50.0
            fig2 = go.Figure()
            for c in signal_cols:
                fig2.add_trace(go.Scatter(
                    x=time_axis, y=merged[c], mode="lines", name=c,
                    line=dict(width=1.2),
                ))
            fig2.update_layout(
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(title="Zeit (s)", gridcolor="rgba(148,163,184,0.15)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Per-window timeline
    with st.expander("🕒 Vorhersage pro Window"):
        win_df = pd.DataFrame({
            "Window": np.arange(len(preds)),
            "Sekunde (Start)": np.arange(len(preds)) * (step_size / 50.0),
            "Vorhersage": preds,
        })
        win_df["Icon"] = win_df["Vorhersage"].map(CLASS_ICONS).fillna("•")
        st.dataframe(win_df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Lade eine ZIP-Datei hoch, um zu starten. "
            "Die App erwartet die Standard-Sensor-Logger-Struktur mit "
            "`Accelerometer.csv`, `Gyroscope.csv` und `Orientation.csv`.")
