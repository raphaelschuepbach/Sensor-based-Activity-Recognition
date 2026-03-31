import streamlit as st
import zipfile
import os
import tempfile
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
USE_TEMP = True  # Für Deployment = True lassen

# =========================
# STYLE
# =========================
st.set_page_config(page_title="Activity Recognition", layout="wide")

st.markdown("""
    <style>
        .main {
            max-width: 900px;
            margin: auto;
        }
        h1 {
            text-align: center;
            font-size: 3em;
        }
        h2, h3 {
            text-align: center;
        }
        p, div {
            font-size: 1.2em;
        }
        .stFileUploader {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# STORAGE LOGIC (HYBRID)
# =========================
if USE_TEMP:
    tmp_dir_obj = tempfile.TemporaryDirectory()
    BASE_DIR = tmp_dir_obj.name
else:
    BASE_DIR = "../Check_Data"
    os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# UI
# =========================
st.title("📱 Sensor-Based Activity Recognition")
st.write("Lade eine ZIP-Datei vom Sensor Logger hoch.")

uploaded_file = st.file_uploader("ZIP-Datei hochladen", type="zip")

# =========================
# FILE HANDLING
# =========================
if uploaded_file is not None:
    st.success("ZIP-Datei erfolgreich hochgeladen ✅")

    zip_path = os.path.join(BASE_DIR, "uploaded.zip")

    # ZIP speichern
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ZIP entpacken
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

    st.subheader("📂 Inhalt der ZIP-Datei")

    file_list = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file == "uploaded.zip":
                continue
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, BASE_DIR)
            file_list.append(relative_path)

    if file_list:
        st.write("Gefundene Dateien:")
        for f in file_list:
            st.write(f"• {f}")
    else:
        st.warning("Keine Dateien gefunden.")

    # =========================
    # CSV HANDLING
    # =========================
    csv_files = [f for f in file_list if f.endswith(".csv")]

    if csv_files:
        st.subheader("📊 CSV-Dateien")
        selected_csv = st.selectbox("Wähle eine CSV-Datei", csv_files)

        if selected_csv:
            csv_path = os.path.join(BASE_DIR, selected_csv)
            df = pd.read_csv(csv_path)

            # =========================
            # Zeitspalten konvertieren (ohne dateutil)
            # =========================
            # Passe 'time' ggf. an den echten Spaltennamen in der CSV an
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
            
            st.write("Preview:")
            st.dataframe(df.head())