import streamlit as st
import zipfile
import os
import shutil

st.set_page_config(page_title="Activity Recognition", layout="wide")

st.title("📱 Sensor-Based Activity Recognition")

st.write("Lade eine ZIP-Datei vom Sensor Logger hoch.")

# Zielordner
TARGET_DIR = "../Check_data"

# Upload
uploaded_file = st.file_uploader("ZIP-Datei hochladen", type="zip")

if uploaded_file is not None:
    st.success("ZIP-Datei erfolgreich hochgeladen ")

    # Ordner neu erstellen (alte Daten löschen)
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    zip_path = os.path.join(TARGET_DIR, "uploaded.zip")

    # ZIP speichern
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ZIP entpacken
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)

    st.subheader(" Inhalt der ZIP-Datei")

    # Dateien anzeigen
    file_list = []
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file == "uploaded.zip":
                continue
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, TARGET_DIR)
            file_list.append(relative_path)

    if file_list:
        st.write("Gefundene Dateien:")
        for f in file_list:
            st.write(f"• {f}")
    else:
        st.warning("Keine Dateien gefunden.")

    # CSV auswählen
    csv_files = [f for f in file_list if f.endswith(".csv")]

    if csv_files:
        st.subheader(" CSV-Dateien")
        selected_csv = st.selectbox("Wähle eine CSV-Datei", csv_files)

        if selected_csv:
            import pandas as pd

            csv_path = os.path.join(TARGET_DIR, selected_csv)
            df = pd.read_csv(csv_path)

            st.write("Preview:")
            st.dataframe(df.head())