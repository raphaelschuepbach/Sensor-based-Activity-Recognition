# Sensor-Based Activity Recognition Challenge  FS26 DS&AI Studenten: Jessica Schmid | Silas Tschopp | Raphael Schüpbach   Fachexperte: Marcel Messerli

Erkennung von 7 menschlichen Bewegungsarten — **Auto, Velo, Lift, Treppe, Zug, Laufen, Roundkick** — aus den Beschleunigungs-, Gyroskop- und Orientierungs-Sensoren eines Smartphones.

---


## Inhaltsverzeichnis

- [Projektbeschrieb](#projektbeschrieb)
- [Ziele & Scope](#ziele--scope)
- [Evaluation](#evaluation)
- [Ergebnis vs. Ziel](#ergebnis-vs-ziel)
- [Klassen](#klassen)
- [Daten-Übersicht](#daten-übersicht)
- [Architektur & Pipeline](#architektur--pipeline)
- [Repo-Struktur](#repo-struktur)
- [Setup](#setup)
- [Daten holen](#daten-holen)
- [Workflow — vom Rohsignal zum Modell](#workflow--vom-rohsignal-zum-modell)
- [Dashboard starten](#dashboard-starten)
- [Modelle](#modelle)
- [Wissenschaftlicher Modellvergleich](#wissenschaftlicher-modellvergleich)
- [Daten beitragen](#daten-beitragen)
- [Team](#team)

---


## Projektbeschrieb

Im Rahmen der Challenge **CDL1 — Sensor-based Activity Recognition** (Data Science & AI, FS 26, FHNW) wird ein Klassifikations-Modell entwickelt, das anhand von Smartphone-Sensordaten zwischen sieben Bewegungstypen unterscheidet:

> **Zug · Auto · Fahrrad · Treppen steigen · Gehen · Lift · Round-Kick**

Schlussendlich wird ein **Dashboard** bereitgestellt, auf dem Nutzende ihre eigenen Bewegungs-Aufnahmen (ZIP aus der App **Sensor Logger**) hochladen können und das Modell ihnen sagt, um welchen Bewegungstyp es sich vermutlich handelt.

Die Aufzeichnungen umfassen drei Sensoren: **Accelerometer**, **Gyroscope** und **Orientation**. Aufgenommen wird via Sensor Logger, gelabelt wird per Sensor-Logger-Tag (gespeichert in `Tags.csv` innerhalb des ZIPs).

---

## Ziele & Scope

### Was wir erreichen wollen

- Mindestens **ein ML-Modell** und **ein DL-Modell** trainieren und systematisch miteinander vergleichen.
- **Weights & Biases** (wandb.ai) zur Protokollierung von Trainingsläufen, Hyperparametern und Evaluationsmetriken verwenden.
- Ein **anschauliches Dashboard** bauen, auf dem Nutzende einfach Daten hochladen können und das Ergebnis sehen.
- Daten **manuell zuschneiden**, damit nur die relevanten Sensordaten des Bewegungstyps enthalten sind.
- Drei Sensoren verwenden: **Accelerometer, Gyroscope, Orientation**.

### Was wir bewusst nicht abdecken

- Ausschliesslich **Smartphone-Daten** — keine Wearable- oder externen Sensoren.
- Fokus auf einem **funktionalen Dashboard**, nicht auf maximaler Modell-Optimierung (Modelle werden verglichen und tunend verbessert, aber nicht erschöpfend optimiert).
- Das Dashboard klassifiziert **nur den aktuellen Bewegungstyp**, sagt keine zukünftigen Bewegungen vorher.

### Optionale Ziele

- Falls am Schluss Zeit bleibt: Modell so erweitern, dass **mehrere Bewegungstypen in einer Aufnahme** erkannt werden können (statt nur einer pro Aufnahme).

### Milestones (gemäss Konzept)

| Meilenstein | Termin |
|---|---|
| Konzept / Planung | 20.03.2026 |
| Datensammlung abgeschlossen | 05.04.2026 |
| Daten einlesen + zuschneiden | 05.04.2026 |
| Feature Engineering + erstes ML-Modell | 15.04.2026 |
| ML-Modelle trainieren + optimieren | 25.04.2026 |
| DL-Modell entwickeln | 15.05.2026 |
| Modelle verglichen + bestes ausgewählt | 20.05.2026 |
| Dashboard erstellt + getestet | 31.05.2026 |
| Abgabe Challenge | 07.06.2026 |
| Präsentation | 18.06.2026 |

---

## Evaluation

### Ziel-Metrik

Die Modell-Leistung wird primär anhand des **Macro-F1-Scores** beurteilt. Begründung:

- Wir gehen von **ungleichen Klassen-Anzahlen** aus (manche Bewegungstypen sind einfacher aufzuzeichnen als andere).
- Macro-F1 gewichtet alle Klassen gleich → eine seltene Klasse wie *Roundkick* zählt gleich viel wie eine häufige wie *Auto*.
- Reine Accuracy oder der gewichtete F1 würden den Score-Vorteil der Mehrheitsklasse(n) zu stark belohnen.

### Zielwert: ~85 % Macro-F1

Begründung aus dem Konzept:

> Andere Human-Activity-Recognition-Projekte mit Smartphone-Sensordaten erreichen häufig zwischen 80 % und 95 % Genauigkeit. Da wir aber gewisse Schwierigkeiten bei ähnlichen Bewegungen (z. B. Zug vs. Auto) erwarten, setzen wir das Ziel etwas tiefer an.

### Evaluations-Methodik

Über reine Punktschätzer hinaus wird im [`Model_Vergleich.ipynb`](Model_Notebooks/Model_Vergleich.ipynb) ein **statistisch belastbarer Vergleich** durchgeführt:

1. **Author-disjunkter 80/20-Split** pro Tag — keine Person im Train *und* im Test eines Tags → die Test-Performance misst Generalisierung auf **unbekannte Personen**, nicht nur auf unbekannte Aufnahmen.
2. **Bootstrap-Konfidenzintervalle** (B = 1000, paired) für F1-Macro pro Modell → quantifiziert die Unsicherheit der Punktschätzer.
3. **McNemar's Test** paarweise zwischen Modellen mit identischem Test-Set → formaler Hypothesentest auf Discordant Pairs.
4. **Holm-Bonferroni-Korrektur** der paarweisen p-Werte → kontrolliert die Family-Wise Error Rate bei 21 Paarvergleichen.


Zusätzlich pro Modell: **Confusion Matrix**, **Per-Klassen-F1**, **Precision/Recall** macro-averaged.

### Tooling

- **Weights & Biases** (`wandb.ai`) trackt sämtliche DL-Trainingsläufe inklusive Hyperparameter, Loss-Verläufe, Val-Metriken und Modell-Artefakte.
- ML-Hyperparameter-Suche: `RandomizedSearchCV` mit 5-facher Stratified K-Fold-CV, Score = Macro-F1.

---



## Ergebnis vs. Ziel

Stand Ende der Modellierungsphase:

| Anforderung | Ziel | Erreicht |
|---|---|---|
| ML-Modell trainiert | ≥ 1 | **6** (RF/HGB/SVM × 50WS/100WS) |
| DL-Modell trainiert | ≥ 1 | **2** (CNN1D Residual, BiLSTM) |
| Macro-F1 (bestes Modell) | ≈ 0.85 | **0.795** (CNN1D Residual 100WS) |
| Statistisch belegter Modellvergleich | gewünscht | ✅ McNemar + Holm-Bonferroni|
| wandb-Tracking | ja | ✅ alle DL-Läufe |
| Funktionierendes Dashboard | ja | ✅ Streamlit, läuft mit Live-Inferenz |
| Manuelle Datentrim | ja | ✅ ±3 s automatisch (in `merge_data.ipynb`) |
| Reproduzierbares Repo + Dokumentation | ja | ✅ DVC + Poetry + README |

**Diskussion:** Das 85 %-Ziel wurde mit 79.5 % knapp verfehlt — wahrscheinliche Ursachen:

- **Verwechslungs-Klassen** wie Zug ↔ Auto (im Konzept explizit als Risiko genannt) sind im Confusion-Matrix-Plot deutlich sichtbar. Ebenfalls bei den Klassen Treppe ↔ Auto
- **Author-disjunkter Test-Split** ist strenger als ein klassischer ID-basierter Split: das Modell sieht im Test ausschliesslich Personen, deren Geh-/Fahr-/Lift-Stil es im Training **nie** gesehen hat. 
- **Klassen-Imbalance:** Roundkick hat deutlich weniger Aufnahmen als Auto/Velo → der Macro-F1 wird vom schwächsten Klassen-Score nach unten gezogen.

Das 95 %-CI für das Best-Model liegt bei **[0.779, 0.808]** — die wahre Generalisierungs-F1 für vergleichbare neue Personen ist mit 95 % Konfidenz in diesem Bereich.

---

## Output

Am Ende der Challenge wird ausgeliefert:

1. **Streamlit-Dashboard** mit funktionierender ZIP-Upload + Live-Klassifikation
2. **Strukturiertes, dokumentiertes Git-Repository** 
3. **Wissenschaftlicher Modellvergleich** als reproduzierbares Notebook
4. **Abschluss-Präsentation** der Ergebnisse (18.06.2026)

---

## Klassen

7 Aktivitäten, die das Modell unterscheidet:

| Icon | Klasse | Beschreibung |
|:---:|---|---|
| 🚗 | **Auto** | Fahrt im Personenwagen |
| 🚴 | **Velo** | Velofahrt |
| 🛗 | **Lift** | Aufzugs-Fahrt (rauf oder runter) |
| 🪜 | **Treppe** | Treppensteigen (rauf oder runter) |
| 🚆 | **Zug** | Zugfahrt |
| 🚶 | **Laufen** | Gehen |
| 🥋 | **Roundkick** | Round-Kick (Karate) |

---

## Daten-Übersicht

**827 Mess-Aufnahmen** von 5 Probanden, je 5–60 Sekunden lang, aufgenommen mit Sensor Logger:

| Proband | Aufnahmen |
|---|---:|
| Jessica Schmid | 490 |
| Renate | 161 |
| Silas Tschopp | 129 |
| Raphael Schüpbach | 37 |
| Tobias | 10 |
| **Total** | **827** |

Jede Aufnahme enthält:
- **Accelerometer** (x, y, z) — Linearbeschleunigung [m/s²]
- **Gyroscope** (x, y, z) — Winkelgeschwindigkeit [rad/s]
- **Orientation** (qx, qy, qz, qw, roll, pitch, yaw) — Quaternion + Euler-Winkel

Datenträger: **DVC** mit lokalem Remote `storage_local`. Die rohen `Daten/`-Ordner sind nicht in Git; Git tracked nur `Daten.dvc` mit dem Hash.

---

## Architektur & Pipeline

```
                Aufnahmen pro Proband
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Daten/<user>/<YYYY-MM-DD_HH-MM-SS>/                 │
│    Accelerometer.csv   Gyroscope.csv                 │
│    Orientation.csv     Tags.csv (Label)              │
└──────────────────────────────────────────────────────┘
                       │
                       │ EDA_Notebooks/merge_data.ipynb
                       │ - Sensoren resampeln (50 Hz)
                       │ - Ersten & letzten 3 s trimmen
                       │ - Mergen über time_elapsed
                       ▼
              merged_df_50Hz.parquet
                       │
                       │ Model_Notebooks/Model_Vorbereitung.ipynb
                       │ - Author-disjunkter 80/20-Split pro Tag
                       │ - Sliding Windows (50 oder 100 Samples, step=50%)
                       ▼
┌──────────────────────────────────────────────────────┐
│  Model_data/                                         │
│    train_split1_<WS>WS.npz  (X: N x WS x 13)         │
│    test_split1_<WS>WS.npz                            │
└──────────────────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼ Feature-Engineering         ▼ Raw-Tensoren
  ML_Daten/features_*WS.npz       train/test_split*.npz
  (165 statistische Features)     (für Deep Learning)
        │                             │
        ▼                             ▼
  RF / HGB / SVM                  CNN1D_Residual / BiLSTM
        │                             │
        └──────────────┬──────────────┘
                       │ Model_Notebooks/Model_Vergleich.ipynb
                       │ - Bootstrap-CIs (B=1000)
                       │ - McNemar paarweise + Holm
                       │ - Best-Model-Auswahl
                       ▼
              Modell/Bestes_Modell/
                       │
                       ▼ Dashboard/app.py
                  Streamlit-UI
```

---

## Repo-Struktur

```
sensor-based-activity-recognition/
│
├── Daten/                              [DVC-tracked, nicht in Git]
│   ├── Jessica_Schmid/<timestamp>/{Accelerometer,Gyroscope,Orientation,Tags}.csv
│   ├── Raphael_Schuepbach/
│   ├── Renate/
│   ├── Silas_Tschopp/
│   └── Tobias/
│
├── EDA_Daten/ Aufnahmen welche für grobe Übersicht über die Daten verwenden werden kann 
│
├── EDA_Notebooks/                      Erste Analysen + Datenaufbereitung
│   ├── EDA.ipynb                       ....
│   ├── Dateneinlesung.ipynb            ....
│   ├── Datenverteilung.ipynb           ....
│   ├── check_data_before_adding.ipynb  Quality-Check neuer Aufnahmen
│   └── merge_data.ipynb                Erster Schritt in der Pipeline: resampeln, trimmen, mergen
│
├── Model_Notebooks/
│   ├── Model_Vorbereitung.ipynb        Train/Test-Split + Sliding Windows
│   ├── Model_Vergleich.ipynb           Statistischer Modellvergleich
│   ├── Deep_Learning.ipynb             CNN1D / CNN1D_Residual / BiLSTM Training
│   ├── ML/
│   │   ├── Machine_Learning.ipynb      Klassisches ML Training
│   │   ├── ML_Model_evaluieren.ipynb   ML-Modell-Evaluation
│   │   └── Notebooks_fuer_colab/       Hyperparameter-Tuning
│   └── Confusion_Matrix/ 
│       ├── DL           Generierte Confusion-Matrizen zu den DL-Modelle
│       └── ML           Generierte Confusion-Matrizen zu den ML-Modelle
│
├── Model_data/
│   ├── merged_df_50Hz.parquet          50-Hz-Resampling, alle Aufnahmen
│   ├── merged_df_40Hz.parquet          40-Hz-Resampling, alle Aufnahmen
│   ├── train_split1_<WS>WS.npz         Raw-Window-Tensoren für DL
│   ├── test_split1_<WS>WS.npz
│   └── ML_Daten/
│       ├── features_train_<WS>WS.npz   165 Statistik-Features pro Window
│       └── features_test_<WS>WS.npz
│
├── Modell/
│   ├── DL_Modell/
│   │   ├── CNN1D_Residual_v7_final.pth
│   │   └── BiLSTM_final.pth
│   ├── ML_Modell/
│   │   ├── final_model_random_forest_<WS>WS.joblib
│   │   ├── final_model_hist_gradient_boosting_<WS>WS.joblib
│   │   ├── final_model_svm_rbf_<WS>WS.joblib
│   │   └── best_hyperparameters_*.json
│   └── Bestes_Modell/                  Vom Vergleichs-Notebook gespeichert
│       ├── <model>.pth oder .joblib
│       └── metadata.json               Vom Dashboard gelesen
│
├── Dashboard/
│   └── app.py                          Streamlit-Dashboard
│
├── pyproject.toml                      Poetry-Konfiguration
├── poetry.lock                         Paket Versionierung
├── Daten.dvc                           DVC-Tracking-Datei für /Daten
├── .dvc/                               DVC-Config
├── commit_and_push.ps1                 Daten+Code in einem Schritt pushen
├── pull_data.ps1                       Daten+Code in einem Schritt pullen
├── Terminal_befehle.txt                Enthält die Terminal Befehle um pull_data und commit_and_push auszuführen
└── README.md
```



---

## Setup

**Voraussetzungen:**
- Python 3.12 oder 3.13
- [Poetry](https://python-poetry.org/) zur Abhängigkeits-Verwaltung
- [Git](https://git-scm.com/) + [DVC](https://dvc.org/) für Daten + Code-Versionierung
- CUDA-fähige GPU (empfohlen) für Deep-Learning-Training

**Installation:**

```bash
# Repo klonen
git clone https://github.com/raphaelschuepbach/Sensor-based-Activity-Recognition.git
cd Sensor-based-Activity-Recognition

# Dependencies installieren
poetry install


Die `pyproject.toml` enthält Dependencies für:
- **Daten-Pipeline:** pandas, numpy, pyarrow, python-dateutil
- **Klassisches ML:** scikit-learn
- **Deep Learning:** torch, torchvision, torchaudio (mit CUDA 12.1 Source)
- **Statistik:** scipy, statsmodels
- **Visualisierung:** matplotlib, seaborn, plotly
- **Dashboard:** streamlit
- **Tracking:** wandb (für DL-Experimente)

---

## Daten holen

Die rohen Sensor-Daten unter `Daten/` sind **DVC-tracked** und nicht in Git. Um sie lokal verfügbar zu machen:

```powershell
# Komplettes Update: Git + DVC
.\pull_data.ps1
```

Was das Skript macht:
1. `git pull` — Code-Updates ziehen
2. `dvc pull -r storage_local` — Daten-Blobs ziehen
3. `dvc checkout --force` — Workspace auf den getrackten Stand bringen (inkl. Löschungen)

Das DVC-Remote ist standardmässig auf `storage_local` konfiguriert (siehe `.dvc/config`). Für die Migration auf einen anderen Storage (Cloud, NAS) muss das angepasst werden.

---

## Workflow — vom Rohsignal zum Modell

Die Notebooks sollten in dieser Reihenfolge laufen:

### 1️⃣ `EDA_Notebooks/merge_data.ipynb`

Liest alle Aufnahmen unter `Daten/<user>/<timestamp>/` rekursiv ein, resampelt jede Sensor-CSV auf **50 Hz** (20-ms-Grid), normalisiert die Zeit auf 0, mergt die drei Sensoren über `time_elapsed`, **trimmt** die ersten und letzten 3 s weg, interpoliert kurze NaN-Lücken linear.

Ausgabe: `Model_data/merged_df_50Hz.parquet` mit den Spalten:
```
time_elapsed, x_acc, y_acc, z_acc, x_gyr, y_gyr, z_gyr,
qx, qy, qz, qw, roll, pitch, yaw, ID, Tag, Author
```

`ID` ist eine fortlaufende Nummer pro Mess-Aufnahme, `Tag` ist die Aktivitätsklasse (aus `Tags.csv`), `Author` ist der Proband (= übergeordneter Ordner).

### 2️⃣ `Model_Notebooks/Model_Vorbereitung.ipynb`

**Author-disjunkter 80/20-Split pro Tag:** Pro Tag wird durch Subset-Selection (alle 2^N Author-Kombinationen) jene Untermenge der Probanden ausgewählt, deren ID-Summe am nähesten an 20% liegt, diese landen komplett im Test, alle anderen komplett im Train. **Garantie: kein Proband ist gleichzeitig in Train und Test innerhalb eines Tags** → wir messen Generalisierungs-Performance auf unbekannte Personen.

Danach: Sliding Windows mit Step-Size = Window-Size / 2 (50% Overlap). Zwei Varianten:
- `<WS>=100` → Window = 2 s @ 50 Hz
- `<WS>=50` → Window = 1 s @ 50 Hz

Ausgabe: `train_split1_<WS>WS.npz` / `test_split1_<WS>WS.npz` mit `X.shape = (N, WS, 13)` und `y` als Tag-Strings.

### 3️⃣ Feature-Engineering (für ML-Modelle)

In `Model_Notebooks/ML/Notebooks_fuer_colab/ML_<WS>WS_Hypertuning_colab.ipynb` werden pro Window 165 statistische Features berechnet (mean, std, min, max, median, IQR, RMS, Range, mean-abs-diff, zero-crossings, slope für jeden der 13 Channels + Magnituden-Features für Accelerometer und Gyroskop).

Ausgabe: `Model_data/ML_Daten/features_<train|test>_<WS>WS.npz`.

### 4️⃣ Modell-Training

**Klassisches ML** (`Model_Notebooks/ML/`):


**Deep Learning** (`Model_Notebooks/Deep_Learning.ipynb`):


### 5️⃣ `Model_Notebooks/Model_Vergleich.ipynb`

Vereint alle 8 trainierten Modelle in einer Vergleichs-Registry. Berechnet pro Modell Test-Metriken (F1-Macro, Accuracy, Per-Klasse-F1, Confusion Matrix), Bootstrap-CIs, McNemar-Paarvergleiche, Holm-Bonferroni-Korrektur. Identifiziert das beste Modell und kopiert es nach `Modell/Bestes_Modell/` + schreibt `metadata.json` für das Dashboard.

Aktuelles Ergebnis: **CNN1D_Residual_100WS mit F1-Macro = 0.795**, signifikant besser als der zweitbeste (RF_100WS, F1 = 0.756; McNemar Holm-p = 0.012).

---

## Dashboard starten

```bash
streamlit run Dashboard/app.py
```

Öffnet automatisch `http://localhost:8501`.

Oder Abrufbar unter https://dashboard-sensor-based-activity-recognition.streamlit.app/

**Was das Dashboard kann:**
- ZIP-Datei aus Sensor Logger hochladen
- Pipeline visualisieren (entpacken → resampeln → trimmen → windowing → inferenz)
- Vorhersage anzeigen (mit Konfidenz, Mehrheits-Vote, Per-Klasse-Wahrscheinlichkeiten)
- Alle 3 Sensor-Signale plotten (Accelerometer, Gyroscope, Orientation) mit getrennten Legenden
- Per-Window-Timeline der Vorhersagen
- Modell-Metadaten anzeigen (F1, CI, Architektur)


**Aufnahme-Anleitung im Dashboard:** Die App zeigt eine 5-Schritt-Anleitung mit App-Store-Links für die Sensor-Logger-App, Sensor-Konfiguration, Aufnahme-Dauer (mind. 10 s) und Export-Format (CSV Zipped).

---

## Modelle

Im Repo trainiert + verglichen:

| Modell | Window | Input | F1-Macro (Test) | Accuracy |
|---|:---:|---|:---:|:---:|
| **CNN1D Residual** | 100 | Raw | **0.795** | 0.847 |
| Random Forest | 100 | 165 Features | 0.756 | 0.831 |
| Hist Gradient Boosting | 100 | 165 Features | 0.753 | 0.817 |
| Random Forest | 50 | 165 Features | 0.737 | 0.812 |
| Hist Gradient Boosting | 50 | 165 Features | 0.703 | 0.771 |
| SVM (RBF) | 100 | 165 Features | 0.669 | 0.714 |
| SVM (RBF) | 50 | 165 Features | 0.650 | 0.686 |
| BiLSTM | 100 | Raw | (siehe Notebook) | — |


---

## Wissenschaftlicher Modellvergleich

Das `Model_Vergleich.ipynb` setzt drei Methoden kombiniert ein:

**Bootstrap-Konfidenzintervalle (Paired)**
- B = 1000 Resamples des Test-Sets mit Zurücklegen
- Indizes identisch über alle Modelle → gepaarte CIs
- 2.5% / 97.5% Perzentil als 95%-CI für F1-Macro
- Forest-Plot zur Visualisierung
- Einzige saubere Methode für Cross-Window-Vergleiche (50WS vs 100WS)

**McNemar's Test (Paired Klassifikator-Vergleich)**
- Nur innerhalb gleicher Window-Size (gleiche Test-Sample-Identitäten)
- n10/n01 = Anzahl Samples, die nur Modell A bzw. B korrekt klassifiziert
- Exakter Binomial-Test bei n10+n01 < 25, sonst Chi² mit Yates-Korrektur
- Holm-Bonferroni-Korrektur der p-Werte für multiple Tests

**Best-Model-Auswahl-Regel**
1. Top-1 nach F1-Macro
2. Wenn paired vergleichbar: signifikant besser als Runner-up (McNemar Holm-p < 0.05)?
3. Wenn Cross-Window: Bootstrap-CI-Overlap als Fallback

Details und Methodik-Beschreibung siehe Markdown-Zellen im Notebook.

---

## Daten beitragen

Der Workflow für neue Probanden / neue Aufnahmen:

1. Aufnahmen via Sensor Logger machen (siehe Dashboard-Anleitung)
2. ZIP entpacken nach `Daten/<DeinName>/<timestamp>/`
3. `Tags.csv` mit der Aktivitätsklasse als einzigem Wert in der `tag`-Spalte schreiben
4. Commit + Push in einem Schritt:

```powershell
.\commit_and_push.ps1 -msg "Neue Daten von <DeinName>"
```

Das Skript erledigt automatisch:
- `dvc add Daten`
- `git add` der DVC-Datei und sonstigen Änderungen
- Commit mit der gewünschten Message
- `dvc push` (Daten ins DVC-Remote)
- `git push` (Code ins Git-Remote)

Wichtig: vor dem Pushen `.\pull_data.ps1` ausführen, damit du auf dem aktuellen Stand bist.

---
