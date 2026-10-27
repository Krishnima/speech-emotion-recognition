# Speech Emotion Recognition (RAVDESS only) — React + Flask

Based on the Kaggle notebook by Shivam Burnwal, using a **single dataset: RAVDESS**
(1,440 speech clips, 24 actors, 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised).

- **Model (Python):** 162 features (ZCR + Chroma + 20 MFCC + RMS + 128 Mel), noise and stretch/pitch augmentation, 1D-CNN.
  The train/test split happens before augmentation, so the accuracy you see is honest.
- **Backend:** Flask API (`/api/status`, `/api/predict`).
- **Frontend:** React (Vite) with four tabs that follow the notebook's sections:
  - **Detect**: record or upload a clip, get scores for all 8 emotions. RAVDESS files show their true label too.
  - **Explore**: your clip's waveform and spectrogram, the four augmentations (noise, stretch, shift, pitch)
    with sliders and playback, the five features (ZCR, Chroma, MFCC, RMS, Mel) and the final 162-number vector.
  - **Dataset**: RAVDESS facts, count of emotions, and a file-name decoder.
  - **Model**: pipeline, layer-by-layer architecture (557,288 parameters), training setup, and after training:
    accuracy/loss curves, confusion matrix and classification report.

## Project structure
```
speech-emotion-recognition/
├── app.py            Flask API (also serves the React build)
├── train.py          Baseline: notebook features + Conv1D -> models/
├── train_ssl.py      High-accuracy model: WavLM (pretrained speech model) + classifier
├── ssl_features.py   WavLM embeddings
├── predict.py        Inference (CLI: python predict.py clip.wav)
├── evaluate.py       Evidence for a demo: held-out accuracy, confusion matrix, demo_clips/
├── DEMO.md           How to show the project works
├── features.py       Feature extraction + augmentation (noise, stretch, shift, pitch)
├── explore.py        Notebook-style EDA figures -> reports/
├── requirements.txt
├── data/             put RAVDESS here
├── models/           created by training
└── frontend/         React app (Vite)
    ├── package.json, vite.config.js, index.html
    └── src/
        ├── App.jsx, main.jsx, styles.css
        ├── views/       DetectView, ExploreView, DatasetView, ModelView
        ├── components/  Waveform, Heatmap, LinePlot, PlayButton, Tabs, DropZone, ResultPanel, StatusBanner
        ├── hooks/useRecorder.js
        └── lib/         dsp.js (STFT, features, augmentations in the browser), acoustics.js (demo mode),
                         audio.js (browser → WAV), api.js, ravdess.js, modelInfo.js, emotions.js
```

## See the UI in one minute (no Python, no training)
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173**. Without the Python server the app runs in **demo mode**: it measures pitch,
pitch variation, brightness and pace in your browser and guesses the emotion with simple rules.
Once you train the model and start `python app.py`, the same UI switches to the trained CNN automatically
(the result shows a *Trained model* tag instead of *Demo estimate*).

## Requirements
- Python **3.9 – 3.12** (the Python built into macOS works; TensorFlow does not support 3.13 yet)
- Node.js **18+**

## 1. Python setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download RAVDESS
Kaggle CLI:
```bash
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p data/ravdess --unzip
```
Or download https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio manually and extract
into `data/ravdess/` (you should see `Actor_01 ... Actor_24`; nested folders are fine).

## 3. Train (about 5–15 min on CPU)
```bash
python train.py --data data/ravdess
```
Saves `models/ser_model.keras`, `scaler.pkl`, `labels.json`, `metrics.json` (accuracy, history,
confusion matrix, per-class report — the Model tab reads this), `features.csv` (like the notebook),
`confusion_matrix.png` and `training_curves.png`. Expect roughly 60–75% test accuracy on 8 classes.

Optional, notebook-style exploration figures (count plot, waveplot + spectrogram per emotion, augmentations):
```bash
python explore.py --data data/ravdess      # saves PNGs to reports/
```

## 3b. High-accuracy model (recommended)
```bash
pip install -r requirements.txt          # adds torch + transformers
python train_ssl.py --data data/ravdess
```
Uses Microsoft's **WavLM Large** speech model (downloaded once, about 1.2 GB) as the "ears" and trains a
classifier on its best layers. Same 288 test clips as `train.py`, plus a separate validation set for tuning,
so the comparison with the CNN is fair. Embedding all clips takes roughly 15–30 minutes the first time; it
is saved per clip in `models/ssl_cache/`, so an interrupted run resumes and later runs take about a minute.
Faster option: `--backbone microsoft/wavlm-base-plus`.

The app switches to this model automatically after you restart `python app.py`; the Model tab then shows
the before/after comparison and which WavLM layers carry emotion. To go back to the CNN, delete
`models/ssl_config.json`.

## 4. Run

### Option A — development (two terminals)
```bash
# terminal 1 (project root, venv active)
python app.py

# terminal 2
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173**. Vite forwards `/api` calls to Flask on port 5000.

### Option B — one server
```bash
cd frontend && npm install && npm run build && cd ..
python app.py
```
Open **http://127.0.0.1:5000** — Flask serves the built React app and the API together.

## Troubleshooting
- **Training crashed with `TerminatedWorkerError` / `SIGSEGV`** → fixed: features are now extracted one file at a time
  (`--jobs 1`, the default) and librosa's code cache lives in `.numba_cache/` inside the project.
- **Training is slow to start** → the first file takes 10–30 s while librosa compiles; after that you get a progress line every 25 files.
  Features are saved to `models/features_cache.npz`, so re-running `train.py` skips extraction (use `--no-cache` to redo it).
- **Banner says "Demo mode"** → the trained model isn’t connected. The banner tells you why (server not running, not trained, or a missing package).
- **"The CNN isn’t trained yet"** → run step 3, then click *Check again*.
- **"The model couldn’t load: ModuleNotFoundError…"** → a Python package is missing; re-run `pip install -r requirements.txt` in the venv.
- **"The Python server isn’t running"** → start `python app.py` (port 5000), then click *Check again*.
- **Mic blocked** → allow microphone access for localhost in the browser. Mic only works on `localhost`/`127.0.0.1` or HTTPS.
- Recordings and uploads (webm, mp3, m4a, ogg, wav…) are converted in the browser to mono 22.05 kHz WAV, so no ffmpeg is needed.
- RAVDESS is acted North American English, so everyday speech will score lower than the test accuracy. Try a RAVDESS `.wav` file first as a sanity check.
