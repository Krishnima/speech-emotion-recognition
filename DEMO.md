# Showing the project works

## Before the demo (once)
```bash
cd ~/Downloads/speech-emotion-recognition
source .venv/bin/activate
python evaluate.py --data data/ravdess
```
This prints, and saves to `models/evaluation.txt`:
1. the saved model's accuracy on the held-out test set (matches train.py),
2. the accuracy of the exact web-app pipeline on all 288 held-out clips, with a confusion matrix,
3. a `demo_clips/` folder of held-out clips named by their true emotion (e.g. `05_angry__03-01-05-...wav`).

Held-out = the model never trained on them, so predicting them correctly is real evidence.

## Live demo (about 5 minutes)
Start the app in two terminal tabs:
```bash
cd ~/Downloads/speech-emotion-recognition && source .venv/bin/activate && python app.py
cd ~/Downloads/speech-emotion-recognition/frontend && npm run dev     # open http://localhost:5173
```
1. **Model tab** – pipeline, the Conv1D layers (557,288 parameters), training curves, confusion matrix, per-class report.
2. **Detect tab** – upload clips from `demo_clips/`. Each result shows the prediction, the scores for all 8 emotions,
   the "Trained model" tag, and whether it matches the true label from the file name.
3. **Detect tab** – record your own voice to show it works on live audio.
4. **Explore tab** – the same clip's spectrogram, the four augmentations and the five features (162 numbers) the model sees.
5. **Dataset tab** – RAVDESS facts and the file-name decoder.

## Points to explain
- Two models, same 288 test clips: the notebook-style CNN (hand-made features) and WavLM + classifier.
  The Model tab shows both scores side by side; the jump comes from using a model pretrained on large
  amounts of speech instead of 162 hand-made numbers.
- The layer chart shows which WavLM layers carry emotion (usually the middle ones).
- 8 emotions, so random guessing is 12.5%. Quote the test accuracy from the Model tab.
- Even human listeners agreed with the intended RAVDESS emotion only about 73% of the time, so ~99% is not a realistic honest target.
- The split happens before augmentation, so augmented copies of test clips never leak into training.
  The original notebook augments before splitting, which inflates its reported accuracy.
- Hardest classes: neutral vs calm, happy vs surprised (acted speech, similar pitch and energy).
- Limits: 24 actors, 2 English sentences, studio audio. Everyday speech and noisy rooms score lower.
