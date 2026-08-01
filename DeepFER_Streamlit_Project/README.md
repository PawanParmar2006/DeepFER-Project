# DeepFER — Facial Emotion Recognition Using Deep Learning

A CNN-based facial emotion recognition system trained on FER2013 (48x48 grayscale images,
7 emotion classes), deployed as an interactive Streamlit dashboard with live prediction,
webcam support, Grad-CAM explainability, and model-performance insights.

## Project Structure

```
deepfer/
├── app.py                 # Streamlit dashboard (main entry point)
├── config.py               # Central config: paths, image size, class names
├── requirements.txt
├── src/
│   ├── data_pipeline.py    # Loads data, builds tf.data pipelines, class weights
│   ├── model.py             # CNN architecture
│   ├── train.py             # Training script
│   ├── evaluate.py          # Computes test-set metrics + confusion matrix
│   └── gradcam.py           # Grad-CAM explainability
├── utils/
│   └── face_utils.py       # OpenCV face detection for real (uncropped) photos
└── models/                 # Saved model, label map, history (created after training)
```

## Setup

```bash
pip install -r requirements.txt
```

Update `BASE_DIR` in `config.py` to point at your dataset:

```python
BASE_DIR = r"archive-3\archive-3"   # contains train/ and test/, 7 class subfolders each
```

## 1. Check your data

```bash
python src/data_pipeline.py
```
Prints image counts per class — confirms the dataset loads and reveals class imbalance.

## 2. Train the model

```bash
python src/train.py
```
Trains the CNN with data augmentation and class-weighted loss (to counter the ~16.5x
imbalance between the largest and smallest emotion classes). Saves the best model to
`models/deepfer_model.keras`.

## 3. Evaluate on the test set

```bash
python src/evaluate.py
```
Saves a per-class precision/recall/F1 report and confusion matrix to `models/` — these
feed directly into the dashboard's "Model Insights" page.

## 4. Launch the dashboard

```bash
streamlit run app.py
```

**Pages:**
- **Live Prediction** — upload a photo or use your webcam; faces are detected automatically,
  cropped, and classified, with a confidence breakdown and a Grad-CAM overlay showing which
  facial regions drove the prediction.
- **Model Insights** — training accuracy/loss curves, per-class precision/recall/F1, and a
  confusion matrix.
- **Dataset Explorer** — class distribution and sample images, read live from your local
  dataset folders.
- **About** — project overview.

## Design Notes

- **Class imbalance** is handled via class weighting during training (`compute_class_weights`
  in `data_pipeline.py`), not oversampling — avoids duplicating data or discarding examples.
- **Grad-CAM** turns the model from a black box into something whose reasoning can be visually
  inspected, both for debugging and for credible reporting.
- **Face detection** (OpenCV Haar cascade) lets the app handle real, uncropped photos instead
  of only pre-cropped 48x48 dataset images — falls back to using the whole image if no face
  is detected.
