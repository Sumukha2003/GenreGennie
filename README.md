# Genre Genie

Genre Genie is a two-stage audio classification project built with TensorFlow, Streamlit, and handcrafted audio features. It predicts whether an uploaded clip belongs to a western or indian music family, then routes it to a specialist classifier for the final genre or instrument prediction.

## What the project does

- Classifies audio with a hierarchical pipeline:
  - Stage 1: `family` classifier predicts `western` or `indian`
  - Stage 2: specialist classifier predicts the final label inside that family
- Includes a Streamlit app for interactive inference
- Trains MobileNetV2-based models on spectrogram-style feature images
- Saves evaluation artifacts such as metrics, confusion matrices, classification reports, and training curves

## Supported labels

Western labels:
- `blues`
- `classical`
- `country`
- `disco`
- `hiphop`
- `jazz`
- `metal`
- `pop`
- `reggae`
- `rock`

Indian labels:
- `mridangam`
- `sitar`
- `tabla`
- `veena`
- `violin_indian`

## Project structure

```text
.
|-- app.py
|-- clean_dataset.py
|-- requirements.txt
|-- src/
|   |-- cnn_model.py
|   |-- evaluate_models.py
|   |-- model_config.py
|   `-- utils.py
`-- models/
    |-- family/
    |-- western/
    `-- indian/
```

## Feature pipeline

The training and inference flow converts audio into image-like inputs for transfer learning:

- Audio is loaded at `22050 Hz`
- Clips are padded or trimmed to a fixed duration
- The model uses mel, delta, and chroma-style representations packed into RGB-like tensors
- Training uses chunk extraction plus augmentation such as noise, gain, shift, and speed changes
- MobileNetV2 is used as the backbone classifier

## Included artifacts

This repository includes:

- source code
- Streamlit app
- trained model files in `models/`
- saved encoders
- evaluation summaries
- training curves and confusion matrices

This repository does not include the full raw dataset, cleaned dataset folders, local virtual environment, or large downloaded media files.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

Then upload an audio file such as `.wav`, `.mp3`, `.m4a`, `.ogg`, or `.flac`.

## Train the models

Training expects a cleaned dataset under `data_clean/`, with one subfolder per class label.

```powershell
python src\cnn_model.py
```

This trains three tasks:

- `family`
- `western`
- `indian`

Outputs are written to `models/`.

## Evaluate saved models

```powershell
python src\evaluate_models.py
```

This writes a summary to `models/evaluation_summary.json`.

## Notes

- The current app loads the best saved model for each task from `models/`
- The repository is set up for local experimentation and demo use
- If you want to retrain from scratch, make sure `data_clean/` exists locally before running training

## Tech stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- librosa
- matplotlib
- scikit-learn
- SciPy
- joblib
