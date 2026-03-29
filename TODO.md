# Genre Genie 85%+ Accuracy Plan
Approved: Yes (with graphs/confusion matrix).

## TODO Steps:
### 1. Dependencies [x]
- Updated requirements.txt (added sklearn, scipy)
- pip install done

### 2. Data Balancing [ ]
- Edit clean_dataset.py: oversample low genres to ~100 each (sitar x12, tabla x3 etc.)

### 3. Features & Utils [ ]
- Edit src/utils.py: add `get_features` (MFCC+Mel+Chroma+SpectralContrast → 128x128x4)

### 4. Core Improvements [ ]
- Edit src/cnn_model.py:
  | Aug: time_stretch, speed/resample, more pitch
  | Features: 4-channel stack
  | Model: deeper + BatchNorm + GlobalAvgPool2D
  | Train: 50 epochs, ReduceLR, graphs (acc/loss plots), confusion_matrix

### 5. App Consistency [ ]
- Edit app.py: use `get_features` instead of mel_only

### 6. Execute [ ]
- `pip install -r requirements.txt`
- `python clean_dataset.py`
- `python src/cnn_model.py` (train & save plots/model)

### 7. Test & Validate [ ]
- `streamlit run app.py`
- Confirm val_acc >=85%

### 8. Complete [ ]
- attempt_completion

Progress tracked here after each step.

