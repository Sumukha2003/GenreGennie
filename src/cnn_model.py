import json
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

try:
    from model_config import FAMILY_MAP, INDIAN_INSTRUMENTS, TASKS, WESTERN_GENRES
    from utils import (
        FEATURE_SIZE,
        TARGET_DURATION,
        TARGET_SR,
        augment_audio,
        choose_chunk_starts,
        extract_chunk,
        get_transfer_input,
        load_audio,
    )
except ImportError:
    from src.model_config import FAMILY_MAP, INDIAN_INSTRUMENTS, TASKS, WESTERN_GENRES
    from src.utils import (
        FEATURE_SIZE,
        TARGET_DURATION,
        TARGET_SR,
        augment_audio,
        choose_chunk_starts,
        extract_chunk,
        get_transfer_input,
        load_audio,
    )


SEED = 42
DATASET_PATH = Path("data_clean")
MODEL_DIR = Path("models")
TRAIN_CHUNK_DURATION = 6.0
TRAIN_CHUNKS_PER_FILE = 2
VAL_CHUNKS_PER_FILE = 1
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
BATCH_SIZE = 32
TASK_CONFIG = {
    "family": {"chunk_duration": 8.0, "train_chunks": 2, "val_chunks": 1},
    "western": {"chunk_duration": 10.0, "train_chunks": 2, "val_chunks": 1},
    "indian": {"chunk_duration": 12.0, "train_chunks": 3, "val_chunks": 1},
}


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def collect_dataset(dataset_path):
    items = []
    for genre_dir in sorted(dataset_path.iterdir()):
        if not genre_dir.is_dir():
            continue
        for audio_path in sorted(genre_dir.iterdir()):
            if audio_path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                continue
            items.append((str(audio_path), genre_dir.name))
    if not items:
        raise RuntimeError(f"No audio files found in {dataset_path}")
    return items


def label_for_task(task_name, raw_label):
    if task_name == "family":
        return FAMILY_MAP[raw_label]
    return raw_label


def filter_items_for_task(items, task_name):
    allowed = set(TASKS[task_name])
    filtered = []
    for path, label in items:
        if task_name == "family":
            filtered.append((path, label_for_task(task_name, label)))
        elif label in allowed:
            filtered.append((path, label))
    return filtered


def split_items(items):
    file_paths = [path for path, _ in items]
    labels = [label for _, label in items]
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    return list(zip(train_paths, train_labels)), list(zip(val_paths, val_labels))


def build_samples(items, train_mode, encoder, chunk_duration, train_chunks, val_chunks):
    x_data = []
    y_labels = []
    rng = np.random.default_rng(SEED)

    for path, label in items:
        y_audio, sr = load_audio(path, sr=TARGET_SR, duration=TARGET_DURATION)
        total_duration = len(y_audio) / sr
        starts = choose_chunk_starts(
            total_duration,
            chunk_duration=chunk_duration,
            count=train_chunks if train_mode else val_chunks,
        )

        for start_sec in starts:
            chunk = extract_chunk(y_audio, sr=sr, start_sec=start_sec, duration=chunk_duration)
            x_data.append(get_transfer_input(chunk, sr))
            y_labels.append(label)

            if train_mode:
                augmented = augment_audio(chunk, sr=sr, rng=rng)
                x_data.append(get_transfer_input(augmented, sr))
                y_labels.append(label)

    x_data = preprocess_input(np.asarray(x_data, dtype=np.float32) * 255.0)
    y_encoded = encoder.transform(y_labels)
    y_cat = to_categorical(y_encoded, num_classes=len(encoder.classes_))
    return x_data, y_cat, np.asarray(y_encoded)


def build_transfer_model(num_classes):
    weights = "imagenet"
    try:
        base_model = MobileNetV2(
            input_shape=(FEATURE_SIZE, FEATURE_SIZE, 3),
            include_top=False,
            weights=weights,
        )
    except Exception:
        base_model = MobileNetV2(
            input_shape=(FEATURE_SIZE, FEATURE_SIZE, 3),
            include_top=False,
            weights=None,
        )

    base_model.trainable = False

    inputs = Input(shape=(FEATURE_SIZE, FEATURE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(label_smoothing=0.02),
        metrics=["accuracy"],
    )
    return model, base_model


def save_training_plots(history, output_dir):
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["accuracy"], label="train")
    plt.plot(epochs, history_dict["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict["loss"], label="train")
    plt.plot(epochs, history_dict["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_confusion_outputs(model, x_val, y_val_idx, classes, output_dir):
    preds = model.predict(x_val, verbose=0)
    pred_idx = np.argmax(preds, axis=1)
    cm = confusion_matrix(y_val_idx, pred_idx)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(y_val_idx, pred_idx, target_names=classes, digits=4)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")


def train_task(task_name, dataset):
    task_dir = MODEL_DIR / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    config = TASK_CONFIG[task_name]

    task_items = filter_items_for_task(dataset, task_name)
    train_items, val_items = split_items(task_items)

    task_labels = [label for _, label in task_items]
    encoder = LabelEncoder()
    encoder.fit(task_labels)

    print(f"\n=== Training {task_name} model ===")
    print(f"Files: {len(task_items)} | Classes: {list(encoder.classes_)}")

    x_train, y_train, y_train_idx = build_samples(
        train_items,
        train_mode=True,
        encoder=encoder,
        chunk_duration=config["chunk_duration"],
        train_chunks=config["train_chunks"],
        val_chunks=config["val_chunks"],
    )
    x_val, y_val, y_val_idx = build_samples(
        val_items,
        train_mode=False,
        encoder=encoder,
        chunk_duration=config["chunk_duration"],
        train_chunks=config["train_chunks"],
        val_chunks=config["val_chunks"],
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_idx),
        y=y_train_idx,
    )
    class_weight_map = dict(enumerate(class_weights))

    model, base_model = build_transfer_model(num_classes=len(encoder.classes_))

    callbacks = [
        ModelCheckpoint(
            filepath=str(task_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        CSVLogger(str(task_dir / "history.csv")),
    ]

    history_frozen = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_map,
        verbose=1,
    )

    if task_name != "family":
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=CategoricalCrossentropy(label_smoothing=0.02),
            metrics=["accuracy"],
        )

        fine_tune_callbacks = [
            ModelCheckpoint(
                filepath=str(task_dir / "best_model.keras"),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ]

        history_tuned = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            initial_epoch=INITIAL_EPOCHS,
            epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=fine_tune_callbacks,
            class_weight=class_weight_map,
            verbose=1,
        )

        combined_history = {}
        for key, values in history_frozen.history.items():
            combined_history[key] = values + history_tuned.history.get(key, [])
        history_frozen.history = combined_history

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    model.save(task_dir / "final_model.keras")
    joblib.dump(encoder, task_dir / "encoder.pkl")
    save_training_plots(history_frozen, task_dir)
    save_confusion_outputs(model, x_val, y_val_idx, encoder.classes_, task_dir)

    metrics = {
        "task": task_name,
        "validation_accuracy": float(val_acc),
        "validation_loss": float(val_loss),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "classes": list(encoder.classes_),
    }
    (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{task_name} validation accuracy: {val_acc:.4f}")
    return metrics


def main():
    set_seed()
    MODEL_DIR.mkdir(exist_ok=True)
    dataset = collect_dataset(DATASET_PATH)
    summary = {}

    for task_name in ("family", "western", "indian"):
        summary[task_name] = train_task(task_name, dataset)

    (MODEL_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nSaved all task metrics to models/training_summary.json")


if __name__ == "__main__":
    main()
