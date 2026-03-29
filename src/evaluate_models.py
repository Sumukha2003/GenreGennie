import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

try:
    from cnn_model import DATASET_PATH, SEED, TASK_CONFIG, build_samples, collect_dataset, filter_items_for_task
except ImportError:
    from src.cnn_model import DATASET_PATH, SEED, TASK_CONFIG, build_samples, collect_dataset, filter_items_for_task


MODEL_DIR = Path("models")


def evaluate_task(task_name):
    task_dir = MODEL_DIR / task_name
    config = TASK_CONFIG[task_name]
    model = load_model(task_dir / "best_model.keras")
    encoder = joblib.load(task_dir / "encoder.pkl")

    dataset = collect_dataset(DATASET_PATH)
    task_items = filter_items_for_task(dataset, task_name)
    file_paths = [path for path, _ in task_items]
    labels = [label for _, label in task_items]

    _, val_paths, _, val_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    x_val, y_val, _ = build_samples(
        list(zip(val_paths, val_labels)),
        train_mode=False,
        encoder=encoder,
        chunk_duration=config["chunk_duration"],
        train_chunks=config["train_chunks"],
        val_chunks=config["val_chunks"],
    )
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    return {"validation_accuracy": float(acc), "validation_loss": float(loss)}


def main():
    results = {}
    for task_name in ("family", "western", "indian"):
        task_dir = MODEL_DIR / task_name
        if (task_dir / "best_model.keras").exists():
            results[task_name] = evaluate_task(task_name)

    output_path = MODEL_DIR / "evaluation_summary.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
