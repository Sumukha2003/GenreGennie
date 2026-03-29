import os
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

INPUT_DIR = Path("data/genres_original")
OUTPUT_DIR = Path("data_clean")
TARGET_SR = 22050
TARGET_DURATION = 25
MIN_DURATION = 5
TARGET_COUNTS = {
    "mridangam": 100,
    "sitar": 100,
    "tabla": 100,
    "veena": 100,
    "violin_indian": 100,
}


def normalize_audio(y):
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def fix_audio_length(y, sr=TARGET_SR, duration=TARGET_DURATION):
    target_len = int(sr * duration)
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y


def process_file(in_path, out_path):
    try:
        if out_path.exists():
            return True
        y, _ = librosa.load(in_path, sr=TARGET_SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)

        if len(y) < TARGET_SR * MIN_DURATION:
            return False

        y = normalize_audio(fix_audio_length(y))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y, TARGET_SR)
        return True
    except Exception:
        print(f"Error: {in_path}")
        return False


def augment_waveform(y, sr=TARGET_SR):
    mode = random.choice(["noise", "gain", "shift", "speed_up", "speed_down"])

    if mode == "noise":
        augmented = y + 0.003 * np.random.randn(len(y))
    elif mode == "gain":
        augmented = y * random.uniform(0.8, 1.2)
    elif mode == "shift":
        shift = random.randint(-sr // 4, sr // 4)
        augmented = np.roll(y, shift)
    elif mode == "speed_up":
        augmented = signal.resample(y, max(1, int(len(y) / 1.1)))
    else:
        augmented = signal.resample(y, int(len(y) / 0.9))

    return normalize_audio(fix_audio_length(augmented, sr=sr))


def oversample_with_augmentation(out_dir, target_counts):
    for genre_dir in sorted(out_dir.iterdir()):
        if not genre_dir.is_dir():
            continue

        target = target_counts.get(genre_dir.name, 100)
        files = sorted(genre_dir.glob("*.wav"))
        current_count = len(files)

        if current_count >= target:
            print(f"{genre_dir.name}: {current_count} (ok)")
            continue

        print(f"Augmenting {genre_dir.name}: {current_count} -> {target}")
        while current_count < target and files:
            src_file = random.choice(files)
            y, sr = sf.read(src_file, always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
            augmented = augment_waveform(y, sr=sr)
            new_name = f"{src_file.stem}_aug{current_count:03d}.wav"
            sf.write(genre_dir / new_name, augmented, TARGET_SR)
            current_count += 1

        print(f"{genre_dir.name}: {current_count}")


def main():
    for genre_dir in sorted(INPUT_DIR.iterdir()):
        if not genre_dir.is_dir():
            continue

        print(f"Processing: {genre_dir.name}")
        out_genre = OUTPUT_DIR / genre_dir.name

        for audio_path in sorted(genre_dir.iterdir()):
            suffix = audio_path.suffix.lower()
            if suffix not in {".wav", ".mp3", ".au", ".ogg", ".flac"}:
                continue
            out_path = out_genre / f"{audio_path.stem}.wav"
            process_file(audio_path, out_path)

    oversample_with_augmentation(OUTPUT_DIR, TARGET_COUNTS)
    print("Dataset cleaned and oversampled.")


if __name__ == "__main__":
    main()
