import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fftpack import dct

TARGET_SR = 22050
TARGET_DURATION = 25
FEATURE_SIZE = 128
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
N_MFCC = 20
def load_audio(audio_source, sr=TARGET_SR, duration=TARGET_DURATION):
    source_path = str(audio_source)

    try:
        if source_path.lower().endswith(".wav"):
            y, file_sr = sf.read(source_path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
            if file_sr != sr:
                y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        else:
            raise RuntimeError("fallback to librosa")
    except Exception:
        y, _ = librosa.load(audio_source, sr=sr, mono=True)

    return fix_audio_length(y, sr, duration), sr


def fix_audio_length(y, sr=TARGET_SR, duration=TARGET_DURATION):
    target_len = int(sr * duration)
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y


def extract_chunk(y, sr=TARGET_SR, start_sec=0.0, duration=6.0):
    chunk_len = int(sr * duration)
    start = max(0, int(start_sec * sr))
    end = start + chunk_len
    chunk = y[start:end]
    if len(chunk) < chunk_len:
        chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
    return chunk


def choose_chunk_starts(total_duration, chunk_duration=6.0, count=2):
    if total_duration <= chunk_duration:
        return [0.0] * count

    max_start = max(0.0, total_duration - chunk_duration)
    if count == 1:
        return [max_start / 2.0]

    return np.linspace(0.0, max_start, num=count).tolist()


def augment_audio(y, sr=TARGET_SR, rng=None):
    rng = rng or np.random.default_rng()
    mode = rng.choice(["noise", "gain", "shift", "speed_up", "speed_down"])

    if mode == "noise":
        augmented = y + 0.003 * rng.standard_normal(len(y))
    elif mode == "gain":
        augmented = y * rng.uniform(0.8, 1.2)
    elif mode == "shift":
        shift = int(rng.integers(low=-sr // 4, high=sr // 4))
        augmented = np.roll(y, shift)
    elif mode == "speed_up":
        augmented = signal.resample(y, max(1, int(len(y) / 1.1)))
    else:
        augmented = signal.resample(y, int(len(y) / 0.9))

    augmented = fix_audio_length(augmented, sr=sr, duration=len(y) / sr)
    peak = np.max(np.abs(augmented))
    if peak > 0:
        augmented = augmented / peak
    return augmented.astype(np.float32)


def _fit_feature_map(feature_map, size=FEATURE_SIZE):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    freq, time = feature_map.shape

    if freq < size:
        feature_map = np.pad(feature_map, ((0, size - freq), (0, 0)))
    else:
        feature_map = feature_map[:size, :]

    if time < size:
        feature_map = np.pad(feature_map, ((0, 0), (0, size - time)))
    else:
        feature_map = feature_map[:, :size]

    return feature_map


def _standardize(feature_map):
    mean = np.mean(feature_map, dtype=np.float32)
    std = np.std(feature_map, dtype=np.float32)
    return (feature_map - mean) / max(std, 1e-6)


def _to_unit_range(feature_map):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    min_val = np.min(feature_map)
    max_val = np.max(feature_map)
    if max_val - min_val < 1e-6:
        return np.zeros_like(feature_map, dtype=np.float32)
    return (feature_map - min_val) / (max_val - min_val)


def _spectrogram_channel(y, sr, nperseg):
    _, _, spec = signal.spectrogram(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="spectrum",
        mode="magnitude",
    )
    return np.log1p(spec).astype(np.float32)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr=TARGET_SR, n_fft=N_FFT, n_mels=N_MELS):
    mel_min = _hz_to_mel(0)
    mel_max = _hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fft_bins = np.clip(fft_bins, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left = fft_bins[i - 1]
        center = fft_bins[i]
        right = fft_bins[i + 1]

        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)

        for j in range(left, center):
            filterbank[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            filterbank[i - 1, j] = (right - j) / max(right - center, 1)

    return filterbank


def _stft_power(y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    _, _, zxx = signal.stft(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    return (np.abs(zxx) ** 2).astype(np.float32)


def _chroma_from_power(power_spec, sr=TARGET_SR, n_fft=N_FFT):
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    chroma = np.zeros((12, power_spec.shape[1]), dtype=np.float32)

    for idx, freq in enumerate(freqs):
        if freq <= 0:
            continue
        midi = int(np.round(69 + 12 * np.log2(freq / 440.0)))
        chroma[midi % 12] += power_spec[idx]

    return chroma


def get_feature_stack(y, sr=TARGET_SR):
    power_spec = _stft_power(y, sr=sr)
    mel_spec = np.dot(_mel_filterbank(sr=sr), power_spec)
    log_mel = np.log1p(mel_spec)
    mfcc = dct(log_mel, axis=0, type=2, norm="ortho")[:N_MFCC]
    chroma = _chroma_from_power(power_spec, sr=sr)
    delta = np.gradient(log_mel, axis=1)

    channels = []
    for feature_map in (log_mel, mfcc, chroma, delta):
        channels.append(_standardize(_fit_feature_map(feature_map)))

    return np.stack(channels, axis=-1).astype(np.float32)


def get_transfer_input(y, sr=TARGET_SR):
    power_spec = _stft_power(y, sr=sr)
    mel_spec = np.dot(_mel_filterbank(sr=sr), power_spec)
    log_mel = _fit_feature_map(np.log1p(mel_spec))
    delta = _fit_feature_map(np.gradient(log_mel, axis=1))
    chroma = _fit_feature_map(_chroma_from_power(power_spec, sr=sr))

    rgb = np.stack(
        [
            _to_unit_range(log_mel),
            _to_unit_range(delta),
            _to_unit_range(chroma),
        ],
        axis=-1,
    )
    return rgb.astype(np.float32)


def get_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = _fit_feature_map(mel_db)
    return mel_db[..., np.newaxis]


def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    return fig


def plot_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax)
    return fig
