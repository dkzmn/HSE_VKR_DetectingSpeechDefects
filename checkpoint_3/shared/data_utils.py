from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import librosa
import pywt
import parselmouth

from . import config


def get_project_root():
    return config.PROJECT_ROOT


def load_dataset_csv():
    """
    Загружает CSV датасета data/dataset.csv
    Колонки: filename, dir, label, duration, <буквы>.
    Возвращает (paths, labels, df) или (None, None, None), если файла нет.
    """
    
    try:
        df = pd.read_csv(config.DATASET_CSV, encoding="utf-8")
        paths = np.array([
            str(config.DATA_DIR / row["dir"] / row["filename"])
            for _, row in df.iterrows()
        ])
        labels = df["label"].values.astype(np.int64)
        df = df.copy()
        df["path"] = paths
        return paths, labels, df
    except Exception:
        return None, None, None


def get_splits():
    """
    Разбиение на train/val/test.
    """
    val_ratio = config.VAL_RATIO
    test_ratio = config.TEST_RATIO
    random_state = config.RANDOM_STATE

    letter_matrix = None
    paths, labels, df = load_dataset_csv()
    base_cols = {"filename", "dir", "label", "duration", "path"}
    letter_cols = [c for c in df.columns if c not in base_cols]
    letter_matrix = df[letter_cols].values.astype(np.float32)

    paths_tt, paths_test, labels_tt, labels_test = train_test_split(
        paths, labels, test_size=test_ratio, stratify=labels, random_state=random_state
    )

    val_ratio_adj = val_ratio / (1 - test_ratio)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tt, labels_tt, test_size=val_ratio_adj, stratify=labels_tt, random_state=random_state
    )

    n = len(paths)
    idx = np.arange(n)
    _, idx_test = train_test_split(
        idx, test_size=test_ratio, stratify=labels, random_state=random_state
    )
    idx_tt = np.setdiff1d(idx, idx_test)
    _, idx_val = train_test_split(
        idx_tt, test_size=val_ratio_adj, stratify=labels[idx_tt], random_state=random_state
    )
    idx_train = np.setdiff1d(idx_tt, idx_val)
    letters_train = letter_matrix[idx_train]
    letters_val = letter_matrix[idx_val]
    letters_test = letter_matrix[idx_test]
    return (
        paths_train, paths_val, paths_test,
        labels_train, labels_val, labels_test,
        letters_train, letters_val, letters_test,
    )


def load_audio(path, sr=None, mono=True, max_sec=None):
    """Загружает один WAV, приводит к sr, моно, опционально обрезает по длительности."""
    sr = sr or config.TARGET_SR
    min_length = config.WIN_LENGTH  # минимум для n_fft в спектральных признаках
    max_samples = int((max_sec or config.MAX_DURATION_SEC) * sr) if max_sec is not None else None
    y, sr_out = librosa.load(path, sr=sr, mono=mono, duration=max_sec)
    # Пустой или слишком короткий сигнал — подставляем тишину, чтобы не ломать n_fft в MFCC/mel
    if len(y) < min_length:
        y = np.zeros(max(min_length, max_samples or sr), dtype=np.float32)
    if max_samples is not None and len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)), mode="constant", constant_values=0)
    elif max_samples is not None and len(y) > max_samples:
        y = y[:max_samples]
    return y, sr_out


def extract_mfcc_stats(path, n_mfcc=None, sr=None, hop_length=None, win_length=None):
    """
    Извлекает MFCC по кадрам и агрегирует в один вектор: mean, std по времени для каждого коэффициента.
    Возвращает вектор формы (n_mfcc * 2,) или с delta (n_mfcc * 4,).
    """
    n_mfcc = n_mfcc or config.N_MFCC
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    win_length = win_length or config.WIN_LENGTH

    y, _ = load_audio(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=win_length)
    # (n_mfcc, time)
    mean_ = mfcc.mean(axis=1)
    std_ = mfcc.std(axis=1)
    # delta требует по времени не меньше width (по умолчанию 9); для очень коротких записей — окно меньше или нули
    n_frames = mfcc.shape[1]
    if n_frames < 3:
        mean_d = np.zeros(n_mfcc, dtype=np.float32)
        std_d = np.zeros(n_mfcc, dtype=np.float32)
    else:
        width = min(9, n_frames)
        if width % 2 == 0:
            width -= 1
        delta = librosa.feature.delta(mfcc, width=width)
        mean_d = delta.mean(axis=1).astype(np.float32)
        std_d = delta.std(axis=1).astype(np.float32)
    return np.concatenate([mean_, std_, mean_d, std_d]).astype(np.float32)


def extract_mfcc_sequence(path, n_mfcc=None, sr=None, hop_length=None, max_frames=None):
    """Извлекает последовательность MFCC (time, n_mfcc) для RNN/ LSTM. При max_frames обрезает или паддит."""
    n_mfcc = n_mfcc or config.N_MFCC
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH

    y, _ = load_audio(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # (n_mfcc, time) -> (time, n_mfcc)
    seq = mfcc.T.astype(np.float32)
    if max_frames is not None:
        if seq.shape[0] < max_frames:
            seq = np.pad(seq, ((0, max_frames - seq.shape[0]), (0, 0)), mode="constant", constant_values=0)
        else:
            seq = seq[:max_frames]
    return seq


def extract_mfcc_image(path, n_mfcc=None, sr=None, hop_length=None, max_frames=None, stack_delta=True):
    """
    MFCC как «изображение» для 2D CNN (Kuo-style, exp_15).
    Возвращает (C, n_mfcc, T): C=2 если stack_delta (mfcc, delta), иначе C=1.
    """
    n_mfcc = n_mfcc or config.N_MFCC
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH

    y, _ = load_audio(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    n_frames = mfcc.shape[1]
    if max_frames is not None:
        if n_frames < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - n_frames)), mode="constant", constant_values=0)
        else:
            mfcc = mfcc[:, :max_frames]
        n_frames = max_frames
    if stack_delta:
        width = min(9, mfcc.shape[1]) if mfcc.shape[1] >= 3 else 3
        if width % 2 == 0:
            width -= 1
        delta = librosa.feature.delta(mfcc, width=width)
        out = np.stack([mfcc.astype(np.float32), delta.astype(np.float32)], axis=0)
    else:
        out = mfcc[np.newaxis, ...].astype(np.float32)
    return out


def extract_cwt_scalogram(path, sr=None, max_frames=None, n_scales=64, wavelet="morl"):
    """
    CWT-скалограмма для exp_20: (1, n_scales, T). При max_frames обрезает/паддит по времени.
    """
    sr = sr or config.TARGET_SR
    y, _ = load_audio(path, sr=sr)
    # Шкалы: от высоких частот к низким (меньше scale = выше частота)
    scales = np.arange(1, n_scales + 1, dtype=np.float64)
    coef, _ = pywt.cwt(y, scales, wavelet)
    # coef: (n_scales, n_samples) — возьмём magnitude для вещественного вейвлета
    img = np.abs(coef).astype(np.float32)
    if max_frames is not None:
        T = img.shape[1]
        if T < max_frames:
            img = np.pad(img, ((0, 0), (0, max_frames - T)), mode="constant", constant_values=0)
        else:
            img = img[:, :max_frames]
    return img[np.newaxis, ...]


def extract_stft_db(path, sr=None, n_fft=2048, hop_length=None, max_frames=None):
    """
    STFT → magnitude → dB для exp_16 (SLINet-style). Возвращает (1, n_fft//2+1, T).
    """
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    y, _ = load_audio(path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S + 1e-10, ref=np.max)
    img = S_db.astype(np.float32)
    if max_frames is not None:
        T = img.shape[1]
        if T < max_frames:
            img = np.pad(img, ((0, 0), (0, max_frames - T)), mode="constant", constant_values=img.min())
        else:
            img = img[:, :max_frames]
    return img[np.newaxis, ...]


def extract_mel_spectrogram(path, n_mels=None, sr=None, hop_length=None, max_frames=None, log=True):
    """Извлекает мел-спектрограмму (n_mels, time). Если log — в логарифме (dB)."""
    n_mels = n_mels or config.N_MELS
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH

    y, _ = load_audio(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    if log:
        mel = librosa.power_to_db(mel, ref=np.max)
    mel = mel.astype(np.float32)
    if max_frames is not None:
        if mel.shape[1] < max_frames:
            mel = np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])), mode="constant", constant_values=mel.min())
        else:
            mel = mel[:, :max_frames]
    return mel


def build_feature_matrix(paths, extractor, n_jobs=-1):
    """Строит матрицу признаков для списка путей. extractor(path) -> vector."""
    try:        
        rows = Parallel(n_jobs=n_jobs)(delayed(extractor)(p) for p in paths)
    except Exception:
        rows = [extractor(p) for p in paths]
    return np.stack(rows)


def extract_vocal_tract_features(path, sr=None, hop_length=None):
    """
    Признаки голосового тракта для скрининга по скороговоркам (exp_19).
    Возвращает вектор: статистики по энергии, ZCR, pitch (F0), спектральные,
    опционально jitter/shimmer/formants если установлен parselmouth.
    """
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    # print(path)
    y, _ = load_audio(path, sr=sr)
    n_fft = 2048

    # Энергия (RMS)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
    # Спектральные: centroid, rolloff, bandwidth
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    cent = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, hop_length=hop_length)[0]
    # Pitch (F0) — pyin даёт NaN в тишине
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=hop_length)
    f0_flat = np.nan_to_num(f0, nan=0.0)

    feats = []
    for arr in [rms, zcr, cent, rolloff, bandwidth, f0_flat]:
        feats.extend([np.mean(arr), np.std(arr), np.max(arr)])
    out = np.array(feats, dtype=np.float32)
    # Parselmouth: jitter, shimmer, formants
    snd = parselmouth.Sound(path)
    pitch_ps = snd.to_pitch(time_step=0.01)
    f0_ps = pitch_ps.selected_array["frequency"]
    f0_ps[f0_ps == 0] = np.nan
    if np.nansum(f0_ps) > 0:
        jitter = np.nanmean(np.abs(np.diff(f0_ps))) / (np.nanmean(f0_ps) + 1e-8)
    else:
        jitter = 0.0
    intensity = parselmouth.praat.call(snd, "To Intensity", 75, 0.01, True)
    # Для Intensity "Get value at time" принимает (time, interpolationMethod)
    int_vals = [parselmouth.praat.call(intensity, "Get value at time", t, "Cubic") for t in np.arange(0, snd.duration, 0.01)]
    int_vals = np.array([x for x in int_vals if not (x is None or np.isnan(x))], dtype=np.float64)
    if len(int_vals) > 1:
        shimmer = np.mean(np.abs(np.diff(int_vals))) / (np.mean(int_vals) + 1e-8)
    else:
        shimmer = 0.0
    formants = snd.to_formant_burg(time_step=0.01)
    F1 = [parselmouth.praat.call(formants, "Get value at time", 1, t, "Hertz", "Linear") for t in np.arange(0, min(0.5, snd.duration), 0.01)]
    F1 = np.array([x for x in F1 if not (x is None or np.isnan(x))])
    F2 = [parselmouth.praat.call(formants, "Get value at time", 2, t, "Hertz", "Linear") for t in np.arange(0, min(0.5, snd.duration), 0.01)]
    F2 = np.array([x for x in F2 if not (x is None or np.isnan(x))])
    out = np.concatenate([out, [
        float(np.mean(F1)) if len(F1) else 0, float(np.std(F1)) if len(F1) else 0,
        float(np.mean(F2)) if len(F2) else 0, float(np.std(F2)) if len(F2) else 0,
        jitter, shimmer
    ]])
    return out
