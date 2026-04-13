# -*- coding: utf-8 -*-
"""
Утилиты загрузки данных и извлечения признаков.

Изменения по сравнению с checkpoint_3/shared/data_utils.py:
  1. get_holdout_split() — сначала отделяем test (holdout), потом CV только на train+val.
     Это исправляет selection bias: тест не участвует в выборе гиперпараметров.
  2. get_splits() сохранён для обратной совместимости.
  3. augment_mel_spectrogram() — SpecAugment (маски по частоте и времени).
  4. augment_waveform() — гауссовский шум и pitch shift.
  5. Все функции нормировки спектрограмм используют только train-статистику
     (mel_mean/mel_std вычисляются на train, применяются к val и test).
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import librosa
import pywt
import parselmouth
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold

from . import config


# ---------------------------------------------------------------------------
# Загрузка датасета
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    return config.PROJECT_ROOT


def load_dataset_csv():
    """
    Загружает data/dataset.csv.
    Возвращает (paths, labels, df) или (None, None, None).
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


def _letter_cols(df: pd.DataFrame):
    base = {"filename", "dir", "label", "duration", "path"}
    return [c for c in df.columns if c not in base]


# ---------------------------------------------------------------------------
# Разбиение данных
# ---------------------------------------------------------------------------

def get_holdout_split():
    """
    Правильный сплит данных:

      1. Сначала отрезаем HOLDOUT TEST (15%) — он больше не трогается до финальной оценки.
      2. Остаток (85%) — train+val для кросс-валидации.

    Возвращает:
        paths_trainval, labels_trainval, letters_trainval,
        paths_test, labels_test, letters_test
    """
    paths, labels, df = load_dataset_csv()
    letters = df[_letter_cols(df)].values.astype(np.float32)

    idx = np.arange(len(paths))
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=config.TEST_RATIO,
        stratify=labels,
        random_state=config.RANDOM_STATE,
    )
    return (
        paths[idx_trainval], labels[idx_trainval], letters[idx_trainval],
        paths[idx_test], labels[idx_test], letters[idx_test],
    )


def get_cv_folds(paths_trainval, labels_trainval, letters_trainval, n_splits=None):
    """
    Генератор фолдов StratifiedKFold для train+val части.

    Yields: (paths_tr, paths_val, labels_tr, labels_val,
              letters_tr, letters_val, fold_idx)
    """
    n_splits = n_splits or config.CV_N_SPLITS
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(paths_trainval, labels_trainval)):
        yield (
            paths_trainval[tr_idx], paths_trainval[val_idx],
            labels_trainval[tr_idx], labels_trainval[val_idx],
            letters_trainval[tr_idx], letters_trainval[val_idx],
            fold,
        )


def get_splits():
    """
    Обратная совместимость с checkpoint_3.
    Возвращает фиксированный сплит 70/15/15 (без CV).
    """
    paths, labels, df = load_dataset_csv()
    letters = df[_letter_cols(df)].values.astype(np.float32)

    idx = np.arange(len(paths))

    idx_tt, idx_test = train_test_split(
        idx, test_size=config.TEST_RATIO, stratify=labels, random_state=config.RANDOM_STATE
    )
    val_ratio_adj = config.VAL_RATIO / (1 - config.TEST_RATIO)
    idx_train, idx_val = train_test_split(
        idx_tt, test_size=val_ratio_adj, stratify=labels[idx_tt], random_state=config.RANDOM_STATE
    )
    return (
        paths[idx_train], paths[idx_val], paths[idx_test],
        labels[idx_train], labels[idx_val], labels[idx_test],
        letters[idx_train], letters[idx_val], letters[idx_test],
    )


# ---------------------------------------------------------------------------
# Загрузка аудио
# ---------------------------------------------------------------------------

def load_audio(path, sr=None, mono=True, max_sec=None):
    """Загружает WAV, приводит к sr, обрезает/паддит до max_sec."""
    sr = sr or config.TARGET_SR
    min_length = config.WIN_LENGTH
    max_samples = int((max_sec or config.MAX_DURATION_SEC) * sr)
    y, _ = librosa.load(path, sr=sr, mono=mono, duration=max_sec or config.MAX_DURATION_SEC)
    if len(y) < min_length:
        y = np.zeros(max(min_length, max_samples), dtype=np.float32)
    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)), mode="constant", constant_values=0)
    elif len(y) > max_samples:
        y = y[:max_samples]
    return y, sr


# ---------------------------------------------------------------------------
# Аугментация
# ---------------------------------------------------------------------------

def augment_waveform(y: np.ndarray, sr: int = None, add_noise: bool = True, pitch_shift: bool = True) -> np.ndarray:
    """
    Аугментация waveform:
      - Гауссовский шум (амплитуда config.NOISE_AMPLITUDE).
      - Pitch shift ±config.PITCH_SHIFT_STEPS полутонов (случайный знак).

    Используется только при обучении (не при валидации/тесте).
    """
    sr = sr or config.TARGET_SR
    y = y.copy()
    if add_noise:
        noise = np.random.normal(0, config.NOISE_AMPLITUDE, size=y.shape).astype(np.float32)
        y = y + noise
    if pitch_shift:
        n_steps = np.random.choice([-config.PITCH_SHIFT_STEPS, config.PITCH_SHIFT_STEPS])
        y = librosa.effects.pitch_shift(y.astype(float), sr=sr, n_steps=n_steps).astype(np.float32)
    return y


def augment_mel_spectrogram(mel: np.ndarray) -> np.ndarray:
    """
    SpecAugment на мел-спектрограмме (H, T) или (1, H, T):
      - Частотная маска: маскируем config.SPEC_N_FREQ_MASKS случайных полос
        шириной до config.SPEC_FREQ_MASK_PARAM по оси H.
      - Временна́я маска: маскируем config.SPEC_N_TIME_MASKS случайных отрезков
        шириной до config.SPEC_TIME_MASK_PARAM по оси T.

    Маски заполняются средним значением спектрограммы.
    Используется только при обучении.
    """
    mel = mel.copy()
    squeeze = mel.ndim == 3  # (1, H, T)
    if squeeze:
        mel = mel[0]  # -> (H, T)

    H, T = mel.shape
    fill = mel.mean()

    for _ in range(config.SPEC_N_FREQ_MASKS):
        f = np.random.randint(0, config.SPEC_FREQ_MASK_PARAM + 1)
        f0 = np.random.randint(0, max(1, H - f))
        mel[f0: f0 + f, :] = fill

    for _ in range(config.SPEC_N_TIME_MASKS):
        t = np.random.randint(0, config.SPEC_TIME_MASK_PARAM + 1)
        t0 = np.random.randint(0, max(1, T - t))
        mel[:, t0: t0 + t] = fill

    if squeeze:
        mel = mel[np.newaxis, ...]
    return mel


# ---------------------------------------------------------------------------
# Извлечение признаков (те же функции, что в checkpoint_3)
# ---------------------------------------------------------------------------

def extract_mfcc_stats(path, n_mfcc=None, sr=None, hop_length=None, win_length=None):
    """MFCC → mean+std+delta_mean+delta_std → вектор (n_mfcc*4,)."""
    n_mfcc = n_mfcc or config.N_MFCC
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    win_length = win_length or config.WIN_LENGTH

    y, _ = load_audio(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=win_length)
    mean_ = mfcc.mean(axis=1)
    std_ = mfcc.std(axis=1)
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
    """MFCC последовательность (time, n_mfcc) для RNN."""
    n_mfcc = n_mfcc or config.N_MFCC
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    y, _ = load_audio(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    seq = mfcc.T.astype(np.float32)
    if max_frames is not None:
        if seq.shape[0] < max_frames:
            seq = np.pad(seq, ((0, max_frames - seq.shape[0]), (0, 0)), mode="constant")
        else:
            seq = seq[:max_frames]
    return seq


def extract_mfcc_image(path, n_mfcc=None, sr=None, hop_length=None, max_frames=None, stack_delta=True):
    """MFCC + delta как 2-канальное изображение (C, n_mfcc, T)."""
    n_mfcc = n_mfcc or config.N_MFCC
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    y, _ = load_audio(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    if max_frames is not None:
        n = mfcc.shape[1]
        if n < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - n)))
        else:
            mfcc = mfcc[:, :max_frames]
    if stack_delta:
        width = min(9, mfcc.shape[1]) if mfcc.shape[1] >= 3 else 3
        if width % 2 == 0:
            width -= 1
        delta = librosa.feature.delta(mfcc, width=width)
        out = np.stack([mfcc.astype(np.float32), delta.astype(np.float32)], axis=0)
    else:
        out = mfcc[np.newaxis, ...].astype(np.float32)
    return out


def extract_mel_spectrogram(path, n_mels=None, sr=None, hop_length=None, max_frames=None, log=True):
    """Мел-спектрограмма (n_mels, T). log=True → dB."""
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
            mel = np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])),
                         mode="constant", constant_values=mel.min())
        else:
            mel = mel[:, :max_frames]
    return mel


def extract_cwt_scalogram(path, sr=None, max_frames=None, n_scales=64, wavelet="morl"):
    """CWT-скалограмма (1, n_scales, T)."""
    sr = sr or config.TARGET_SR
    y, _ = load_audio(path, sr=sr)
    scales = np.arange(1, n_scales + 1, dtype=np.float64)
    coef, _ = pywt.cwt(y, scales, wavelet)
    img = np.abs(coef).astype(np.float32)
    if max_frames is not None:
        T = img.shape[1]
        if T < max_frames:
            img = np.pad(img, ((0, 0), (0, max_frames - T)))
        else:
            img = img[:, :max_frames]
    return img[np.newaxis, ...]


def extract_stft_db(path, sr=None, n_fft=2048, hop_length=None, max_frames=None):
    """STFT magnitude в dB (1, n_fft//2+1, T)."""
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    y, _ = load_audio(path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S + 1e-10, ref=np.max).astype(np.float32)
    if max_frames is not None:
        T = S_db.shape[1]
        if T < max_frames:
            S_db = np.pad(S_db, ((0, 0), (0, max_frames - T)),
                          mode="constant", constant_values=S_db.min())
        else:
            S_db = S_db[:, :max_frames]
    return S_db[np.newaxis, ...]


def extract_vocal_tract_features(path, sr=None, hop_length=None):
    """Признаки голосового тракта: RMS, ZCR, F0, спектральные + jitter/shimmer/форманты."""
    sr = sr or config.TARGET_SR
    hop_length = hop_length or config.HOP_LENGTH
    y, _ = load_audio(path, sr=sr)
    n_fft = 2048

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    cent = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, hop_length=hop_length)[0]
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                            sr=sr, hop_length=hop_length)
    f0_flat = np.nan_to_num(f0, nan=0.0)

    feats = []
    for arr in [rms, zcr, cent, rolloff, bandwidth, f0_flat]:
        feats.extend([np.mean(arr), np.std(arr), np.max(arr)])
    out = np.array(feats, dtype=np.float32)

    snd = parselmouth.Sound(path)
    pitch_ps = snd.to_pitch(time_step=0.01)
    f0_ps = pitch_ps.selected_array["frequency"]
    f0_ps[f0_ps == 0] = np.nan
    jitter = (
        np.nanmean(np.abs(np.diff(f0_ps))) / (np.nanmean(f0_ps) + 1e-8)
        if np.nansum(f0_ps) > 0 else 0.0
    )
    intensity = parselmouth.praat.call(snd, "To Intensity", 75, 0.01, True)
    int_vals = [
        parselmouth.praat.call(intensity, "Get value at time", t, "Cubic")
        for t in np.arange(0, snd.duration, 0.01)
    ]
    int_vals = np.array([x for x in int_vals if x is not None and not np.isnan(x)])
    shimmer = (
        np.mean(np.abs(np.diff(int_vals))) / (np.mean(int_vals) + 1e-8)
        if len(int_vals) > 1 else 0.0
    )
    formants = snd.to_formant_burg(time_step=0.01)
    F1 = np.array([
        parselmouth.praat.call(formants, "Get value at time", 1, t, "Hertz", "Linear")
        for t in np.arange(0, min(0.5, snd.duration), 0.01)
    ])
    F1 = F1[~np.isnan(F1)]
    F2 = np.array([
        parselmouth.praat.call(formants, "Get value at time", 2, t, "Hertz", "Linear")
        for t in np.arange(0, min(0.5, snd.duration), 0.01)
    ])
    F2 = F2[~np.isnan(F2)]
    out = np.concatenate([out, [
        float(F1.mean()) if len(F1) else 0, float(F1.std()) if len(F1) else 0,
        float(F2.mean()) if len(F2) else 0, float(F2.std()) if len(F2) else 0,
        jitter, shimmer,
    ]])
    return out


def build_feature_matrix(paths, extractor, n_jobs=-1):
    """Параллельно строит матрицу признаков: extractor(path) -> вектор."""
    try:
        rows = Parallel(n_jobs=n_jobs)(delayed(extractor)(p) for p in paths)
    except Exception:
        rows = [extractor(p) for p in paths]
    return np.stack(rows)
