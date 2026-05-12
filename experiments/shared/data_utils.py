import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold

from . import config


# Загрузка датасета

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


def _aux_feature_cols(df: pd.DataFrame) -> list[str]:
    """Числовые вспомогательные признаки из CSV (буквенные флаги + duration и т.п.).

    Исключает служебные поля и строковые колонки.
    Результат используется как дополнительный вход к аудиопризнакам.
    """
    base = {"filename", "dir", "label", "path"}
    return [
        c for c in df.columns
        if c not in base and df[c].dtype != object
    ]


# Разбиение данных

def get_test_split():
    """
    Правильный сплит данных:

      1. Сначала отрезаем test (15%) — он не трогается до финальной оценки.
      2. Остаток (85%) — train+val для кросс-валидации.

    Возвращает:
        paths_trainval, labels_trainval, letters_trainval,
        paths_test, labels_test, letters_test
    """
    paths, labels, df = load_dataset_csv()
    letters = df[_aux_feature_cols(df)].values.astype(np.float32)

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


# Загрузка аудио

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


# Аугментация

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


# Извлечение признаков

def extract_mfcc_stats(path, n_mfcc=None, sr=None, hop_length=None, win_length=None):
    """MFCC  mean+std+delta_mean+delta_std  вектор (n_mfcc*4,)."""
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


def extract_mel_spectrogram(path, n_mels=None, sr=None, hop_length=None, max_frames=None, log=True):
    """Мел-спектрограмма (n_mels, T). log=True  dB."""
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


def build_feature_matrix(paths, extractor, n_jobs=-1):
    """Параллельно строит матрицу признаков: extractor(path) -> вектор."""
    try:
        rows = Parallel(n_jobs=n_jobs)(delayed(extractor)(p) for p in paths)
    except Exception:
        rows = [extractor(p) for p in paths]
    return np.stack(rows)