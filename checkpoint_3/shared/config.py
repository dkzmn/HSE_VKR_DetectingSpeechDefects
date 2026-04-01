# -*- coding: utf-8 -*-
"""
Общая конфигурация экспериментов.
Пути относительно корня проекта (папка VKR).
"""
from pathlib import Path

# Корень проекта: родитель папки experiments
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GOOD_DIR = DATA_DIR / "good"
BAD_DIR = DATA_DIR / "bad"
# Папка экспериментов, где лежит shared
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
METRICS_FILE = EXPERIMENTS_DIR / "metrics_summary.csv"
# CSV датасета: создаётся 00_dataset_analysis (filename, dir, label, duration, <буквы>)
DATASET_CSV = DATA_DIR / "dataset.csv"

# Разбиение данных
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# Аудио по умолчанию (для признаков и моделей)
TARGET_SR = 16000
N_MFCC = 20
N_MELS = 80
HOP_LENGTH = 512
WIN_LENGTH = 2048
MAX_DURATION_SEC = 10.0  # обрезка/паддинг до этой длительности

# Метки классов
CLASS_GOOD = 0
CLASS_BAD = 1
CLASS_NAMES = ["good", "bad"]

# Обучение DL (по умолчанию)
DEFAULT_GRAD_CLIP = 1.0
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 10
