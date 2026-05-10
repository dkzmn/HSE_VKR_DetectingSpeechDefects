from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GOOD_DIR = DATA_DIR / "good"
BAD_DIR = DATA_DIR / "bad"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
METRICS_FILE = EXPERIMENTS_DIR / "metrics_summary.csv"
DATASET_CSV = DATA_DIR / "dataset.csv"

CLASS_GOOD = 0
CLASS_BAD = 1
CLASS_NAMES = ["good", "bad"]

TRAIN_VAL_RATIO = 0.85
TEST_RATIO = 0.15
RANDOM_STATE = 42

# Аудио
TARGET_SR = 16000
N_MFCC = 20
N_MELS = 80
HOP_LENGTH = 512
WIN_LENGTH = 2048
MAX_DURATION_SEC = 10.0

# Обучение DL
DEFAULT_GRAD_CLIP = 1.0
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 10

CV_N_SPLITS = 5

# Аугментация
SPEC_FREQ_MASK_PARAM = 15 # макс. полос по частоте, которые маскируются
SPEC_TIME_MASK_PARAM = 30 # макс. кадров по времени, которые маскируются
SPEC_N_FREQ_MASKS = 2 # сколько раз применять частотную маску
SPEC_N_TIME_MASKS = 2 # сколько раз применять временну́ю маску
NOISE_AMPLITUDE = 0.005 # ампл. гауссовского шума
PITCH_SHIFT_STEPS = 2 # +/- полутона для pitch shift

THRESHOLD_GRID = [i / 100 for i in range(10, 90)]
