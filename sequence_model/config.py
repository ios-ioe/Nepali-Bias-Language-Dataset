import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
ATTENTION_DIR   = os.path.join(OUTPUT_DIR, "attention_plots")
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

for d in [DATA_DIR, OUTPUT_DIR, ATTENTION_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_FILE       = os.path.join(DATA_DIR, "bias_labeled_dataset.json")   # swap with your full file
LABEL_COLUMNS   = [
    "gender", "regional", "caste", "religion",
    "appearance", "socialstatus", "ambiguity",
    "political", "age", "disability", "none"
]
NUM_LABELS      = len(LABEL_COLUMNS)

MAX_VOCAB_SIZE  = 30_000
MAX_SEQ_LEN     = 64        # tokens; pad/truncate to this length
PAD_TOKEN       = "<PAD>"
UNK_TOKEN       = "<UNK>"

EMBEDDING_DIM   = 128
HIDDEN_DIM      = 256       # per LSTM direction (total = 2 × HIDDEN_DIM)
ATTENTION_DIM   = 128       # Bahdanau attention projection size
DROPOUT         = 0.4

BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15

THRESHOLD       = 0.50

SEED            = 42

MAX_PLOT_SAMPLES = 3
