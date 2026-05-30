import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = ROOT / "data" / "bias_labeled_dataset.json"
OUTPUT_DIR = ROOT / "data"


def main() -> None:
    with SOURCE_FILE.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    random.seed(42)
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)

    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:train_size + val_size]
    test_data = shuffled[train_size + val_size:]

    for filename, data in (
        ("train.json", train_data),
        ("validation.json", val_data),
        ("test.json", test_data),
    ):
        with (OUTPUT_DIR / filename).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train_data)} train samples")
    print(f"Wrote {len(val_data)} validation samples")
    print(f"Wrote {len(test_data)} test samples")


if __name__ == "__main__":
    main()
