import argparse
import csv
import sys
from pathlib import Path

import numpy as np

QUANTUM_ROOT = Path(__file__).resolve().parents[0]
PROJECT_ROOT = QUANTUM_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum.conditioning import validate_labels, validate_feature_matrix
from quantum.features_schema import FEATURE_NAMES


def convert_csv_to_npz(csv_path: Path, output_path: Path) -> None:
    rows = []
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        header = reader.fieldnames
        if not header or header[0] != "split" or header[1] != "label":
            raise SystemExit(f"CSV must start with columns 'split,label', got {header}")
        if header[2:] != FEATURE_NAMES:
            raise SystemExit(
                f"Feature columns must be exactly {FEATURE_NAMES}, got {header[2:]}"
            )
        for row in reader:
            rows.append(row)

    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row["split"]
        if split not in splits:
            raise SystemExit(f"Unknown split '{split}' (must be train/val/test)")
        splits[split].append(row)

    result = {}
    for split, split_rows in splits.items():
        if not split_rows:
            raise SystemExit(f"Split '{split}' has no rows")
        X = np.asarray([[float(row[f]) for f in FEATURE_NAMES] for row in split_rows], dtype=np.float32)
        y = np.asarray([int(row["label"]) for row in split_rows], dtype=np.int64)
        validate_feature_matrix(X)
        validate_labels(y)
        result[f"X_{split}"] = X
        result[f"y_{split}"] = y

    result["feature_names"] = np.asarray(FEATURE_NAMES, dtype=object)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)
    for split in splits:
        print(f"{split}: {result[f'X_{split}'].shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert rPPG feature CSV to the quantum component's NPZ format.")
    parser.add_argument("--csv", required=True, help="Input CSV with columns split,label,<9 features>")
    parser.add_argument("--out", default=str(QUANTUM_ROOT / "output" / "data.npz"), help="Output NPZ path")
    args = parser.parse_args()

    convert_csv_to_npz(Path(args.csv), Path(args.out))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
