import csv

import numpy as np

from quantum.config import DataConfig
from quantum.conditioning import FeatureConditioner, validate_labels, validate_feature_matrix
from quantum.features_schema import FEATURE_NAMES, LABEL_FAKE, LABEL_REAL


class SyntheticRPPGGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _real_sample(self) -> np.ndarray:
        latent = self.rng.normal(0.7, 0.15)
        primary = np.clip(latent + self.rng.normal(0.0, 0.06, size=6), 0.1, 1.0)
        aux = np.clip(self.rng.beta(6.0, 2.0, size=3), 0.1, 1.0)
        return np.concatenate([primary, aux])

    def _fake_sample(self) -> np.ndarray:
        latent = self.rng.normal(0.3, 0.18)
        primary = np.clip(latent + self.rng.normal(0.0, 0.16, size=6), 0.0, 1.0)
        aux = np.clip(self.rng.beta(3.5, 3.5, size=3), 0.0, 1.0)
        return np.concatenate([primary, aux])

    def generate(self, n_real: int, n_fake: int) -> tuple[np.ndarray, np.ndarray]:
        real_x = np.stack([self._real_sample() for _ in range(n_real)])
        fake_x = np.stack([self._fake_sample() for _ in range(n_fake)])
        X = np.vstack([real_x, fake_x])
        y = np.concatenate([np.full(n_real, LABEL_REAL), np.full(n_fake, LABEL_FAKE)])
        order = self.rng.permutation(len(y))
        return validate_feature_matrix(X[order]), validate_labels(y[order])


def generate_dataset(cfg: DataConfig) -> dict[str, np.ndarray]:
    generator = SyntheticRPPGGenerator(seed=cfg.seed)
    X_train, y_train = generator.generate(cfg.train_per_class, cfg.train_per_class)
    X_val, y_val = generator.generate(cfg.val_per_class, cfg.val_per_class)
    X_test, y_test = generator.generate(cfg.test_per_class, cfg.test_per_class)

    conditioner = FeatureConditioner().fit(X_train)
    X_train = conditioner.transform(X_train)
    X_val = conditioner.transform(X_val)
    X_test = conditioner.transform(X_test)

    cfg.data_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg.data_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
    )
    conditioner.save(cfg.scaler_file)
    write_csv(cfg.csv_file, X_train, y_train, X_val, y_val, X_test, y_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def write_csv(
    path,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["split", "label"] + FEATURE_NAMES)
        for split, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
            for row, label in zip(X, y):
                writer.writerow([split, int(label), *[f"{v:.6f}" for v in row]])


def load_dataset(data_file) -> dict[str, np.ndarray]:
    data = np.load(data_file, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "feature_names": list(data["feature_names"]),
    }


if __name__ == "__main__":
    data = generate_dataset(DataConfig())
    for key in ("X_train", "X_val", "X_test"):
        print(key, data[key].shape)
