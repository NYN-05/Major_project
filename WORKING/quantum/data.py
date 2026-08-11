import numpy as np

from quantum.config import DataConfig, FEATURE_NAMES, LABEL_FAKE, LABEL_REAL

EFFECTS = np.array([0.30, 0.28, 0.26, 0.24, 0.22, 0.26, 0.06, 0.08, 0.05])


def _sample_class(n_samples, shift, rng):
    features = rng.uniform(0.15, 0.85, size=(n_samples, len(FEATURE_NAMES)))
    features = features + shift * EFFECTS + rng.normal(0.0, 0.10, size=features.shape)
    return np.clip(features, 0.0, 1.0)


def _split(n_per_class, shift_real, rng):
    X = np.concatenate(
        [_sample_class(n_per_class, shift_real, rng), _sample_class(n_per_class, 0.0, rng)]
    )
    y = np.concatenate(
        [np.full(n_per_class, LABEL_REAL), np.full(n_per_class, LABEL_FAKE)]
    )
    order = rng.permutation(len(y))
    return X[order], y[order]


def generate_dataset(cfg=None):
    cfg = cfg or DataConfig()
    rng = np.random.default_rng(cfg.seed)
    X_train, y_train = _split(cfg.train_per_class, 0.30, rng)
    X_val, y_val = _split(cfg.val_per_class, 0.30, rng)
    X_test, y_test = _split(cfg.test_per_class, 0.30, rng)
    cfg.data_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.data_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=FEATURE_NAMES,
    )
    return load_dataset(cfg.data_file)


def load_dataset(path=None):
    path = path or DataConfig().data_file
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it first with: python -m quantum.run --gen-data"
        )
    data = np.load(path)
    return {key: data[key] for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}