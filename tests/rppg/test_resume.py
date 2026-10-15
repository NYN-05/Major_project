"""Tests for rPPG extraction resume logic (identity vs runtime config split)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "WORKING"))

import json

from RPPG import extract_dataset_features as edf


def _make_identity_config():
    return {"method": "POS", "min_sqi": 0.05, "max_nan_features": 2,
            "blur_threshold": 3.0, "brightness_min": 30, "brightness_max": 235,
            "min_usable_frames": 24, "target_fps": None, "max_per_class": 20}


def _make_full_config(**overrides):
    base = {**_make_identity_config(), "gpu": True, "gpu_workers": 8, "workers": 0}
    base.update(overrides)
    return base


def _make_samples(n=10):
    """Create proper 4-tuple samples: (label, Path, source, root)."""
    return [(1, Path(f"v{i}.mp4"), "Fake", Path(".")) for i in range(n)]


class TestResumeConfigSplit(unittest.TestCase):
    """Resume should reject identity mismatches but tolerate runtime changes."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def _write_checkpoint(self, out_dir, config_dict, processed=5):
        ckpt = Path(out_dir) / "checkpoint.json"
        identity = {k: config_dict.get(k) for k in edf.CheckpointManager._IDENTITY_CONFIG_KEYS if k in config_dict}
        data = {
            "last_completed_index": processed - 1,
            "total_samples": 10,
            "processed_count": processed,
            "failed_count": 0,
            "no_features_count": 0,
            "gated_count": 0,
            "config_hash": edf.CheckpointManager._compute_config_hash(None, identity),
            "config": identity,
            "sample_hashes": ["h" + str(i) for i in range(10)],
            "completed_samples": [f"v{i}.mp4" for i in range(processed)],
            "completed_indices": list(range(processed)),
            "paused": False,
            "resume_from_index": -1,
        }
        ckpt.write_text(json.dumps(data), encoding="utf-8")
        return ckpt

    def test_runtime_gpu_change_loads_without_warning(self):
        """Changing --gpu from True to False should load cleanly (runtime key)."""
        out_dir = Path(self._dir.name)
        orig_config = _make_full_config()
        self._write_checkpoint(out_dir, orig_config)

        resume_config = _make_full_config(gpu=False, gpu_workers=0)

        mgr = edf.CheckpointManager(out_dir / "checkpoint.json", _make_samples(), resume_config)
        mgr._sample_hashes = ["h" + str(i) for i in range(10)]
        mgr._config_hash = mgr._compute_config_hash({k: resume_config.get(k) for k in mgr._IDENTITY_CONFIG_KEYS if k in resume_config})

        with mock.patch("builtins.print") as mp:
            ok = mgr.load()
        self.assertTrue(ok)
        printed = " ".join(str(c) for c in mp.call_args_list)
        self.assertNotIn("Config hash mismatch", printed)

    def test_identity_method_change_refuses(self):
        """Changing --method from POS to CHROM must refuse to resume."""
        out_dir = Path(self._dir.name)
        orig_config = _make_full_config()
        self._write_checkpoint(out_dir, orig_config)

        resume_config = _make_full_config(method="CHROM")

        mgr = edf.CheckpointManager(out_dir / "checkpoint.json", _make_samples(), resume_config)
        mgr._sample_hashes = ["h" + str(i) for i in range(10)]
        mgr._config_hash = mgr._compute_config_hash({k: resume_config.get(k) for k in mgr._IDENTITY_CONFIG_KEYS if k in resume_config})

        with mock.patch("builtins.print") as mp:
            ok = mgr.load()
        self.assertFalse(ok)
        printed = " ".join(str(c) for c in mp.call_args_list)
        self.assertIn("Config mismatch on identity keys", printed)
        self.assertIn("method", printed)

    def test_identity_max_per_class_change_refuses(self):
        """Changing --max-per-class must refuse."""
        out_dir = Path(self._dir.name)
        orig_config = _make_full_config()
        self._write_checkpoint(out_dir, orig_config)

        resume_config = _make_full_config(max_per_class=10)

        mgr = edf.CheckpointManager(out_dir / "checkpoint.json", _make_samples(), resume_config)
        mgr._sample_hashes = ["h" + str(i) for i in range(10)]
        mgr._config_hash = mgr._compute_config_hash({k: resume_config.get(k) for k in mgr._IDENTITY_CONFIG_KEYS if k in resume_config})

        with mock.patch("builtins.print") as mp:
            ok = mgr.load()
        self.assertFalse(ok)
        printed = " ".join(str(c) for c in mp.call_args_list)
        self.assertIn("Config mismatch on identity keys", printed)
        self.assertIn("max_per_class", printed)

    def test_sample_list_changed_refuses(self):
        """Sample list hash mismatch (e.g., different dataset root) refuses."""
        out_dir = Path(self._dir.name)
        orig_config = _make_full_config()
        self._write_checkpoint(out_dir, orig_config)

        resume_config = _make_full_config()

        mgr = edf.CheckpointManager(out_dir / "checkpoint.json", _make_samples(), resume_config)
        mgr._sample_hashes = ["different" + str(i) for i in range(10)]
        mgr._config_hash = mgr._compute_config_hash({k: resume_config.get(k) for k in mgr._IDENTITY_CONFIG_KEYS if k in resume_config})

        with mock.patch("builtins.print") as mp:
            ok = mgr.load()
        self.assertFalse(ok)
        printed = " ".join(str(c) for c in mp.call_args_list)
        self.assertIn("Sample list changed", printed)

    def test_resume_from_index_normalized_to_minus_one(self):
        """Resume_from_index should default to -1, not 0."""
        out_dir = Path(self._dir.name)
        ckpt = out_dir / "checkpoint.json"
        identity = _make_identity_config()
        data = {
            "last_completed_index": 4,
            "total_samples": 10,
            "processed_count": 5,
            "failed_count": 0,
            "no_features_count": 0,
            "gated_count": 0,
            "config_hash": edf.CheckpointManager._compute_config_hash(None, identity),
            "config": identity,
            "sample_hashes": ["h" + str(i) for i in range(10)],
            "completed_samples": [],
            "completed_indices": [0, 1, 2, 3, 4],
            "paused": False,
            # resume_from_index intentionally omitted
        }
        ckpt = out_dir / "checkpoint.json"
        ckpt.write_text(json.dumps(data), encoding="utf-8")

        mgr = edf.CheckpointManager(ckpt, _make_samples(), identity)  # Pass identity config for hash match
        mgr._sample_hashes = ["h" + str(i) for i in range(10)]
        mgr._config_hash = edf.CheckpointManager._compute_config_hash(None, identity)

        with mock.patch("builtins.print"):
            ok = mgr.load()
        self.assertTrue(ok)
        self.assertEqual(mgr.state.resume_from_index, -1)


if __name__ == "__main__":
    unittest.main()
