"""Tests for rPPG dataset CSV durability helpers (retry, batching, lock)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "WORKING"))

import pandas as pd

from RPPG import extract_dataset_features as edf


def _entry(video, label=1):
    return {"label": label, "video_path": video, "heart_rate_bpm": 72.0}


class TestAtomicCsvReplace(unittest.TestCase):
    """_atomic_csv_replace retries transient locks, then gives up."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.csv = Path(self._dir.name) / "dataset_features.csv"
        self._old_base = edf._CSV_RETRY_BASE_S
        edf._CSV_RETRY_BASE_S = 0.01
        self.addCleanup(setattr, edf, "_CSV_RETRY_BASE_S", self._old_base)
        self.addCleanup(self._dir.cleanup)

    def test_retries_transient_lock_then_succeeds(self):
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            calls.append((str(src), str(dst)))
            if len(calls) < 3:
                raise PermissionError(13, "Access is denied")
            return real_replace(src, dst)

        df = pd.DataFrame([_entry("b.mp4"), _entry("a.mp4")])
        with mock.patch("os.replace", side_effect=flaky):
            edf._atomic_csv_replace(df, self.csv)
        self.assertEqual(len(calls), 3)
        out = pd.read_csv(str(self.csv))
        self.assertEqual(len(out), 2)

    def test_exhaustion_reraises_and_cleans_tmp(self):
        df = pd.DataFrame([_entry("a.mp4")])
        with mock.patch("os.replace", side_effect=PermissionError(13, "denied")):
            with self.assertRaises(PermissionError):
                edf._atomic_csv_replace(df, self.csv)
        leftovers = list(Path(self._dir.name).glob("*.tmp"))
        self.assertEqual(leftovers, [])
        self.assertFalse(self.csv.exists())

    def test_batch_append_is_sorted_and_complete(self):
        edf._append_entries_to_csv_atomic([_entry("c.mp4"), _entry("a.mp4")], self.csv)
        edf._append_entries_to_csv_atomic([_entry("b.mp4")], self.csv)
        out = pd.read_csv(str(self.csv))
        self.assertEqual(list(out["video_path"]), ["a.mp4", "b.mp4", "c.mp4"])

    def test_batch_append_empty_is_noop(self):
        edf._append_entries_to_csv_atomic([], self.csv)
        self.assertFalse(self.csv.exists())


class TestExtractionLock(unittest.TestCase):
    """_acquire_extraction_lock refuses a second writer, reclaims stale locks."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.addCleanup(edf._release_extraction_lock)

    def test_second_writer_refused_then_released(self):
        out_dir = Path(self._dir.name)
        edf._acquire_extraction_lock(out_dir)
        with self.assertRaises(SystemExit):
            edf._acquire_extraction_lock(out_dir)
        edf._release_extraction_lock()
        edf._acquire_extraction_lock(out_dir)  # works again after release

    def test_stale_lock_reclaimed(self):
        out_dir = Path(self._dir.name)
        stale = out_dir / edf._EXTRACTION_LOCK_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        stale.write_text("pid=999999", encoding="utf-8")
        old_mtime = stale.stat().st_mtime - (edf._EXTRACTION_LOCK_STALE_S + 3600)
        os.utime(str(stale), (old_mtime, old_mtime))
        edf._acquire_extraction_lock(out_dir)
        self.assertTrue(stale.exists())


if __name__ == "__main__":
    unittest.main()
