"""Pytest wrapper for quantum layer contract tests.

These tests are defined in WORKING/quantum/tests.py and re-exported here
for pytest discovery. The two pytest-specific additions are:
- test_metadata_grouping_prevents_correlated_clip_split
- test_split_manifest_integrity_rejects_group_overlap
"""

import sys
from pathlib import Path

# Ensure WORKING root is importable for quantum.* modules
WORKING_ROOT = str(Path(__file__).resolve().parent.parent.parent / "WORKING")
if WORKING_ROOT not in sys.path:
    sys.path.insert(0, WORKING_ROOT)

# Import all shared contract tests from the canonical source
from quantum.tests import (
    test_beta_alive,
    test_checkpoint_state_dict_compat,
    test_dfdc_subject_grouping,
    test_feature_contract_sync,
    test_hamiltonian_matches_classical,
    test_label_conversion,
    test_no_group_leakage,
    test_qaoa_sim_matches_pennylane,
    test_real_hamiltonian_verification,
    test_split_determinism,
    test_torch_layer_matches_pennylane,
)

# Pytest-specific additions that need fixtures from tests/conftest.py
def test_metadata_grouping_prevents_correlated_clip_split():
    """Verify split groups prevent correlated clips from leaking across splits."""
    from quantum.data import _grouped_train_val_test_split, DataConfig
    import numpy as np
    import pandas as pd
    
    # Create synthetic data with known group structure
    # 6 subjects, 2 clips each (12 total)
    n_subjects = 6
    clips_per_subject = 2
    n_samples = n_subjects * clips_per_subject
    
    X = np.random.randn(n_samples, 59)
    y = np.array([0, 1] * (n_samples // 2))
    groups = np.repeat(np.arange(n_subjects), clips_per_subject)
    paths = [f"subj{g}_clip{c}.mp4" for g, c in zip(groups, np.tile(np.arange(clips_per_subject), n_subjects))]
    
    cfg = DataConfig(seed=42, val_ratio=0.2, test_ratio=0.2)
    split = _grouped_train_val_test_split(X, y, groups, paths, cfg)
    
    # Verify no group appears in more than one split
    for split_name in ["train", "val", "test"]:
        split_groups = set(split[f"groups_{split_name}"])
        for other_name in ["train", "val", "test"]:
            if other_name != split_name:
                other_groups = set(split[f"groups_{other_name}"])
                assert split_groups.isdisjoint(other_groups), (
                    f"Group leakage between {split_name} and {other_name}"
                )

def test_split_manifest_integrity_rejects_group_overlap():
    """Verify split manifest records group assignment and rejects overlap."""
    import json
    import tempfile
    from pathlib import Path
    
    from quantum.data import _grouped_train_val_test_split, DataConfig
    from quantum.config import sha256_file
    import numpy as np
    
    # Create data with clear group structure
    n_subjects = 4
    clips_per_subject = 3
    n_samples = n_subjects * clips_per_subject
    
    X = np.random.randn(n_samples, 59)
    y = np.array([0, 1] * (n_samples // 2))
    groups = np.repeat(np.arange(n_subjects), clips_per_subject)
    paths = [f"subj{g}_clip{c}.mp4" for g, c in zip(groups, np.tile(np.arange(3), 4))]
    
    cfg = DataConfig(seed=123, val_ratio=0.25, test_ratio=0.25)
    split = _grouped_train_val_test_split(X, y, groups, paths, cfg)
    
    # Write manifest to temp file
    with tempfile.TemporaryDirectory() as d:
        manifest_path = Path(d) / "manifest.json"
        import json
        manifest = {
            "seed": cfg.seed,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
            "grouping": "metadata-first-grouped-split",
            "rows": {},
        }
        for split_name in ["train", "val", "test"]:
            for p, g in zip(split[f"paths_{split_name}"], split[f"groups_{split_name}"]):
                manifest["rows"][str(p)] = {"split": split_name, "group": str(g)}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        # Read back and verify integrity
        loaded = json.loads(manifest_path.read_text())
        
        # Verify each path appears exactly once
        all_paths = []
        for split_name in ["train", "val", "test"]:
            all_paths.extend(split[f"paths_{split_name}"])
        assert set(all_paths) == set(loaded["rows"].keys())
        
        # Verify no group overlap
        split_groups = {}
        for split_name in ["train", "val", "test"]:
            split_groups[split_name] = set(
                row["group"] for row in loaded["rows"].values()
                if row["split"] == split_name
            )
        
        for name1, groups1 in split_groups.items():
            for name2, groups2 in split_groups.items():
                if name1 != name2:
                    assert groups1.isdisjoint(groups2), f"Group overlap: {name1} & {name2}"

# Re-export for pytest discovery
__all__ = [
    "test_beta_alive",
    "test_checkpoint_state_dict_compat",
    "test_dfdc_subject_grouping",
    "test_feature_contract_sync",
    "test_hamiltonian_matches_classical",
    "test_label_conversion",
    "test_metadata_grouping_prevents_correlated_clip_split",
    "test_no_group_leakage",
    "test_qaoa_sim_matches_pennylane",
    "test_real_hamiltonian_verification",
    "test_split_determinism",
    "test_split_manifest_integrity_rejects_group_overlap",
    "test_torch_layer_matches_pennylane",
]
