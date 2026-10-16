"""
tests/leakage_test.py — Phase 2 leakage test.

Verifies that the labelled training dataset has no leakage:
  * GDACS is never used as a training label (external alert evidence only).
  * Train/validation/test splits are disjoint.
  * No sample appears in more than one split.
"""

import sys
sys.path.insert(0, 'F:\mp-latest\disaster-ai-api')
from global_mesh.label_fusion import build_label_dataset

def test_no_gdacs_labels():
    """GDACS must never be a training label."""
    ds = build_label_dataset(cells_per_dim=1)
    for m in ds["metadata"]:
        # GDACS must not appear as a label source
        assert "GDACS" not in str(m.get("label_reasons", "")), "GDACS used as label"
    print("PASS: No GDACS labels in training data")

def test_split_disjoint():
    """Train/val/test splits must be disjoint."""
    ds = build_label_dataset(cells_per_dim=1)
    train = set(ds["train_idx"])
    val = set(ds["val_idx"])
    test = set(ds["test_idx"])
    assert len(train & val) == 0, "train/val overlap"
    assert len(train & test) == 0, "train/test overlap"
    assert len(val & test) == 0, "val/test overlap"
    print("PASS: Train/val/test splits are disjoint")

if __name__ == "__main__":
    test_no_gdacs_labels()
    test_split_disjoint()
    print("ALL LEAKAGE TESTS PASSED")
