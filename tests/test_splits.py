import numpy as np

from src.flsys.data.mnist import dirichlet_splits, load_full


def test_dirichlet_splits_cover_and_disjoint():
    """Test that Dirichlet splits cover all data and have no overlaps."""
    train, _ = load_full()
    y = np.array(train.targets)
    parts = dirichlet_splits(y, n_clients=3, alpha=0.5)
    all_idx = np.concatenate(parts)

    # verify no duplicate indices (disjoint splits)
    assert len(set(all_idx)) == len(all_idx)
    # verify all indices are present (complete coverage)
    assert set(all_idx) == set(range(len(train)))
