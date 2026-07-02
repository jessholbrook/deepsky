import numpy as np
import pytest
import torch
from PIL import Image

from deepsky.data.dataset import CropDataset
from deepsky.data.preprocess import crop_is_informative


@pytest.fixture
def crops_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(4):
        arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"fake_{i:03d}.webp", "WEBP")
    return tmp_path


def test_dataset_output_contract(crops_dir):
    ds = CropDataset(crops_dir, image_size=64, seed=0)
    assert len(ds) == 4
    x = ds[0]
    assert x.shape == (3, 64, 64)
    assert x.dtype == torch.float32
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_dataset_deterministic_with_seed(crops_dir):
    a = CropDataset(crops_dir, image_size=64, seed=7)[1]
    b = CropDataset(crops_dir, image_size=64, seed=7)[1]
    assert torch.equal(a, b)


def test_dataset_native_resolution(crops_dir):
    """image_size == crop size: no sub-crop, still valid output."""
    x = CropDataset(crops_dir, image_size=256, seed=0)[0]
    assert x.shape == (3, 256, 256)


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CropDataset(tmp_path, image_size=64)


def test_informative_filter_rejects_black_and_flat():
    black = np.zeros((256, 256, 3), dtype=np.uint8)
    flat_gray = np.full((256, 256, 3), 128, dtype=np.uint8)
    noisy = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert not crop_is_informative(black)
    assert not crop_is_informative(flat_gray)
    assert crop_is_informative(noisy)
