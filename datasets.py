from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AUDIO_DATASETS = ("cello",)
IMAGE_DATASETS = ("camera", "coffee", "astronaut")
DATASETS = AUDIO_DATASETS + IMAGE_DATASETS


def grid(shape) -> torch.Tensor:
    """Regular grid over [-1, 1]^d, raster (row-major) order: [prod(shape), d]."""
    axes = [torch.linspace(-1, 1, steps=int(n)) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(axes))


def load_cello(sound_file=None):
    import soundfile as sf

    data, _ = sf.read(str(sound_file or HERE / "sound" / "gt_bach.wav"), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)[20000:25000:5]
    data = data / np.max(np.abs(data))
    target = torch.as_tensor(data, dtype=torch.float32).view(-1, 1)
    return grid((len(data),)).unsqueeze(0), target.unsqueeze(0)


def load_image(name: str, img_size: int = 32):
    import skimage
    from PIL import Image
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

    if name == "camera":
        image = Image.fromarray(skimage.data.camera())
    elif name == "coffee":
        image = Image.fromarray(skimage.color.rgb2gray(skimage.data.coffee()))
    elif name == "astronaut":
        image = Image.fromarray(skimage.color.rgb2gray(skimage.data.astronaut()))
    else:
        raise ValueError(f"unknown image {name!r}; choose from {IMAGE_DATASETS}")
    transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor(), Normalize([0.5], [0.5])])
    pixels = transform(image)                                   # [1, H, W]
    target = pixels.permute(1, 2, 0).reshape(-1, pixels.shape[0])
    return grid((img_size, img_size)).unsqueeze(0), target.unsqueeze(0)


def load_dataset(name: str, img_size: int = 32):
    """Returns (coords [1, N, d], target [1, N, C], grid_shape)."""
    if name in AUDIO_DATASETS:
        coords, target = load_cello()
        return coords, target, (coords.shape[1],)
    coords, target = load_image(name, img_size)
    return coords, target, (img_size, img_size)
