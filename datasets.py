#---------------------------------------
#Since : 2024/09/05
#Update: 2025/04/05
# -*- coding: utf-8 -*-
# datasets.py
#---------------------------------------
import os
import subprocess
import shutil
import tarfile
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import datasets


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 0

BSDS500_URL = os.environ.get(
    "BSDS500_URL",
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
)

DATASET_ALIASES = {
    "mnist": "mnist",
    "kmnist": "kmnist",
    "k_mnist": "kmnist",
    "k-mnist": "kmnist",
    "k mnist": "kmnist",
    "fashion_mnist": "fashion_mnist",
    "fashion mnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
    "cifar10": "cifar10",
    "cifar_10": "cifar10",
    "cifar-10": "cifar10",
    "cifar 10": "cifar10",
    "cifar10_gray": "cifar10_gray",
    "cifar10_grayscale": "cifar10_gray",
    "cifar_10_gray": "cifar10_gray",
    "cifar_10_grayscale": "cifar10_gray",
    "cifar-10-gray": "cifar10_gray",
    "cifar-10-grayscale": "cifar10_gray",
    "cifar 10 gray": "cifar10_gray",
    "cifar 10 grayscale": "cifar10_gray",
    "bsds500": "bsds500",
    "bsds_500": "bsds500",
    "bsds-500": "bsds500",
    "bsds 500": "bsds500",
    "bsds500_patch": "bsds500_patch",
    "bsds500_patches": "bsds500_patch",
    "bsds500_patch_only": "bsds500_patch",
    "bsds_500_patch": "bsds500_patch",
    "bsds-500-patch": "bsds500_patch",
    "bsds 500 patch": "bsds500_patch",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _datasets_root() -> Path:
    return _repo_root() / "datasets"


def normalize_dataset_name(dataset_type: str) -> str:
    if not isinstance(dataset_type, str):
        raise TypeError("dataset_type must be a string")

    normalized = dataset_type.strip().lower()
    collapsed = " ".join(normalized.replace("_", " ").replace("-", " ").split())

    if normalized in DATASET_ALIASES:
        return DATASET_ALIASES[normalized]
    if collapsed in DATASET_ALIASES:
        return DATASET_ALIASES[collapsed]

    return normalized.replace(" ", "_")


def _read_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _should_auto_download_bsds500() -> bool:
    # Default to auto-download so BSDS500 runs work out of the box.
    # Users can still opt out explicitly with BSDS500_AUTO_DOWNLOAD=0.
    return _read_bool_env("BSDS500_AUTO_DOWNLOAD", default=True)


def _kmnist_mirrors() -> list[str]:
    return _read_csv_env(
        "KMNIST_MIRRORS",
        default=[
            "https://codh.rois.ac.jp/kmnist/dataset/kmnist/",
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
        ],
    )


def _kmnist_huggingface_urls() -> dict[str, str]:
    return {
        "train": os.environ.get(
            "KMNIST_HF_TRAIN_URL",
            "https://huggingface.co/datasets/tanganke/kmnist/resolve/main/kmnist/train-00000-of-00001.parquet?download=true",
        ),
        "test": os.environ.get(
            "KMNIST_HF_TEST_URL",
            "https://huggingface.co/datasets/tanganke/kmnist/resolve/main/kmnist/test-00000-of-00001.parquet?download=true",
        ),
    }


def _kmnist_root() -> Path:
    return _datasets_root()


def _download_file_with_curl(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    result = subprocess.run(
        ["curl", "-L", "--fail", "--output", str(tmp_path), url],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        stderr = result.stderr.strip() or result.stdout.strip() or f"curl exited with {result.returncode}"
        raise RuntimeError(f"curl download failed for {url}: {stderr}")
    tmp_path.replace(destination)


def _download_file_with_fallbacks(url: str, destination: Path) -> None:
    try:
        _download_file(url, destination)
    except Exception:
        _download_file_with_curl(url, destination)


def _kmnist_legacy_paths(root: Path) -> tuple[Path, Path]:
    processed_dir = root / "KMNIST" / "processed"
    return processed_dir / datasets.KMNIST.training_file, processed_dir / datasets.KMNIST.test_file


def _kmnist_legacy_files_exist(root: Path) -> bool:
    training_path, test_path = _kmnist_legacy_paths(root)
    return training_path.is_file() and test_path.is_file()


def _convert_kmnist_parquet_to_legacy(parquet_path: Path, output_path: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "KMNIST Hugging Face fallback requires pyarrow. Install it with `pip install pyarrow`."
        ) from exc

    records = pq.read_table(parquet_path, columns=["image", "label"]).to_pylist()
    images = []
    labels = []

    for record in records:
        image_info = record["image"]
        image_bytes = image_info.get("bytes") if isinstance(image_info, dict) else None
        if image_bytes is None:
            raise ValueError(f"Missing image bytes in {parquet_path}")

        with Image.open(BytesIO(image_bytes)) as image:
            image_array = np.array(image.convert("L"), dtype=np.uint8)

        images.append(torch.from_numpy(image_array))
        labels.append(int(record["label"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        (torch.stack(images), torch.tensor(labels, dtype=torch.long)),
        output_path,
    )


def _prepare_kmnist_legacy_dataset_from_huggingface(root: Path) -> None:
    if _kmnist_legacy_files_exist(root):
        return

    hf_dir = root / "KMNIST" / "huggingface"
    urls = _kmnist_huggingface_urls()
    parquet_paths = {
        "train": hf_dir / "train-00000-of-00001.parquet",
        "test": hf_dir / "test-00000-of-00001.parquet",
    }

    for split, parquet_path in parquet_paths.items():
        if not parquet_path.is_file():
            _download_file_with_fallbacks(urls[split], parquet_path)

    training_path, test_path = _kmnist_legacy_paths(root)
    if not training_path.is_file():
        _convert_kmnist_parquet_to_legacy(parquet_paths["train"], training_path)
    if not test_path.is_file():
        _convert_kmnist_parquet_to_legacy(parquet_paths["test"], test_path)


def _prepare_kmnist_raw_dataset(train: bool):
    original_mirrors = list(datasets.KMNIST.mirrors)
    datasets.KMNIST.mirrors = _kmnist_mirrors()
    root = _kmnist_root()

    try:
        return datasets.KMNIST(root=root, train=train, download=True)
    except RuntimeError as exc:
        try:
            _prepare_kmnist_legacy_dataset_from_huggingface(root)
            return datasets.KMNIST(root=root, train=train, download=False)
        except Exception as fallback_exc:
            mirrors = ", ".join(datasets.KMNIST.mirrors)
            raise RuntimeError(
                "KMNIST download failed from the CODH mirrors and Hugging Face fallback. "
                "Set KMNIST_MIRRORS to a reachable mirror list, set KMNIST_HF_TRAIN_URL and "
                "KMNIST_HF_TEST_URL to alternate Parquet URLs, or place the original KMNIST raw files "
                "under ./datasets/KMNIST/raw or legacy files under ./datasets/KMNIST/processed. "
                f"Tried mirrors: {mirrors}"
            ) from fallback_exc
    finally:
        datasets.KMNIST.mirrors = original_mirrors


def _candidate_bsds500_roots() -> list[Path]:
    repo_root = _repo_root()
    env_root = os.environ.get("BSDS500_ROOT")
    candidates = []

    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            repo_root / "datasets" / "BSR" / "BSDS500",
            repo_root / "datasets" / "BSDS500",
            repo_root / "data" / "BSR" / "BSDS500",
            repo_root / "data" / "BSDS500",
            repo_root / "BSR" / "BSDS500",
            repo_root / "BSDS500",
        ]
    )
    return candidates


def _find_bsds500_images_root_from_candidate(candidate: Path) -> Path | None:
    candidate = candidate.expanduser().resolve()

    if (candidate / "data" / "images").is_dir():
        return candidate / "data" / "images"
    if (candidate / "images").is_dir():
        return candidate / "images"
    if (candidate / "train").is_dir() and (candidate / "test").is_dir():
        return candidate

    return None


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    path = path.resolve()

    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path)):
            raise RuntimeError(f"Unsafe tar member path detected: {member.name}")

    tar.extractall(path)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)

    tmp_path.replace(destination)


def _download_and_extract_bsds500() -> Path:
    download_root = Path(
        os.environ.get("BSDS500_DOWNLOAD_ROOT", _datasets_root())
    ).expanduser().resolve()

    archive_path = download_root / "downloads" / "BSR_bsds500.tgz"
    extract_root = download_root

    if not archive_path.exists():
        print(f"[BSDS500] downloading from: {BSDS500_URL}")
        _download_file(BSDS500_URL, archive_path)
        print(f"[BSDS500] downloaded to: {archive_path}")
    else:
        print(f"[BSDS500] archive already exists: {archive_path}")

    # 展開済み候補があれば再展開しない
    post_extract_candidates = [
        extract_root / "BSR" / "BSDS500",
        extract_root / "BSDS500",
        extract_root / "BSR",
        extract_root,
    ]
    for candidate in post_extract_candidates:
        images_root = _find_bsds500_images_root_from_candidate(candidate)
        if images_root is not None:
            print(f"[BSDS500] found existing extracted dataset: {images_root}")
            return images_root

    print(f"[BSDS500] extracting: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract_tar(tar, extract_root)

    for candidate in post_extract_candidates:
        images_root = _find_bsds500_images_root_from_candidate(candidate)
        if images_root is not None:
            print(f"[BSDS500] extracted dataset found: {images_root}")
            return images_root

    raise FileNotFoundError(
        "BSDS500 archive was downloaded/extracted, but images directory still "
        "could not be found."
    )


def _resolve_bsds500_images_root() -> Path:
    for candidate in _candidate_bsds500_roots():
        images_root = _find_bsds500_images_root_from_candidate(candidate)
        if images_root is not None:
            return images_root

    if _should_auto_download_bsds500():
        return _download_and_extract_bsds500()

    searched = "\n  - ".join(str(path) for path in _candidate_bsds500_roots())
    raise FileNotFoundError(
        "BSDS500 images directory not found. "
        "Set BSDS500_ROOT to the dataset root, or re-enable auto-download with "
        "BSDS500_AUTO_DOWNLOAD=1 if you disabled it.\n"
        f"Searched:\n  - {searched}"
    )


def _list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []

    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _prepare_stratify_targets(targets):
    if targets is None:
        return None

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().reshape(-1).numpy()
    else:
        targets = np.asarray(targets).reshape(-1)

    _, counts = np.unique(targets, return_counts=True)
    if counts.size == 0 or np.any(counts < 2):
        return None

    return targets


def _split_indices(
    n_samples: int,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_state: int = DEFAULT_SPLIT_SEED,
    stratify_targets=None,
):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the open interval (0, 1)")
    if n_samples < 2:
        raise ValueError("At least two samples are required to create a validation split")

    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=_prepare_stratify_targets(stratify_targets),
    )
    return torch.from_numpy(train_idx), torch.from_numpy(val_idx)


def _split_train_validation_tensors(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_state: int = DEFAULT_SPLIT_SEED,
):
    train_idx, val_idx = _split_indices(
        n_samples=X_train.size(0),
        validation_fraction=validation_fraction,
        random_state=random_state,
        stratify_targets=y_train,
    )

    return (
        X_train[train_idx],
        X_train[val_idx],
        y_train[train_idx],
        y_train[val_idx],
    )


def _split_train_validation_paths(
    image_paths: list[Path],
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_state: int = DEFAULT_SPLIT_SEED,
):
    train_idx, val_idx = _split_indices(
        n_samples=len(image_paths),
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    train_paths = [image_paths[index] for index in train_idx.tolist()]
    val_paths = [image_paths[index] for index in val_idx.tolist()]
    return train_paths, val_paths


class BSDS500PatchDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        patch_size: int = 32,
        patches_per_epoch: int = 10000,
        flatten: bool = False,
        deterministic: bool = False,
        seed: int = 0,
    ):
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patches_per_epoch <= 0:
            raise ValueError("patches_per_epoch must be positive")

        self.image_paths = image_paths
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.flatten = flatten
        self.deterministic = deterministic
        self.seed = seed
        self.cached_images = [self._load_grayscale_image(path) for path in self.image_paths]

    def __len__(self):
        return self.patches_per_epoch

    @staticmethod
    def _load_grayscale_image(image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            # Cache decoded grayscale pixels once to avoid repeated JPEG decode costs.
            return np.asarray(image.convert("L"), dtype=np.uint8)

    def _rng(self, index: int):
        if self.deterministic:
            return np.random.default_rng(self.seed + index)
        return np.random.default_rng()

    def _sample_image_index(self, index: int, rng) -> int:
        if self.deterministic:
            return index % len(self.cached_images)
        return int(rng.integers(0, len(self.cached_images)))

    def __getitem__(self, index):
        rng = self._rng(index)
        image_index = self._sample_image_index(index, rng)
        image_array = self.cached_images[image_index]

        height, width = image_array.shape
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Image {self.image_paths[image_index]} is smaller than patch_size={self.patch_size}: "
                f"got {(height, width)}"
            )

        max_top = height - self.patch_size
        max_left = width - self.patch_size
        top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
        left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0

        patch = image_array[top:top + self.patch_size, left:left + self.patch_size]
        patch = patch.astype(np.float32, copy=False) / 255.0
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)

        if self.flatten:
            patch_tensor = patch_tensor.reshape(-1)

        return patch_tensor, torch.tensor(0, dtype=torch.long)


def _prepare_grayscale_classification_data(dataset_cls, batch_size=64, flatten=False):
    dataset_root = str(_datasets_root())
    train_ds_raw = dataset_cls(root=dataset_root, train=True, download=True)
    test_ds_raw = dataset_cls(root=dataset_root, train=False, download=True)

    X_train = train_ds_raw.data.to(dtype=torch.float32) / 255.0
    X_test = test_ds_raw.data.to(dtype=torch.float32) / 255.0

    if flatten:
        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)
    else:
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

    y_train = train_ds_raw.targets.long()
    y_test = test_ds_raw.targets.long()

    X_train, X_val, y_train, y_val = _split_train_validation_tensors(
        X_train,
        y_train,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        random_state=DEFAULT_SPLIT_SEED,
    )

    train_td = TensorDataset(X_train, y_train)
    val_td = TensorDataset(X_val, y_val)
    test_td = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_td, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_td, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_td, batch_size=batch_size, shuffle=False, pin_memory=False)

    return train_loader, val_loader, test_loader

# MNISTデータの準備
def prepare_mnist_data(batch_size=64, flatten=False):
    return _prepare_grayscale_classification_data(
        datasets.MNIST,
        batch_size=batch_size,
        flatten=flatten,
    )


def prepare_kmnist_data(batch_size=64, flatten=False):
    train_ds_raw = _prepare_kmnist_raw_dataset(train=True)
    test_ds_raw = _prepare_kmnist_raw_dataset(train=False)

    X_train = train_ds_raw.data.to(dtype=torch.float32) / 255.0
    X_test = test_ds_raw.data.to(dtype=torch.float32) / 255.0

    if flatten:
        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)
    else:
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

    y_train = train_ds_raw.targets.long()
    y_test = test_ds_raw.targets.long()

    X_train, X_val, y_train, y_val = _split_train_validation_tensors(
        X_train,
        y_train,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        random_state=DEFAULT_SPLIT_SEED,
    )

    train_td = TensorDataset(X_train, y_train)
    val_td = TensorDataset(X_val, y_val)
    test_td = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_td, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_td, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_td, batch_size=batch_size, shuffle=False, pin_memory=False)

    return train_loader, val_loader, test_loader

# Fashion MNISTデータの準備
def prepare_fashion_mnist_data(batch_size=64, flatten=False):
    return _prepare_grayscale_classification_data(
        datasets.FashionMNIST,
        batch_size=batch_size,
        flatten=flatten,
    )

# CIFAR-10データの準備
def _prepare_cifar10_tensor_data(batch_size=64, flatten=False, grayscale=False):
    dataset_root = str(_datasets_root())
    train_ds_raw = datasets.CIFAR10(root=dataset_root, train=True, download=True)
    test_ds_raw  = datasets.CIFAR10(root=dataset_root, train=False, download=True)

    # データを0-1の範囲に正規化し、float32に変換
    X_train = torch.from_numpy(train_ds_raw.data).permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0  # (N,3,32,32)
    X_test  = torch.from_numpy(test_ds_raw.data).permute(0, 3, 1, 2).to(dtype=torch.float32)  / 255.0

    if grayscale:
        rgb_to_gray = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
        X_train = (X_train * rgb_to_gray).sum(dim=1, keepdim=True)
        X_test = (X_test * rgb_to_gray).sum(dim=1, keepdim=True)

    # 必要に応じて形状を変更
    if flatten:
        X_train = X_train.view(X_train.size(0), -1)  # (N, 3*32*32)
        X_test  = X_test.view(X_test.size(0), -1)

    # tensorに変換
    y_train = torch.tensor(train_ds_raw.targets, dtype=torch.long)  # integer labels
    y_test  = torch.tensor(test_ds_raw.targets,  dtype=torch.long)

    X_train, X_val, y_train, y_val = _split_train_validation_tensors(
        X_train,
        y_train,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        random_state=DEFAULT_SPLIT_SEED,
    )

    # TensorDatasetを使用してDataLoaderを作成
    train_td = TensorDataset(X_train, y_train)
    val_td = TensorDataset(X_val, y_val)
    test_td  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_td, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_td, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader  = DataLoader(test_td,  batch_size=batch_size, shuffle=False, pin_memory=False)

    return train_loader, val_loader, test_loader


def prepare_cifar10_data(batch_size=64, flatten=False):
    return _prepare_cifar10_tensor_data(
        batch_size=batch_size,
        flatten=flatten,
        grayscale=False,
    )


def prepare_cifar10_gray_data(batch_size=64, flatten=False):
    return _prepare_cifar10_tensor_data(
        batch_size=batch_size,
        flatten=flatten,
        grayscale=True,
    )


def prepare_bsds500_data(
    batch_size=64,
    flatten=False,
    pin_memory=False,
    train_patches_default=200,
    val_patches_default=100,
    test_patches_default=200,
):
    patch_size = _read_int_env("BSDS500_PATCH_SIZE", 32)
    train_patches = _read_int_env("BSDS500_TRAIN_PATCHES", train_patches_default)
    val_patches = _read_int_env("BSDS500_VAL_PATCHES", val_patches_default)
    test_patches = _read_int_env("BSDS500_TEST_PATCHES", test_patches_default)
    val_seed = _read_int_env("BSDS500_VAL_SEED", 4321)
    test_seed = _read_int_env("BSDS500_TEST_SEED", 1234)

    images_root = _resolve_bsds500_images_root()
    train_paths = _list_image_files(images_root / "train")
    val_paths = _list_image_files(images_root / "val")
    test_paths = _list_image_files(images_root / "test")

    if not train_paths:
        raise FileNotFoundError(f"No BSDS500 training images found under {images_root / 'train'}")

    if not val_paths:
        train_paths, val_paths = _split_train_validation_paths(
            train_paths,
            validation_fraction=DEFAULT_VALIDATION_FRACTION,
            random_state=DEFAULT_SPLIT_SEED,
        )

    if not test_paths:
        raise FileNotFoundError(f"No BSDS500 test images found under {images_root / 'test'}")

    train_dataset = BSDS500PatchDataset(
        image_paths=train_paths,
        patch_size=patch_size,
        patches_per_epoch=train_patches,
        flatten=flatten,
        deterministic=False,
    )
    val_dataset = BSDS500PatchDataset(
        image_paths=val_paths,
        patch_size=patch_size,
        patches_per_epoch=val_patches,
        flatten=flatten,
        deterministic=True,
        seed=val_seed,
    )
    test_dataset = BSDS500PatchDataset(
        image_paths=test_paths,
        patch_size=patch_size,
        patches_per_epoch=test_patches,
        flatten=flatten,
        deterministic=True,
        seed=test_seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def prepare_bsds500_patch_data(batch_size=64, flatten=False, pin_memory=False):
    return prepare_bsds500_data(
        batch_size=batch_size,
        flatten=flatten,
        pin_memory=pin_memory,
        train_patches_default=54000,
        val_patches_default=6000,
        test_patches_default=10000,
    )

# メイン関数で選択したデータセットに基づくDataLoaderを生成
def load_dataset(dataset_type="mnist", batch_size=64, n_bits=None, flatten=False):
    dataset_type = normalize_dataset_name(dataset_type)

    if dataset_type == "mnist":
        return prepare_mnist_data(batch_size, flatten=flatten)

    elif dataset_type == "kmnist":
        return prepare_kmnist_data(batch_size, flatten=flatten)

    elif dataset_type == "fashion_mnist":
        return prepare_fashion_mnist_data(batch_size, flatten=flatten)

    elif dataset_type == "cifar10":
        return prepare_cifar10_data(batch_size, flatten=flatten)

    elif dataset_type == "cifar10_gray":
        return prepare_cifar10_gray_data(batch_size, flatten=flatten)

    elif dataset_type == "bsds500":
        return prepare_bsds500_data(batch_size, flatten=flatten)

    elif dataset_type == "bsds500_patch":
        return prepare_bsds500_patch_data(batch_size, flatten=flatten)

    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
