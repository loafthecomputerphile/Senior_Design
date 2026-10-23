import numpy as np
import polars as pl
import soundfile as sf
import torch, torchaudio
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from torch.utils.data import Dataset, random_split, Subset
from typing import Sequence
from modules.preprocessing.audio_process import EnforceFixedLength, WaveformAugmentor, segment_cough_fast


LABELS: list[str] = ["healthy", "COVID-19", "symptomatic"]

def make_dataset(df: pl.DataFrame, sample_rate: int, out: str = 'final_processed_dataset.npz', max_workers: int = 8) -> None:
    label: str
    file_path: str

    master_collections: dict[str, list[np.ndarray]] = {k: [] for k in LABELS}
    master_paths: dict[str, list[str]] = {k: [] for k in LABELS}
    master_labels: dict[str, list[str]] = {k: [] for k in LABELS}

    df = df.select(["audio_name", "status"])
    length_fitter: EnforceFixedLength = EnforceFixedLength(sample_rate, to_numpy=True)

    def process_row(row: dict) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        label = row["status"]
        file_path = row["audio_name"]

        # ponytail: torchaudio.load needs FFmpeg/torchcodec DLLs absent in this env;
        # soundfile reads wav natively. Swap back if torchaudio decoding is fixed.
        wave, rate = sf.read(file_path, dtype="float32")
        if wave.ndim > 1:
            wave = wave[:, 0]
        if rate != sample_rate:
            wave = torchaudio.functional.resample(
                torch.tensor(wave), rate, sample_rate
            ).numpy()

        waves, _ = segment_cough_fast(
            wave, sample_rate,
            cough_padding=0.1, min_cough_len=0.1
        )

        fitted = np.array([length_fitter(w) for w in waves])
        labels = np.full(shape=(len(waves),), fill_value=label, dtype=object)
        paths = np.full(shape=(len(waves),), fill_value=file_path, dtype=object)
        return label, fitted, labels, paths

    # ponytail: ThreadPoolExecutor — file decode + numpy segmentation release the GIL,
    # so threads scale without pickle overhead. ProcessPool if profiling says CPU-bound.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(tqdm(
            pool.map(process_row, df.iter_rows(named=True)),
            total=len(df), desc="processing audio"
        ))

    final_npz_data: dict[str, np.ndarray] = {}
    total_length: int = 0
    for label, data, labels, paths in results:
        if len(data) == 0:
            continue
        master_collections[label].append(data)
        master_labels[label].append(labels)
        master_paths[label].append(paths)
        total_length += len(data)

    for label in master_collections.keys():
        if not master_collections[label]:
            continue

        final_npz_data[f"{label}_data"] = np.concatenate(master_collections[label], axis=0)
        final_npz_data[f"{label}_paths"] = np.concatenate(master_paths[label], axis=0)
        final_npz_data[f"{label}_labels"] = np.concatenate(master_labels[label], axis=0)
        master_collections[label] = None
        master_paths[label] = None
        master_labels[label] = None

    final_npz_data["length"] = total_length
    np.savez(out, **final_npz_data)


class AugmentedSubset(Subset):

    def __init__(self, dataset: Dataset, indices: Sequence[int], augmentor: torch.nn.Module | None = None,) -> None:
        super().__init__(dataset, indices)
        self.augmentor: torch.nn.Module | None = augmentor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.NumberType]:
        waveform: torch.Tensor
        label: torch.NumberType

        waveform, label = self.dataset[self.indices[idx]]

        if self.augmentor is not None:
            return self.augmentor(waveform, label), label

        return waveform, label
    
    def __getitems__(self, indices: Sequence[int]) -> list[tuple[torch.Tensor, torch.NumberType]]:
        return [self.__getitem__(idx) for idx in indices]


class NpzDataPipeline(Dataset):

    def __init__(self, npz_path: str) -> None:
        self.npz_path: str = npz_path
        with np.load(self.npz_path, allow_pickle=True) as data:
            size: int  = data["length"]

        self.length  : int = size
        self.features: torch.Tensor
        self.labels  : torch.Tensor
        self.paths   : list[str]

    def load_data(self) -> None:

        data: dict[str, np.ndarray]
        with np.load(self.npz_path, allow_pickle=True) as data:
            # Combine the 3 separate label blocks into single unified matrices
            all_data: np.ndarray   = np.concatenate(
                [data["healthy_data"], data["COVID-19_data"], data["symptomatic_data"]],
                axis=0
            )

            raw_labels: np.ndarray = np.concatenate(
                [data["healthy_labels"], data["COVID-19_labels"], data["symptomatic_labels"]],
                axis=0
            )

            label_mapping: dict[str, int] = {
                "healthy": 0, "COVID-19": 1, "symptomatic": 2
            }

            numeric_labels: np.ndarray = np.array(
                [label_mapping[lbl] for lbl in raw_labels],
                dtype=np.int64
            )

        self.features = torch.tensor(all_data, dtype=torch.float32)
        self.labels   = torch.tensor(numeric_labels, dtype=torch.long)

    def load_paths(self) -> None:

        data: dict[str, np.ndarray]
        with np.load(self.npz_path, allow_pickle=True) as data:
            all_paths: np.ndarray  = np.concatenate(
                [data["healthy_paths"], data["COVID-19_paths"], data["symptomatic_paths"]],
                axis=0
            )

        self.paths = list(all_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.NumberType]:
        return self.features[idx], self.labels[idx]

    def get_path(self, idx: int) -> str:
        return self.paths[idx]



def dataset_split(
    dataset: Dataset, train: float = 0.8, val: float = 0.1,
    test: float | None = None, seed: int | None = None
) -> tuple[Subset, Subset, Subset]:

    if test is None:
        test = 1.0 - train - val
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1.0")

    train_size: int = int(train * len(dataset))
    val_size:   int = int(val * len(dataset))
    test_size:  int = len(dataset) - train_size - val_size

    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed) if seed is not None else None
    )


def persist_indicies(train_dataset: Subset, val_dataset: Subset, test_dataset: Subset, save_path: str = "data/splits/dataset_split.pt") -> None:
    torch.save(
        {
            "train": train_dataset.indices,
            "val": val_dataset.indices,
            "test": test_dataset.indices,
        },
        save_path,
    )

def load_indicies(dataset: Dataset, path: str) -> tuple[Subset, Subset, Subset]:
    splits: dict[str, torch.Tensor] = torch.load(
        path, weights_only=True,
    )

    return (
        Subset(dataset, splits["train"]),
        Subset(dataset, splits["val"]),
        Subset(dataset, splits["test"])
    )