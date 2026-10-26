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

def print_class_counts(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    for label, n in counts.items():
        print(f"{label}: {n} ({n / total:.1%})")
    print(f"total: {total}")

def _subsample(data: np.ndarray, labels: np.ndarray, paths: np.ndarray, cap: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(data) <= cap:
        return data, labels, paths
    keep = np.sort(rng.choice(len(data), size=cap, replace=False))
    return data[keep], labels[keep], paths[keep]


def make_dataset(df: pl.DataFrame, sample_rate: int, out: str = 'final_processed_dataset.npz', max_workers: int = 8, max_per_clip: int | None = None, max_per_label: dict[str, int] | None = None) -> None:
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

        # ponytail: per-clip cap kills near-duplicate segments from one recording;
        # raise max_per_clip if per-class totals drop below target.
        if max_per_clip is not None and len(fitted) > max_per_clip:
            keep = np.sort(np.random.default_rng(0).choice(len(fitted), size=max_per_clip, replace=False))
            fitted, labels, paths = fitted[keep], labels[keep], paths[keep]
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
    rng = np.random.default_rng(0)
    for label, data, labels, paths in results:
        if len(data) == 0:
            continue
        master_collections[label].append(data)
        master_labels[label].append(labels)
        master_paths[label].append(paths)

    for label in master_collections.keys():
        if not master_collections[label]:
            continue

        # ponytail: subsample AFTER concatenating — a per-row cap never fires since
        # each row carries only a handful of segments.
        final_data = np.concatenate(master_collections[label], axis=0)
        final_paths = np.concatenate(master_paths[label], axis=0)
        final_labels = np.concatenate(master_labels[label], axis=0)
        if max_per_label and label in max_per_label:
            final_data, final_labels, final_paths = _subsample(
                final_data, final_labels, final_paths, max_per_label[label], rng
            )

        final_npz_data[f"{label}_data"] = final_data
        final_npz_data[f"{label}_paths"] = final_paths
        final_npz_data[f"{label}_labels"] = final_labels
        total_length += len(final_data)
        master_collections[label] = None
        master_paths[label] = None
        master_labels[label] = None

    final_npz_data["length"] = total_length    
    np.savez(out, **final_npz_data)
    print_class_counts({label: len(final_npz_data[f"{label}_data"]) for label in LABELS if f"{label}_data" in final_npz_data})


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
            
            print_class_counts({label: len(data[f"{label}_labels"]) for label in LABELS})

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


def dataset_split_by_recording(
    dataset: Dataset, paths: Sequence[str],
    train: float = 0.8, val: float = 0.1, seed: int | None = None
) -> tuple[Subset, Subset, Subset]:
    """Split by SOURCE RECORDING, not by segment: all segments from one audio file
    stay in the same split, so near-duplicate sibling segments can't leak from
    train into val/test. Split is decided on recordings but measured in segments."""
    if train + val >= 1.0:
        raise ValueError("train + val must be < 1.0 (rest goes to test)")

    # group segment indices by source recording, preserving dataset order
    groups: dict[str, list[int]] = {}
    for idx, path in enumerate(paths):
        groups.setdefault(path, []).append(idx)

    recording_order = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(recording_order)

    n: int = len(dataset)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    # walk recordings in shuffled order, filling splits until each hits its
    # segment-count target; remainder lands in test
    for i, rec in enumerate(recording_order):
        segs = groups[rec]
        if len(train_idx) < train * n:
            train_idx.extend(segs)
        elif len(val_idx) < val * n:
            val_idx.extend(segs)
        else:
            test_idx.extend(segs)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
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


def binary_labels(labels: torch.Tensor, positive: int) -> torch.Tensor:
    """One-vs-rest target: 1 where label == positive, else 0."""
    return (labels == positive).long()


def binary_labels_grouped(labels: torch.Tensor, positives: frozenset[int]) -> torch.Tensor:
    """One-vs-group target: 1 where label in positives (e.g. sick = {1, 2})."""
    return torch.isin(labels, torch.tensor(sorted(positives))).long()


def indices_by_label(labels: torch.Tensor, keep: frozenset[int]) -> list[int]:
    """Dataset indices whose label is in `keep` — builds Stage-2's standalone dataset."""
    lab = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    return [int(i) for i in torch.nonzero(torch.isin(lab, torch.tensor(sorted(keep)))).flatten()]


# binary task views over the 3-class labels: 0=healthy, 1=COVID-19, 2=symptomatic
# lookup tensor created on raw.device — targets get computed on-GPU during training
TASK_TARGETS: dict[str, callable] = {
    "covid_symp_vs_healthy": lambda raw: torch.isin(raw, torch.tensor([1, 2], device=raw.device)).long(),
    "covid_vs_symp": lambda raw: (raw == 1).long(),
}


class TaskDataset(Dataset):
    """Binary task view over NpzDataPipeline — loads the usual data and relabels.

    - 'covid_symp_vs_healthy': all samples; covid+symp merged as positive vs healthy.
    - 'covid_vs_symp': covid+symp samples ONLY (healthy filtered out entirely);
      target 1 = COVID, 0 = symptomatic — a standalone covid-vs-symp dataset.
    __getitem__ still returns the RAW 3-class label so WaveformAugmentor / spec-aug
    prob dicts keep working; apply TASK_TARGETS[task] only at the loss/metrics step."""

    def __init__(self, base: NpzDataPipeline, task: str) -> None:
        self.base: NpzDataPipeline = base
        self.task: str = task
        if task not in TASK_TARGETS:
            raise ValueError(f"unknown task: {task}")
        if task == "covid_vs_symp":
            self.indices: list[int] = indices_by_label(base.labels, frozenset({1, 2}))
        else:
            self.indices: list[int] = list(range(len(base)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.NumberType]:
        return self.base[self.indices[idx]]

    def get_path(self, idx: int) -> str:
        return self.base.get_path(self.indices[idx])

    def positions_for(self, base_indices: Sequence[int]) -> list[int]:
        """Task-dataset row positions for these base-dataset indices (split mapping)."""
        lookup = {b: p for p, b in enumerate(self.indices)}
        return [lookup[b] for b in base_indices]