import numpy as np
import polars as pl
import librosa, torch, math, torchaudio
from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, random_split, Subset
from modules.preprocessing.audio_process import segment_cough, EnforceFixedLength



def make_dataset(df: pl.DataFrame, sample_rate: int, chunk_size: int = 100, out: str = 'final_processed_dataset.npz') -> None:    
    rate: str
    label: str
    file_path: str
    num_waves: int
    wave: np.ndarray
    waves: list[np.ndarray]
    
    master_collections: dict[str, list[np.ndarray]] = {
        "healthy": [], "COVID-19": [], "symptomatic": []
    }
    
    master_paths: dict[str, list[np.ndarray]] = {
        "healthy": [], "COVID-19": [], "symptomatic": []
    }
    
    master_labels: dict[str, list[np.ndarray]] = {
        "healthy": [], "COVID-19": [], "symptomatic": []
    }
        
    
    final_npz_data: dict[str, np.ndarray] = {}
    df = df.select(["audio_name", "status"])
    length_fitter: EnforceFixedLength = EnforceFixedLength(sample_rate)
    
    total_length: int = 0
    chunk_total: int = math.ceil(len(df) / chunk_size)
    for chunk_df in tqdm(df.iter_slices(n_rows=chunk_size), total=chunk_total, desc="processing df chunks"):
        for row in chunk_df.iter_rows(named=True):
            total_length += 1
            label = row["status"]
            file_path = row["audio_name"]
            
            wave, rate = torchaudio.load(file_path)
            if rate != sample_rate:
                wave = torchaudio.functional.resample(
                    wave, rate, sample_rate
                )

            waves,  _ = segment_cough(
                wave.squeeze(0).numpy(), sample_rate, 
                cough_padding=0.1, min_cough_len=0.1
            )
            
            num_waves = len(waves)
            
            master_collections[label].append(np.array([length_fitter(wave) for wave in waves]))
            master_labels[label].append(np.full(shape=(num_waves,), fill_value=file_path, dtype=object))
            master_paths[label].append(np.full(shape=(num_waves,), fill_value=label, dtype=object))  

    
    for label in master_collections.keys():
        if not master_collections[label]:  
            continue
        
        final_npz_data[f"{label}_data"] = np.concatenate(master_collections[label], axis=0)
        master_collections[label] = None 
        
        final_npz_data[f"{label}_paths"] = np.concatenate(master_paths[label], axis=0)
        master_paths[label] = None
        
        final_npz_data[f"{label}_labels"] = np.concatenate(master_labels[label], axis=0)
        master_labels[label] = None
        
    final_npz_data["length"] = total_length
    np.savez_compressed(out, **final_npz_data)



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
    test: float | None = None, gen_seed: torch.Generator | None = None
) -> tuple[Subset, Subset, Subset]:
    
    if sum([train, val, (test if isinstance(test, float) else 0.0)]) != 1.0:
        raise ValueError()
    
    train_size: int = int(train * len(dataset))
    val_size:   int = int(val * len(dataset))
    test_size:  int = len(dataset) - train_size - val_size

    return random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=gen_seed
    )