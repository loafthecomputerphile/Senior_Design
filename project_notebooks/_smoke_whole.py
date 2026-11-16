"""Smoke check for make_dataset_whole: window rules + real small build -> npz -> NpzDataPipeline.
Run from project_notebooks/ with the project venv."""
import os

import numpy as np
import polars as pl

from modules.dataset.make_dataset import NpzDataPipeline
from modules.dataset.make_dataset_whole import _fit_window, make_dataset_whole, nonexpert_df
from modules.preprocessing.dataframe_process import filter_coughvid_df_expert

SR = 16_000

# 1) window rules (synthetic)
assert _fit_window(np.arange(2 * SR, dtype=np.float32), 2 * SR, 5 * SR) is not None  # exactly min -> kept
assert _fit_window(np.arange(SR, dtype=np.float32), 2 * SR, 5 * SR) is None          # <2s dropped
mid = _fit_window(np.arange(9 * SR, dtype=np.float32), 2 * SR, 5 * SR)               # middle 5s
assert len(mid) == 5 * SR and mid[0] == 2 * SR and mid[-1] == 7 * SR - 1              # crop is centered
short = _fit_window(np.arange(3 * SR, dtype=np.float32), 2 * SR, 5 * SR)             # right-padded
assert len(short) == 5 * SR and short[3 * SR] == 0.0 and short[3 * SR - 1] == 3 * SR - 1
print("1) window rules OK (drop <2s / middle 5s / right-pad)")

# 2) real build on a small expert subset
df = pl.read_csv(
    "data/public_dataset_v3/coughvid_20211012/metadata_compiled.csv",
    infer_schema_length=0,
)
df = filter_coughvid_df_expert(df).with_columns(
    (pl.col("uuid").map_elements(
        lambda u: f"data/public_dataset_v3_wav/{u}.wav",
        return_dtype=pl.String,
    )).alias("wav_path")
)
# existence guard: expert df may reference wavs that aren't on disk
df = df.filter(pl.col("wav_path").map_elements(os.path.exists, return_dtype=pl.Boolean))
df = df.group_by("status").head(6)
print(f"smoke df: {len(df)} rows, labels {df['status'].unique().to_list()}")

out = "data/_smoke_whole.npz"
make_dataset_whole(df, SR, out, max_workers=4, yamnet_threshold=0.1)

# 3) round-trip through the unmodified loader
pipe = NpzDataPipeline(out)
pipe.load_data()
pipe.load_paths()
n = len(pipe)
f0, l0 = pipe[0]
assert f0.shape == (5 * SR,), f0.shape
assert set(pipe.labels.unique().tolist()) <= {0, 1, 2}
assert len(pipe.paths) == n
# nonexpert anti-join on the same df (df_all needs wav_path to join on)
df_all = pl.read_csv("data/public_dataset_v3/coughvid_20211012/metadata_compiled.csv",
                     infer_schema_length=0).with_columns(
    (pl.col("uuid").map_elements(
        lambda u: f"data/public_dataset_v3_wav/{u}.wav",
        return_dtype=pl.String,
    )).alias("wav_path")
)
ne = nonexpert_df(df_all, df)
assert ne.height <= df_all.height and not (
    set(ne["wav_path"]) & set(df["wav_path"])), "anti-join leaked an expert recording"
print(f"nonexpert anti-join OK: {ne.height} rows, disjoint from expert subset")
print(f"3) NpzDataPipeline round-trip OK: {n} recordings, features {tuple(f0.shape)}")

os.remove(out)
print("smoke OK")
