import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

dataset_dir: Path = Path("data/public_dataset_v3/coughvid_20211012")
output_dir: Path = Path("data/public_dataset_v3_wav")
output_dir.mkdir(parents=True, exist_ok=True)

def process_file(file_path: Path) -> None:
    if not file_path.is_file():
        return
    if file_path.suffix.lower() == "json":
        new_out: Path = output_dir/f"{file_path.stem}.json"
        if new_out.exists(): return
        shutil.copy2(file_path, new_out)
        return

    wav_path: str = output_dir / f"{file_path.stem}.wav"

    if wav_path.exists():
        return

    # Direct copy if already WAV, otherwise convert via ffmpeg
    if file_path.suffix.lower() == ".wav":
        shutil.copy2(file_path, wav_path)
    else:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(file_path), "-ar", "16000", "-ac", "1", str(wav_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(str(file_path), str(wav_path))
            raise e

if __name__ == "__main__":
    files: list[Path] = [f for f in dataset_dir.iterdir() if f.is_file()]
    
    with ProcessPoolExecutor() as executor:
        list(executor.map(process_file, files))