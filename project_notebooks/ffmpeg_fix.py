import os
import sys
import shutil
import platform
import urllib.request
import subprocess
import tarfile
import zipfile

def setup_ffmpeg_shared():
    print("============================================================")
    print(" Automated Cross-Platform FFmpeg Setup for TorchCodec / Python ")
    print("============================================================")

    system = platform.system()
    
    # 1. Platform Detection & Path Definitions
    if system == "Windows":
        install_dir = os.path.expanduser(r"~\AppData\Local\ffmpeg-shared")
        bin_dir = os.path.join(install_dir, "bin")
        download_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z"
        archive_path = os.path.join(os.getenv("TEMP", "C:\\Temp"), "ffmpeg-release-full-shared.7z")
    elif system == "Darwin":
        install_dir = os.path.expanduser("~/Library/Application Support/ffmpeg-shared")
        bin_dir = os.path.join(install_dir, "bin")
        download_url = "https://evermeet.cx/ffmpeg/getrelease/zip"
        archive_path = "/tmp/ffmpeg-release.zip"
    elif system == "Linux":
        install_dir = os.path.expanduser("~/.local/share/ffmpeg-shared")
        bin_dir = os.path.join(install_dir, "bin")
        download_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        archive_path = "/tmp/ffmpeg-release.tar.xz"
    else:
        raise OSError(f"Unsupported operating system: {system}")

    temp_extract_dir = os.path.join(os.getenv("TEMP", "/tmp"), "ffmpeg_temp")

    # 2. Check Existing Installation
    if os.path.exists(bin_dir) and os.listdir(bin_dir):
        print(f"[1/3] FFmpeg binaries already installed at: {bin_dir}")
    else:
        print(f"[1/3] Downloading FFmpeg build for {system}...")
        os.makedirs(install_dir, exist_ok=True)
        os.makedirs(bin_dir, exist_ok=True)
        
        req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(archive_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        print("      Download complete.")

        # 3. Universal Archive Extraction
        print("[2/3] Extracting binaries...")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        extracted = False
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            extracted = True
        elif archive_path.endswith((".tar.xz", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(temp_extract_dir)
            extracted = True
        else:
            # Fallback to system tools for format extensions like .7z (Windows)
            try:
                subprocess.run(["tar", "-xf", archive_path, "-C", temp_extract_dir], check=True)
                extracted = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(["7z", "x", archive_path, f"-o{temp_extract_dir}", "-y"], check=True)
                    extracted = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

        if not extracted:
            raise RuntimeError("Extraction failed. Install system `tar` or `7z` to extract the archive.")

        # Relocate extracted files to bin directory
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                if file.startswith("ffmpeg") or file.startswith("ffprobe") or file.endswith((".dll", ".so", ".dylib")):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(bin_dir, file)
                    shutil.copy2(src_path, dst_path)
                    if system != "Windows":
                        os.chmod(dst_path, 0o755)

        # Cleanup Temporary Files
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        if os.path.exists(archive_path):
            os.remove(archive_path)
        print("      Binaries successfully configured.")

    # 4. Register Binaries in Environment PATH Across Systems
    print("[3/3] Registering binaries to Environment PATH...")

    # Inject into live Python runtime execution
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    if system == "Windows" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_dir)

    # Persist PATH updates across system sessions
    if system == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS) as key:
            try:
                current_path, _ = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                current_path = ""
            path_list = [p.strip() for p in current_path.split(";") if p.strip()]
            if bin_dir not in path_list:
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(path_list + [bin_dir]))
    else:
        shell_rc = os.path.expanduser("~/.bashrc") if os.path.exists(os.path.expanduser("~/.bashrc")) else os.path.expanduser("~/.zshrc")
        export_line = f'\nexport PATH="{bin_dir}:$PATH"\n'
        if os.path.exists(shell_rc):
            with open(shell_rc, "r+") as f:
                if bin_dir not in f.read():
                    f.write(export_line)

    print("\n============================================================")
    print(" SETUP COMPLETE!")
    print(f" Binaries active in live environment at: {bin_dir}")
    print(" Restart your active IDE or notebook kernel if PATH changes persist.")
    print("============================================================")

if __name__ == "__main__":
    setup_ffmpeg_shared()