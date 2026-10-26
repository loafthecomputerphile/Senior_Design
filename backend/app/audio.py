import io
import librosa
import soundfile as sf

TARGET_SR = 16000

def to_wav_16k(raw: bytes) -> bytes:
    """Decode any supported audio upload, resample to 16 kHz mono, return wav bytes."""
    y, _ = librosa.load(io.BytesIO(raw), sr=TARGET_SR, mono=True)
    buf = io.BytesIO()
    sf.write(buf, y, TARGET_SR, format="WAV")
    return buf.getvalue()
