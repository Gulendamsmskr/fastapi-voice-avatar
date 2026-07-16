# app/services/stt.py
import numpy as np
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", compute_type="int8")

def transcribe_file(file_path: str) -> str:
    audio, sr = sf.read(file_path, dtype="float32")

    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)

    segments, _ = model.transcribe(file_path, language="tr")
    text = "".join(seg.text for seg in segments).strip()
    return text

print("🔄 Whisper Large V3 yükleniyor...")
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
print("✅ Whisper hazır!")

def transcribe(audio_np: np.ndarray) -> tuple[str, str]:
    segments, info = model.transcribe(audio_np, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    return text, info.language