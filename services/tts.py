# services/tts.py
import os
import uuid
from pathlib import Path
from TTS.api import TTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # app/.. -> proje kökü
REFERENCE_AUDIO = PROJECT_ROOT / "data" / "audio" / "reference.wav"

# ✅ Modeli 1 kere yükle
print("🔄 XTTS v2 yükleniyor...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
print("✅ TTS hazır!")

def tts_to_wav_file(text: str, lang: str = "tr") -> str:
    out_dir = PROJECT_ROOT / "data" / "audio" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"reply_{uuid.uuid4().hex}.wav"

    tts_model.tts_to_file(
        text=text,
        file_path=str(out_path),
        speaker_wav=str(REFERENCE_AUDIO),
        language=lang,
    )
    return str(out_path)