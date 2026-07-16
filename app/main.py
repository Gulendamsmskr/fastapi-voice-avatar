from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse
import io
import numpy as np
import soundfile as sf
import tempfile

from faster_whisper import WhisperModel
from services.tts import tts_to_wav_file

app = FastAPI()  # ✅ app önce

model = WhisperModel("large-v3", compute_type="int8")

def read_wav_upload(file_bytes: bytes):
    audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

def transcribe(audio: np.ndarray, samplerate: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, samplerate)
        segments, _info = model.transcribe(tmp.name, language="tr")
        return "".join(seg.text for seg in segments).strip()

@app.get("/")
def root():
    return {"message": "ok"}
@app.post("/api/echo_raw")
async def echo_raw(request: Request):
    wav_bytes = await request.body()

    audio, sr = read_wav_upload(wav_bytes)
    text = transcribe(audio, sr)
    if not text:
        return {"ok": False, "error": "Boş transcript"}

    out_audio_path = tts_to_wav_file(text, lang="tr")
    return FileResponse(out_audio_path, media_type="audio/wav", filename="reply.wav")

# ✅ Whisper model: 1 kere yükle
model = WhisperModel("large-v3", compute_type="int8")

def read_wav_upload(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """multipart'tan gelen wav bytes -> (mono float32 audio, sr)"""
    audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

def transcribe(audio: np.ndarray, samplerate: int) -> str:
    """numpy audio + sample rate -> text"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, samplerate)
        segments, _info = model.transcribe(tmp.name, language="tr")
        text = "".join(seg.text for seg in segments).strip()
    return text

# ✅ Seçenek B: sen konuş -> aynı metni avatar söylesin (WAV döner)
@app.post("/api/echo")
async def echo(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio, sr = read_wav_upload(audio_bytes)

    text = transcribe(audio, sr)
    if not text:
        return {"ok": False, "error": "Boş transcript"}

    wav_path = tts_to_wav_file(text, lang="tr")
    return FileResponse(wav_path, media_type="audio/wav", filename="reply.wav")

# (Opsiyonel) sadece STT
@app.post("/api/stt")
async def api_stt(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio, sr = read_wav_upload(audio_bytes)
    text = transcribe(audio, sr)
    return {"text": text}
