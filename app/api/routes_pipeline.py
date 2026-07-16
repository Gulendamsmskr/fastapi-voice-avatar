from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid



from services.stt import transcribe_file
from services.ai import ollama_chat
from services.tts import tts_to_wav_file

router = APIRouter()  # ✅ MUTLAKA DECORATOR'LARDAN ÖNCE

async def save_upload(upload_file: UploadFile, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)

    ext = os.path.splitext(upload_file.filename)[1] or ".wav"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(folder, filename)

    content = await upload_file.read()
    with open(path, "wb") as f:
        f.write(content)

    return path


@router.post("/echo")
async def echo(audio: UploadFile = File(...)):
    in_path = await save_upload(audio, "data/audio/in")

    transcript = transcribe_file(in_path)
    if not transcript:
        return {"ok": False, "error": "Boş transcript"}

    out_audio_path = tts_to_wav_file(transcript, lang="tr")

    return FileResponse(out_audio_path, media_type="audio/wav", filename="reply.wav")


@router.post("/pipeline")
async def pipeline(audio: UploadFile = File(...)):
    in_path = await save_upload(audio, "data/audio/in")

    transcript = transcribe_file(in_path)
    if not transcript:
        return {"ok": False, "error": "Boş transcript"}

    assistant_text = ollama_chat(
        system="Cevaplarını sadece Türkçe ver. Kısa ve doğal konuşma dili kullan.",
        user=transcript
    )

    out_audio_path = tts_to_wav_file(assistant_text, lang="tr")

    return FileResponse(out_audio_path, media_type="audio/wav", filename="reply.wav")