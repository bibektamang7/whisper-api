from fastapi import FastAPI
from pydantic import BaseModel
import whisper
import requests
import tempfile
import os
import subprocess

app = FastAPI()
model = whisper.load_model("base")

class TranscriptionRequest(BaseModel):
    url: str

class TranscriptionResponse(BaseModel):
    transcript: str
    segments: list

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_from_url(req: TranscriptionRequest):
    response = requests.get(req.url)
    if response.status_code != 200:
        return {"error": "Failed to fetch file from URL"}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(response.content)
        video_path = temp_video.name

    audio_path = video_path.replace(".mp4", ".wav")

    try:
        # Extract audio
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ], check=True)

        # Transcribe
        result = model.transcribe(audio_path)

        return {
            "transcript": result["text"],
            "segments": result["segments"]
        }

    finally:
        os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
