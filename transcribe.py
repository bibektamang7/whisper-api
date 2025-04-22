import os
import tempfile
import whisper
import uvicorn
import logging
import requests
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from typing import Optional
import subprocess
from pydantic import BaseModel, AnyHttpUrl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Whisper Transcription API")

# Initialize Whisper model
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base") 
model = None

# Define request model
class TranscriptionRequest(BaseModel):
    url: AnyHttpUrl
    task: Optional[str] = "transcribe"  # Can be 'transcribe' or 'translate'
    language: Optional[str] = None
    output_format: Optional[str] = "json"  # Can be 'json', 'text', 'srt', 'vtt'

def load_model():
    global model
    logger.info(f"Loading Whisper model: {MODEL_SIZE}")
    model = whisper.load_model(MODEL_SIZE)
    logger.info("Model loaded successfully")

@app.on_event("startup")
async def startup_event():
    load_model()

def download_video(url, output_path):
    """Download video from URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading video: {e}")
        return False

def extract_audio(video_path, output_path):
    """Extract audio from video file."""
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', output_path, '-y'
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        return False

@app.post("/transcribe/")
async def transcribe_url(request: TranscriptionRequest):
    logger.info(f"Received URL: {request.url}")
    
    # Create temp files for video and audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
        video_path = video_file.name
    
    audio_path = f"{video_path}.wav"
    
    try:
        # Download video from URL
        if not download_video(request.url, video_path):
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to download video from URL"}
            )
        
        # Extract audio from video
        if not extract_audio(video_path, audio_path):
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to extract audio from video file"}
            )
        
        # Transcribe audio
        transcribe_options = {
            "task": request.task,
            "fp16": False
        }
        
        if request.language:
            transcribe_options["language"] = request.language
            
        logger.info(f"Starting transcription with options: {transcribe_options}")
        result = model.transcribe(audio_path, **transcribe_options)
        
        # Format output based on request
        response_content = {}
        if request.output_format == "json":
            response_content = {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"]
            }
        elif request.output_format == "text":
            response_content = {"text": result["text"]}
        elif request.output_format in ["srt", "vtt"]:
            # Convert segments to SRT or VTT format
            formatted_text = ""
            for i, segment in enumerate(result["segments"]):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Format timestamps as HH:MM:SS,mmm
                start_formatted = f"{int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}:{int(start_time % 60):02d},{int((start_time % 1) * 1000):03d}"
                end_formatted = f"{int(end_time // 3600):02d}:{int((end_time % 3600) // 60):02d}:{int(end_time % 60):02d},{int((end_time % 1) * 1000):03d}"
                
                formatted_text += f"{i+1}\n{start_formatted} --> {end_formatted}\n{text}\n\n"
                
            response_content = {"formatted_text": formatted_text}
            
        return response_content
    
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {str(e)}"}
        )
    finally:
        # Clean up temp files
        try:
            os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "model": MODEL_SIZE}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("transcribe:app", host="0.0.0.0", port=port, reload=False)