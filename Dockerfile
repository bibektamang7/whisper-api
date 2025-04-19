FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Whisper needs to be installed from Git
RUN pip install git+https://github.com/openai/whisper.git

# Add app
COPY app ./app

# Expose API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
