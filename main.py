from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import io
import traceback

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    genre: str = Form(...),
    daw: str = Form(...)
):
    try:
        # Log file type for debugging
        print(f"Uploaded file content type: {file.content_type}")

        # Read and decode audio
        contents = await file.read()
        y, sr = librosa.load(io.BytesIO(contents), sr=None, mono=True)

        # Limit duration if needed
        duration = librosa.get_duration(y=y, sr=sr)
        if duration > 300:
            return {"error": "Track is too long. Please upload a file under 5 minutes."}

        rms = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        return {
            "duration_sec": round(duration, 2),
            "rms_level": round(rms, 4),
            "tempo_bpm": round(tempo, 2),
            "spectral_centroid": round(centroid, 2),
            "genre": genre,
            "daw": daw
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
