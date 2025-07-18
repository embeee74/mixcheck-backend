from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import io
import traceback
import soundfile as sf

app = FastAPI()

# CORS setup to accept requests from altguitar.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://altguitar.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional CORS preflight handler
@app.options("/analyze")
async def preflight(request: Request):
    return JSONResponse(status_code=200, content={"message": "CORS preflight OK"})

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    genre: str = Form("Unknown"),
    daw: str = Form("Unknown")
):
    try:
        print(f"📥 File received: {file.filename} | Type: {file.content_type}")

        contents = await file.read()
        audio_buffer = io.BytesIO(contents)

        # Attempt to decode with librosa
        try:
            y, sr = librosa.load(audio_buffer, sr=None, mono=True)
            print("✅ Audio decoded with librosa.")
        except Exception:
            print("⚠️ Librosa failed, trying soundfile fallback...")
            audio_buffer.seek(0)
            try:
                y, sr = sf.read(audio_buffer)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                print("✅ Audio decoded with soundfile.")
            except Exception:
                print("❌ Both decode attempts failed.")
                traceback.print_exc()
                return {"error": "Unable to process audio. Try WAV or MP3 under 5 mins."}

        duration = librosa.get_duration(y=y, sr=sr)
        if duration > 300:
            return {"error": "Track is too long. Please upload a file under 5 minutes."}

        rms = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        print(f"🎚 Analysis complete — Duration: {duration:.2f}s | RMS: {rms:.4f} | Tempo: {tempo:.2f} BPM | Centroid: {centroid:.2f}")

        return {
            "duration_sec": round(duration, 2),
            "rms_level": round(rms, 4),
            "tempo_bpm": round(tempo, 2),
            "spectral_centroid": round(centroid, 2),
            "genre": genre,
            "daw": daw
        }

    except Exception:
        print("🔥 Unexpected backend error:")
        traceback.print_exc()
        return {"error": "Server error. Please try again later."}

