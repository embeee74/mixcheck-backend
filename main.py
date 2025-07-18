from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import io
import traceback
import soundfile as sf

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
        print(f"⏺️ Received file: {file.filename}, type: {file.content_type}")

        # Load file into memory
        contents = await file.read()
        audio_data = io.BytesIO(contents)

        # Try primary decode method
        try:
            y, sr = librosa.load(audio_data, sr=None, mono=True)
        except Exception as decode_err:
            print("⚠️ Librosa decode failed, trying fallback...")
            audio_data.seek(0)
            try:
                y, sr = sf.read(audio_data)
                if len(y.shape) > 1:  # Convert stereo to mono
                    y = np.mean(y, axis=1)
            except Exception as fallback_err:
                print("❌ Fallback decode failed.")
                traceback.print_exc()
                return {"error": "Failed to decode audio. Please try a different file format (WAV recommended)."}

        # Analyze audio
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"✅ Loaded audio, duration: {duration:.2f} sec")

        if duration > 300:
            return {"error": "Track is too long. Please upload a file under 5 minutes."}

        rms = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        print(f"🎵 Analysis complete: RMS={rms}, Tempo={tempo}, Centroid={centroid}")

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
        return {"error": f"Unexpected error: {str(e)}"}
