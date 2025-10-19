import io,threading
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import io, os, tempfile, traceback,subprocess
from pathlib import Path
import torchaudio
import numpy as np
from fastapi import HTTPException,File,UploadFile
import soundfile as sf

MODEL_ID = "Mukhil06/my-stt-model"

app = FastAPI(title="STT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ASR pipeline once at startup
_asr = None
_asr_lock = threading.Lock()

def get_asr():
    global _asr
    if _asr is None:
        with _asr_lock:
            if _asr is None:
                _asr = pipeline(
                    task="automatic-speech-recognition",
                    model=MODEL_ID,
                    # remove device_map to avoid needing accelerate, or keep if you installed it
                    device_map="auto",
                    device=-1
                )
    return _asr

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_path = None
    wav_path = None
    try:
        raw = await file.read()

        # Save the uploaded blob to a temp file (supports webm/ogg/wav/mp3/etc.)
        suffix = Path(file.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        # Transcode to 16kHz mono WAV using ffmpeg (robust for all browser formats)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_tmp:
            wav_path = wav_tmp.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", tmp_path,
            "-ac", "1",          # mono
            "-ar", "16000",      # 16 kHz
            "-f", "wav",
            wav_path
        ]
        # run ffmpeg quietly; capture stderr for debugging
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(wav_path):
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')}")

        # Read the wav as float32
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if sr != 16000:
            raise RuntimeError(f"Unexpected sampling rate {sr}; expected 16000")

        # Ensure 1D float32 numpy
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        audio = audio.astype(np.float32)

        # Run ASR
        result = get_asr()(
            {"array": audio, "sampling_rate": 16000},
            return_timestamps=False,
            generate_kwargs={"task": "transcribe", "language": "en"}  # set to your language if needed
        )
        text = result["text"] if isinstance(result, dict) else str(result)
        return {"text": text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to transcribe: {e}")

    finally:
        # Cleanup temp files
        for p in (tmp_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass



from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

