import torch
import os
import warnings
import re
import gc

from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import uvicorn
import json

from importlib.resources import files
from cached_path import cached_path
from omegaconf import OmegaConf
from hydra.utils import get_class

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

from faster_whisper import WhisperModel

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
REF_AUDIO = "ref3.wav"
REF_TEXT = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "F5TTS_Base"
VOCODER_NAME = "vocos"
SPEED = 2.0
CROSS_FADE_DURATION = 0.15
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0


models = {}


def split_into_sentences(text):
    # Splits by punctuation (. ! ?) followed by whitespace
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global REF_TEXT
    print(f"🚀 Initializing F5-TTS Server on {DEVICE}...")

    if not os.path.exists(REF_AUDIO):
        print(
            f"\n❌ Error: Could not find '{REF_AUDIO}' in current directory: {os.getcwd()}"
        )

        print(
            "    Please run this command from the directory containing your ref.wav file."
        )

    # -------------------------------------------------------
    # 1. Automatic Transcription
    # -------------------------------------------------------
    print(f"\n🎧 Transcribing '{REF_AUDIO}' using faster-whisper...")
    try:
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        whisper = WhisperModel("medium", device=DEVICE, compute_type=compute_type)

        segments, info = whisper.transcribe(REF_AUDIO, beam_size=5)

        transcribed_text = " ".join([segment.text for segment in segments]).strip()

        if not transcribed_text:
            raise ValueError("Transcription returned empty text.")

        REF_TEXT = transcribed_text
        print(f'✅ Transcript detected: "{REF_TEXT}"')

        del whisper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Whisper model unloaded and VRAM cleared.")

    except Exception as e:
        print(f"❌ Transcription failed: {e}")

    # -------------------------------------------------------
    # 2. Load F5-TTS Model
    # -------------------------------------------------------
    print("\n⬇️  Loading Model Configuration...")
    config_path = files("f5_tts").joinpath(f"configs/{MODEL_NAME}.yaml")
    model_cfg = OmegaConf.load(str(config_path))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    repo_name = "F5-TTS"
    ckpt_step = 1200000
    ckpt_type = "safetensors"

    print("⬇️  Checking weights (cached_path)...")
    ckpt_file = str(
        cached_path(
            f"hf://SWivid/{repo_name}/{MODEL_NAME}/model_{ckpt_step}.{ckpt_type}"
        )
    )

    print("📦 Loading Vocoder...")
    vocoder = load_vocoder(vocoder_name=VOCODER_NAME, is_local=False, device=DEVICE)

    print(f"📦 Loading {MODEL_NAME}...")
    model = load_model(
        model_cls,
        model_arc,
        ckpt_file,
        mel_spec_type=VOCODER_NAME,
        vocab_file="",
        device=DEVICE,
    )

    models["model"] = model
    models["vocoder"] = vocoder

    print(f"\n✅ Server Ready! Ref Audio: '{REF_AUDIO}'")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # 1. Receive text from client
            text_to_gen = await websocket.receive_text()
            print(f"\n📝 Received: {text_to_gen}")

            sentences = split_into_sentences(text_to_gen)

            # Preprocess reference once
            final_ref_audio, final_ref_text = preprocess_ref_audio_text(
                REF_AUDIO, REF_TEXT
            )

            for i, sentence in enumerate(sentences):
                print(
                    f"  ⏳ Generating sentence {i + 1}/{len(sentences)}: '{sentence[:20]}...'"
                )

                audio, sample_rate, spectrogram = infer_process(
                    final_ref_audio,
                    final_ref_text,
                    sentence,
                    models["model"],
                    models["vocoder"],
                    mel_spec_type=VOCODER_NAME,
                    nfe_step=NFE_STEP,
                    speed=SPEED,
                    cfg_strength=CFG_STRENGTH,
                    sway_sampling_coef=SWAY_SAMPLING_COEF,
                    device=DEVICE,
                )

                await websocket.send_text(
                    json.dumps({"sample_rate": sample_rate, "sentence": sentence})
                )

                await websocket.send_bytes(audio.tobytes())
                print(f"  📤 Sent audio for sentence {i + 1}")

            # Signal end of generation for this input
            await websocket.send_text(json.dumps({"status": "done"}))

    except Exception as e:
        print(f"❌ Connection Error: {e}")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
