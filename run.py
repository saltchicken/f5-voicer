import sounddevice as sd
import soundfile as sf
import torch
import time
import os
import sys
import warnings
import re


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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
REF_AUDIO = "ref.wav"
REF_TEXT = "Do you know how many books will be signed and when can we buy them. So I have signed pages and I had to send them to."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_NAME = "F5TTS_Base"
VOCODER_NAME = "vocos"
SPEED = 1.0
CROSS_FADE_DURATION = 0.15
NFE_STEP = 16  # Reduced to 16 for speed (CLI default is 32)
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0


def chunk_text(text):
    """
    ‼️ Simple helper to split text into sentences for streaming.
    Splits by . ? ! or newline, keeping the punctuation.
    """
    # Split by punctuation followed by space or end of string
    pattern = r"([.?!:;]+(?:\s+|$))"
    chunks = re.split(pattern, text)

    # Recombine split parts (text + punctuation)
    result = []
    current_chunk = ""
    for part in chunks:
        current_chunk += part
        # If the part ends with punctuation/space, assume it's a complete sentence
        if re.search(r"[.?!:;]\s*$", part):
            if current_chunk.strip():
                result.append(current_chunk.strip())
            current_chunk = ""

    if current_chunk.strip():
        result.append(current_chunk.strip())

    return result


def main():
    print(f"🚀 Initializing F5-TTS on {DEVICE}...")

    # 1. Validation
    if not os.path.exists(REF_AUDIO):
        print(f"\n❌ Error: Could not find '{REF_AUDIO}'")
        return

    global REF_TEXT
    if not REF_TEXT.strip():
        REF_TEXT = input(f"   👉 Please type what is said in '{REF_AUDIO}': ").strip()
        if not REF_TEXT:
            print("❌ Error: Reference text is required.")
            return

    # ------------------------------------------------------------------
    # 2. LOAD MODEL (The Official Way)
    # ------------------------------------------------------------------
    print("\n⬇️  Loading Model Configuration...")

    config_path = files("f5_tts").joinpath(f"configs/{MODEL_NAME}.yaml")
    model_cfg = OmegaConf.load(str(config_path))

    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    repo_name = "F5-TTS"
    ckpt_step = 1200000
    ckpt_type = "safetensors"  # or 'pt'

    # Download weights using cached_path (handles caching automatically)
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

    print(f"\n✅ Ready! Ref Audio: '{REF_AUDIO}'")
    print("--------------------------------------------------")

    while True:
        try:
            text_to_gen = input("\n📝 Text to generate: ")

            if text_to_gen.lower() in ["exit", "quit"]:
                break
            if not text_to_gen.strip():
                continue

            final_ref_audio, final_ref_text = preprocess_ref_audio_text(
                REF_AUDIO, REF_TEXT
            )

            chunks = chunk_text(text_to_gen)
            print(f"⏳ Generating {len(chunks)} chunks...")

            total_start = time.time()

            # Loop through chunks
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                chunk_start = time.time()

                # Generate audio for THIS chunk
                audio, sample_rate, spectrogram = infer_process(
                    final_ref_audio,
                    final_ref_text,
                    chunk,
                    model,
                    vocoder,
                    mel_spec_type=VOCODER_NAME,
                    nfe_step=NFE_STEP,
                    speed=SPEED,
                    cfg_strength=CFG_STRENGTH,
                    sway_sampling_coef=SWAY_SAMPLING_COEF,
                    device=DEVICE,
                )

                chunk_time = time.time() - chunk_start
                print(
                    f"   ▶️  Playing chunk {i + 1}/{len(chunks)} ({chunk_time:.2f}s gen time)"
                )

                # sd.play is non-blocking (fire and forget), but we call sd.wait()
                # AFTER starting the NEXT generation to overlap them.
                # However, for a simple script, blocking here ensures smooth playback
                # without race conditions on the audio device.
                sd.play(audio, sample_rate)
                sd.wait()  # Block until this chunk finishes playing

            print(f"✅ Done. Total time: {time.time() - total_start:.2f}s")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()

