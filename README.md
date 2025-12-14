# Create a conda env with python_version>=3.10 (you could also use virtualenv)

conda create -n f5-tts python=3.11
conda activate f5-tts

# Install pytorch with your CUDA version, e.g

pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url <https://download.pytorch.org/whl/cu124>

pip install f5-tts faster-whisper
