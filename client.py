import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import threading
import queue


audio_queue = queue.Queue()


# This ensures playback blocks locally for each sentence without blocking the network receive loop
def audio_player_thread():
    while True:
        item = audio_queue.get()
        if item is None:
            break

        audio_data, sample_rate = item
        print(f"🔊 Playing sentence... ({len(audio_data) / sample_rate:.2f}s)")

        try:
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until this sentence finishes
        except Exception as e:
            print(f"Playback error: {e}")

        audio_queue.task_done()


# Start the player thread
player_thread = threading.Thread(target=audio_player_thread, daemon=True)
player_thread.start()


async def communicate():
    uri = "ws://localhost:8000/ws"

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to F5-TTS Server!")

            # Input loop
            while True:
                text = await asyncio.to_thread(input, "\n📝 Enter text (or 'exit'): ")
                if text.lower() in ["exit", "quit"]:
                    break
                if not text.strip():
                    continue

                await websocket.send(text)

                # Receive loop for this specific request
                print("⏳ Waiting for audio stream...")
                current_sample_rate = 24000  # Default fallback

                while True:
                    message = await websocket.recv()

                    if isinstance(message, str):
                        data = json.loads(message)
                        if "status" in data and data["status"] == "done":
                            break
                        if "sample_rate" in data:
                            current_sample_rate = data["sample_rate"]

                    elif isinstance(message, bytes):
                        audio_np = np.frombuffer(message, dtype=np.float32)
                        audio_queue.put((audio_np, current_sample_rate))

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(communicate())
    except KeyboardInterrupt:
        print("\nExiting...")
        audio_queue.put(None)  # Signal thread to stop

