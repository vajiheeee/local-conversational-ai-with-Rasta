"""
Voice Assistant Client Interface
================================
This module acts as the voice-enabled frontend for the Rasa conversational agent.
As intended, it operates entirely locally, ensuring maximum privacy and offline capability.

Architecture Pipeline:
1. Voice Activity Detection (VAD): Uses RMS audio thresholding to detect when the user speaks.
2. Speech-to-Text (ASR): Utilizes OpenAI's `faster-whisper` for local transcription.
3. Dialogue Management: Forwards transcripts to the Rasa Action Server via REST webhook.
4. Text-to-Speech (TTS): Synthesizes the bot's text responses into natural voice using PiperTTS.

Hardware Optimization:
Whisper automatically attempts to utilize CUDA (GPU) for inference but gracefully 
falls back to INT8 CPU computation if a compatible GPU is not detected.

Setup Note:
To run this locally, ensure you have downloaded the `piper` executable and 
the `en_US-ryan-high` ONNX voice model into your project directory.
"""

import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
import time
import os
import subprocess
import tempfile
from faster_whisper import WhisperModel

# Faster-whisper + PiperTTS parts are similar to their github repository codes.

# --- CONFIGURATION ---
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

# --- PIPER TTS CONFIG ---
MODEL_DIR = "models/en_US-ryan-high"
MODEL_ONNX = os.path.join(MODEL_DIR, "en_US-ryan-high.onnx")
MODEL_JSON = os.path.join(MODEL_DIR, "en_US-ryan-high.onnx.json")

# --- AUDIO SETTINGS ---
samplerate = 16000 
block_duration = 0.06 
channels = 1
frames_per_block = int(samplerate * block_duration)

audio_queue = queue.Queue()

# ==========================================
# 1. PIPER TTS FUNCTIONS
# ==========================================

def add_silence(data, samplerate, pre_s=0.6, post_s=0.6):
    pre = np.zeros(int(samplerate * pre_s), dtype=data.dtype)
    post = np.zeros(int(samplerate * post_s), dtype=data.dtype)
    return np.concatenate([pre, data, post])

def tts(text: str):
    """Text-to-Speech using Piper TTS engine"""
    if not os.path.exists(MODEL_ONNX):
        print(f"Error: Model file not found: {MODEL_ONNX}")
        return
    if not os.path.exists(MODEL_JSON):
        print(f"Error: Config file not found: {MODEL_JSON}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        out_wav = temp_wav.name

    try:
        cmd = [
            "piper",
            "--model", MODEL_ONNX,
            "--config", MODEL_JSON,
            "--output_file", out_wav
        ]

        # Run Piper process
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            print("Piper failed:", result.stderr.decode(errors="ignore"))
            return

        # Small wait for file system
        for _ in range(20):
            if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
                break
            time.sleep(0.01)

        data, sr = sf.read(out_wav, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]  # mono

        data = add_silence(data, sr, pre_s=0.6, post_s=0.6)

        print(f"🤖 Bot: {text}")
        sd.play(data, sr)
        sd.wait()

    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        if os.path.exists(out_wav):
            os.remove(out_wav)

# ==========================================
# 2. WHISPER & RECORDER
# ==========================================

def init_whisper(model_size: str = "large-v3"): 
    if os.getenv("WHISPER_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        print("[STT] Forced CPU.")
        return WhisperModel(model_size, device="cpu", compute_type="int8")

    try:
        print(f"[STT] Attempting CUDA with {model_size}...")
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        return model
    except Exception as e:
        print(f"[STT] CUDA failed ({e}). Using CPU (int8).")
        return WhisperModel(model_size, device="cpu", compute_type="int8")

# Initialize Model
print("Loading Whisper... This may take a moment.")
model = init_whisper("large-v3") 
print("Whisper Loaded.")

def rms(block: np.ndarray) -> float:
    x = block.astype(np.float32).flatten()
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def audio_callback(indata, frames, t, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=audio_callback,
        blocksize=frames_per_block,
        dtype="float32",
    ):
        print("\n🎙️  Your AI assistant for weather and calendar is here.")
        while True:
            sd.sleep(1000)

# ==========================================
# 3. Connecting above with Rasa
# ==========================================

def process_text_with_rasa(user_text):
    """Sends text to Rasa and speaks the response."""
    try:
        payload = {"sender": "user", "message": user_text}
        resp = requests.post(RASA_URL, json=payload)
        
        if resp.status_code == 200:
            bot_responses = resp.json()
            for msg in bot_responses:
                text_response = msg.get("text")
                if text_response:
                    tts(text_response)
                    
                    exit_phrases = ["bye", "goodbye", "shut off", "have a nice day", "see you"]
                    if any(phrase in text_response.lower() for phrase in exit_phrases):
                        print("System shutting down... :(")
                        time.sleep(1)
                        os._exit(0)
        else:
            print(f"Rasa Error: {resp.status_code}")
    except Exception as e:
        print(f"Connection Error: {e}")

def transcriber_loop():
    """
    Records -> Transcribes -> Sends to Rasa
    """
    # Endpointing parameters
    silence_rms_threshold = 0.010
    start_speech_seconds = 0.18
    start_speech_blocks = max(1, int(start_speech_seconds / block_duration))
    end_silence_seconds = 1.3
    
    # Whisper params
    beam_size = 5
    whisper_vad_min_silence_ms = 300 

    in_utt = False
    speech_blocks = 0
    utterance_blocks = []
    
    utt_start_time = None
    last_speech_time = None
    max_utt_seconds = 15.0
    min_utt_seconds = 0.25

    while True:
        block = audio_queue.get()
        block_rms = rms(block)
        is_speechish = block_rms >= silence_rms_threshold
        now = time.time()

        if not in_utt:
            if is_speechish:
                speech_blocks += 1
                utterance_blocks.append(block)
                if speech_blocks >= start_speech_blocks:
                    in_utt = True
                    utt_start_time = now
                    last_speech_time = now
            else:
                speech_blocks = 0
                utterance_blocks.clear()
        else:
            utterance_blocks.append(block)
            if is_speechish:
                last_speech_time = now

            utt_len = now - (utt_start_time or now)
            silence_len = now - (last_speech_time or now)
            should_end = (silence_len >= end_silence_seconds) or (utt_len >= max_utt_seconds)

            if should_end:
                if utt_len >= min_utt_seconds:
                    # 1. Transcribe
                    print("Processing audio...", end="\r")
                    audio = np.concatenate(utterance_blocks).flatten().astype(np.float32)
                    
                    segments, _ = model.transcribe(
                        audio,
                        language="en",
                        beam_size=beam_size,
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": whisper_vad_min_silence_ms},
                        word_timestamps=False,
                        condition_on_previous_text=False,
                        initial_prompt="Frankfurt, Marburg, Calendar, Weather" 
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                    
                    if text:
                        print(f"👤 You: {text}")
                        # 2. Send to Rasa & Speak
                        process_text_with_rasa(text)
                        print("🎙️  Ready...")

                # Reset
                in_utt = False
                speech_blocks = 0
                utterance_blocks.clear()
                utt_start_time = None
                last_speech_time = None

if __name__ == "__main__":
    threading.Thread(target=recorder, daemon=True).start()
    transcriber_loop()