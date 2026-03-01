"""
CyBot Server - Runs on Windows PC with GPU
Receives audio from Raspberry Pi, processes STT -> LLM -> TTS, returns audio
"""

import os
import sys

# Fix CUDA DLL path BEFORE importing anything else
def setup_cuda_path():
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        cublas_path = os.path.dirname(nvidia.cublas.lib.__file__)
        cudnn_path = os.path.dirname(nvidia.cudnn.lib.__file__)
        os.add_dll_directory(cublas_path)
        os.add_dll_directory(cudnn_path)
        os.environ['PATH'] = cublas_path + os.pathsep + cudnn_path + os.pathsep + os.environ.get('PATH', '')
        return True
    except (ImportError, Exception):
        return False

CUDA_AVAILABLE = setup_cuda_path()

import json
import tempfile
import wave
import io
import numpy as np
import requests
from pathlib import Path
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# Configuration
CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    default_config = {
        "butler_name": "Susan",
        "wake_word_required": True,
        "system_prompt": "You are {name}, a witty English butler. Respond in 1-2 sentences MAX.",
        "ollama_model": "llama3",
        "whisper_model": "base",
        "max_tokens": 75
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
    return default_config

CONFIG = load_config()
BUTLER_NAME = CONFIG["butler_name"]
WAKE_WORD_REQUIRED = CONFIG["wake_word_required"]
SYSTEM_PROMPT = CONFIG["system_prompt"].replace("{name}", BUTLER_NAME)
MAX_TOKENS = CONFIG["max_tokens"]
OLLAMA_MODEL = CONFIG["ollama_model"]
OLLAMA_URL = "http://localhost:11434/api/generate"

# Global models (loaded once at startup)
whisper_model = None
piper_voice = None
conversation_history = []


def load_whisper_model():
    from faster_whisper import WhisperModel
    model_name = CONFIG["whisper_model"]
    
    if CUDA_AVAILABLE:
        try:
            print(f"🔄 Loading Whisper '{model_name}' on GPU...")
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
            print("✅ Whisper loaded on GPU")
            return model
        except Exception as e:
            print(f"⚠️  GPU failed: {e}")
    
    print(f"🔄 Loading Whisper '{model_name}' on CPU...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print("✅ Whisper loaded on CPU")
    return model


def load_piper_voice():
    voice_dir = Path.home() / ".piper-voices"
    default_voice = voice_dir / "amy.onnx"
    
    if not default_voice.exists():
        print("⚠️  Piper voice not found")
        return None
    
    try:
        from piper import PiperVoice
        print("🔊 Loading Piper voice...")
        voice = PiperVoice.load(str(default_voice))
        print(f"✅ Voice loaded @ {voice.config.sample_rate}Hz")
        return voice
    except Exception as e:
        print(f"❌ Piper load error: {e}")
        return None


def transcribe_audio(audio_data, sample_rate):
    """Transcribe audio bytes to text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        with wave.open(f.name, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_data)
    
    try:
        segments, _ = whisper_model.transcribe(temp_path, language="en")
        text = " ".join([seg.text for seg in segments]).strip()
        return text
    finally:
        os.unlink(temp_path)


def query_ollama(prompt):
    """Query the LLM."""
    global conversation_history
    
    # Check for detailed answer
    detail_keywords = ['explain', 'detail', 'elaborate', 'tell me more', 'extensive', 
                       'in depth', 'longer', 'full answer', 'more info', 'describe fully']
    wants_detail = any(kw in prompt.lower() for kw in detail_keywords)
    tokens = MAX_TOKENS * 4 if wants_detail else MAX_TOKENS
    
    context = "\n".join([
        f"User: {h['user']}\n{BUTLER_NAME}: {h['assistant']}" 
        for h in conversation_history[-5:]
    ])
    
    sys_prompt = SYSTEM_PROMPT
    if wants_detail:
        sys_prompt = sys_prompt.replace("1-2 sentences MAX", "a detailed answer (3-5 sentences)")
    
    if context:
        full_prompt = f"{sys_prompt}\n\nConversation:\n{context}\n\nUser: {prompt}\n{BUTLER_NAME}:"
    else:
        full_prompt = f"{sys_prompt}\n\nUser: {prompt}\n{BUTLER_NAME}:"
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": tokens}
            },
            timeout=60
        )
        result = response.json().get("response", "").strip()
        
        conversation_history.append({"user": prompt, "assistant": result})
        return result
    except Exception as e:
        return f"I apologize, but I encountered an error: {e}"


def synthesize_speech(text):
    """Convert text to audio bytes."""
    if piper_voice is None:
        return None, 22050
    
    audio_bytes = b''
    for chunk in piper_voice.synthesize(text):
        audio_bytes += chunk.audio_int16_bytes
    
    return audio_bytes, piper_voice.config.sample_rate


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "butler_name": BUTLER_NAME,
        "whisper_loaded": whisper_model is not None,
        "piper_loaded": piper_voice is not None
    })


@app.route('/process', methods=['POST'])
def process_audio():
    """
    Main endpoint: receive audio, return audio response.
    
    Input: WAV audio file
    Output: WAV audio response
    """
    global conversation_history
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    sample_rate = int(request.form.get('sample_rate', 44100))
    
    # Read audio data
    audio_data = audio_file.read()
    
    print(f"\n📥 Received {len(audio_data)} bytes @ {sample_rate}Hz")
    
    # 1. Transcribe
    print("🔄 Transcribing...")
    user_text = transcribe_audio(audio_data, sample_rate)
    print(f"📝 Heard: {user_text}")
    
    if not user_text:
        return jsonify({"error": "Could not transcribe", "text": ""}), 200
    
    # 2. Check wake word
    if WAKE_WORD_REQUIRED:
        if BUTLER_NAME.lower() not in user_text.lower():
            print(f"💤 No wake word '{BUTLER_NAME}'")
            return jsonify({
                "text": user_text,
                "response": "",
                "wake_word_missing": True
            }), 200
        
        # Remove wake word
        import re
        user_text = re.sub(rf'\b{BUTLER_NAME}\b[,\s]*', '', user_text, flags=re.IGNORECASE).strip()
        
        if not user_text:
            response_text = "Yes? How may I assist you?"
            audio_bytes, sr = synthesize_speech(response_text)
            return send_audio_response(audio_bytes, sr, user_text, response_text)
    
    # 3. Check commands
    if user_text.lower() in ["quit", "exit", "bye", "goodbye"]:
        response_text = "Very good. It has been a pleasure serving you. Do take care!"
        audio_bytes, sr = synthesize_speech(response_text)
        return send_audio_response(audio_bytes, sr, user_text, response_text, quit=True)
    
    if user_text.lower() in ["clear", "reset", "start over"]:
        conversation_history.clear()
        response_text = "Consider it done. Our previous conversation has been discreetly forgotten."
        audio_bytes, sr = synthesize_speech(response_text)
        return send_audio_response(audio_bytes, sr, user_text, response_text)
    
    # 4. Query LLM
    print(f"🎩 {BUTLER_NAME} is pondering...")
    response_text = query_ollama(user_text)
    print(f"🎩 {BUTLER_NAME}: {response_text}")
    
    # 5. Synthesize response
    print("🔊 Synthesizing...")
    audio_bytes, sr = synthesize_speech(response_text)
    
    return send_audio_response(audio_bytes, sr, user_text, response_text)


def send_audio_response(audio_bytes, sample_rate, user_text, response_text, quit=False):
    """Package audio response as WAV file."""
    if audio_bytes is None:
        return jsonify({
            "text": user_text,
            "response": response_text,
            "quit": quit
        }), 200
    
    # Create WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_bytes)
    wav_buffer.seek(0)
    
    response = send_file(
        wav_buffer,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='response.wav'
    )
    response.headers['X-User-Text'] = user_text
    response.headers['X-Response-Text'] = response_text
    response.headers['X-Quit'] = str(quit)
    
    return response


@app.route('/greeting', methods=['GET'])
def get_greeting():
    """Get the butler's greeting."""
    greeting = f"Good day! I am {BUTLER_NAME}, your digital butler. How may I be of service?"
    audio_bytes, sr = synthesize_speech(greeting)
    return send_audio_response(audio_bytes, sr, "", greeting)


if __name__ == '__main__':
    print("=" * 60)
    print(f"🎩 {BUTLER_NAME} Server - CyBot Network Mode")
    print("=" * 60)
    
    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        print("✅ Ollama running")
    except:
        print("❌ Ollama not running! Start with: ollama serve")
        sys.exit(1)
    
    # Load models
    whisper_model = load_whisper_model()
    piper_voice = load_piper_voice()
    
    print("\n" + "=" * 60)
    print("🌐 Starting server on http://0.0.0.0:5000")
    print("   Raspberry Pi should connect to this PC's IP address")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
