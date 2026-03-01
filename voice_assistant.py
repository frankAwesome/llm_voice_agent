"""
CyBot - Local Voice Assistant
Speech-to-Text (faster-whisper) -> LLM (Ollama) -> Text-to-Speech (piper-tts)
Runs entirely on your local machine with NVIDIA GPU acceleration.
"""

import os
import sys

# Fix CUDA DLL path BEFORE importing anything else
def setup_cuda_path():
    """Add NVIDIA CUDA DLLs to PATH if installed via pip."""
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

import pyaudio
import numpy as np
import requests
import subprocess
import tempfile
import wave
import threading
import queue
from pathlib import Path

# Configuration - loaded from config.json
CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    """Load configuration from config.json."""
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
            import json
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                default_config.update(config)
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
    
    return default_config

CONFIG = load_config()
WHISPER_MODEL = CONFIG["whisper_model"]
OLLAMA_MODEL = CONFIG["ollama_model"]
OLLAMA_URL = "http://localhost:11434/api/generate"
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

# Butler Personality from config
BUTLER_NAME = CONFIG["butler_name"]
WAKE_WORD_REQUIRED = CONFIG["wake_word_required"]
SYSTEM_PROMPT = CONFIG["system_prompt"].replace("{name}", BUTLER_NAME)
MAX_TOKENS = CONFIG["max_tokens"]

# Audio settings (will be auto-detected)
SAMPLE_RATE = 44100  # Volt 2 uses 44100Hz
CHANNELS = 1
INPUT_DEVICE = None
OUTPUT_DEVICE = None
CHUNK_SIZE = 1024

# PyAudio instance
pa = None

# Global state
is_recording = False
audio_queue = queue.Queue()
VAD_THRESHOLD = 500  # Will be calibrated at startup
PIPER_VOICE = None  # Pre-loaded TTS voice for instant speech


def setup_audio():
    """Configure Volt 2 as input and output device."""
    global SAMPLE_RATE, INPUT_DEVICE, OUTPUT_DEVICE, pa
    
    print("🔍 Detecting Volt 2 audio devices...")
    
    pa = pyaudio.PyAudio()
    
    # Find Volt 2 devices
    volt_input = None
    volt_output = None
    
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        name = dev['name'].lower()
        
        if 'volt' in name:
            if dev['maxInputChannels'] > 0 and volt_input is None:
                volt_input = i
                print(f"  ✅ Found Volt input: [{i}] {dev['name']}")
            if dev['maxOutputChannels'] > 0 and volt_output is None:
                volt_output = i
                print(f"  ✅ Found Volt output: [{i}] {dev['name']}")
    
    if volt_input is None:
        print("❌ Volt 2 microphone not found!")
        return False
    
    if volt_output is None:
        print("❌ Volt 2 speakers not found!")
        return False
    
    INPUT_DEVICE = volt_input
    OUTPUT_DEVICE = volt_output
    
    print(f"\n✅ Using Volt 2 @ {SAMPLE_RATE} Hz")
    print(f"   Input device: [{INPUT_DEVICE}]")
    print(f"   Output device: [{OUTPUT_DEVICE}]")
    return True


def check_dependencies():
    """Check if all required services are available."""
    print("🔍 Checking dependencies...")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            if not any(OLLAMA_MODEL in m for m in models):
                print(f"⚠️  Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
                return False
            print(f"✅ Ollama running with {OLLAMA_MODEL}")
        else:
            print("❌ Ollama not responding")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running. Start it with: ollama serve")
        return False
    
    # Check faster-whisper
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper available")
    except ImportError:
        print("❌ faster-whisper not installed. Run: pip install faster-whisper")
        return False
    
    # Check piper
    try:
        result = subprocess.run(["piper", "--help"], capture_output=True, timeout=5)
        print("✅ piper-tts available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠️  piper CLI not found - will use Python fallback")
    
    return True


def load_whisper_model():
    """Load the Whisper model with GPU acceleration if available."""
    from faster_whisper import WhisperModel
    
    # Try GPU first, fall back to CPU
    if CUDA_AVAILABLE:
        try:
            print(f"🔄 Loading Whisper model '{WHISPER_MODEL}' on GPU...")
            model = WhisperModel(
                WHISPER_MODEL,
                device="cuda",
                compute_type="float16"  # Use float16 for 4070 Ti
            )
            print("✅ Whisper model loaded on GPU")
            return model
        except Exception as e:
            print(f"⚠️  GPU loading failed: {e}")
    
    print(f"🔄 Loading Whisper model '{WHISPER_MODEL}' on CPU...")
    model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8"  # int8 is faster on CPU
    )
    print("✅ Whisper model loaded on CPU (slower but works)")
    return model


def calibrate_microphone():
    """Calibrate microphone noise floor once at startup."""
    global VAD_THRESHOLD
    
    print("\n🎤 Calibrating microphone... (stay quiet for 2 seconds)")
    
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=INPUT_DEVICE,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Measure ambient noise for 2 seconds
        noise_samples = []
        chunks_for_calibration = int(SAMPLE_RATE / CHUNK_SIZE * 2)
        for _ in range(chunks_for_calibration):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            noise_samples.append(np.abs(audio_chunk).mean())
        
        stream.stop_stream()
        stream.close()
        
        # Set threshold at 2.5x the average noise floor
        noise_floor = np.mean(noise_samples)
        VAD_THRESHOLD = max(noise_floor * 2.5, 200)  # At least 200 to avoid false triggers
        print(f"   ✅ Noise floor: {noise_floor:.0f}, Threshold: {VAD_THRESHOLD:.0f}")
        return True
        
    except Exception as e:
        print(f"❌ Calibration error: {e}")
        VAD_THRESHOLD = 500  # Fallback
        return False


def record_audio_vad():
    """Record audio with voice activity detection using PyAudio."""
    print("\n🎤 Listening... (speak now)")
    
    audio_chunks = []
    silence_counter = 0
    has_speech = False
    max_duration = 30
    chunks_per_second = SAMPLE_RATE / CHUNK_SIZE
    
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=INPUT_DEVICE,
            frames_per_buffer=CHUNK_SIZE
        )
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            
            volume = np.abs(audio_chunk).mean()
            
            # Visual feedback
            bar_length = int(min(volume / 100, 30))
            bar = "█" * bar_length
            status = "🔴 SPEECH" if volume > VAD_THRESHOLD else "⚪ quiet"
            print(f"\r   {status} [{bar:<30}] {volume:.0f}  ", end="", flush=True)
            
            if volume > VAD_THRESHOLD:
                has_speech = True
                silence_counter = 0
                audio_chunks.append(audio_chunk)
            elif has_speech:
                silence_counter += 1 / chunks_per_second
                audio_chunks.append(audio_chunk)
                
                if silence_counter >= SILENCE_DURATION:
                    print()  # Newline after progress
                    break
            
            # Timeout check
            total_recorded = len(audio_chunks) / chunks_per_second
            if total_recorded >= max_duration:
                print("\n⏱️  Max duration reached")
                break
                
            if not has_speech and total_recorded >= 10:
                print("\n⏱️  No speech detected, try again...")
                stream.stop_stream()
                stream.close()
                return None
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"\n❌ Audio recording error: {e}")
        return None
    
    if not audio_chunks:
        return None
    
    audio = np.concatenate(audio_chunks)
    # Convert to float32 for Whisper
    audio_float = audio.astype(np.float32) / 32768.0
    print(f"📝 Recorded {len(audio) / SAMPLE_RATE:.1f} seconds")
    return audio_float


def record_audio_push_to_talk():
    """Record audio while Enter is held (press Enter to start, Enter to stop)."""
    print("\n🎤 Press ENTER to start recording...")
    
    # Pre-open the stream so it's ready immediately when user presses Enter
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=INPUT_DEVICE,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"❌ Could not open audio stream: {e}")
        return None
    
    input()  # Wait for first Enter
    
    print("🔴 Recording... Press ENTER to stop")
    audio_chunks = []
    stop_recording = threading.Event()
    
    def wait_for_stop():
        input()
        stop_recording.set()
    
    stop_thread = threading.Thread(target=wait_for_stop)
    stop_thread.start()
    
    try:
        while not stop_recording.is_set():
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_chunks.append(audio_chunk)
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"❌ Audio recording error: {e}")
        stop_recording.set()
        stream.stop_stream()
        stream.close()
        return None
    
    stop_thread.join()
    
    if not audio_chunks:
        return None
    
    audio = np.concatenate(audio_chunks)
    # Convert to float32 for Whisper
    audio_float = audio.astype(np.float32) / 32768.0
    print(f"📝 Recorded {len(audio) / SAMPLE_RATE:.1f} seconds")
    return audio_float


def transcribe_audio(model, audio):
    """Transcribe audio using faster-whisper."""
    print("🔄 Transcribing...")
    
    # Save to temp file for faster-whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(f.name, 'wb') as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio_int16.tobytes())
    
    try:
        segments, info = model.transcribe(temp_path, language="en")
        text = " ".join([segment.text for segment in segments]).strip()
        print(f"📝 You said: {text}")
        return text
    finally:
        os.unlink(temp_path)


def query_ollama(prompt, conversation_history):
    """Send prompt to Ollama and get streaming response."""
    print(f"🎩 {BUTLER_NAME} is pondering...")
    
    # Check if user wants a detailed answer
    detail_keywords = ['explain', 'detail', 'elaborate', 'tell me more', 'extensive', 
                       'in depth', 'longer', 'full answer', 'more info', 'describe fully']
    wants_detail = any(kw in prompt.lower() for kw in detail_keywords)
    tokens = MAX_TOKENS * 4 if wants_detail else MAX_TOKENS  # 300 vs 75 tokens
    
    # Build context from history
    context = "\n".join([
        f"User: {h['user']}\n{BUTLER_NAME}: {h['assistant']}" 
        for h in conversation_history[-5:]  # Last 5 exchanges
    ])
    
    # Adjust system prompt if detailed answer requested
    if wants_detail:
        sys_prompt = SYSTEM_PROMPT.replace("1-2 sentences MAX", "a detailed answer (3-5 sentences)")
    else:
        sys_prompt = SYSTEM_PROMPT
    
    # Include system prompt for personality
    if context:
        full_prompt = f"{sys_prompt}\n\nConversation so far:\n{context}\n\nUser: {prompt}\n{BUTLER_NAME}:"
    else:
        full_prompt = f"{sys_prompt}\n\nUser: {prompt}\n{BUTLER_NAME}:"
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": tokens
                }
            },
            stream=True,
            timeout=60
        )
        
        full_response = ""
        print(f"🎩 {BUTLER_NAME}: ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                chunk = data.get("response", "")
                full_response += chunk
                print(chunk, end="", flush=True)
                
                if data.get("done", False):
                    break
        
        print()  # Newline after response
        return full_response.strip()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error querying Ollama: {e}")
        return "Sorry, I couldn't process that request."


def load_piper_voice():
    """Pre-load Piper voice model for instant TTS."""
    global PIPER_VOICE
    
    voice_dir = Path.home() / ".piper-voices"
    default_voice = voice_dir / "amy.onnx"
    
    if not default_voice.exists():
        print("⚠️  Piper voice not found. Run setup_piper_voice.py first.")
        return False
    
    try:
        from piper import PiperVoice
        print("🔊 Loading TTS voice (one-time)...")
        PIPER_VOICE = PiperVoice.load(str(default_voice))
        print(f"   ✅ Voice loaded: {default_voice.name} @ {PIPER_VOICE.config.sample_rate}Hz")
        return True
    except Exception as e:
        print(f"❌ Failed to load Piper voice: {e}")
        return False


def speak_text_piper(text, voice_model=None):
    """Convert text to speech using pre-loaded Piper voice with streaming."""
    global PIPER_VOICE
    
    if PIPER_VOICE is None:
        # Try loading if not already loaded
        if not load_piper_voice():
            speak_text_fallback(text)
            return
    
    try:
        # Stream audio directly to PyAudio - no temp file!
        piper_sample_rate = PIPER_VOICE.config.sample_rate  # Usually 22050
        
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=piper_sample_rate,
            output=True,
            output_device_index=OUTPUT_DEVICE
        )
        
        # Stream each chunk as it's generated - starts playing immediately
        for audio_chunk in PIPER_VOICE.synthesize(text):
            stream.write(audio_chunk.audio_int16_bytes)
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"⚠️  TTS streaming error: {e}")
        speak_text_fallback(text)


def speak_text_fallback(text):
    """Fallback TTS using Windows SAPI or espeak."""
    if sys.platform == "win32":
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            return
        except ImportError:
            pass
        
        # PowerShell fallback
        try:
            subprocess.run([
                "powershell", "-Command",
                f'Add-Type -AssemblyName System.Speech; '
                f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$speak.Speak("{text.replace(chr(34), chr(39))}")'
            ], timeout=60)
            return
        except subprocess.TimeoutExpired:
            pass
    else:
        # Linux/Mac: try espeak
        try:
            subprocess.run(["espeak", text], timeout=60)
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    print(f"[TTS unavailable - response text above]")


def play_audio_file(filepath):
    """Play a WAV file through Volt 2."""
    try:
        wf = wave.open(filepath, 'rb')
        
        stream = pa.open(
            format=pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            output_device_index=OUTPUT_DEVICE  # Use Volt 2 output
        )
        
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        while data:
            stream.write(data)
            data = wf.readframes(chunk_size)
        
        stream.stop_stream()
        stream.close()
        wf.close()
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")


def main():
    """Main conversation loop."""
    print("=" * 60)
    print(f"🎩 {BUTLER_NAME} - Your Distinguished Digital Butler")
    print("=" * 60)
    print("Powered by: faster-whisper + Ollama + piper-tts")
    print("All processing runs locally on your GPU, naturally.")
    print("=" * 60)
    
    # Setup audio first
    if not setup_audio():
        print("\n❌ Failed to setup audio. Please check your microphone.")
        return
    
    if not check_dependencies():
        print("\n❌ Please fix the issues above and try again.")
        return
    
    # Pre-load models for instant response
    whisper_model = load_whisper_model()
    load_piper_voice()  # Pre-load TTS for instant speech
    
    # Conversation history
    conversation_history = []
    
    print("\n" + "=" * 60)
    print(f"🎩 {BUTLER_NAME} at your service!")
    print("Options:")
    print("  1. Voice Activity Detection (I shall listen attentively)")
    print("  2. Push-to-Talk (press Enter when you wish to speak)")
    print("  Type 'quit' to dismiss me, 'clear' to forget our chat")
    print("=" * 60)
    
    # Choose recording mode
    print("\nSelect mode (1 or 2): ", end="")
    mode = input().strip()
    use_vad = mode != "2"
    
    if use_vad:
        # Calibrate microphone once for VAD mode
        calibrate_microphone()
        print("📢 Very good. I shall detect when you speak.")
    else:
        print("📢 Very good. Press Enter when you wish to address me.")
    
    # Butler greeting
    greeting = f"Good day! I am {BUTLER_NAME}, your digital butler. How may I be of service?"
    print(f"\n🎩 {BUTLER_NAME}: {greeting}")
    speak_text_piper(greeting)
    
    while True:
        try:
            # Record audio
            if use_vad:
                audio = record_audio_vad()
            else:
                audio = record_audio_push_to_talk()
            
            if audio is None:
                continue
            
            # Transcribe
            user_text = transcribe_audio(whisper_model, audio)
            
            if not user_text:
                print("🤷 Couldn't understand that, try again...")
                continue
            
            # Check for wake word if required
            if WAKE_WORD_REQUIRED:
                if BUTLER_NAME.lower() not in user_text.lower():
                    print(f"💤 (No wake word '{BUTLER_NAME}' detected, ignoring)")
                    continue
                # Remove wake word from query for cleaner processing
                import re
                user_text = re.sub(rf'\b{BUTLER_NAME}\b[,\s]*', '', user_text, flags=re.IGNORECASE).strip()
                if not user_text:
                    # Just said the name with nothing else
                    response = "Yes? How may I assist you?"
                    print(f"🎩 {BUTLER_NAME}: {response}")
                    speak_text_piper(response)
                    continue
            
            # Check for commands
            if user_text.lower() in ["quit", "exit", "bye", "goodbye"]:
                farewell = "Very good. It has been a pleasure serving you. Do take care!"
                print(f"🎩 {BUTLER_NAME}: {farewell}")
                speak_text_piper(farewell)
                break
            
            if user_text.lower() in ["clear", "reset", "start over"]:
                conversation_history.clear()
                response = "Consider it done. Our previous conversation has been... discreetly forgotten."
                print(f"🎩 {BUTLER_NAME}: {response}")
                speak_text_piper(response)
                continue
            
            # Query LLM
            response = query_ollama(user_text, conversation_history)
            
            # Save to history
            conversation_history.append({
                "user": user_text,
                "assistant": response
            })
            
            # Speak response
            speak_text_piper(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
