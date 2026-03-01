# CyBot - Local Voice Assistant 🤖🎤

A completely **free, local** voice assistant that runs entirely on your NVIDIA GPU.

**Stack:**
- 🎤 **Speech-to-Text:** faster-whisper (GPU accelerated)
- 🧠 **LLM:** Ollama with Llama3
- 🔊 **Text-to-Speech:** piper-tts

## Quick Start

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install.

```bash
# Pull the Llama3 model
ollama pull llama3
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Piper Voice (for TTS)

```bash
# Download a voice model
python setup_piper_voice.py amy
```

### 4. Run the Assistant

```bash
# Make sure Ollama is running first!
ollama serve  # In a separate terminal

# Run the voice assistant
python voice_assistant.py
```

## Usage

1. Choose your recording mode:
   - **Voice Activity Detection (1):** Automatically detects when you start/stop speaking
   - **Push-to-Talk (2):** Press Enter to start recording, Enter again to stop

2. Speak your question/command

3. The assistant will:
   - Transcribe your speech (faster-whisper)
   - Generate a response (Llama3)
   - Speak the response (piper-tts)

### Voice Commands
- Say "quit", "exit", or "goodbye" to exit
- Say "clear" or "reset" to clear conversation history

## Configuration

Edit `voice_assistant.py` to customize:

```python
WHISPER_MODEL = "base"    # Options: tiny, base, small, medium, large-v3
OLLAMA_MODEL = "llama3"   # Any Ollama model
SILENCE_THRESHOLD = 0.01  # Adjust for your mic
SILENCE_DURATION = 1.5    # Seconds of silence before stopping
```

### Whisper Model Sizes

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~1GB | Fastest | Good |
| base | ~1GB | Fast | Better |
| small | ~2GB | Medium | Great |
| medium | ~5GB | Slower | Excellent |
| large-v3 | ~10GB | Slowest | Best |

For a 4070 Ti (12GB), you can run `large-v3` + `llama3:8b` together!

### Alternative LLM Models

```bash
ollama pull mistral        # Fast, good quality
ollama pull phi3           # Small, fast
ollama pull llama3:70b     # Large, best quality (needs 48GB+ VRAM or will use CPU)
```

## Troubleshooting

### "Ollama not running"
Start Ollama in a separate terminal:
```bash
ollama serve
```

### "No speech detected"
- Check your microphone is working
- Adjust `SILENCE_THRESHOLD` in the script

### TTS not working
1. Run `python setup_piper_voice.py amy` to download a voice
2. If piper doesn't work, the script falls back to Windows SAPI

### CUDA errors
Make sure you have:
- NVIDIA GPU drivers installed
- CUDA toolkit (usually comes with PyTorch)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Cost: $0/month 💰

Everything runs locally on your GPU. No API keys, no subscriptions, no data sent to the cloud!

## License

MIT - Do whatever you want with it!
