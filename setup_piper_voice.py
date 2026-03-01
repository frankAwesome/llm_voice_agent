"""
Download and setup a Piper voice model.
Run this once to download a voice for text-to-speech.
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path

# Voice options (English) - pick one
VOICES = {
    "amy": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "description": "US English female (medium quality)"
    },
    "lessac": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "description": "US English male (medium quality)"
    },
    "jenny": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json",
        "description": "British English female (medium quality)"
    }
}

def download_voice(voice_name="amy"):
    """Download a Piper voice model."""
    if voice_name not in VOICES:
        print(f"Unknown voice: {voice_name}")
        print(f"Available: {', '.join(VOICES.keys())}")
        return None
    
    voice = VOICES[voice_name]
    voice_dir = Path.home() / ".piper-voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = voice_dir / f"{voice_name}.onnx"
    config_path = voice_dir / f"{voice_name}.onnx.json"
    
    if model_path.exists() and config_path.exists():
        print(f"✅ Voice '{voice_name}' already downloaded at {model_path}")
        return str(model_path)
    
    print(f"📥 Downloading voice: {voice_name} ({voice['description']})")
    
    try:
        # Download model
        print(f"   Downloading model... (this may take a minute)")
        urllib.request.urlretrieve(voice["url"], model_path)
        print(f"   ✅ Model saved to {model_path}")
        
        # Download config
        print(f"   Downloading config...")
        urllib.request.urlretrieve(voice["config"], config_path)
        print(f"   ✅ Config saved to {config_path}")
        
        print(f"\n✅ Voice '{voice_name}' ready!")
        return str(model_path)
        
    except Exception as e:
        print(f"❌ Error downloading voice: {e}")
        return None


def list_voices():
    """List available and downloaded voices."""
    voice_dir = Path.home() / ".piper-voices"
    
    print("\n📢 Available Piper Voices:\n")
    for name, info in VOICES.items():
        model_path = voice_dir / f"{name}.onnx"
        status = "✅ Downloaded" if model_path.exists() else "⬜ Not downloaded"
        print(f"  {name}: {info['description']}")
        print(f"         {status}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_voices()
        else:
            download_voice(sys.argv[1])
    else:
        print("Piper Voice Setup")
        print("=" * 40)
        list_voices()
        print("\nUsage:")
        print("  python setup_piper_voice.py list      - List voices")
        print("  python setup_piper_voice.py amy       - Download 'amy' voice")
        print("  python setup_piper_voice.py lessac    - Download 'lessac' voice")
        print("\nDownloading default voice (amy)...")
        download_voice("amy")
