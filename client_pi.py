#!/usr/bin/env python3
"""
CyBot Client - Runs on Raspberry Pi
Captures audio, sends to server, plays response audio
"""

import pyaudio
import wave
import requests
import numpy as np
import io
import sys
import threading
import argparse

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

pa = pyaudio.PyAudio()


def find_audio_devices():
    """Find default input and output devices."""
    input_dev = None
    output_dev = None
    
    print("🔍 Finding audio devices...")
    
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        
        if dev['maxInputChannels'] > 0 and input_dev is None:
            input_dev = i
            print(f"  🎤 Input: [{i}] {dev['name']}")
        
        if dev['maxOutputChannels'] > 0 and output_dev is None:
            output_dev = i
            print(f"  🔊 Output: [{i}] {dev['name']}")
    
    return input_dev, output_dev


def calibrate_microphone(input_device):
    """Calibrate noise threshold."""
    global SILENCE_THRESHOLD
    
    print("\n🎤 Calibrating... (stay quiet for 2 seconds)")
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CHUNK_SIZE
    )
    
    noise_samples = []
    chunks_needed = int(SAMPLE_RATE / CHUNK_SIZE * 2)
    
    for _ in range(chunks_needed):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        noise_samples.append(np.abs(audio).mean())
    
    stream.stop_stream()
    stream.close()
    
    noise_floor = np.mean(noise_samples)
    SILENCE_THRESHOLD = max(noise_floor * 2.5, 200)
    print(f"   ✅ Threshold: {SILENCE_THRESHOLD:.0f}")


def record_audio_vad(input_device):
    """Record with voice activity detection."""
    print("\n🎤 Listening... (speak now)")
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CHUNK_SIZE
    )
    
    audio_chunks = []
    silence_counter = 0
    has_speech = False
    chunks_per_second = SAMPLE_RATE / CHUNK_SIZE
    
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio).mean()
        
        # Visual feedback
        bar = "█" * int(min(volume / 100, 30))
        status = "🔴" if volume > SILENCE_THRESHOLD else "⚪"
        print(f"\r   {status} [{bar:<30}] {volume:.0f}  ", end="", flush=True)
        
        if volume > SILENCE_THRESHOLD:
            has_speech = True
            silence_counter = 0
            audio_chunks.append(data)
        elif has_speech:
            silence_counter += 1 / chunks_per_second
            audio_chunks.append(data)
            if silence_counter >= SILENCE_DURATION:
                print()
                break
        
        # Timeout
        if len(audio_chunks) / chunks_per_second >= 30:
            print("\n⏱️  Max duration")
            break
        
        if not has_speech and len(audio_chunks) > 0 and len(audio_chunks) / chunks_per_second >= 10:
            print("\n💤 No speech detected")
            stream.stop_stream()
            stream.close()
            return None
    
    stream.stop_stream()
    stream.close()
    
    if not audio_chunks:
        return None
    
    return b''.join(audio_chunks)


def record_audio_ptt(input_device):
    """Push-to-talk recording."""
    print("\n🎤 Press ENTER to start recording...")
    
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CHUNK_SIZE
    )
    
    input()
    print("🔴 Recording... Press ENTER to stop")
    
    audio_chunks = []
    stop = threading.Event()
    
    def wait_stop():
        input()
        stop.set()
    
    t = threading.Thread(target=wait_stop)
    t.start()
    
    while not stop.is_set():
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_chunks.append(data)
    
    stream.stop_stream()
    stream.close()
    t.join()
    
    return b''.join(audio_chunks) if audio_chunks else None


def send_to_server(server_url, audio_data):
    """Send audio to server, receive response."""
    print("📤 Sending to server...")
    
    # Create WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_data)
    wav_buffer.seek(0)
    
    try:
        response = requests.post(
            f"{server_url}/process",
            files={'audio': ('audio.wav', wav_buffer, 'audio/wav')},
            data={'sample_rate': SAMPLE_RATE},
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Server error: {response.status_code}")
            return None, None, False
        
        # Check if JSON response (no audio)
        if 'application/json' in response.headers.get('Content-Type', ''):
            data = response.json()
            if data.get('wake_word_missing'):
                print(f"💤 (No wake word detected)")
                return None, None, False
            if data.get('error'):
                print(f"❌ {data['error']}")
                return None, None, False
            return None, data.get('response'), data.get('quit', False)
        
        # Audio response
        user_text = response.headers.get('X-User-Text', '')
        response_text = response.headers.get('X-Response-Text', '')
        quit_flag = response.headers.get('X-Quit', 'False') == 'True'
        
        print(f"📝 You: {user_text}")
        print(f"🎩 Response: {response_text}")
        
        return response.content, response_text, quit_flag
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        return None, None, False
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, False


def play_audio(audio_data, output_device):
    """Play WAV audio data."""
    if not audio_data:
        return
    
    wav_buffer = io.BytesIO(audio_data)
    wf = wave.open(wav_buffer, 'rb')
    
    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
        output_device_index=output_device
    )
    
    chunk_size = 1024
    data = wf.readframes(chunk_size)
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)
    
    stream.stop_stream()
    stream.close()
    wf.close()


def main():
    parser = argparse.ArgumentParser(description='CyBot Client for Raspberry Pi')
    parser.add_argument('--server', '-s', required=True, 
                        help='Server URL (e.g., http://192.168.1.100:5000)')
    parser.add_argument('--mode', '-m', choices=['vad', 'ptt'], default='vad',
                        help='Recording mode: vad (voice activity) or ptt (push-to-talk)')
    parser.add_argument('--input', '-i', type=int, help='Input device index')
    parser.add_argument('--output', '-o', type=int, help='Output device index')
    args = parser.parse_args()
    
    server_url = args.server.rstrip('/')
    
    print("=" * 50)
    print("🎩 CyBot Client - Raspberry Pi")
    print("=" * 50)
    print(f"Server: {server_url}")
    
    # Check server connection
    try:
        r = requests.get(f"{server_url}/health", timeout=5)
        info = r.json()
        print(f"✅ Connected to {info['butler_name']}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        sys.exit(1)
    
    # Find audio devices
    if args.input is not None and args.output is not None:
        input_dev, output_dev = args.input, args.output
    else:
        input_dev, output_dev = find_audio_devices()
    
    if input_dev is None or output_dev is None:
        print("❌ Audio devices not found!")
        sys.exit(1)
    
    # Calibrate for VAD mode
    if args.mode == 'vad':
        calibrate_microphone(input_dev)
    
    # Get greeting
    print("\n🔄 Getting greeting...")
    try:
        r = requests.get(f"{server_url}/greeting", timeout=30)
        if r.status_code == 200:
            play_audio(r.content, output_dev)
    except Exception as e:
        print(f"⚠️  Could not get greeting: {e}")
    
    print("\n" + "=" * 50)
    print("Ready! Speak to your butler.")
    print("=" * 50)
    
    # Main loop
    while True:
        try:
            # Record
            if args.mode == 'vad':
                audio_data = record_audio_vad(input_dev)
            else:
                audio_data = record_audio_ptt(input_dev)
            
            if audio_data is None:
                continue
            
            print(f"📝 Recorded {len(audio_data) / (SAMPLE_RATE * 2):.1f}s")
            
            # Send to server
            response_audio, response_text, quit_flag = send_to_server(server_url, audio_data)
            
            # Play response
            if response_audio:
                play_audio(response_audio, output_dev)
            elif response_text:
                print(f"🎩 {response_text}")
            
            if quit_flag:
                print("👋 Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    pa.terminate()


if __name__ == '__main__':
    main()
