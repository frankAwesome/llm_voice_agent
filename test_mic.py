"""Test microphone recording."""
import pyaudio
import wave

p = pyaudio.PyAudio()

print("=== Available Audio Devices ===\n")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    print(f"[{i}] {dev['name']}")
    print(f"    Input Channels: {dev['maxInputChannels']}")
    print(f"    Output Channels: {dev['maxOutputChannels']}")
    print(f"    Default Sample Rate: {int(dev['defaultSampleRate'])}")
    print()

print("\n=== Testing Microphone ===")
print("Enter device index to test (or press Enter for default): ", end="")
try:
    idx = input().strip()
    device_index = int(idx) if idx else None
except:
    device_index = None

# Get device info
if device_index is not None:
    dev_info = p.get_device_info_by_index(device_index)
else:
    dev_info = p.get_default_input_device_info()
    device_index = dev_info['index']

print(f"\nUsing device: [{device_index}] {dev_info['name']}")
sample_rate = int(dev_info['defaultSampleRate'])
print(f"Sample rate: {sample_rate}")

print("\n🎤 Recording 3 seconds... Speak now!")

try:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=1024
    )
    
    frames = []
    for _ in range(int(sample_rate / 1024 * 3)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    print("✅ Recording successful!")
    
    # Save to file
    output_file = "test_recording.wav"
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"✅ Saved to {output_file}")
    print("   Play it to verify: start test_recording.wav")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPossible fixes:")
    print("1. Check Windows Settings > Privacy > Microphone permissions")
    print("2. Make sure your microphone is not muted")
    print("3. Try a different device index")
    print("4. Close other apps using the microphone")

p.terminate()
