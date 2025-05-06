import wave
import numpy as np
import pyaudio
from scipy.signal import resample
import time

t = time.time()
alpha = 1.0
last_speed = 1.0

def get_speed_factor(target_speed=None):
    if not target_speed:
        return 1.0  # Default speed

    global last_speed

    smoothed_speed = alpha * target_speed + (1 - alpha) * last_speed
    last_speed = smoothed_speed
    return smoothed_speed


def get_volume_factor(volume=None):
    if not volume:
        return 0.1
    return volume

# Load WAV file
wav = wave.open("song.wav", 'rb')
framerate = wav.getframerate()
n_channels = wav.getnchannels()
sample_width = wav.getsampwidth()

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(sample_width),
                channels=n_channels,
                rate=framerate,
                output=True)

chunk_size = 1024

while True:
    # Read chunk
    data = wav.readframes(chunk_size)
    if not data:
        break

    # Convert bytes to numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Get dynamic speed factor from gesture (for now, constant)
    speed = get_speed_factor()
    volume = get_volume_factor()

    # Resample to change speed
    new_length = int(len(audio_data) / speed)
    audio_resampled = resample(audio_data, new_length).astype(np.int16)

    # Apply volume change
    audio_volume_adjusted = (audio_resampled * volume).clip(-32768, 32767).astype(np.int16)

    # Write to stream
    stream.write(audio_volume_adjusted.tobytes())

stream.stop_stream()
stream.close()
p.terminate()
wav.close()
