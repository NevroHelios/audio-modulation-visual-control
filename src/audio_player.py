from scipy.signal import resample
import pyaudio
import wave
import numpy as np
from src.shared_state import shared_state, lock

from src.config import CFG


def play_audio(path="audio/song.wav", chunk_size=1024):
    wav = wave.open(path, 'rb')
    framerate = wav.getframerate()
    n_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()

    spectrum = np.zeros(CFG.BINS)

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=n_channels,
                    rate=framerate,
                    output=True)

    while True:
        data = wav.readframes(chunk_size)
        if not data:
            break
        audio_data = np.frombuffer(data, dtype=np.int16)
        if n_channels > 1:
            audio_data = audio_data.flatten()

        with lock:
            volume = shared_state['volume']
            speed = shared_state['speed']
            max_height = shared_state['max_plane_height']

        new_length = int(len(audio_data) / speed)
        audio_resampled = resample(audio_data, new_length).astype(np.int16)
        audio_volume_adjusted = (audio_resampled * volume).clip(-32768, 32767).astype(np.int16)
        stream.write(audio_volume_adjusted.tobytes())

        fft_result = np.abs(np.fft.rfft(audio_data))
        fft_result = fft_result[:CFG.BINS]
        spectrum = np.interp(fft_result, (0, np.max(fft_result)), (0, max_height // 2))
        print(spectrum)
        with lock:
            shared_state['spectrum'] = spectrum

    stream.stop_stream()
    stream.close()
    p.terminate()
    wav.close()


