import threading
from src.audio_player import play_audio
from src.gesture import hand_tracking


t_audio = threading.Thread(target=play_audio, daemon=True)
t_audio.start()

hand_tracking()