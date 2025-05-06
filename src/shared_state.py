import threading

shared_state = {
    'volume': 0.5,
    'speed': 1.0,
    'max_plane_height': 0,
}
lock = threading.Lock()
