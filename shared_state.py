import threading

shared_state = {
    'volume': 1.0,
    'speed': 1.0
}
lock = threading.Lock()
