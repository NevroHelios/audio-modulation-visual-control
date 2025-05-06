# Vision Project

This project uses hand gesture tracking and audio playback to create an interactive experience. It leverages libraries like MediaPipe, OpenCV, and PyAudio for gesture recognition and audio manipulation.

## Setup Instructions

1. Ensure you have Python 3.12 installed.
2. Install the required dependencies:
   ```bash
   uv sync
   ```
3. Create a directory named `audio` in the project root:
   ```bash
   mkdir audio
   ```
4. Place your `song.wav` file inside the `audio` directory.

## How to Run

1. Start the application:
   ```bash
   uv run main.py
   ```
2. Use hand gestures to interact with the application:
   - Adjust volume using hand plane height.
   - Visualize audio spectrum based on gestures.

## Accomplishments

- [x] Real-time hand gesture tracking using MediaPipe.
- [x] Dynamic audio playback with adjustable volume and speed.
- [x] Visual representation of audio spectrum synchronized with gestures.

## Future Goals

- [ ] Automate the creation of the `audio` directory and handle missing `song.wav` gracefully.
- [ ] Add support for multiple audio formats.
- [ ] Enhance gesture recognition for more complex interactions.
- [ ] Improve UI/UX for better visualization and feedback.
- [ ] Integrate with cloud services for gesture data analytics.