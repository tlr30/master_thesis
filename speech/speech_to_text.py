"""
Detect Navigation commands

This script captures audio input from the microphone, transcribes it using OpenAI's Whisper model, 
and passes recognized voice commands to an external Python script for execution. It continuously 
listens in 5-second windows and reacts to spoken commands.

Requirements:
- whisper
- pyaudio
- wave
- numpy
- subprocess

Ensure the directory 'speech/audio_files/' exists before running.

Author: Tim Riekeles
Date: 2025-08-05
"""
import whisper
import pyaudio
import wave
import numpy as np
import subprocess

# Load Whisper model (Choose from: tiny, base, small, medium, large)
model = whisper.load_model("base")

# Configure microphone input
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz audio
CHUNK = 1024
RECORD_SECONDS = 5  # Adjust based on expected command length

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening for commands... (Press Ctrl+C to stop)")

try:
    while True:
        print("New command window starts")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Save recorded audio as a temporary file
        wf = wave.open("speech/audio_files/command.wav", "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Transcribe audio using Whisper
        result = model.transcribe("speech/audio_files/command.wav")
        command = result["text"].strip().lower()

        if command:
            print(f"Recognized Command: {command}")
            subprocess.run(["python", "speech/command_execution.py", command])

except KeyboardInterrupt:
    print("\nStopping voice recognition...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
