import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import sys
import time
import math
import os
import random
import tkinter as tk
from tkinter import filedialog


# Configuration class to hold vital parameters
class Config:
    frame_size = 512        # Number of audio samples per frame for FFT analysis
    hop_size = 128          # How many samples to jump for the next frame (overlap depends on this)
    threshold_k = 1.1      # Sensitivity for beat detection; higher means fewer beats detected
    refractory_ms = 80     # Minimum interval (in ms) before detecting a new beat
    sample_rate = 44100     # Updated from audio file sample rate
    bass_cutoff = 800       # Only frequencies under 150 Hz (bass) are used for beat detection


# BeatDetector finds beats by analyzing bass energy in audio frames
class BeatDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self.energy_history = []
        self.max_history = 20  # smaller history for responsiveness
        self.last_beat_time = 0
        self.prev_energy = None

    def detect(self, frame, t_ms):
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs = np.fft.rfftfreq(len(frame), 1 / self.cfg.sample_rate)

        # Frequency bands: bass, mid, treble
        bass = spectrum[freqs <= 150]
        mid = spectrum[(freqs > 150) & (freqs <= 1000)]
        treble = spectrum[freqs > 1000]

        energy_bass = np.sum(bass**2)
        energy_mid = np.sum(mid**2)
        energy_treble = np.sum(treble**2)

        total_energy = energy_bass + energy_mid + energy_treble

        # Energy onset = difference between current and previous total energy frame
        if self.prev_energy is None:
            self.prev_energy = total_energy
            return False
        onset = total_energy - self.prev_energy
        self.prev_energy = total_energy

        # Keep track of recent onsets for adaptive threshold
        self.energy_history.append(onset)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
        if len(self.energy_history) < self.max_history:
            return False

        avg_onset = np.mean(self.energy_history)
        std_onset = np.std(self.energy_history)
        threshold = avg_onset + std_onset  # threshold depends on average + spread

        # Detect beat if onset is above threshold and sufficient time passed
        if onset > threshold * self.cfg.threshold_k:
            if t_ms - self.last_beat_time > self.cfg.refractory_ms:
                self.last_beat_time = t_ms
                return True

        return False

# Main function to run the audio visualizer
def run_visualizer(audio_file):
    data, samplerate = sf.read(audio_file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Convert stereo to mono for analysis

    Config.sample_rate = samplerate
    detector = BeatDetector(Config())

    pygame.init()
    width, height = 1000, 700
    center_x, center_y = width // 2, height // 2
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    neon_colors = [
        (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (255, 120, 0), (0, 255, 50), (255, 0, 100), (50, 0, 255), (0, 150, 255)
    ]

    NUM_BARS = 100
    MAX_SPECTRUM = 150
    beat_pulse = 0.0
    rotation_angle = 0.0
    audio_name = os.path.basename(audio_file)

    # Start audio playback
    sd.play(data, samplerate)

    frame_start = 0
    start_time = time.time()
    particles = []

    # Fonts for displaying time and audio name on screen
    font_center = pygame.font.SysFont("Arial", 32)
    font_name = pygame.font.SysFont("Arial", 24)

    while True:
        # Event handling to allow closing window properly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sd.stop()
                pygame.quit()
                sys.exit(0)

        # Extract current frame slice from audio data
        frame_end = frame_start + Config.frame_size
        if frame_end >= len(data):
            break
        frame = data[frame_start:frame_end]
        frame_start += Config.hop_size

        now = time.time() - start_time
        t_ms = int(now * 1000)

        # Detect beat for current frame
        beat_occurred = detector.detect(frame, t_ms)
        if beat_occurred:
            beat_pulse = 1.0
            # Generate colorful particles on beat
            for _ in range(30):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(50, 180)
                speed = random.uniform(0.005, 0.015)
                color = random.choice(neon_colors)
                particles.append({
                    "angle": angle,
                    "radius": radius,
                    "speed": speed,
                    "color": color,
                    "life": random.randint(40, 70)
                })
        else:
            beat_pulse *= 0.85  # Smooth decay of pulse

        # Calculate spectrum for visualization bars
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))[:MAX_SPECTRUM]
        bars = spectrum[np.linspace(0, len(spectrum) - 1, NUM_BARS).astype(int)]
        bars /= np.max(bars) + 1e-9  # Normalize to max value

        screen.fill((5, 5, 10))  # Dark background

        # Draw pulsing central circle with elapsed time text
        circle_radius = int(100 + 60 * beat_pulse)
        pygame.draw.circle(screen, (0, 255, 0), (center_x, center_y), circle_radius, 6)
        minutes = int(now // 60)
        seconds = int(now % 60)
        time_text = font_center.render(f"{minutes:02d}:{seconds:02d}", True, (0, 255, 0))
        screen.blit(time_text, time_text.get_rect(center=(center_x, center_y)))

        # Draw circular spectrum bars rotating around center
        rotation_angle += (0.03 + 0.15 * beat_pulse) / 2
        radius_base = 200
        for i, val in enumerate(bars):
            angle_bar = i * (360 / NUM_BARS) + math.degrees(rotation_angle)
            rad = math.radians(angle_bar)
            bar_len = int(val * 150 * beat_pulse)
            x_start = center_x + int(radius_base * math.cos(rad))
            y_start = center_y + int(radius_base * math.sin(rad))
            x_end = center_x + int((radius_base + bar_len) * math.cos(rad))
            y_end = center_y + int((radius_base + bar_len) * math.sin(rad))
            pygame.draw.line(screen, neon_colors[i % len(neon_colors)], (x_start, y_start), (x_end, y_end), 4)

        # Update and draw beat-triggered particles
        for p in particles[:]:
            p["angle"] += p["speed"]
            p["radius"] += 0.1
            x = int(center_x + p["radius"] * math.cos(p["angle"]))
            y = int(center_y + p["radius"] * math.sin(p["angle"]))
            pygame.draw.circle(screen, p["color"], (x, y), 3)
            p["life"] -= 1
            if p["life"] <= 0:
                particles.remove(p)

        # Show audio file name in the corner
        name_text = font_name.render(f"Playing: {audio_name}", True, (0, 255, 0))
        screen.blit(name_text, (20, 20))

        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS

    # Stop audio and close window properly
    sd.stop()
    pygame.quit()


if __name__ == "__main__":
    audio_file = None
    if len(sys.argv) >= 2:
        audio_file = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        audio_file = filedialog.askopenfilename(title="Select audio file", filetypes=[("WAV files", "*.wav")])
        if not audio_file:
            print("No file selected.")
            input("Press Any Key to EXIT...")
            sys.exit(1)

    run_visualizer(audio_file)
