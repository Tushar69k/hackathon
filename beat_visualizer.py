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

# Config
# -------------------------------
# Configuration (essential parameters)
# -------------------------------
class Config:
    frame_size = 512        # audio frame size for FFT
    hop_size = 128          # step between frames
    threshold_k = 1.3       # beat detection sensitivity
    refractory_ms = 100     # minimum time between beats (ms)
    sample_rate = 44100
    bass_cutoff = 150       # only bass frequencies considered

# Beat Detector
# -------------------------------
# Beat detection (spectral energy in bass)
# -------------------------------
class BeatDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self.energy_history = []
        self.max_history = 43
        self.last_beat_time = 0

    def detect(self, frame, t_ms):
        # FFT and bass energy
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs = np.fft.rfftfreq(len(frame), 1/self.cfg.sample_rate)
        bass = spectrum[freqs <= self.cfg.bass_cutoff]
        energy = np.sum(bass**2)

        # maintain energy history
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
        if len(self.energy_history) < self.max_history:
            return False

        # adaptive threshold
        avg_energy = np.mean(self.energy_history)
        threshold = (-0.0025714 * len(self.energy_history) + 1.5142857) * avg_energy

        # beat occurs if energy > threshold & outside refractory period
        if energy > threshold * self.cfg.threshold_k:
            if t_ms - self.last_beat_time > self.cfg.refractory_ms:
                self.last_beat_time = t_ms
                return True
        return False

# Neon Visualizer
# -------------------------------
# Visualizer (main graphics + audio)
# -------------------------------
def run_visualizer(audio_file):
    # Load audio
    data, samplerate = sf.read(audio_file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # stereo → mono
    Config.sample_rate = samplerate
    detector = BeatDetector(Config())

    # Pygame setup
    pygame.init()
    width, height = 1000, 700
    center_x, center_y = width//2, height//2
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    neon_colors = [(0,255,0), (0,255,255), (255,0,255), (255,255,0)]

    NUM_BARS = 200
    MAX_SPECTRUM = 150
    beat_pulse = 0.0
    rotation_angle = 0.0

    # Start audio
    sd.play(data, samplerate)
    frame_start = 0
    start_time = time.time()

    font_center = pygame.font.SysFont("Arial", 32)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sd.stop()
                pygame.quit()
                sys.exit(0)

        frame_end = frame_start + Config.frame_size
        if frame_end >= len(data):
            break
        frame = data[frame_start:frame_end]
        frame_start += Config.hop_size

        now = time.time() - start_time
        t_ms = int(now*1000)

        # Beat detection
        beat_occurred = detector.detect(frame, t_ms)
        beat_pulse = 1.0 if beat_occurred else beat_pulse * 0.85

        # FFT spectrum
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        spectrum = spectrum[:100]

        # Background
        bg_r = int(50 + 100 * abs(math.sin(time.time() * 0.5)))
        bg_g = int(50 + 100 * abs(math.sin(time.time() * 0.7)))
        bg_b = int(50 + 100 * abs(math.sin(time.time() * 0.9)))
        screen.fill((bg_r, bg_g, bg_b))
        # FFT for spectrum bars
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))[:MAX_SPECTRUM]
        bars = spectrum[np.linspace(0, len(spectrum)-1, NUM_BARS).astype(int)]
        bars /= np.max(bars)+1e-9

        screen.fill((5,5,10))  # background

        # Pulsing central circle with time
        circle_radius = int(100 + 60*beat_pulse)
        pygame.draw.circle(screen, (0,255,0), (center_x, center_y), circle_radius, 6)
        minutes, seconds = divmod(int(now), 60)
        time_text = font_center.render(f"{minutes:02d}:{seconds:02d}", True, (0,255,0))
        screen.blit(time_text, time_text.get_rect(center=(center_x, center_y)))

        bar_w = width // len(spectrum)
        for idx, val in enumerate(spectrum):
            h = int((val / (np.max(spectrum) + 1e-6)) * (height // 2))
            r = pygame.Rect(idx * bar_w, height - h, bar_w, h)
            color = neon_colors[idx % len(neon_colors)] # will use later
            pygame.draw.rect(screen, (10, 10, 10), r)  # dark base
            pygame.draw.rect(screen, color, r.inflate(-2, 0)) # neon bar

        cx, cy = width // 2, height // 2
        radius = int(80 + 60 * beat_pulse)
        color = neon_colors[int(time.time() * 2) % len(neon_colors)]
        for glow in range(6, 0, -1):
            pygame.draw.circle(
                screen,
                color,
                (cx, cy),
                radius + glow * 4,
                width=2
            )
        # Circular spectrum bars
        rotation_angle += (0.03 + 0.15*beat_pulse)/2
        radius_base = 200
        for i, val in enumerate(bars):
            angle_bar = i * (360/NUM_BARS) + math.degrees(rotation_angle)
            rad = math.radians(angle_bar)
            bar_len = int(val*150*beat_pulse)
            x_start = center_x + int(radius_base*math.cos(rad))
            y_start = center_y + int(radius_base*math.sin(rad))
            x_end = center_x + int((radius_base+bar_len)*math.cos(rad))
            y_end = center_y + int((radius_base+bar_len)*math.sin(rad))
            pygame.draw.line(screen, neon_colors[i % len(neon_colors)], (x_start,y_start), (x_end,y_end), 4)

        pygame.display.flip()
        clock.tick(60)

    sd.stop()
    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python beat_visualizer.py yourfile.wav")
        sys.exit(1)
    run_visualizer(sys.argv[1])
# -------------------------------
# Main: CLI or file dialog
# -------------------------------
if __name__=="__main__":
    audio_file = None

    if len(sys.argv) >= 2:  # command line
        audio_file = sys.argv[1]
    else:  # file dialog
        root = tk.Tk()
        root.withdraw()
        audio_file = filedialog.askopenfilename(title="Select audio file", filetypes=[("WAV files", "*.wav")])
        if not audio_file:
            print("No file selected. Exiting.")
            sys.exit(1)

    run_visualizer(audio_file)
