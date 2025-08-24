import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import sys
import time
import math

# -------------------------------
# Config
# -------------------------------
class Config:
    frame_size = 1024
    hop_size = 512
    threshold_k = 1.5
    refractory_ms = 200
    sample_rate = 44100

# -------------------------------
# Beat Detector
# -------------------------------
class BeatDetector:
    def __init__(self, config: Config):
        self.cfg = config
        self.energy_history = []
        self.max_history = 43
        self.last_beat_time = 0

    def detect(self, frame, t_ms):
        energy = np.sum(frame ** 2)
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)

        if len(self.energy_history) < self.max_history:
            return False

        avg_energy = np.mean(self.energy_history)
        c = -0.0025714 * len(self.energy_history) + 1.5142857
        threshold = c * avg_energy

        if energy > threshold * self.cfg.threshold_k:
            if t_ms - self.last_beat_time > self.cfg.refractory_ms:
                self.last_beat_time = t_ms
                return True
        return False

# -------------------------------
# Neon Visualizer
# -------------------------------
def run_visualizer(audio_file):
    data, samplerate = sf.read(audio_file)
    if len(data.shape) > 1:  # stereo → mono
        data = np.mean(data, axis=1)
    Config.sample_rate = samplerate

    detector = BeatDetector(Config)

    pygame.init()
    width, height = 900, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("🎶 Neon Beat Visualizer")
    clock = pygame.time.Clock()

    # Neon color palette
    neon_colors = [
        (57, 255, 20),   # neon green
        (0, 255, 255),   # cyan
        (255, 20, 147),  # pink
        (255, 140, 0),   # orange
        (138, 43, 226),  # purple
        (0, 191, 255),   # sky blue
        (255, 255, 0)    # yellow
    ]

    # Start audio
    sd.play(data, samplerate)
    start_time = time.time()

    running = True
    beat_pulse = 0.0
    frame_idx = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time() - start_time
        t_ms = int(now * 1000)
        start = frame_idx * Config.hop_size
        end = start + Config.frame_size
        if end >= len(data):
            break
        frame = data[start:end]
        frame_idx += 1

        # Beat detection
        if detector.detect(frame, t_ms):
            beat_pulse = 1.0

        beat_pulse *= 0.9  # smooth decay

        # FFT spectrum
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        spectrum = spectrum[:100]

        # ---- Background ----
        bg_r = int(50 + 100 * abs(math.sin(time.time() * 0.5)))
        bg_g = int(50 + 100 * abs(math.sin(time.time() * 0.7)))
        bg_b = int(50 + 100 * abs(math.sin(time.time() * 0.9)))
        screen.fill((bg_r, bg_g, bg_b))

        # ---- Spectrum Bars ----
        bar_w = width // len(spectrum)
        for idx, val in enumerate(spectrum):
            h = int((val / (np.max(spectrum) + 1e-6)) * (height // 2))
            r = pygame.Rect(idx * bar_w, height - h, bar_w, h)
            color = neon_colors[idx % len(neon_colors)]
            pygame.draw.rect(screen, (10, 10, 10), r)  # dark base
            pygame.draw.rect(screen, color, r.inflate(-2, 0))  # neon bar

        # ---- Neon Pulse Circle ----
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

        pygame.display.flip()
        clock.tick(60)

    sd.stop()
    pygame.quit()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python beat_visualizer.py yourfile.wav")
        sys.exit(1)
    run_visualizer(sys.argv[1])
