import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import sys
import time
import math
import os

class Config:
    frame_size = 1024
    hop_size = 512
    threshold_k = 1.5
    refractory_ms = 200
    sample_rate = 44100

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
        threshold = (-0.0025714 * len(self.energy_history) + 1.5142857) * avg_energy
        if energy > threshold * self.cfg.threshold_k:
            if t_ms - self.last_beat_time > self.cfg.refractory_ms:
                self.last_beat_time = t_ms
                return True
        return False

def run_visualizer(audio_file):
    data, samplerate = sf.read(audio_file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    Config.sample_rate = samplerate
    detector = BeatDetector(Config)

    pygame.init()
    width, height = 1000, 700
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    neon_colors = [(0,255,255),(0,255,100),(255,0,255),(255,255,0),(255,120,0)]
    
    NUM_BARS = 50
    MAX_SPECTRUM = 120
    caps = [0.0]*NUM_BARS
    cap_fall_speed = 0.1
    sd.play(data, samplerate)
    start_time = time.time()
    beat_pulse = 0.0
    frame_idx = 0
    rotation_angle = 0.0  # rotation of circular bars

    audio_name = os.path.basename(audio_file)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time() - start_time
        t_ms = int(now*1000)
        start = frame_idx*Config.hop_size
        end = start+Config.frame_size
        if end >= len(data):
            break
        frame = data[start:end]
        frame_idx += 1

        # Detect beat
        beat_occurred = detector.detect(frame, t_ms)
        if beat_occurred:
            beat_pulse = 1.0
        else:
            beat_pulse *= 0.85  # smooth decay

        spectrum = np.abs(np.fft.rfft(frame*np.hanning(len(frame))))[:MAX_SPECTRUM]
        bars = spectrum[np.linspace(0,len(spectrum)-1,NUM_BARS).astype(int)]
        bars /= np.max(bars)+1e-9

        screen.fill((5,5,10))  # dark background
        center_x, center_y = width//2, height//2

        # --- pulsing circle reacts to beat intensity ---
        pulse_radius = int(80 + 120 * beat_pulse)
        pulse_surf = pygame.Surface((pulse_radius*2, pulse_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(pulse_surf, (0,255,255,100), (pulse_radius,pulse_radius), pulse_radius, 6)
        screen.blit(pulse_surf, (center_x-pulse_radius, center_y-pulse_radius))

        # --- circular bars: move faster with strong beat ---
        radius = 180
        rotation_angle += 0.5 + 3*beat_pulse  # faster rotation when beat is strong
        for i, val in enumerate(bars):
            if beat_pulse > 0.05:
                bar_len = int(val*150*beat_pulse)
            else:
                bar_len = 0
            angle_bar = i * (360/NUM_BARS) + rotation_angle
            rad = math.radians(angle_bar)
            x_start = center_x + int(radius*math.cos(rad))
            y_start = center_y + int(radius*math.sin(rad))
            x_end = center_x + int((radius+bar_len)*math.cos(rad))
            y_end = center_y + int((radius+bar_len)*math.sin(rad))
            color = neon_colors[i % len(neon_colors)]
            pygame.draw.line(screen, color, (x_start,y_start), (x_end,y_end), 6)

        # --- display timestamp ---
        elapsed_sec = int(now)
        minutes = elapsed_sec // 60
        seconds = elapsed_sec % 60
        timestamp = font.render(f"{minutes:02d}:{seconds:02d}", True, (255,255,255))
        screen.blit(timestamp, (width-100, 20))

        # --- display audio name ---
        name_text = font.render(f"Playing: {audio_name}", True, (0,255,255))
        screen.blit(name_text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

    sd.stop()
    pygame.quit()

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python beat_visualizer.py yourfile.wav")
        sys.exit(1)
    run_visualizer(sys.argv[1])
