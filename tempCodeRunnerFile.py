#!/usr/bin/env python3
"""
Beat Detection & Rhythm Visualizer
----------------------------------
- Plays a WAV file.
- Detects beats in real-time using Spectral Flux + adaptive threshold.
- Visualizes spectrum bars and a pulsing circle on each beat.

Dependencies: numpy, sounddevice, soundfile, pygame
Install: pip install numpy sounddevice soundfile pygame

Run:
    python beat_visualizer.py path/to/your.wav
or:
    python beat_visualizer.py
    (auto-generates a small demo WAV named sample.wav)
"""

import sys
import time
import threading
import queue
import math
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame


# ---------------------------
# Config (tweak as you like)
# ---------------------------
class Config:
    # Analysis
    frame_size = 1024          # samples per analysis frame
    hop_size = 512             # step between frames (overlap = 50%)
    n_fft = 1024               # FFT size (power of 2, >= frame_size)
    min_freq_hz = 50           # ignore super low bins (<50Hz) in flux
    threshold_k = 1.5          # threshold = mean + k*std of past flux
    threshold_window = 43      # ~0.5s at hop=512, fs=44100 (adjust)
    refractory_ms = 120        # min gap between beats (avoid double trigger)
    # Visualization
    width = 960
    height = 540
    fps = 60


# ---------------------------
# Utility: create a demo WAV
# ---------------------------
def generate_demo_wav(path="sample.wav", seconds=10, sr=44100):
    """Generate a simple metronome + kick-like thump so you can test quickly."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = np.zeros_like(t)

    bpm = 100
    beat_period = 60.0 / bpm
    click_len = int(0.010 * sr)
    kick_len = int(0.060 * sr)

    # Metronome clicks (every beat)
    for b in range(int(seconds / beat_period)):
        start = int(b * beat_period * sr)
        end = min(start + click_len, len(audio))
        # short high click
        audio[start:end] += np.hanning(end - start) * 0.8

    # Low kick on every beat (adds spectral contrast)
    for b in range(int(seconds / beat_period)):
        start = int(b * beat_period * sr)
        end = min(start + kick_len, len(audio))
        env = np.hanning(end - start)
        sine = np.sin(2 * np.pi * 80 * np.linspace(0, (end - start) / sr, end - start, endpoint=False))
        audio[start:end] += 0.6 * env * sine

    # Normalize
    peak = np.max(np.abs(audio)) + 1e-9
    audio = (audio / peak * 0.9).astype(np.float32)
    sf.write(path, audio, sr)
    return path


# ---------------------------
# Beat Detector (Spectral Flux)
# ---------------------------
class BeatDetector:
    def __init__(self, samplerate, cfg: Config):
        self.fs = samplerate
        self.cfg = cfg
        self.window = np.hanning(cfg.frame_size).astype(np.float32)
        self.prev_mag = None
        self.flux_hist = deque(maxlen=cfg.threshold_window)
        self.last_flux = 0.0
        self.last_increasing = False
        self.last_beat_time = -1e9  # very negative to allow first beat
        # compute index to ignore low freqs
        self.k_min = int(cfg.min_freq_hz * cfg.n_fft / self.fs)

    def spectral_flux(self, frame):
        # Window, zero-pad if needed, FFT -> magnitude
        xw = frame * self.window
        mag = np.abs(np.fft.rfft(xw, n=self.cfg.n_fft))

        # Positive changes only vs previous mag
        if self.prev_mag is None:
            flux = 0.0
        else:
            diff = mag - self.prev_mag
            # Ignore super low frequencies that muddy onset detection
            pos = np.maximum(diff[self.k_min:], 0.0)
            flux = np.sum(pos)

        self.prev_mag = mag
        return float(flux), mag

    def adaptive_threshold(self):
        if len(self.flux_hist) < 4:
            return float('inf')  # delay until we have some history
        m = np.mean(self.flux_hist)
        s = np.std(self.flux_hist)
        return m + self.cfg.threshold_k * s

    def update_and_check_peak(self, flux_value, frame_time_sec):
        """Return True if this frame is a beat (local peak above threshold)."""
        # Update moving history (for threshold)
        self.flux_hist.append(flux_value)
        thr = self.adaptive_threshold()

        # Simple online peak-picking using slope change
        increasing = flux_value > self.last_flux
        beat = False

        # detect local peak: last was rising, now falling or flat, and above threshold
        if self.last_increasing and not increasing:
            # last point (self.last_flux) was the local max; use it for comparison
            local_peak_value = self.last_flux
            if local_peak_value > thr:
                # Refractory period (avoid double fires within short time)
                if (frame_time_sec - self.last_beat_time) * 1000.0 >= self.cfg.refractory_ms:
                    beat = True
                    self.last_beat_time = frame_time_sec

        self.last_increasing = increasing
        self.last_flux = flux_value
        return beat, thr


# ---------------------------
# Real-time Analyzer Thread
# ---------------------------
class AnalyzerThread(threading.Thread):
    def __init__(self, audio, fs, cfg: Config, beat_queue, shared_state, stop_event):
        """
        audio: mono float32 numpy array
        beat_queue: queue to send beat events (timestamp)
        shared_state: dict to share current spectrum magnitudes with UI
        stop_event: signal to stop
        """
        super().__init__(daemon=True)
        self.audio = audio
        self.fs = fs
        self.cfg = cfg
        self.beat_queue = beat_queue
        self.shared = shared_state
        self.stop_event = stop_event

    def run(self):
        bd = BeatDetector(self.fs, self.cfg)
        N = len(self.audio)
        i = 0  # frame start index
        start_time = time.perf_counter()

        # Timing: target wall-clock when we should process frame i
        def idx_to_time(idx):
            return idx / float(self.fs)

        while not self.stop_event.is_set() and i + self.cfg.frame_size <= N:
            frame = self.audio[i:i + self.cfg.frame_size]
            flux, mag = bd.spectral_flux(frame)
            frame_time_sec = idx_to_time(i)

            # Peak picking
            is_beat, thr = bd.update_and_check_peak(flux, frame_time_sec)
            if is_beat:
                # send event for the current frame time
                self.beat_queue.put(frame_time_sec)

            # Share current spectrum for visualization (downsample bars)
            # Convert mag (rfft) to ~64 bars equally spaced in frequency
            mag_db = 20 * np.log10(mag + 1e-8)
            bars = 64
            # interpolate magnitudes to fixed number of bars
            idxs = np.linspace(0, len(mag_db) - 1, bars).astype(np.int32)
            self.shared["bars"] = mag_db[idxs]
            self.shared["flux"] = flux
            self.shared["thr"] = thr
            self.shared["time"] = frame_time_sec

            # Pace to real-time: wait until wall-clock >= frame_time
            now = time.perf_counter()
            target = start_time + frame_time_sec
            delay = target - now
            if delay > 0:
                time.sleep(delay)

            # advance by hop
            i += self.cfg.hop_size

        # signal end (optional)
        self.shared["done"] = True


# ---------------------------
# Visualization (Pygame)
# ---------------------------
def run_visualizer(audio, fs, cfg: Config):
    # Convert to mono float32 if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Shared structures
    beat_queue = queue.Queue()
    shared = {"bars": np.zeros(64), "flux": 0.0, "thr": float('inf'), "time": 0.0, "done": False}
    stop_event = threading.Event()

    # Start playback
    sd.stop()
    sd.play(audio, fs)  # async

    # Start analyzer thread (real-time streaming paced to playback)
    analyzer = AnalyzerThread(audio, fs, cfg, beat_queue, shared, stop_event)
    analyzer.start()

    # Init Pygame UI
    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Beat Detection & Rhythm Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    # Beat pulse state
    pulse = 0.0   # grows to 1.0 on beat, then decays
    pulse_decay = 3.0  # per second

    running = True
    start_t = time.perf_counter()

    while running:
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Pull beat events that should have happened by now
        now = time.perf_counter() - start_t
        try:
            while True:
                beat_t = beat_queue.get_nowait()
                # trigger a pulse for any beat (we don't check times again here because analyzer is paced)
                pulse = 1.0
        except queue.Empty:
            pass

        # Background color modulated by pulse
        base = 20
        bg_val = min(255, int(base + 235 * pulse))
        screen.fill((bg_val, bg_val, bg_val))

        # Draw spectrum bars (centered horizontally)
        bars = shared.get("bars", np.zeros(64))
        bars = np.clip(bars, -80, 10)  # dB range
        bars = (bars - (-80)) / (10 - (-80))  # normalize 0..1
        bar_w = cfg.width // (len(bars) + 10)
        max_h = int(cfg.height * 0.5)
        x0 = (cfg.width - (bar_w + 4) * len(bars)) // 2
        y_base = int(cfg.height * 0.75)
        for idx, v in enumerate(bars):
            h = int(v * max_h)
            r = pygame.Rect(x0 + idx * (bar_w + 4), y_base - h, bar_w, h)
            pygame.draw.rect(screen, (30, 30, 30), r)          # bar background
            pygame.draw.rect(screen, (220, 220, 220), r.inflate(-4, 0))  # filled bar

        # Draw pulsing circle in the middle
        cx, cy = cfg.width // 2, int(cfg.height * 0.35)
        radius = int(40 + 120 * pulse)
        pygame.draw.circle(screen, (0, 0, 0), (cx, cy), radius, width=4)

        # Text HUD: flux and threshold
        flux = shared.get("flux", 0.0)
        thr = shared.get("thr", float('inf'))
        txt = f"Flux: {flux:7.3f} | Thr: {thr if thr != float('inf') else 0:7.3f}"
        surface = font.render(txt, True, (0, 0, 0))
        screen.blit(surface, (12, 12))

        # Update + decay pulse
        pygame.display.flip()
        dt = clock.tick(cfg.fps) / 1000.0
        pulse = max(0.0, pulse - pulse_decay * dt)

        # Stop when audio finished and analyzer marked done
        if shared.get("done", False) and (sd.get_stream() is None or not sd.get_stream().active):
            running = False

    # Cleanup
    stop_event.set()
    sd.stop()
    pygame.quit()


# ---------------------------
# Entry Point
# ---------------------------
def load_wav_mono(path):
    audio, fs = sf.read(path, dtype='float32', always_2d=True)
    # If multi-channel, average to mono for analysis and playback (or you could keep stereo for playback)
    audio_mono = audio.mean(axis=1)
    return audio_mono, fs

def main():
    cfg = Config()

    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        print("No WAV provided. Generating a demo file 'sample.wav'...")
        wav_path = generate_demo_wav("sample.wav")

    # Load
    try:
        audio, fs = load_wav_mono(wav_path)
    except Exception as e:
        print(f"Failed to load WAV: {e}")
        return

    print(f"Loaded: {wav_path}  fs={fs}Hz  duration={len(audio)/fs:.2f}s")
    run_visualizer(audio, fs, cfg)


if __name__ == "__main__":
    main()
