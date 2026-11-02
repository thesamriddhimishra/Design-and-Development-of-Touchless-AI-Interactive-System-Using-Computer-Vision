"""
generate_assets.py
-------------------
Creates simple, clean placeholder icons and piano tones so the project
runs immediately without needing any external image/sound downloads.

Run once after cloning:
    python generate_assets.py

Replace anything in assets/images or assets/sounds later with your own
files (same filenames) whenever you want nicer art/audio - the rest of
the code doesn't care where the files came from.
"""
import math
import os
import random
import struct
import wave

from PIL import Image, ImageDraw

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
IMG_DIR = os.path.join(ASSETS_DIR, "images")
SND_DIR = os.path.join(ASSETS_DIR, "sounds")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(SND_DIR, exist_ok=True)

W = H = 200


def new_canvas():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def save(img, name):
    img.save(os.path.join(IMG_DIR, name))
    print(f"  wrote images/{name}")


def make_pencil():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.polygon([(40, 160), (140, 60), (160, 80), (60, 180)], fill=(255, 205, 60, 255))
    d.polygon([(140, 60), (160, 40), (180, 60), (160, 80)], fill=(90, 90, 90, 255))
    d.polygon([(40, 160), (60, 180), (35, 190)], fill=(60, 40, 20, 255))
    save(img, "pencil.png")


def make_eraser():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([40, 70, 160, 140], radius=18, fill=(255, 120, 150, 255), outline=(120, 40, 60, 255), width=4)
    d.rectangle([40, 105, 100, 140], fill=(255, 235, 240, 255))
    save(img, "eraser.png")


def make_mole():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([30, 60, 170, 190], fill=(120, 80, 50, 255))
    d.ellipse([55, 30, 90, 65], fill=(120, 80, 50, 255))
    d.ellipse([110, 30, 145, 65], fill=(120, 80, 50, 255))
    d.ellipse([65, 100, 85, 120], fill=(20, 20, 20, 255))
    d.ellipse([115, 100, 135, 120], fill=(20, 20, 20, 255))
    d.ellipse([85, 130, 115, 150], fill=(230, 150, 160, 255))
    save(img, "mole.png")


def make_start_button():
    img = Image.new("RGBA", (210, 90), (40, 160, 70, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 209, 89], outline=(20, 90, 40, 255), width=4)
    d.polygon([(80, 25), (80, 65), (140, 45)], fill=(255, 255, 255, 255))
    save(img, "startgame.png")


def make_hit_effect():
    """An 8-point starburst 'POW' shape shown briefly when the mole is hit."""
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    outer_r, inner_r = 95, 38
    num_spikes = 8
    points = []
    for i in range(num_spikes * 2):
        angle = math.pi * i / num_spikes - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    d.polygon(points, fill=(40, 190, 255, 255), outline=(20, 110, 200, 255))
    save(img, "hit_effect.png")


def _write_wav(path, samples, sample_rate=44100):
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        frames = bytearray()
        for s in samples:
            s = max(-1.0, min(1.0, s))
            frames += struct.pack("<h", int(s * 32767))
        f.writeframes(frames)


def make_piano_tone(freq_hz, name, seconds=1.1, volume=0.9, sample_rate=44100, damping=0.9985):
    """Karplus-Strong plucked/struck-string synthesis.

    The earlier version summed a handful of sine harmonics with a single
    global decay envelope - every harmonic faded at the same rate, which
    is exactly what an organ (sustained, pure additive tones) sounds
    like. A piano is a struck *string*: higher harmonics die out much
    faster than the fundamental, which is what actually reads as
    'percussive' rather than 'sustained'. Karplus-Strong produces that
    naturally: seed a short noise burst, then repeatedly feed it through
    a tiny averaging filter that both delays and low-pass-filters it.
    The delay length sets the pitch, and the built-in low-pass is what
    makes the higher harmonics decay faster than the fundamental.
    """
    n_samples = int(seconds * sample_rate)
    buf_len = max(2, int(round(sample_rate / freq_hz)))
    rng = random.Random(int(freq_hz * 1000))
    # A noise burst plus a touch of DC-centered impulse gives a brighter,
    # more "struck" onset than pure noise alone.
    buf = [rng.uniform(-1.0, 1.0) for _ in range(buf_len)]

    samples = []
    prev = buf[-1]
    idx = 0
    for i in range(n_samples):
        current = buf[idx]
        # Randomly flipping the sign occasionally (classic "Extended KS")
        # adds a little inharmonicity/brightness rather than a too-clean tone.
        sign = -1.0 if rng.random() < 0.001 else 1.0
        averaged = damping * 0.5 * (current + prev) * sign
        buf[idx] = averaged
        prev = current
        samples.append(averaged)
        idx = (idx + 1) % buf_len

    # Fast attack ramp (avoids a click) and a gentle tail fade (avoids a
    # hard cutoff at the end of the sample).
    attack_samples = int(0.003 * sample_rate)
    release_samples = int(0.05 * sample_rate)
    for i in range(min(attack_samples, len(samples))):
        samples[i] *= i / attack_samples
    for i in range(min(release_samples, len(samples))):
        samples[-(i + 1)] *= i / release_samples

    peak = max(1e-6, max(abs(s) for s in samples))
    samples = [volume * s / peak for s in samples]
    _write_wav(os.path.join(SND_DIR, name), samples, sample_rate)
    print(f"  wrote sounds/{name} ({freq_hz:.2f} Hz)")


def make_hit_sound(name="hit.wav", seconds=0.22, sample_rate=44100):
    """A short percussive 'thwack': a fast noise burst plus a quick
    downward-pitched pop, so hitting the mole feels responsive."""
    rng = random.Random(7)
    n_samples = int(seconds * sample_rate)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        noise = rng.uniform(-1, 1) * math.exp(-30 * t)
        chirp_freq = 90 + 500 * math.exp(-18 * t)
        pop = math.sin(2 * math.pi * chirp_freq * t) * math.exp(-14 * t)
        samples.append(0.5 * noise + 0.6 * pop)
    _write_wav(os.path.join(SND_DIR, name), samples, sample_rate)
    print(f"  wrote sounds/{name}")


def main():
    print("Generating placeholder images...")
    make_pencil()
    make_eraser()
    make_mole()
    make_start_button()
    make_hit_effect()

    print("Generating placeholder piano tones (two octaves, C4-B5)...")
    notes = {
        "C4.wav": 261.63, "D4.wav": 293.66, "E4.wav": 329.63, "F4.wav": 349.23,
        "G4.wav": 392.00, "A4.wav": 440.00, "B4.wav": 493.88,
        "C5.wav": 523.25, "D5.wav": 587.33, "E5.wav": 659.25, "F5.wav": 698.46,
        "G5.wav": 783.99, "A5.wav": 880.00, "B5.wav": 987.77,
    }
    for name, freq in notes.items():
        make_piano_tone(freq, name)

    print("Generating mole-hit sound effect...")
    make_hit_sound()

    print("\nDone. Swap any file in assets/images or assets/sounds for your own later.")


if __name__ == "__main__":
    main()
