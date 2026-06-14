import os
import numpy as np
from scipy.io import wavfile

def generate_genuine_signal(duration=2.0, sr=16000):
    """
    Generates a simulated genuine human speech signal.
    Features: Smooth pitch variation (natural intonation), natural formant structure,
    and realistic speech noise.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Pitch variation (smooth frequency modulation for natural voice)
    # Fundamental frequency f0 around 120-160Hz for male/female blend, smoothly changing
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t) + 5 * np.cos(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    
    # Fundamental tone
    signal = np.sin(phase)
    
    # 2. Harmonics with natural exponential decay
    for harmonic in range(2, 6):
        signal += (1.0 / (harmonic ** 1.5)) * np.sin(phase * harmonic)
        
    # 3. Add simulated formants (vowel-like resonances) by applying spectral shaping
    # We apply bandpass resonances at 600Hz, 1700Hz, 2800Hz
    # Formants are simulated by adding damped sinusoids that modulate the signal
    formants = (np.sin(2 * np.pi * 600 * t) * np.exp(-10 * t % 0.1) +
                0.5 * np.sin(2 * np.pi * 1700 * t) * np.exp(-15 * t % 0.08) +
                0.25 * np.sin(2 * np.pi * 2800 * t) * np.exp(-20 * t % 0.05))
    signal = 0.7 * signal + 0.3 * formants * signal
    
    # 4. Add natural breathing/unvoiced background noise (low-level white noise)
    noise = np.random.normal(0, 0.02, len(t))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    return signal

def generate_deepfake_signal(duration=2.0, sr=16000):
    """
    Generates a simulated deepfake AI-generated signal.
    Features: Robotic/perfectly constant pitch (no micro-variability),
    spectral buzz, and periodic phase artifacts.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Monotone pitch (perfectly constant frequency with no variance)
    f0 = 150.0  # static pitch
    phase = 2 * np.pi * f0 * t
    
    signal = np.sin(phase)
    
    # 2. Harmonics with rigid decay and high frequency buzz
    for harmonic in range(2, 10):
        signal += (1.0 / harmonic) * np.sin(phase * harmonic)
        
    # 3. Phase vocoder distortion: add periodic phase shifts or "glitches"
    # Create phase jumps every 0.2 seconds
    glitch_mask = np.ones_like(t)
    for i in range(1, int(duration / 0.2)):
        glitch_idx = int(i * 0.2 * sr)
        # Apply a short ring modulation or volume dip to simulate synthesis artifact
        glitch_mask[glitch_idx:glitch_idx + 80] = 0.1
    signal *= glitch_mask
    
    # 4. High-frequency digital noise/buzz (metal hiss)
    high_freq_noise = np.sin(2 * np.pi * 4000 * t) * np.random.normal(0, 0.05, len(t))
    signal += high_freq_noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    return signal

def main():
    # Set output directories
    base_dir = os.path.join("D:\\", "MARS_ML", "data")
    gen_dir = os.path.join(base_dir, "genuine")
    fake_dir = os.path.join(base_dir, "deepfake")
    
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    print("Generating simulated dataset...")
    
    num_samples = 100  # 100 samples per class
    sr = 16000
    duration = 2.0
    
    for i in range(num_samples):
        # Generate genuine
        gen_sig = generate_genuine_signal(duration, sr)
        gen_path = os.path.join(gen_dir, f"genuine_{i:03d}.wav")
        wavfile.write(gen_path, sr, (gen_sig * 32767).astype(np.int16))
        
        # Generate deepfake
        fake_sig = generate_deepfake_signal(duration, sr)
        fake_path = os.path.join(fake_dir, f"deepfake_{i:03d}.wav")
        wavfile.write(fake_path, sr, (fake_sig * 32767).astype(np.int16))
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{num_samples} samples per class.")
            
    print(f"Dataset generated successfully!")
    print(f"Genuine folder: {gen_dir} ({len(os.listdir(gen_dir))} files)")
    print(f"Deepfake folder: {fake_dir} ({len(os.listdir(fake_dir))} files)")

if __name__ == "__main__":
    main()
