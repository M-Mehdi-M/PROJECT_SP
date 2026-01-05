import numpy as np
import librosa
import soundfile as sf
import os
import random

# --- CONFIGURATION ---
CLEAN_DIR = "dataset/clean_16k"
NOISE_DIR = "dataset/noise_16k"
OUTPUT_NOISY = "dataset/train_noisy"
OUTPUT_CLEAN = "dataset/train_clean"

# Parameters
TARGET_SR = 16000
DURATION = 10.0        # Seconds per clip (keep it short for faster training)
NUM_SAMPLES = 5000    # How many training pairs to generate

# Create output folders
os.makedirs(OUTPUT_NOISY, exist_ok=True)
os.makedirs(OUTPUT_CLEAN, exist_ok=True)

# Get file lists
clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.wav')]
noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')]

print(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files.")
print(f"Generating {NUM_SAMPLES} training pairs...")

def get_random_snr():
    # The paper uses a Gaussian distribution N(5, 10)
    # We will clamp it between -5dB and 20dB to avoid extremely loud noise or silence
    snr = random.gauss(5, 10)
    return max(-5, min(snr, 20))

for i in range(NUM_SAMPLES):
    try:
        # 1. Load random clean speech
        clean_filename = random.choice(clean_files)
        clean_path = os.path.join(CLEAN_DIR, clean_filename)
        clean_audio, _ = librosa.load(clean_path, sr=TARGET_SR)
        
        # 2. Load random noise
        noise_filename = random.choice(noise_files)
        noise_path = os.path.join(NOISE_DIR, noise_filename)
        noise_audio, _ = librosa.load(noise_path, sr=TARGET_SR)
        
        # 3. CUT or PAD to exact duration
        target_len = int(DURATION * TARGET_SR)
        
        # Fix Clean Audio Length
        if len(clean_audio) > target_len:
            start = random.randint(0, len(clean_audio) - target_len)
            clean_audio = clean_audio[start:start+target_len]
        else:
            # Pad with zeros if too short
            padding = target_len - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, padding), 'constant')
            
        # Fix Noise Audio Length
        while len(noise_audio) < target_len:
            noise_audio = np.concatenate((noise_audio, noise_audio))
        start = random.randint(0, len(noise_audio) - target_len)
        noise_audio = noise_audio[start:start+target_len]

        # 4. Calculate Energy (RMS)
        clean_rms = np.sqrt(np.mean(clean_audio**2))
        noise_rms = np.sqrt(np.mean(noise_audio**2))
        
        # Skip silence
        if clean_rms < 1e-4 or noise_rms < 1e-4:
            continue

        # 5. MIXING STRATEGY (SNR)
        target_snr_db = get_random_snr()
        target_snr_linear = 10 ** (target_snr_db / 20)
        
        # Formula: Noise_new = Noise_old * (Clean_RMS / (Noise_RMS * SNR))
        noise_gain = clean_rms / (noise_rms * target_snr_linear)
        adjusted_noise = noise_audio * noise_gain
        
        # Add them
        noisy_audio = clean_audio + adjusted_noise
        
        # 6. Normalize to prevent clipping (keep max volume at 0.9)
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 0:
            scale_factor = 0.9 / max_val
            noisy_audio = noisy_audio * scale_factor
            clean_audio = clean_audio * scale_factor 

        # 7. Save
        out_filename = f"sample_{i:04d}.wav"
        sf.write(os.path.join(OUTPUT_NOISY, out_filename), noisy_audio, TARGET_SR)
        sf.write(os.path.join(OUTPUT_CLEAN, out_filename), clean_audio, TARGET_SR)
        
        if i % 100 == 0:
            print(f"Generated {i} / {NUM_SAMPLES} pairs")

    except Exception as e:
        print(f"Skipping sample {i} due to error: {e}")

print("Dataset generation complete!")