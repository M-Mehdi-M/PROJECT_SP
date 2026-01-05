import librosa
import soundfile as sf
import numpy as np
import os
import math

# --- CONFIGURATION ---
# Replace this with the actual name of your downloaded file
INPUT_FILE = "new_data.mp3"

# Output folder for the chunks
OUTPUT_DIR = "dataset/noise_16k" 

# Paper Settings
TARGET_SR = 16000
CHUNK_DURATION = 10.0 # Seconds per chunk [cite: 83, 149]

def slice_long_audio():
    # 1. Check if file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find '{INPUT_FILE}'.")
        print("Make sure the downloaded file is in this folder and the name matches exactly!")
        return

    print(f"Loading large audio file: {INPUT_FILE}...")
    print("This might take a minute because it creates a huge array in RAM...")
    
    # 2. Load and Resample (Handles MP3 automatically via ffmpeg/librosa)
    # mono=True mixes it down to 1 channel as required
    audio, sr = librosa.load(INPUT_FILE, sr=TARGET_SR, mono=True)
    
    total_samples = len(audio)
    chunk_samples = int(CHUNK_DURATION * TARGET_SR)
    total_chunks = math.floor(total_samples / chunk_samples)
    
    print(f"Audio Loaded. Total Duration: {total_samples/sr/60:.2f} minutes.")
    print(f"Splitting into {total_chunks} clips of {CHUNK_DURATION} seconds each...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Slicing Loop
    for i in range(total_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        
        chunk = audio[start:end]
        
        # 4. Save as WAV
        # We name it 'nyc_ambience' so you can easily identify it later
        out_name = f"nyc_ambience_part_{i:04d}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        sf.write(out_path, chunk, TARGET_SR)
        
        if i % 50 == 0:
            print(f"Saved {i}/{total_chunks} chunks...")

    print(f"Done! {total_chunks} new noise files added to '{OUTPUT_DIR}'.")
    print("You can now run 'create_dataset.py' to mix these into your training set.")

if __name__ == "__main__":
    slice_long_audio()
