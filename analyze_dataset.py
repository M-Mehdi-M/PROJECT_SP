import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

NOISY_DIR = "dataset/train_noisy"
CLEAN_DIR = "dataset/train_clean"
NUM_SAMPLES = 100  # Analyze first 100 samples

def calculate_snr(clean, noisy):
    """Calculate input SNR"""
    noise = noisy - clean
    power_clean = np.sum(clean ** 2)
    power_noise = np.sum(noise ** 2)
    if power_noise < 1e-10:
        return 100.0
    return 10 * np.log10(power_clean / power_noise)

def analyze_dataset():
    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith('.wav')])[: NUM_SAMPLES]
    
    snr_values = []
    clean_rms_values = []
    noisy_rms_values = []
    
    print(f"Analyzing {len(noisy_files)} samples...")
    
    for filename in tqdm(noisy_files):
        noisy_path = os.path. join(NOISY_DIR, filename)
        clean_path = os.path.join(CLEAN_DIR, filename)
        
        if not os.path.exists(clean_path):
            continue
            
        noisy, sr = sf.read(noisy_path)
        clean, _ = sf.read(clean_path)
        
        # Calculate SNR
        snr = calculate_snr(clean, noisy)
        snr_values.append(snr)
        
        # Calculate RMS (volume level)
        clean_rms = np.sqrt(np.mean(clean ** 2))
        noisy_rms = np.sqrt(np.mean(noisy ** 2))
        clean_rms_values.append(clean_rms)
        noisy_rms_values.append(noisy_rms)
    
    # Statistics
    print("\n" + "="*60)
    print("DATASET ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n--- SNR Distribution (Input Noise Level) ---")
    print(f"  Mean SNR:   {np.mean(snr_values):.2f} dB")
    print(f"  Std SNR:    {np.std(snr_values):.2f} dB")
    print(f"  Min SNR:     {np.min(snr_values):.2f} dB")
    print(f"  Max SNR:    {np.max(snr_values):.2f} dB")
    
    # Paper target:  N(5, 10) means mean=5, std=10
    print(f"\n  Paper Target: Mean=5 dB, Std=10 dB")
    print(f"  Your Data:     Mean={np.mean(snr_values):.1f} dB, Std={np.std(snr_values):.1f} dB")
    
    if 0 <= np.mean(snr_values) <= 10:
        print(f"  [OK] SNR range is SUITABLE for this project!")
    elif np.mean(snr_values) > 15:
        print(f"  [WARNING] SNR too high - noise is too weak, model won't learn much")
    else:
        print(f"  [WARNING] SNR too low - noise is very heavy")
    
    print(f"\n--- Volume Levels (RMS) ---")
    print(f"  Clean RMS Mean:   {np.mean(clean_rms_values):.4f}")
    print(f"  Noisy RMS Mean:  {np.mean(noisy_rms_values):.4f}")
    
    # Categorize samples by difficulty
    easy = sum(1 for s in snr_values if s > 10)
    medium = sum(1 for s in snr_values if 0 <= s <= 10)
    hard = sum(1 for s in snr_values if s < 0)
    
    print(f"\n--- Difficulty Distribution ---")
    print(f"  Easy (SNR > 10 dB):     {easy} samples ({100*easy/len(snr_values):.1f}%)")
    print(f"  Medium (0-10 dB):       {medium} samples ({100*medium/len(snr_values):.1f}%)")
    print(f"  Hard (SNR < 0 dB):      {hard} samples ({100*hard/len(snr_values):.1f}%)")
    
    # Plot histogram
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(snr_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=5, color='red', linestyle='--', label='Paper Target Mean (5 dB)')
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('Number of Samples')
    plt.title('SNR Distribution of Your Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    categories = ['Easy\n(>10 dB)', 'Medium\n(0-10 dB)', 'Hard\n(<0 dB)']
    counts = [easy, medium, hard]
    colors = ['green', 'orange', 'red']
    plt.bar(categories, counts, color=colors, edgecolor='black')
    plt.ylabel('Number of Samples')
    plt.title('Dataset Difficulty Distribution')
    for i, v in enumerate(counts):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150)
    print(f"\n[SAVED] Histogram saved to:  dataset_analysis.png")
    plt.show()
    
    return snr_values

if __name__ == "__main__":
    analyze_dataset()
