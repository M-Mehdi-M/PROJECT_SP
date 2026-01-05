"""
Batch Evaluation Script for NSnet2 Speech Enhancement
- Evaluates model on multiple samples
- Saves clean/noisy/enhanced audio files together for comparison
- Calculates all metrics (SNR, SI-SDR, PESQ, STOI)
- Generates summary statistics for documentation
"""

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import os
import math
import shutil
import time
from tqdm import tqdm
import pandas as pd

# Fix Windows encoding
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "nsnet2_epoch_10.pth"
NOISY_DIR = "dataset/train_noisy"
CLEAN_DIR = "dataset/train_clean"
OUTPUT_DIR = "enhanced"  # Output directory for comparison files
NUM_TEST_SAMPLES = 50    # Number of samples to evaluate
N_FFT = 320
HOP_LENGTH = 160
SAMPLE_RATE = 16000

# =============================================================================
# MODEL DEFINITION (Must match training!)
# =============================================================================
class NSnet2(nn.Module):
    """NSnet2: FC-GRU-GRU-FC-FC-FC"""
    def __init__(self, input_dim=161, gru_dim=400, fc_dim=600):
        super(NSnet2, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.gru = nn.GRU(fc_dim, gru_dim, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(gru_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, fc_dim)
        self.fc4 = nn.Linear(fc_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x, _ = self.gru(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.sigmoid(self.fc4(x))

# =============================================================================
# METRICS FUNCTIONS
# =============================================================================
def calculate_snr(clean, signal):
    """Signal-to-Noise Ratio in dB"""
    noise = signal - clean
    power_clean = np.sum(clean ** 2)
    power_noise = np.sum(noise ** 2)
    if power_noise < 1e-10:
        return 100.0
    return 10 * math.log10(power_clean / power_noise)


def calculate_si_sdr(reference, estimation):
    """Scale-Invariant Signal-to-Distortion Ratio in dB"""
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    dot = np.dot(estimation, reference)
    s_target = dot * reference / (np.dot(reference, reference) + 1e-8)
    e_noise = estimation - s_target
    
    return 10 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-8))


# =============================================================================
# ENHANCEMENT FUNCTION
# =============================================================================
def enhance_audio(model, noisy_audio, device):
    """
    Enhance a single audio file using the trained model. 
    Returns enhanced audio and inference time.
    """
    audio_t = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).to(device)
    window = torch.sqrt(torch.hann_window(N_FFT)).to(device)
    
    # STFT
    stft = torch.stft(audio_t, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      window=window, return_complex=True)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Log power spectrum (must match training!)
    log_mag = torch.log10(magnitude ** 2 + 1e-8)
    model_input = log_mag.transpose(1, 2)
    
    # Inference with timing
    start_time = time.time()
    with torch.no_grad():
        mask = model(model_input)
    inference_time = time.time() - start_time
    
    # Apply mask and reconstruct
    mask = mask.transpose(1, 2)
    enhanced_mag = magnitude * mask
    enhanced_stft = torch.polar(enhanced_mag, phase)
    enhanced_audio = torch.istft(enhanced_stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                  window=window, length=len(noisy_audio))
    
    return enhanced_audio. squeeze().cpu().numpy(), inference_time


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================
def main():
    print("="*70)
    print("NSnet2 BATCH EVALUATION")
    print("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    
    # Create output directory structure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}/")
    
    # Load model
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = NSnet2().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[OK] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")
    
    # Get test files
    noisy_files = sorted([f for f in os.listdir(NOISY_DIR) if f.endswith('.wav')])[:NUM_TEST_SAMPLES]
    print(f"[INFO] Testing on {len(noisy_files)} samples\n")
    
    # Check for optional metric libraries
    try:
        from pesq import pesq
        has_pesq = True
        print("[OK] PESQ library available")
    except ImportError:
        has_pesq = False
        print("[WARNING] PESQ not installed.  Run: pip install pesq")
    
    try:
        from pystoi import stoi
        has_stoi = True
        print("[OK] STOI library available")
    except ImportError: 
        has_stoi = False
        print("[WARNING] STOI not installed. Run: pip install pystoi")
    
    print("\n" + "-"*70)
    print("Processing samples...")
    print("-"*70)
    
    # Store results
    results = []
    total_inference_time = 0
    total_audio_duration = 0
    
    # Process each sample
    for filename in tqdm(noisy_files, desc="Evaluating"):
        # Get sample name (without extension)
        sample_name = os.path.splitext(filename)[0]
        
        # Build paths
        noisy_path = os.path.join(NOISY_DIR, filename)
        clean_path = os.path. join(CLEAN_DIR, filename)
        
        # Skip if clean file doesn't exist
        if not os. path.exists(clean_path):
            print(f"[SKIP] No clean reference for:  {filename}")
            continue
        
        # Load audio files
        noisy, sr = sf.read(noisy_path)
        clean, _ = sf.read(clean_path)
        
        # Enhance audio
        enhanced, inf_time = enhance_audio(model, noisy, device)
        
        # Track timing
        total_inference_time += inf_time
        total_audio_duration += len(noisy) / sr
        
        # Ensure same length for all files
        min_len = min(len(clean), len(noisy), len(enhanced))
        clean = clean[:min_len]
        noisy = noisy[: min_len]
        enhanced = enhanced[:min_len]
        
        # =================================================================
        # SAVE FILES FOR MANUAL COMPARISON
        # =================================================================
        # Save all three versions with clear naming
        clean_output_path = os.path.join(OUTPUT_DIR, f"{sample_name}_1_clean.wav")
        noisy_output_path = os. path.join(OUTPUT_DIR, f"{sample_name}_2_noisy.wav")
        enhanced_output_path = os. path.join(OUTPUT_DIR, f"{sample_name}_3_enhanced.wav")
        
        sf.write(clean_output_path, clean, sr)
        sf.write(noisy_output_path, noisy, sr)
        sf.write(enhanced_output_path, enhanced, sr)
        
        # =================================================================
        # CALCULATE METRICS
        # =================================================================
        result = {
            'sample': sample_name,
            'input_snr': calculate_snr(clean, noisy),
            'output_snr': calculate_snr(clean, enhanced),
            'input_sisdr': calculate_si_sdr(clean, noisy),
            'output_sisdr': calculate_si_sdr(clean, enhanced),
            'inference_time_ms': inf_time * 1000,
        }
        
        # Calculate improvements
        result['snr_improvement'] = result['output_snr'] - result['input_snr']
        result['sisdr_improvement'] = result['output_sisdr'] - result['input_sisdr']
        
        # PESQ (if available)
        if has_pesq:
            try:
                result['input_pesq'] = pesq(sr, clean, noisy, 'wb')
                result['output_pesq'] = pesq(sr, clean, enhanced, 'wb')
                result['pesq_improvement'] = result['output_pesq'] - result['input_pesq']
            except Exception as e:
                result['input_pesq'] = np.nan
                result['output_pesq'] = np.nan
                result['pesq_improvement'] = np.nan
        
        # STOI (if available)
        if has_stoi: 
            try:
                result['input_stoi'] = stoi(clean, noisy, sr, extended=False)
                result['output_stoi'] = stoi(clean, enhanced, sr, extended=False)
                result['stoi_improvement'] = result['output_stoi'] - result['input_stoi']
            except Exception as e: 
                result['input_stoi'] = np.nan
                result['output_stoi'] = np.nan
                result['stoi_improvement'] = np.nan
        
        results.append(result)
    
    # =================================================================
    # CREATE RESULTS DATAFRAME
    # =================================================================
    df = pd.DataFrame(results)
    
    # Calculate Real-Time Factor
    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0
    
    # =================================================================
    # PRINT SUMMARY STATISTICS
    # =================================================================
    print("\n" + "="*70)
    print("EVALUATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n[INFO] Samples evaluated: {len(df)}")
    print(f"[INFO] Total audio processed: {total_audio_duration:.1f} seconds")
    print(f"[INFO] Total inference time: {total_inference_time:.2f} seconds")
    print(f"[INFO] Real-Time Factor: {rtf:.4f} ({'Real-time capable!' if rtf < 1 else 'Not real-time'})")
    
    print(f"\n--- SNR (Signal-to-Noise Ratio) ---")
    print(f"  Input Mean:         {df['input_snr']. mean():.2f} dB")
    print(f"  Output Mean:       {df['output_snr'].mean():.2f} dB")
    print(f"  Improvement Mean:  {df['snr_improvement'].mean():+.2f} dB")
    print(f"  Improvement Std:   {df['snr_improvement'].std():.2f} dB")
    print(f"  Best Improvement:  {df['snr_improvement']. max():+.2f} dB")
    print(f"  Worst Improvement: {df['snr_improvement'].min():+.2f} dB")
    
    print(f"\n--- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) ---")
    print(f"  Input Mean:        {df['input_sisdr']. mean():.2f} dB")
    print(f"  Output Mean:       {df['output_sisdr'].mean():.2f} dB")
    print(f"  Improvement Mean:   {df['sisdr_improvement'].mean():+.2f} dB")
    print(f"  Improvement Std:   {df['sisdr_improvement']. std():.2f} dB")
    
    if has_pesq and 'pesq_improvement' in df.columns:
        print(f"\n--- PESQ (Perceptual Evaluation of Speech Quality) ---")
        print(f"  Input Mean:        {df['input_pesq'].mean():.3f}")
        print(f"  Output Mean:       {df['output_pesq'].mean():.3f}")
        print(f"  Improvement Mean:  {df['pesq_improvement'].mean():+.3f}")
        print(f"  (PESQ range: 1. 0 = bad, 4.5 = excellent)")
    
    if has_stoi and 'stoi_improvement' in df. columns:
        print(f"\n--- STOI (Short-Time Objective Intelligibility) ---")
        print(f"  Input Mean:        {df['input_stoi'].mean():.3f}")
        print(f"  Output Mean:       {df['output_stoi'].mean():.3f}")
        print(f"  Improvement Mean:  {df['stoi_improvement'].mean():+.3f}")
        print(f"  (STOI range: 0.0 = unintelligible, 1.0 = perfect)")
    
    print(f"\n--- Inference Performance ---")
    print(f"  Mean inference time: {df['inference_time_ms'].mean():.2f} ms per 10s audio")
    print(f"  Real-Time Factor:     {rtf:.4f}")
    print(f"  Speed:                {1/rtf:.1f}x faster than real-time")
    
    # =================================================================
    # SAVE RESULTS
    # =================================================================
    
    # Save detailed CSV
    csv_path = os. path.join(OUTPUT_DIR, "evaluation_results.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n[SAVED] Detailed results:  {csv_path}")
    
    # Save summary statistics
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NSnet2 SPEECH ENHANCEMENT - EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Samples Evaluated: {len(df)}\n")
        f.write(f"Total Audio Duration: {total_audio_duration:.1f} seconds\n\n")
        
        f. write("-"*50 + "\n")
        f.write("RESULTS TABLE (for documentation)\n")
        f.write("-"*50 + "\n\n")
        
        f.write("| Metric | Input (Noisy) | Output (Enhanced) | Improvement |\n")
        f.write("|--------|---------------|-------------------|-------------|\n")
        f.write(f"| SNR (dB) | {df['input_snr'].mean():.2f} | {df['output_snr'].mean():.2f} | {df['snr_improvement'].mean():+.2f} |\n")
        f.write(f"| SI-SDR (dB) | {df['input_sisdr'].mean():.2f} | {df['output_sisdr'].mean():.2f} | {df['sisdr_improvement'].mean():+.2f} |\n")
        
        if has_pesq: 
            f.write(f"| PESQ | {df['input_pesq']. mean():.3f} | {df['output_pesq'].mean():.3f} | {df['pesq_improvement'].mean():+.3f} |\n")
        if has_stoi:
            f.write(f"| STOI | {df['input_stoi'].mean():.3f} | {df['output_stoi'].mean():.3f} | {df['stoi_improvement']. mean():+.3f} |\n")
        
        f.write(f"\nReal-Time Factor: {rtf:.4f} ({1/rtf:.1f}x real-time)\n")
        f.write(f"Model Parameters: {total_params:,}\n")
    
    print(f"[SAVED] Summary report: {summary_path}")
    
    # =================================================================
    # PRINT TABLE FOR DOCUMENTATION
    # =================================================================
    print("\n" + "="*70)
    print("TABLE FOR YOUR DOCUMENTATION (copy this!):")
    print("="*70)
    print("""
| Metric | Input (Noisy) | Output (Enhanced) | Improvement |
|--------|---------------|-------------------|-------------|""")
    print(f"| SNR (dB) | {df['input_snr'].mean():.2f} | {df['output_snr'].mean():.2f} | {df['snr_improvement'].mean():+.2f} |")
    print(f"| SI-SDR (dB) | {df['input_sisdr']. mean():.2f} | {df['output_sisdr'].mean():.2f} | {df['sisdr_improvement'].mean():+.2f} |")
    if has_pesq:
        print(f"| PESQ | {df['input_pesq']. mean():.3f} | {df['output_pesq'].mean():.3f} | {df['pesq_improvement'].mean():+.3f} |")
    if has_stoi: 
        print(f"| STOI | {df['input_stoi'].mean():.3f} | {df['output_stoi'].mean():.3f} | {df['stoi_improvement']. mean():+.3f} |")
    print(f"\nComputational Performance:")
    print(f"  - Model Parameters: {total_params:,}")
    print(f"  - Real-Time Factor: {rtf:.4f}")
    print(f"  - Processing Speed: {1/rtf:.1f}x real-time")
    print("="*70)
    
    # =================================================================
    # OUTPUT DIRECTORY STRUCTURE INFO
    # =================================================================
    print(f"\n[INFO] Output files saved in: {OUTPUT_DIR}/")
    print("  File naming convention:")
    print("    - sample_XXXX_1_clean.wav    (original clean speech)")
    print("    - sample_XXXX_2_noisy.wav    (noisy input)")
    print("    - sample_XXXX_3_enhanced. wav (model output)")
    print("\n  Open these files in any audio player to compare!")
    print("  Tip: Play them in order (1, 2, 3) to hear the difference.")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
