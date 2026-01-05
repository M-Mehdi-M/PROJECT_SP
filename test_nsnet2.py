import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
import time
import os
import math

# Fix Windows console encoding for Unicode
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
MODEL_PATH = "nsnet2_epoch_10.pth"
INPUT_FILE = "dataset/train_noisy/sample_0024.wav"
CLEAN_FILE = "dataset/train_clean/sample_0024.wav"
OUTPUT_FILE = "enhanced_nsnet2.wav"
SAMPLE_RATE = 16000
N_FFT = 320
HOP_LENGTH = 160

# --- MODEL CLASS (MUST MATCH TRAINING EXACTLY!) ---
class NSnet2(nn.Module):
    """
    NSnet2: FC-GRU-GRU-FC-FC-FC
    Paper standard:  gru_dim=400, fc_dim=600
    """
    def __init__(self, input_dim=161, gru_dim=400, fc_dim=600):
        super(NSnet2, self).__init__()
        
        # FC input layer (161 -> 600)
        self.fc1 = nn.Linear(input_dim, fc_dim)
        
        # GRU layers (600 -> 400 internally)
        self.gru = nn.GRU(fc_dim, gru_dim, num_layers=2, batch_first=True)
        
        # FC output layers (400 -> 600 -> 600 -> 161)
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
        mask = self.sigmoid(self.fc4(x))
        return mask


# --- METRICS ---
def calculate_snr(clean, signal):
    """Signal-to-Noise Ratio"""
    noise = signal - clean
    power_clean = np.sum(clean ** 2)
    power_noise = np.sum(noise ** 2)
    if power_noise < 1e-10:
        return 100.0
    return 10 * math.log10(power_clean / power_noise)


def calculate_si_sdr(reference, estimation):
    """Scale-Invariant Signal-to-Distortion Ratio"""
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    dot = np.dot(estimation, reference)
    s_target = dot * reference / (np.dot(reference, reference) + 1e-8)
    e_noise = estimation - s_target
    
    si_sdr = 10 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-8))
    return si_sdr


# --- MAIN INFERENCE ---
def run_test():
    device = torch.device("cpu")
    
    # 1. Load Model
    print("Loading NSnet2 model...")
    model = NSnet2(input_dim=161, gru_dim=400, fc_dim=600).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[OK] Model loaded successfully!")
    except RuntimeError as e:
        print(f"[ERROR] Error loading model: {e}")
        print("\nMake sure the model architecture matches your training script!")
        return
    
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # 2. Load Audio
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    noisy_audio, sr = sf.read(INPUT_FILE)
    print(f"\n--- Input Audio ---")
    print(f"  File:  {INPUT_FILE}")
    print(f"  Duration: {len(noisy_audio)/sr:.2f}s")
    print(f"  Sample Rate: {sr} Hz")
    
    # Load clean reference if available
    clean_audio = None
    if os.path.exists(CLEAN_FILE):
        clean_audio, _ = sf.read(CLEAN_FILE)
        print(f"  Clean reference: {CLEAN_FILE}")

    # 3. Pre-processing (STFT with sqrt-Hann window)
    audio_t = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0)
    window = torch.sqrt(torch.hann_window(N_FFT))
    
    stft = torch.stft(audio_t, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                      window=window, return_complex=True)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Log magnitude spectrum (input features) - MUST match training! 
    # Training uses:  torch.log10(noisy_mag ** 2 + 1e-8)
    log_mag = torch.log10(magnitude ** 2 + 1e-8)
    
    # Reshape: [Batch, Freq, Time] -> [Batch, Time, Freq]
    model_input = log_mag.transpose(1, 2)

    # 4. Run Inference & Measure Time
    print(f"\n--- Inference ---")
    
    # Warm-up run (for accurate timing)
    with torch.no_grad():
        _ = model(model_input)
    
    # Timed run
    start_time = time.time()
    with torch.no_grad():
        mask = model(model_input)
    end_time = time.time()
    
    inference_time = end_time - start_time
    audio_duration = len(noisy_audio) / sr
    rtf = inference_time / audio_duration
    
    print(f"  Inference Time: {inference_time*1000:.2f} ms")
    print(f"  Audio Duration: {audio_duration:.2f} s")
    print(f"  Real-Time Factor: {rtf:.4f}")
    if rtf < 1.0:
        print(f"  [OK] Real-time capable!")
    else:
        print(f"  [SLOW] Not real-time")

    # 5. Reconstruct Audio
    # Reshape mask back:  [Batch, Time, Freq] -> [Batch, Freq, Time]
    mask = mask.transpose(1, 2)
    
    # Apply mask to original magnitude
    enhanced_mag = magnitude * mask
    
    # Reconstruct with original phase
    enhanced_stft = torch.polar(enhanced_mag, phase)
    
    # Inverse STFT
    enhanced_audio = torch.istft(
        enhanced_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        length=len(noisy_audio)
    )
    enhanced_audio = enhanced_audio.squeeze().numpy()

    # 6. Save Output
    sf.write(OUTPUT_FILE, enhanced_audio, sr)
    print(f"\n--- Output ---")
    print(f"  Saved to: {OUTPUT_FILE}")

    # 7. Calculate Metrics
    print(f"\n--- Evaluation Metrics ---")
    
    if clean_audio is not None: 
        min_len = min(len(clean_audio), len(noisy_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio_trimmed = noisy_audio[:min_len]
        enhanced_audio_trimmed = enhanced_audio[:min_len]
        
        input_snr = calculate_snr(clean_audio, noisy_audio_trimmed)
        output_snr = calculate_snr(clean_audio, enhanced_audio_trimmed)
        snr_improvement = output_snr - input_snr
        
        print(f"  Input SNR:         {input_snr:.2f} dB")
        print(f"  Output SNR:       {output_snr:.2f} dB")
        print(f"  SNR Improvement:   {snr_improvement:+.2f} dB")
        
        input_sisdr = calculate_si_sdr(clean_audio, noisy_audio_trimmed)
        output_sisdr = calculate_si_sdr(clean_audio, enhanced_audio_trimmed)
        sisdr_improvement = output_sisdr - input_sisdr
        
        print(f"\n  Input SI-SDR:         {input_sisdr:.2f} dB")
        print(f"  Output SI-SDR:       {output_sisdr:.2f} dB")
        print(f"  SI-SDR Improvement:  {sisdr_improvement:+.2f} dB")
        
        try:
            from pesq import pesq
            input_pesq = pesq(sr, clean_audio, noisy_audio_trimmed, 'wb')
            output_pesq = pesq(sr, clean_audio, enhanced_audio_trimmed, 'wb')
            print(f"\n  Input PESQ:  {input_pesq:.3f}")
            print(f"  Output PESQ: {output_pesq:.3f}")
            print(f"  PESQ Improvement: {output_pesq - input_pesq:+.3f}")
        except ImportError:
            print("\n  (Install 'pesq' package for PESQ metric:  pip install pesq)")
        except Exception as e:
            print(f"\n  PESQ calculation error: {e}")
            
        try: 
            from pystoi import stoi
            input_stoi = stoi(clean_audio, noisy_audio_trimmed, sr, extended=False)
            output_stoi = stoi(clean_audio, enhanced_audio_trimmed, sr, extended=False)
            print(f"\n  Input STOI:  {input_stoi:.3f}")
            print(f"  Output STOI: {output_stoi:.3f}")
            print(f"  STOI Improvement: {output_stoi - input_stoi:+.3f}")
        except ImportError:
            print("\n  (Install 'pystoi' package for STOI metric:  pip install pystoi)")
        except Exception as e:
            print(f"\n  STOI calculation error: {e}")
    else:
        print("  (No clean reference found - cannot calculate SNR/PESQ)")
        print("  To get metrics, ensure clean file exists at:", CLEAN_FILE)
    
    print("\n" + "="*50)
    print("Done!  Compare the audio files:")
    print(f"  Noisy:     {INPUT_FILE}")
    print(f"  Enhanced: {OUTPUT_FILE}")
    if clean_audio is not None: 
        print(f"  Clean:     {CLEAN_FILE}")
    print("="*50)


if __name__ == "__main__":
    run_test()
