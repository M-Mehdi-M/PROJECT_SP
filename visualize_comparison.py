"""
Visualization Script for NSnet2 Speech Enhancement
- Processes multiple samples from enhanced/ directory
- Creates spectrograms, waveforms, and metrics visualizations
- Generates publication-ready figures for documentation
"""

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
ENHANCED_DIR = "enhanced"
OUTPUT_DIR = "figures"
NUM_SAMPLES_TO_VISUALIZE = 5  # Number of detailed spectrograms
SAMPLE_RATE = 16000
N_FFT = 320
HOP_LENGTH = 160

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_sample_names(directory, num_samples=50):
    """Get unique sample names from enhanced directory"""
    if not os.path.exists(directory):
        print(f"[ERROR] Directory not found: {directory}")
        return []
    
    files = [f for f in os.listdir(directory) if f.endswith('_1_clean.wav')]
    sample_names = [f.replace('_1_clean.wav', '') for f in files]
    return sorted(sample_names)[:num_samples]


def load_audio_triplet(sample_name):
    """Load clean, noisy, enhanced audio for a sample"""
    clean_path = os.path.join(ENHANCED_DIR, f"{sample_name}_1_clean.wav")
    noisy_path = os.path.join(ENHANCED_DIR, f"{sample_name}_2_noisy.wav")
    enhanced_path = os.path. join(ENHANCED_DIR, f"{sample_name}_3_enhanced.wav")
    
    # Check if all files exist
    for path in [clean_path, noisy_path, enhanced_path]: 
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    clean, _ = sf.read(clean_path)
    noisy, _ = sf.read(noisy_path)
    enhanced, _ = sf.read(enhanced_path)
    
    return clean, noisy, enhanced


def calculate_snr(clean, signal):
    """Calculate SNR in dB"""
    # Ensure same length
    min_len = min(len(clean), len(signal))
    clean = clean[:min_len]
    signal = signal[:min_len]
    
    noise = signal - clean
    power_clean = np.sum(clean ** 2)
    power_noise = np.sum(noise ** 2)
    if power_noise < 1e-10:
        return 100.0
    return 10 * np.log10(power_clean / power_noise)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_spectrogram_grid(sample_names, num_samples=5):
    """Create a grid of spectrograms for multiple samples"""
    print(f"Creating spectrogram grid for {num_samples} samples...")
    
    actual_samples = min(num_samples, len(sample_names))
    if actual_samples == 0:
        print("  [SKIP] No samples available")
        return
    
    fig, axes = plt.subplots(actual_samples, 3, figsize=(15, 4*actual_samples))
    
    # Handle single sample case (axes not 2D)
    if actual_samples == 1:
        axes = axes.reshape(1, -1)
    
    img = None
    for idx, sample_name in enumerate(sample_names[: actual_samples]):
        try:
            clean, noisy, enhanced = load_audio_triplet(sample_name)
            
            audio_files = [
                (clean, "Clean (Reference)", axes[idx, 0]),
                (noisy, "Noisy (Input)", axes[idx, 1]),
                (enhanced, "Enhanced (Output)", axes[idx, 2]),
            ]
            
            for audio, title, ax in audio_files:
                D = librosa. stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis='time',
                                               y_axis='hz', ax=ax, cmap='magma', vmin=-80, vmax=0)
                
                if idx == 0:
                    ax.set_title(title, fontsize=12, fontweight='bold')
                else:
                    ax.set_title('')
                
                ax.set_xlabel('')
                ax.set_ylabel(f'Sample {idx+1}' if title == "Clean (Reference)" else '')
        except Exception as e:
            print(f"  [WARNING] Error processing {sample_name}: {e}")
            continue
    
    # Add colorbar
    if img is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    
    plt.suptitle('Spectrogram Comparison:  Clean vs Noisy vs Enhanced', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{OUTPUT_DIR}/spectrogram_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {OUTPUT_DIR}/spectrogram_grid.png")


def create_single_comparison(sample_name):
    """Create detailed comparison for a single sample"""
    try:
        clean, noisy, enhanced = load_audio_triplet(sample_name)
    except Exception as e:
        print(f"  [WARNING] Cannot create comparison for {sample_name}: {e}")
        return
    
    fig = plt.figure(figsize=(16, 10))
    
    # Spectrograms (top row)
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig. add_subplot(2, 3, 3)
    
    for audio, title, ax in [(clean, "Clean (Reference)", ax1),
                              (noisy, "Noisy (Input)", ax2),
                              (enhanced, "Enhanced (Output)", ax3)]:
        D = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display. specshow(S_db, sr=SAMPLE_RATE, x_axis='time',
                                       y_axis='hz', ax=ax, cmap='magma', vmin=-80, vmax=0)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Waveforms (bottom row)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    time_clean = np.arange(len(clean)) / SAMPLE_RATE
    time_noisy = np.arange(len(noisy)) / SAMPLE_RATE
    time_enhanced = np.arange(len(enhanced)) / SAMPLE_RATE
    
    for audio, time_arr, title, ax, color in [
        (clean, time_clean, "Clean Waveform", ax4, '#27ae60'),
        (noisy, time_noisy, "Noisy Waveform", ax5, '#e74c3c'),
        (enhanced, time_enhanced, "Enhanced Waveform", ax6, '#3498db')
    ]:
        ax.plot(time_arr, audio, color=color, linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    input_snr = calculate_snr(clean, noisy)
    output_snr = calculate_snr(clean, enhanced)
    improvement = output_snr - input_snr
    
    fig.suptitle(f'{sample_name} | SNR:  {input_snr:.1f} dB -> {output_snr:.1f} dB (Improvement: +{improvement:.1f} dB)',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/detailed_{sample_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_metrics_summary(sample_names):
    """Create summary visualization of metrics across all samples"""
    print("Calculating metrics for all samples...")
    
    metrics = []
    for sample_name in tqdm(sample_names, desc="Processing", leave=False):
        try:
            clean, noisy, enhanced = load_audio_triplet(sample_name)
            
            # Ensure same length
            min_len = min(len(clean), len(noisy), len(enhanced))
            clean = clean[:min_len]
            noisy = noisy[:min_len]
            enhanced = enhanced[:min_len]
            
            input_snr = calculate_snr(clean, noisy)
            output_snr = calculate_snr(clean, enhanced)
            
            metrics.append({
                'sample': sample_name,
                'input_snr': input_snr,
                'output_snr': output_snr,
                'improvement': output_snr - input_snr
            })
        except Exception as e:
            print(f"  [WARNING] Skipping {sample_name}: {e}")
            continue
    
    # Check if we have any metrics
    if len(metrics) == 0:
        print("  [ERROR] No metrics calculated!  Check your enhanced/ directory.")
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics)
    print(f"  [INFO] Successfully processed {len(df)} samples")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. SNR Distribution (Before/After)
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width/2, df['input_snr']. values, width, label='Input (Noisy)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, df['output_snr'].values, width, label='Output (Enhanced)', color='#27ae60', alpha=0.8)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('SNR Comparison: Input vs Output', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement Distribution Histogram
    ax2 = axes[0, 1]
    ax2.hist(df['improvement']. values, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    mean_improvement = df['improvement'].mean()
    ax2.axvline(x=mean_improvement, color='red', linestyle='--', linewidth=2,
                label=f'Mean:  {mean_improvement:.2f} dB')
    ax2.set_xlabel('SNR Improvement (dB)')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Distribution of SNR Improvement', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter Plot: Input SNR vs Improvement
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['input_snr'].values, df['improvement'].values,
                          c=df['improvement'].values, cmap='RdYlGn', s=60, edgecolors='black', alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Input SNR (dB)')
    ax3.set_ylabel('SNR Improvement (dB)')
    ax3.set_title('Improvement vs Input Quality', fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Improvement (dB)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics Box
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    ============================================
            EVALUATION SUMMARY                 
    ============================================
      Samples Analyzed:      {len(df):>6}             
                                              
      INPUT SNR:                              
        Mean:                {df['input_snr']. mean():>6.2f} dB         
        Std:                {df['input_snr']. std():>6.2f} dB         
                                              
      OUTPUT SNR:                             
        Mean:               {df['output_snr'].mean():>6.2f} dB         
        Std:                {df['output_snr'].std():>6.2f} dB         
                                              
      IMPROVEMENT:                             
        Mean:              {df['improvement'].mean():>+6.2f} dB         
        Best:              {df['improvement'].max():>+6.2f} dB         
        Worst:              {df['improvement'].min():>+6.2f} dB         
    ============================================
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('NSnet2 Speech Enhancement - Metrics Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {OUTPUT_DIR}/metrics_summary.png")
    
    return df


def create_best_worst_comparison(df, sample_names):
    """Create comparison of best and worst performing samples"""
    print("Creating best/worst comparison...")
    
    if df.empty or len(df) < 2:
        print("  [SKIP] Not enough data for best/worst comparison")
        return
    
    best_idx = df['improvement'].idxmax()
    worst_idx = df['improvement'].idxmin()
    
    best_sample = df.loc[best_idx, 'sample']
    worst_sample = df.loc[worst_idx, 'sample']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for row, (sample_name, label) in enumerate([(best_sample, 'BEST'), (worst_sample, 'WORST')]):
        try:
            clean, noisy, enhanced = load_audio_triplet(sample_name)
            
            improvement = calculate_snr(clean, enhanced) - calculate_snr(clean, noisy)
            
            for col, (audio, title) in enumerate([
                (clean, "Clean"), (noisy, "Noisy"), (enhanced, "Enhanced")
            ]):
                ax = axes[row, col]
                D = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display. specshow(S_db, sr=SAMPLE_RATE, x_axis='time',
                                               y_axis='hz', ax=ax, cmap='magma', vmin=-80, vmax=0)
                ax.set_title(f"{label} Case - {title}" if col == 0 else title, fontsize=11)
                
                if col == 0:
                    ax.set_ylabel(f'{label}\n(+{improvement:.1f} dB)')
        except Exception as e:
            print(f"  [WARNING] Error with {sample_name}: {e}")
            continue
    
    plt.suptitle('Best vs Worst Enhancement Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/best_worst_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {OUTPUT_DIR}/best_worst_comparison. png")


def create_presentation_figure(sample_names):
    """Create a clean figure suitable for presentation"""
    print("Creating presentation figure...")
    
    if len(sample_names) == 0:
        print("  [SKIP] No samples available")
        return
    
    # Pick a good sample (middle of the list)
    sample_idx = min(len(sample_names)//4, len(sample_names)-1)
    sample_name = sample_names[sample_idx]
    
    try:
        clean, noisy, enhanced = load_audio_triplet(sample_name)
    except Exception as e:
        print(f"  [ERROR] Cannot load sample: {e}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    titles = ['Clean (Reference)', 'Noisy (Input)', 'Enhanced (Output)']
    audios = [clean, noisy, enhanced]
    
    img = None
    for ax, audio, title in zip(axes, audios, titles):
        D = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis='time',
                                       y_axis='hz', ax=ax, cmap='magma', vmin=-80, vmax=0)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
    
    # Add colorbar
    if img is not None:
        cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.8)
        cbar.set_label('Magnitude (dB)', fontsize=11)
    
    input_snr = calculate_snr(clean, noisy)
    output_snr = calculate_snr(clean, enhanced)
    
    plt.suptitle(f'Speech Enhancement Result: SNR {input_snr:.1f} dB -> {output_snr:.1f} dB (+{output_snr-input_snr:.1f} dB)',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/presentation_spectrogram.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {OUTPUT_DIR}/presentation_spectrogram.png")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    print("="*60)
    print("NSnet2 VISUALIZATION GENERATOR")
    print("="*60)
    
    # Get sample names
    sample_names = get_sample_names(ENHANCED_DIR, num_samples=50)
    print(f"\n[INFO] Found {len(sample_names)} samples in {ENHANCED_DIR}/")
    print(f"[INFO] Output directory: {OUTPUT_DIR}/\n")
    
    if len(sample_names) == 0:
        print("[ERROR] No samples found!  Run batch_evaluation.py first.")
        print("\nMake sure you have files like:")
        print("  enhanced/sample_0000_1_clean.wav")
        print("  enhanced/sample_0000_2_noisy.wav")
        print("  enhanced/sample_0000_3_enhanced.wav")
        return
    
    # Generate visualizations
    print("Generating visualizations.. .\n")
    
    # 1. Metrics summary (processes all samples)
    df = create_metrics_summary(sample_names)
    
    if df.empty:
        print("\n[ERROR] Could not calculate metrics.  Exiting.")
        return
    
    # 2. Spectrogram grid (top N samples)
    create_spectrogram_grid(sample_names, num_samples=NUM_SAMPLES_TO_VISUALIZE)
    
    # 3. Best/Worst comparison
    create_best_worst_comparison(df, sample_names)
    
    # 4. Presentation figure
    create_presentation_figure(sample_names)
    
    # 5. Detailed single comparisons (only for first 3)
    print(f"Creating detailed comparisons for {min(3, len(sample_names))} samples...")
    for sample_name in sample_names[:3]: 
        create_single_comparison(sample_name)
    print(f"  [SAVED] {OUTPUT_DIR}/detailed_*. png")
    
    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated figures in '{OUTPUT_DIR}/':")
    print("  - metrics_summary.png        (Overall metrics analysis)")
    print("  - spectrogram_grid.png       (Multiple sample comparison)")
    print("  - best_worst_comparison.png  (Best vs worst cases)")
    print("  - presentation_spectrogram.png (Clean slide figure)")
    print("  - detailed_*.png             (Individual sample analysis)")
    print("\nUse these figures in your documentation and presentation!")
    print("="*60)


if __name__ == "__main__":
    main()
