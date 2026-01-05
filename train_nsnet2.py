import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import soundfile as sf
import numpy as np

# --- 1. CONFIGURATION (Based on Paper Section 4.3) ---
TRAIN_NOISY_DIR = "dataset/train_noisy"
TRAIN_CLEAN_DIR = "dataset/train_clean"
SAMPLE_RATE = 16000
N_FFT = 320            # 20ms at 16kHz
HOP_LENGTH = 160       # 50% overlap
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 8e-5   # Paper uses 8e-5 with AdamW
WEIGHT_DECAY = 0.1     # Paper Section 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Pre-compute window (sqrt-Hann as per paper)
WINDOW = torch.sqrt(torch. hann_window(N_FFT))

# --- 2. LOSS FUNCTION (Paper Equation 1) ---
class CompressedMSELoss(nn.Module):
    def __init__(self, c=0.3, lambda_phase=0.3):
        super().__init__()
        self.c = c
        self.lambda_phase = lambda_phase
    
    def forward(self, pred_mag, target_mag):
        pred_compressed = (pred_mag + 1e-8) ** self.c
        target_compressed = (target_mag + 1e-8) ** self.c
        return torch.mean((pred_compressed - target_compressed) ** 2)

# --- 3. DATASET CLASS ---
class AudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.window = torch.sqrt(torch.hann_window(N_FFT))

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        
        noisy_audio, _ = sf.read(noisy_path)
        clean_audio, _ = sf.read(clean_path)

        noisy_t = torch.tensor(noisy_audio, dtype=torch.float32)
        clean_t = torch. tensor(clean_audio, dtype=torch.float32)

        # STFT with sqrt-Hann window (Paper Section 4.3)
        noisy_stft = torch. stft(noisy_t, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                window=self.window, return_complex=True)
        clean_stft = torch.stft(clean_t, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                window=self.window, return_complex=True)

        noisy_mag = torch. abs(noisy_stft)
        clean_mag = torch. abs(clean_stft)

        # Log Power Spectrum as input features
        noisy_log = torch.log10(noisy_mag ** 2 + 1e-8)  # Log power spectrum
        
        # Transpose to (Time, Frequency) for RNN:  [T, 161]
        return noisy_log.transpose(0, 1), noisy_mag.transpose(0, 1), clean_mag.transpose(0, 1)

# --- 4. THE MODEL (NSnet2 Architecture from Paper Section 3. 1) ---
class NSnet2(nn.Module):
    """
    NSnet2: FC-GRU-GRU-FC-FC-FC
    Standard dimensions: 400 for GRUs, 600 for FC layers
    """
    def __init__(self, input_dim=161, gru_dim=400, fc_dim=600):
        super(NSnet2, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.gru = nn.GRU(fc_dim, gru_dim, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(gru_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, fc_dim)
        self.fc4 = nn.Linear(fc_dim, input_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Constrained suppression gain [0, 1]

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x, _ = self.gru(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        mask = self.sigmoid(self.fc4(x))
        return mask

# --- 5. TRAINING LOOP ---
def train():
    dataset = AudioDataset(TRAIN_NOISY_DIR, TRAIN_CLEAN_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = NSnet2().to(device)
    
    # Paper uses AdamW with weight decay 0.1
    optimizer = optim.AdamW(model. parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = CompressedMSELoss(c=0.3)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (noisy_log, noisy_mag, clean_mag) in enumerate(dataloader):
            noisy_log = noisy_log.to(device)
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag. to(device)
            
            optimizer.zero_grad()
            
            # Forward:  predict suppression mask
            mask = model(noisy_log)
            
            # Apply mask to noisy magnitude
            predicted_mag = mask * noisy_mag
            
            # Compressed MSE loss
            loss = criterion(predicted_mag, clean_mag)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f} ===")
        
        torch.save(model.state_dict(), f"nsnet2_epoch_{epoch+1}.pth")

    print("Training Complete!")

if __name__ == "__main__":
    train()
