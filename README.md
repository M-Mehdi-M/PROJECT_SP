# Real-Time Noise Suppression Using Deep Neural Networks

Reimplementation of the NSnet2 method from the paper "Towards Efficient Models for Real-Time Deep Noise Suppression" (Braun et al., Microsoft Research, 2021).

The project uses a recurrent neural network (FC-GRU-GRU-FC-FC-FC) to improve the quality of speech signals affected by noise. The model takes the noisy audio spectrum as input and generates a suppression mask applied to reconstruct the clean signal.

The dataset contains 5000 audio pairs (clean + noisy) of 10 seconds each, with an SNR distribution between -5 and 20 dB. Training was done over 10 epochs using the AdamW optimizer and the Compressed MSE loss function.

**Results obtained on 50 test samples:**
- SNR: 7.47 dB → 14.96 dB (+7.48 dB improvement)
- SI-SDR: 7.47 dB → 14.75 dB (+7.28 dB improvement)
- PESQ: 1.675 → 2.202 (+0.527 improvement)
- STOI: 0.899 → 0.923 (+0.024 improvement)

The model has 2.96M parameters and a Real-Time Factor of 0.0157, making it 63.8 times faster than real time. The results fall within the values reported in the original paper, demonstrating the method's effectiveness for real-time communication applications.

**Running:** `python train_nsnet2.py` (training), `python batch_evaluation.py` (evaluation), `python visualize_comparison.py` (visualizations).

**Reference:** Braun, S., Gamper, H., Reddy, C. K., & Tashev, I. (2021). Towards Efficient Models for Real-Time Deep Noise Suppression. arXiv: 2101.09249.
