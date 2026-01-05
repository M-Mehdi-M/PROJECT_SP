# Suprimarea Zgomotului in Timp Real folosind Retele Neuronale Profunde

Reimplementare a metodei NSnet2 din articolul "Towards Efficient Models for Real-Time
Deep Noise Suppression" (Braun et al., Microsoft Research, 2021).

Proiectul foloseste o retea neuronala recurenta (FC-GRU-GRU-FC-FC-FC) pentru imbunatatirea
calitatii semnalelor vocale afectate de zgomot.  Modelul primeste spectrul audio zgomotos
si genereaza o masca de suprimare aplicata pentru reconstructia semnalului curat. 

Setul de date contine 5000 de perechi audio (curat + zgomotos) de cate 10 secunde, cu
distributia SNR intre -5 si 20 dB. Antrenarea s-a realizat pe 10 epoci folosind 
optimizatorul AdamW si functia de cost Compressed MSE. 

Rezultate obtinute pe 50 de esantioane de test:
- SNR:  7.47 dB -> 14.96 dB (imbunatatire +7.48 dB)
- SI-SDR: 7.47 dB -> 14.75 dB (imbunatatire +7.28 dB)
- PESQ: 1.675 -> 2.202 (imbunatatire +0.527)
- STOI: 0.899 -> 0.923 (imbunatatire +0.024)

Modelul are 2.96M parametri si un Real-Time Factor de 0.0157, fiind de 63.8 ori mai
rapid decat timpul real.  Rezultatele se incadreaza in valorile raportate in articolul
original, demonstrand eficienta metodei pentru aplicatii de comunicare in timp real. 

Rulare:  python train_nsnet2.py (antrenare), python batch_evaluation.py (evaluare),
python visualize_comparison.py (vizualizari).

Referinta: Braun, S., Gamper, H., Reddy, C.  K., & Tashev, I. (2021). Towards
Efficient Models for Real-Time Deep Noise Suppression. arXiv: 2101.09249.
