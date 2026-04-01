# Klasyfikacja arytmii EKG — porównanie modeli klasycznych i głębokich sieci neuronowych

Projekt porównuje skuteczność klasycznych metod uczenia maszynowego (Random Forest) z architekturami głębokiego uczenia (Dense, LSTM, Conv1D) w zadaniu klasyfikacji uderzeń serca na podstawie sygnału EKG.

## Zbiór danych

Wykorzystano [ECG Arrhythmia Dataset (INCART 2-lead)](https://www.kaggle.com/datasets) z platformy Kaggle:
- **175 729** sekwencji uderzeń serca
- **187** punktów czasowych na sekwencję
- **4 klasy** (po usunięciu klasy Q jako nieistotnej):
  - **N** — uderzenie normalne
  - **VEB** — przedwczesne pobudzenie komorowe
  - **SVEB** — przedwczesne pobudzenie nadkomorowe
  - **F** — uderzenie fuzyjne

Zbiór jest silnie niezbalansowany — klasa N dominuje. Problem ten adresowany jest przez SMOTE (model bazowy) oraz augmentację danych (sieci neuronowe).

## Modele

| Model | Najlepszy F1 Macro (walidacja) | Uwagi |
|---|---|---|
| **Random Forest** (baseline) | **87.00%** | Z oversampling SMOTE |
| Conv1D | 81.41% | 30 epok, regularyzacja L2 |
| LSTM | 80.17% | 10 epok, silny overfitting |
| Dense | 75.09% | 10 epok, early stopping |

Model bazowy Random Forest z oversampling SMOTE okazał się najskuteczniejszy. Sieci neuronowe wykazują tendencję do overfittingu — różnica między F1 na zbiorze treningowym a walidacyjnym sięga 10–25 p.p.

## Struktura projektu

```
├── scripts/
│   └── prepare_data.py          # Pipeline danych: pobieranie, preprocessing, augmentacja
├── notebooks/
│   ├── baseline_model.ipynb     # Random Forest (baseline)
│   ├── lstm.ipynb               # Eksperymenty LSTM
│   ├── conv1d.ipynb             # Eksperymenty Conv1D
│   ├── dense.ipynb              # Eksperymenty Dense
│   ├── visualize_data.ipynb     # Eksploracja danych
│   ├── grapghs.ipynb            # Generowanie wykresów porównawczych
│   ├── projekt.ipynb            # Notebook zbiorczy
│   └── models/                  # Wytrenowane modele (.keras)
├── figures/                     # Wygenerowane wykresy
├── logs/fit/                    # Logi TensorBoard
└── pyproject.toml               # Zależności (uv)
```

## Uruchomienie

```bash
# Instalacja zależności
uv sync

# Uruchomienie notebooków
jupyter notebook notebooks/
```

Dane pobierane są automatycznie z Kaggle przy pierwszym uruchomieniu (wymaga skonfigurowanego `kagglehub`).

## Technologie

- Python 3.10
- TensorFlow / Keras
- scikit-learn, imbalanced-learn
- pandas, matplotlib
