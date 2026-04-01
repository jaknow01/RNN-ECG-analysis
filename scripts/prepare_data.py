from tqdm import tqdm
import kagglehub
from typing import List
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE


def download_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("sadmansakib7/ecg-arrhythmia-classification-dataset")
    return pd.read_csv(f"{path}/INCART 2-lead Arrhythmia Database.csv")

def preproces_baseline_forest(df: pd.DataFrame):

    #drop unneccessary columns
    df = df.drop('record', axis = 1)
    df = df.dropna()
    df = df[df["type"] != 'Q']

    df["type"] = df["type"].map({'N': 0, 'VEB': 1, 'SVEB': 2, 'F': 3})

    #remove insignificant class

    X = df.drop('type', axis = 1)
    Y = df['type']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    ros = SMOTE()
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, Y_train_resampled, y_test

def preproces_without_oversampling(df: pd.DataFrame):
    #drop unneccessary columns
    df = df.drop('record', axis = 1)
    df = df.dropna()
    df = df[df["type"] != 'Q']

    df["type"] = df["type"].map({'N': 0, 'VEB': 1, 'SVEB': 2, 'F': 3})

    #remove insignificant class

    X = df.drop('type', axis = 1)
    Y = df['type']

    return train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

import numpy as np

def augment_minority_classes(X, y, multiplier=1):
    """
    X: (N, time_steps, 1)
    multiplier: Ile razy powielić mniejszość? 
                1 = Podwojenie (Oryginał + 1 kopia)
                2 = Potrojenie (Oryginał + 2 kopie)
    """
    time_steps = X.shape[1] 
    
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    
    X_augmented = [X]
    y_augmented = [y]
    
    for cls in unique_classes:
        # Pomijamy klasę większościową
        # if counts[cls] == max_count:
        #     continue
            
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]
        
        n_needed = int(len(X_cls) * multiplier)
        
        print(f"Klasa {cls}: Oryginał {len(X_cls)} -> Generuję {n_needed} nowych próbek (Razem: {len(X_cls) + n_needed})")
        
        if n_needed <= 0: continue
            
        new_samples = []
        
        for _ in range(n_needed):
            idx = np.random.randint(len(X_cls))
            sample = X_cls[idx].copy()
                        
            # A. Skalowanie (Szansa 50%)
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.99, 1.01)
                sample = sample * scale

            # B. Szum (Szansa 50%)
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.01, sample.shape) 
                sample = sample + noise
            
            # C. Przesunięcie (Szansa 50%)
            if np.random.rand() < 0.5:
                shift = np.random.randint(-1, 2)
                if shift != 0:
                    if shift > 0:
                        sample = np.pad(sample, ((shift, 0), (0, 0)), mode='constant', constant_values=0)[:time_steps]
                    else:
                        sample = np.pad(sample, ((0, -shift), (0, 0)), mode='constant', constant_values=0)[-time_steps:]
            
            new_samples.append(sample)
            
        X_augmented.append(np.array(new_samples))
        y_augmented.append(np.full(n_needed, cls))
        
    return np.concatenate(X_augmented), np.concatenate(y_augmented)

def augment_to_target_counts(X, y, target_counts_dict):
    """
    Dostosowuje liczebność klas do wskazanych wartości.
    
    target_counts_dict: Słownik {label_klasy: docelowa_liczebnosc}
    Np. {0: 20000, 2: 15000, 3: 15000}
    
    - Jeśli target > current: augmentacja (szum, skalowanie, shift)
    - Jeśli target < current: losowy subsampling
    - Jeśli klasa nie jest w słowniku: pozostaje bez zmian
    """
    time_steps = X.shape[1] 
    
    X_final = []
    y_final = []
    
    unique_classes, counts = np.unique(y, return_counts=True)
    current_counts = dict(zip(unique_classes, counts))
    
    print(f"Stan początkowy: {current_counts}")
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]
        y_cls = y[cls_indices]
        
        if cls in target_counts_dict:
            target = target_counts_dict[cls]
            current = len(X_cls)
            
            if target < current:
                # SUBSAMPLING - losowy wybór próbek
                indices = np.random.choice(current, size=target, replace=False)
                X_final.append(X_cls[indices])
                y_final.append(y_cls[indices])
                print(f"Klasa {cls}: Subsampling z {current} do {target} (-{current - target} próbek)")
                
            elif target > current:
                # AUGMENTACJA - dodajemy oryginały + nowe próbki
                X_final.append(X_cls)
                y_final.append(y_cls)
                
                n_needed = target - current
                print(f"Klasa {cls}: Augmentacja z {current} do {target} (+{n_needed} próbek)")
                
                new_samples = []
                for _ in range(n_needed):
                    idx = np.random.randint(len(X_cls))
                    sample = X_cls[idx].copy()
                    
                    if np.random.rand() < 0.6:
                        scale = np.random.uniform(0.98, 1.02)
                        sample = sample * scale
                    if np.random.rand() < 0.4:
                        noise = np.random.normal(0, 0.005, sample.shape)
                        sample = sample + noise
                    if np.random.rand() < 0.6:
                        shift = np.random.randint(-2, 3)
                        if shift != 0:
                            if shift > 0:
                                sample = np.pad(sample, ((shift, 0), (0, 0)), mode='constant')[:time_steps]
                            else:
                                sample = np.pad(sample, ((0, -shift), (0, 0)), mode='constant')[-time_steps:]
                    
                    new_samples.append(sample)
                
                X_final.append(np.array(new_samples))
                y_final.append(np.full(n_needed, cls))
            else:
                # target == current
                print(f"Klasa {cls}: Już ma dokładnie {target} próbek. Bez zmian.")
                X_final.append(X_cls)
                y_final.append(y_cls)
        else:
            print(f"Klasa {cls}: Brak celu w słowniku. Pozostaje bez zmian ({len(X_cls)}).")
            X_final.append(X_cls)
            y_final.append(y_cls)

    return np.concatenate(X_final), np.concatenate(y_final)