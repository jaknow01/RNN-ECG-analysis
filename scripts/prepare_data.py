from tqdm import tqdm
import kagglehub
from typing import List
import pandas as pd

from sklearn.model_selection import train_test_split


def download_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("sadmansakib7/ecg-arrhythmia-classification-dataset")
    return pd.read_csv(f"{path}/INCART 2-lead Arrhythmia Database.csv")

def preproces_baseline_forest(df: pd.DataFrame):
    df = df.drop('record', axis = 1)
    df = df.dropna()
    df["type"] = df["type"].map({'N': 0, 'VEB': 1, 'SVEB': 2, 'F': 3, 'Q': 4})

    X = df.drop('type', axis = 1)
    Y = df['type']

    return train_test_split(X, Y, test_size=0.2, random_state=42)

