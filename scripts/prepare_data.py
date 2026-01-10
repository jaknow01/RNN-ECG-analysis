from tqdm import tqdm
import kagglehub
from typing import List
import pandas as pd

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

