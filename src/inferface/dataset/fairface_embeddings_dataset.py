import csv

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


KEY_EMBEDDING = 'embedding'
KEY_FILE = 'file'
KEY_AGE = 'age'
KEY_GENDER = 'gender'
KEY_RACE = 'race'

AGE_9_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
GENDER_2_LABELS = ['Male', 'Female']
RACE_7_LABELS = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']


class FairFaceEmbeddingsDataset(Dataset):
    """FairFace dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string):  Path to the csv file containing the FairFace embeddings.
        """
        self.fair_face_embeddings = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL, delimiter=',')
        # convert embedding array saved as string to float array
        self.fair_face_embeddings["embedding"] = self.fair_face_embeddings["embedding"].apply(
            lambda embedding: np.array(','.join(','.join(embedding[2:-2].split(' \n')).split()).split(','),
                                       dtype=float))
        # convert other columns to class indeces so CrossEntropyLoss can be used
        self.fair_face_embeddings["age"] = self.fair_face_embeddings["age"].apply(
            lambda age: self._get_age_class_idx(age))
        self.fair_face_embeddings["gender"] = self.fair_face_embeddings["gender"].apply(
            lambda gender: self._get_gender_class_idx(gender))
        self.fair_face_embeddings["race"] = self.fair_face_embeddings["race"].apply(
            lambda race: self._get_race_class_idx(race))

    @staticmethod
    def _get_age_class_idx(age_class_name: str):
        return AGE_9_LABELS.index(age_class_name)

    @staticmethod
    def _get_gender_class_idx(gender_class_name: str):
        return GENDER_2_LABELS.index(gender_class_name)

    @staticmethod
    def _get_race_class_idx(race_class_name: str):
        return RACE_7_LABELS.index(race_class_name)

    def __len__(self):
        return len(self.fair_face_embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.fair_face_embeddings.iloc[idx, :]
        sample = {KEY_EMBEDDING: row[KEY_EMBEDDING],
                  KEY_FILE: row[KEY_FILE],
                  KEY_AGE: row[KEY_AGE],
                  KEY_GENDER: row[KEY_GENDER],
                  KEY_RACE: row[KEY_RACE]}
        return sample
