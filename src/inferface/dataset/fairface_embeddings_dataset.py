import csv

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from inferface.config import FairFaceColumnKeys, FairFaceLabels


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
        return FairFaceLabels.AGE_9_LABELS.value.index(age_class_name)

    @staticmethod
    def _get_gender_class_idx(gender_class_name: str):
        return FairFaceLabels.GENDER_2_LABELS.value.index(gender_class_name)

    @staticmethod
    def _get_race_class_idx(race_class_name: str):
        return FairFaceLabels.RACE_7_LABELS.value.index(race_class_name)

    def __len__(self):
        return len(self.fair_face_embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.fair_face_embeddings.iloc[idx, :]
        sample = {FairFaceColumnKeys.KEY_EMBEDDING.value: row[FairFaceColumnKeys.KEY_EMBEDDING.value],
                  FairFaceColumnKeys.KEY_FILE.value: row[FairFaceColumnKeys.KEY_FILE.value],
                  FairFaceColumnKeys.KEY_AGE.value: row[FairFaceColumnKeys.KEY_AGE.value],
                  FairFaceColumnKeys.KEY_GENDER.value: row[FairFaceColumnKeys.KEY_GENDER.value],
                  FairFaceColumnKeys.KEY_RACE.value: row[FairFaceColumnKeys.KEY_RACE.value]}
        return sample
