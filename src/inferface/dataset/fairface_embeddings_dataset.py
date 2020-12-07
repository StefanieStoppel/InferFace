import csv

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class FairFaceEmbeddingsDataset(Dataset):
    """FairFace dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file containing the FairFace embeddings.
                on a sample.
        """
        self.fair_face_embeddings = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL, delimiter=',')

    def __len__(self):
        return len(self.fair_face_embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.fair_face_embeddings.iloc[idx, :]
        embedding = np.array(row["Embedding"])
        image_path = row["Image Path"]
        age = row["Age"]
        gender = row["Gender"]
        race = row["Race"]
        sample = {'embedding': embedding, 'image_path': image_path, 'age': age, 'gender': gender, 'race': race}
        print(sample)
        return sample
