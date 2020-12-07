from torch.utils.data import DataLoader

from inferface.dataset.fairface_embeddings_dataset import FairFaceEmbeddingsDataset


class FairFaceEmbeddingDataLoaderFactory:

    def __init__(self, dataset_path='/home/steffi/dev/independent_study/fairface_margin_025/embeddings/train.csv'):
        self.fairFaceDataset = FairFaceEmbeddingsDataset(dataset_path)

    def train_loader(self, batch_size=4, num_workers=0):
        return DataLoader(self.fairFaceDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def val_loader(self, batch_size=512, num_workers=0):
        return DataLoader(self.fairFaceDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def test_loader(self, batch_size=1, num_workers=0):
        return DataLoader(self.fairFaceDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
