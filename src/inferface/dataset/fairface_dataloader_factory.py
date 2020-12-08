import logging
import torch
import pytorch_lightning as pl

from torch.nn.functional import one_hot
from typing import Any
from torch.utils.data import DataLoader, random_split
from inferface.dataset.fairface_embeddings_dataset import FairFaceEmbeddingsDataset, KEY_FILE, KEY_EMBEDDING, KEY_AGE, \
    KEY_GENDER, KEY_RACE
from inferface.network import GENDER_2_OUT_SIZE

_logger = logging.getLogger(__name__)


class FairFaceDataModule(pl.LightningDataModule):

    def __init__(self, train_csv: str, test_csv: str, batch_size: int = 128, num_workers: int = 8):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.fairface_train = None
        self.fairface_val = None
        self.fairface_test = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        _logger.info("Loading FairFace data set...")
        self.fairface_test = FairFaceEmbeddingsDataset(self.test_csv)
        fairface_full = FairFaceEmbeddingsDataset(self.train_csv)
        ff_size = len(fairface_full)
        ff_train_size = int(ff_size / 100 * 80)
        ff_val_size = ff_size - ff_train_size
        self.fairface_train, self.fairface_val = random_split(fairface_full, [ff_train_size, ff_val_size])

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        batch[KEY_EMBEDDING] = batch[KEY_EMBEDDING].float().to(device)
        batch[KEY_AGE] = batch[KEY_AGE].long().to(device)
        batch[KEY_GENDER] = one_hot(batch[KEY_GENDER], GENDER_2_OUT_SIZE).float().to(device)
        batch[KEY_RACE] = batch[KEY_RACE].to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self.fairface_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.fairface_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.fairface_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
