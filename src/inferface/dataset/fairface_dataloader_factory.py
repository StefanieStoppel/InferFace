import logging
import torch
import pytorch_lightning as pl

from torch.nn.functional import one_hot
from typing import Any
from torch.utils.data import DataLoader, random_split

from inferface.config import FairFaceColumnKeys, NetworkLayerSizes
from inferface.dataset.fairface_embeddings_dataset import FairFaceEmbeddingsDataset

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
        batch[FairFaceColumnKeys.KEY_EMBEDDING.value] = batch[FairFaceColumnKeys.KEY_EMBEDDING.value].float().to(device)
        batch[FairFaceColumnKeys.KEY_AGE.value] = batch[FairFaceColumnKeys.KEY_AGE.value].long().to(device)
        batch[FairFaceColumnKeys.KEY_GENDER.value] = one_hot(batch[FairFaceColumnKeys.KEY_GENDER.value],
                                                       NetworkLayerSizes.GENDER_2_OUTPUT.value).float().to(device)
        batch[FairFaceColumnKeys.KEY_RACE.value] = batch[FairFaceColumnKeys.KEY_RACE.value].to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self.fairface_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.fairface_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.fairface_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
