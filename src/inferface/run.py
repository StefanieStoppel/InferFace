import os

import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

from inferface.config import LossNames
from inferface.dataset.fairface_dataloader_factory import FairFaceDataModule
from inferface.network import AgeGenderRaceClassifier
from src import ROOT_DIR


def run():
    # init model
    lr = 1e-3
    dropout = 0.3
    model = AgeGenderRaceClassifier(lr=lr, dropout=dropout)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(ROOT_DIR, 'lightning_logs/'))

    train_csv_path = '/home/steffi/dev/independent_study/fairface_margin_025/embeddings/train_labels_and_embeddings.csv'
    test_csv_path = '/home/steffi/dev/independent_study/fairface_margin_025/embeddings/val_labels_and_embeddings.csv'
    fairface_data_module = FairFaceDataModule(train_csv_path, test_csv_path)

    # Early stopping based on val_loss
    early_stop_callback = EarlyStopping(
        monitor=LossNames.VAL_LOSS_TOTAL.value,
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer(gpus=1, logger=tb_logger, callbacks=[early_stop_callback])
    trainer.fit(model, datamodule=fairface_data_module)


if __name__ == "__main__":
    run()
