import pytorch_lightning as pl

from inferface.dataset.fairface_dataloader_factory import FairFaceEmbeddingDataLoaderFactory
from inferface.network import AgeGenderRaceClassifier


def train():
    # init model
    model = AgeGenderRaceClassifier()

    train_loader = FairFaceEmbeddingDataLoaderFactory('/home/steffi/dev/independent_study/InferFace/data/embeddings'
                                                      '/embeddings.csv').train_loader(batch_size=128)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    print('hi')
    train()
