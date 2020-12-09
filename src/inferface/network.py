import torch
from pytorch_lightning import LightningModule
from torch import nn

from inferface.config import NetworkLayerSizes, LossNames, FairFaceColumnKeys


class AgeGenderRaceClassifier(LightningModule):
    def __init__(self,
                 input_size: int = NetworkLayerSizes.INPUT.value,
                 output_size_age: int = NetworkLayerSizes.AGE_9_OUTPUT.value,
                 output_size_gender: int = NetworkLayerSizes.GENDER_2_OUTPUT.value,
                 output_size_race: int = NetworkLayerSizes.RACE_7_OUTPUT.value,
                 lr: float = 1e-3,
                 dropout: float = 0.4
                 ):
        super().__init__()
        self.lr = lr
        self.dropout = dropout

        self.fc_age = nn.Sequential(nn.Linear(input_size, 256),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(256, 64),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(64, output_size_age),
                                    nn.LogSoftmax(dim=1))
        self.fc_gender = nn.Sequential(nn.Linear(input_size, 256),
                                       nn.ReLU(),
                                       nn.Dropout(self.dropout),
                                       nn.Linear(256, 64),
                                       nn.ReLU(),
                                       nn.Dropout(self.dropout),
                                       nn.Linear(64, output_size_gender),
                                       nn.Sigmoid())
        self.fc_race = nn.Sequential(nn.Linear(input_size, 256),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(256, 64),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(64, output_size_race),
                                     nn.LogSoftmax(dim=1))
        self.criterion_binary = nn.BCELoss()
        self.criterion_multioutput = nn.CrossEntropyLoss()

    def forward(self, x):
        age = self.fc_age(x)
        gender = self.fc_gender(x)
        race = self.fc_race(x)
        return age, gender, race

    def _loop(self, batch, batch_idx, stage):
        image_path, embedding, age, gender, race = batch[FairFaceColumnKeys.KEY_FILE.value], \
                                                   batch[FairFaceColumnKeys.KEY_EMBEDDING.value], \
                                                   batch[FairFaceColumnKeys.KEY_AGE.value], \
                                                   batch[FairFaceColumnKeys.KEY_GENDER.value], \
                                                   batch[FairFaceColumnKeys.KEY_RACE.value]
        age_hat, gender_hat, race_hat = self(embedding)

        loss_age = self.criterion_multioutput(age_hat, age)
        self.log(f"{stage}_{LossNames.LOSS_AGE.value}", loss_age)
        loss_gender = self.criterion_binary(gender_hat, gender)
        self.log(f"{stage}_{LossNames.LOSS_GENDER.value}", loss_gender)
        loss_race = self.criterion_multioutput(race_hat, race)
        self.log(f"{stage}_{LossNames.LOSS_RACE.value}", loss_race)
        loss = loss_age + loss_gender + loss_race
        self.log(f"{stage}_{LossNames.LOSS_TOTAL.value}", loss)
        return loss

    def training_step(self, batch, batch_idx):
        stage = 'train'
        return self._loop(batch, batch_idx, stage)

    def validation_step(self, batch, batch_idx):
        stage = 'val'
        return self._loop(batch, batch_idx, stage)

    def test_step(self, batch, batch_idx):
        stage = 'test'
        return self._loop(batch, batch_idx, stage)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
