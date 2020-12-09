import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import one_hot
from inferface.dataset.fairface_embeddings_dataset import KEY_FILE, KEY_EMBEDDING, KEY_AGE, KEY_GENDER, KEY_RACE

VGG_FACE2_EMBEDDING_SIZE = 512
AGE_9_OUT_SIZE = 9
GENDER_2_OUT_SIZE = 2
RACE_7_OUT_SIZE = 7


class AgeGenderRaceClassifier(LightningModule):
    def __init__(self,
                 input_size: int = VGG_FACE2_EMBEDDING_SIZE,
                 output_size_age: int = AGE_9_OUT_SIZE,
                 output_size_gender: int = GENDER_2_OUT_SIZE,
                 output_size_race: int = RACE_7_OUT_SIZE,
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

    def _loop(self, batch, batch_idx):
        image_path, embedding, age, gender, race = batch[KEY_FILE], batch[KEY_EMBEDDING], batch[KEY_AGE], \
                                                   batch[KEY_GENDER], batch[KEY_RACE]
        age_hat, gender_hat, race_hat = self(embedding)

        loss_age = self.criterion_multioutput(age_hat, age)
        self.log('train_loss_age', loss_age)
        loss_gender = self.criterion_binary(gender_hat, gender)
        self.log('train_loss_gender', loss_gender)
        loss_race = self.criterion_multioutput(race_hat, race)
        self.log('train_loss_race', loss_race)

        loss = loss_age + loss_gender + loss_race
        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self._loop(batch, batch_idx)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._loop(batch, batch_idx)
        self.log('val_loss', val_loss, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self._loop(batch, batch_idx)
        self.log('test_loss', test_loss, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
