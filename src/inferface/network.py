import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import cross_entropy

VGG_FACE2_EMBEDDING_SIZE = 512
AGE_9_OUT_SIZE = 9
GENDER_2_OUT_SIZE = 2
RACE_7_OUT_SIZE = 7


class AgeGenderRaceClassifier(LightningModule):
    def __init__(self, input_size=VGG_FACE2_EMBEDDING_SIZE, output_size_age=AGE_9_OUT_SIZE,
                 output_size_gender=GENDER_2_OUT_SIZE, output_size_race=RACE_7_OUT_SIZE):
        super().__init__()
        self.fc_age = nn.Sequential(nn.Linear(input_size, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 64),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(64, output_size_age),
                                    nn.LogSoftmax(dim=1))
        self.fc_gender = nn.Sequential(nn.Linear(input_size, 256),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Linear(256, 64),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Linear(64, output_size_gender),
                                       nn.LogSoftmax(dim=1))
        self.fc_race = nn.Sequential(nn.Linear(input_size, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 64),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(64, output_size_race),
                                     nn.LogSoftmax(dim=1))
        self.criterion_binary = nn.BCELoss()
        self.criterion_multioutput = nn.CrossEntropyLoss()

    def forward(self, x):
        age = self.fc_age(x)
        gender = torch.sigmoid(self.fc_gender(x))
        race = self.fc_race(x)
        return age, gender, race

    def _loop(self, batch, batch_idx):
        print(batch)
        image_path, embedding, age, gender, race = batch['image_path'], batch['embedding'], batch['age'], batch['gender'], batch['race']
        print(image_path)
        embedding = embedding.squeeze().type(torch.FloatTensor)
        print(embedding)
        age_hat, gender_hat, race_hat = self(embedding)

        loss_age = self.criterion_multioutput(age_hat, age.squeeze().type(torch.FloatTensor))
        loss_gender = self.criterion_binary(gender_hat, gender.squeeze().type(torch.FloatTensor))
        loss_race = self.criterion_multioutput(race_hat, race.squeeze().type(torch.FloatTensor))
        loss = loss_age + loss_gender + loss_race
        return loss

    def training_step(self, batch, batch_idx):
        return self._loop(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._loop(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._loop(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
