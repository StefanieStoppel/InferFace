from torch import nn

VGG_FACE2_EMBEDDING_SIZE = 512
AGE_9_OUT_SIZE = 9
GENDER_2_OUT_SIZE = 2
RACE_7_OUT_SIZE = 7


class VggFace2CustomHead(nn.Module):
    def __init__(self, input_size=VGG_FACE2_EMBEDDING_SIZE, output_size=None):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64, output_size),
                                nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.fc(x)
        return out


class Age9Head(VggFace2CustomHead):
    def __init__(self, input_size=VGG_FACE2_EMBEDDING_SIZE, output_size=AGE_9_OUT_SIZE):
        super().__init__(input_size, output_size)


class Gender2Head(VggFace2CustomHead):
    def __init__(self, input_size=VGG_FACE2_EMBEDDING_SIZE, output_size=GENDER_2_OUT_SIZE):
        super().__init__(input_size, output_size)


class Race7Head(VggFace2CustomHead):
    def __init__(self, input_size=VGG_FACE2_EMBEDDING_SIZE, output_size=RACE_7_OUT_SIZE):
        super().__init__(input_size, output_size)
