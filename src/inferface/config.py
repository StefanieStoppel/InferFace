from enum import Enum


class NetworkLayerSizes(Enum):
    """ Network input & output layer size definitions """
    INPUT: int = 512
    AGE_9_OUTPUT: int = 9
    GENDER_2_OUTPUT: int = 2
    RACE_7_OUTPUT: int = 7


class LossNames(Enum):
    """ Loss name definitions for logging """
    TRAIN_LOSS_TOTAL: str = "train_loss_total"
    TRAIN_LOSS_AGE: str = "train_loss_age"
    TRAIN_LOSS_GENDER: str = "train_loss_gender"
    TRAIN_LOSS_RACE: str = "train_loss_race"

    VAL_LOSS_TOTAL: str = "val_loss_total"
    VAL_LOSS_AGE: str = "val_loss_age"
    VAL_LOSS_GENDER: str = "val_loss_gender"
    VAL_LOSS_RACE: str = "val_loss_race"

    TEST_LOSS_TOTAL: str = "test_loss_total"
    TEST_LOSS_AGE: str = "test_loss_age"
    TEST_LOSS_GENDER: str = "test_loss_gender"
    TEST_LOSS_RACE: str = "test_loss_race"


class FairFaceColumnKeys(Enum):
    KEY_EMBEDDING: str = 'embedding'
    KEY_FILE: str = 'file'
    KEY_AGE: str = 'age'
    KEY_GENDER: str = 'gender'
    KEY_RACE: str = 'race'


class FairFaceLabels(Enum):
    AGE_9_LABELS: str = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
    GENDER_2_LABELS: str = ['Male', 'Female']
    RACE_7_LABELS: str = ['White', 'Black', 'Latino_Hispanic', 'East Asian',
                          'Southeast Asian', 'Indian', 'Middle Eastern']
