from enum import Enum


class NetworkLayerSizes(Enum):
    """ Network input & output layer size definitions """
    INPUT: int = 512
    AGE_9_OUTPUT: int = 9
    GENDER_2_OUTPUT: int = 2
    RACE_7_OUTPUT: int = 7


class LossNames(Enum):
    """ Loss name postfix definitions for logging """
    LOSS_TOTAL: str = "loss_total"
    LOSS_AGE: str = "loss_age"
    LOSS_GENDER: str = "loss_gender"
    LOSS_RACE: str = "loss_race"


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
