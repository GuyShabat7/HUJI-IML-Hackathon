import joblib
import numpy as np
import pandas as pd

from base_model import BaseModel
from model import BikeDemandModel


class Model(BaseModel):
    """
    Fixed evaluator-facing wrapper.

    The evaluator will call:

        model = Model()
        model.load("weights.joblib")
        preds = model.predict(hidden_test_df)

    You should usually NOT edit this file.
    Put your model logic in model.py.
    Put your training logic in train.py.
    """

    def __init__(self):
        self.model = BikeDemandModel()

    def load(self, weights_path: str) -> None:
        artifacts = joblib.load(weights_path)
        self.model.load_artifacts(artifacts)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test_df)